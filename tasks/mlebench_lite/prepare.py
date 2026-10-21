#!/usr/bin/env python3
"""Prepare the official Lite split using a Kaggle access token and verified ZIPs.

Only the download transport is replaced: all splits, preparation functions,
archive checksums, public/private checksums and graders remain upstream's.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "mle-bench"
DEFAULT_DATA = ROOT / "data" / "mlebench"
STATE = ROOT / ".cache" / "mlebench-lite"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


@contextmanager
def cluster_lock(path, timeout=10800):
    """Atomic mkdir works across this BeeGFS mount; flock is node-local here."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            path.mkdir(mode=0o700)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for {path}; inspect its owner.json before removing a stale lock")
            time.sleep(5)
    owner = path / "owner.json"
    try:
        owner.write_text(json.dumps({"hostname": socket.gethostname(), "pid": os.getpid(),
                                     "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                     "created_at": timestamp()}) + "\n")
        yield
    finally:
        owner.unlink(missing_ok=True)
        path.rmdir()


def ranged_download(url, total, partial, connections=8, chunk_size=512 * 1024 * 1024):
    """Resume bounded byte ranges, then assemble exactly the official archive."""
    parts = partial.with_suffix(".parts")
    parts.mkdir(exist_ok=True)
    ranges = [(start, min(start + chunk_size, total) - 1) for start in range(0, total, chunk_size)]
    # Reuse complete chunks from a previous sequential transfer.
    if partial.exists() and partial.stat().st_size < total:
        with partial.open("rb") as source:
            for start, end in ranges:
                if end >= partial.stat().st_size:
                    break
                part = parts / f"{start:012d}"
                if not part.exists():
                    with part.open("wb") as handle:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            data = source.read(min(8 * 1024 * 1024, remaining))
                            handle.write(data)
                            remaining -= len(data)

    def fetch(bounds):
        start, end = bounds
        part = parts / f"{start:012d}"
        expected = end - start + 1
        for attempt in range(4):
            size = part.stat().st_size if part.exists() else 0
            if size == expected:
                return part
            if size > expected:
                raise RuntimeError("Invalid cached download range length")
            try:
                with requests.get(url, headers={"Range": f"bytes={start + size}-{end}"}, stream=True, timeout=(30, 120)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start + size}-{end}/{total}":
                        raise RuntimeError(f"Unexpected ranged download response: HTTP {response.status_code}")
                    with part.open("ab") as handle:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            handle.write(chunk)
                if part.stat().st_size == expected:
                    return part
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError("Ranged download failed after four attempts") from None
            time.sleep(2 * (attempt + 1))
        raise RuntimeError("Incomplete download range")

    with ThreadPoolExecutor(max_workers=connections) as pool:
        completed = list(pool.map(fetch, ranges))
    with partial.open("wb") as handle:
        for part in completed:
            with part.open("rb") as source:
                shutil.copyfileobj(source, handle, 8 * 1024 * 1024)
    return parts


def download(task, data_dir, token_file):
    target = data_dir / task / f"{task}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    checksums = yaml.safe_load((UPSTREAM / "mlebench" / "competitions" / task / "checksums.yaml").read_text())
    expected = checksums["zip"]
    with cluster_lock(target.parent / ".download.guard"):
        if target.exists():
            if digest(target) == expected:
                return target
            raise RuntimeError(f"Existing archive checksum mismatch for {task}; preserved for inspection")
        token = token_file.read_text().strip()
        if not token.startswith("KGAT_") or any(c.isspace() for c in token):
            raise ValueError("Expected a single Kaggle access token in the credential file")
        partial = target.with_suffix(".zip.partial")
        url = "https://www.kaggle.com/api/v1/competitions/data/download-all/" + task
        with requests.get(url, headers={"Authorization": "Bearer " + token, "Range": "bytes=0-0"}, stream=True, timeout=(30, 120)) as probe:
            total = int(probe.headers.get("Content-Range", "/0").rsplit("/", 1)[-1])
            if probe.status_code == 206 and total >= 1024 ** 3:
                parts = ranged_download(probe.url, total, partial, connections=16 if total >= 32 * 1024 ** 3 else 8)
                actual = digest(partial)
                if actual != expected:
                    raise RuntimeError(f"Archive checksum mismatch for {task}: expected {expected}, got {actual}")
                partial.replace(target)
                shutil.rmtree(parts)
                return target
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Authorization": "Bearer " + token}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                    if response.status_code not in (200, 206):
                        hint = " (check competition rules and token permissions)" if response.status_code == 403 else ""
                        raise RuntimeError(f"Kaggle download for {task}: HTTP {response.status_code}{hint}")
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with partial.open(mode) as handle:
                        for chunk in response.iter_content(8 * 1024 * 1024):
                            handle.write(chunk)
                actual = digest(partial)
                if actual != expected:
                    raise RuntimeError(f"Archive checksum mismatch for {task}: expected {expected}, got {actual}")
                partial.replace(target)
                return target
            except requests.RequestException as error:
                if attempt == 2:
                    raise RuntimeError(f"Download failed for {task}: {type(error).__name__}") from None
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"Download failed for {task}")


def prepare_one(task, data_dir):
    directory = data_dir / task
    directory.mkdir(parents=True, exist_ok=True)
    with cluster_lock(directory / ".prepare.guard"):
        _prepare_one_locked(task, data_dir)


def _prepare_one_locked(task, data_dir):
    # Import after setting up the project environment. No credential is passed
    # to the preparer or any subsequently generated ML program.
    from mlebench import data
    from mlebench.registry import registry

    archive = data_dir / task / f"{task}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Download the archive first: {archive}")

    def cached_download(competition_id, download_dir, quiet=False, force=False):
        if competition_id != task or Path(download_dir).resolve() != archive.parent.resolve() or force:
            raise ValueError("Unexpected download request from upstream preparer")
        return archive

    data.download_dataset = cached_download
    competition = registry.set_data_dir(data_dir).get_competition(task)
    if data.is_dataset_prepared(competition):
        expected = yaml.safe_load(competition.checksums.read_text())
        actual = {"zip": digest(archive), "public": data.generate_checksums(competition.public_dir),
                  "private": data.generate_checksums(competition.private_dir)}
        if actual != expected:
            raise RuntimeError(f"Existing prepared data checksum mismatch for {task}")
        print(f"Verified existing prepared data: {task}")
        return
    # No --skip-verification and no --overwrite-checksums: fail on any mismatch.
    data.download_and_prepare_dataset(competition, keep_raw=False)
    assert data.is_dataset_prepared(competition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--token-file", type=Path, default=ROOT.parent / "kaggle_token")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--status-file", type=Path, help="Use a separate status report for an independently scheduled subset")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent downloads; preparation is serial")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download-only", action="store_true")
    modes.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--wait-for-downloads", action="store_true", help="Wait for the separate downloader (up to 3 hours per archive)")
    parser.add_argument("--prepare-one", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.data_dir = args.data_dir.resolve()
    if args.prepare_one:
        prepare_one(args.prepare_one, args.data_dir)
        return
    tasks = (ROOT / "config" / "mlebench_lite22.txt").read_text().split()
    official = (UPSTREAM / "experiments" / "splits" / "low.txt").read_text().split()
    if tasks != official:
        raise RuntimeError("Repository task list differs from the official Lite split")
    if args.tasks:
        if set(args.tasks) - set(tasks):
            parser.error("All tasks must belong to the official Lite split")
        tasks = args.tasks
    else:
        first = ["leaf-classification", "spooky-author-identification"]
        tasks = first + [task for task in tasks if task not in first]
        if args.prepare_only:
            tasks.remove("siim-isic-melanoma-classification")
            tasks.append("siim-isic-melanoma-classification")
    STATE.mkdir(parents=True, exist_ok=True)
    status_file = args.status_file or STATE / ("download-status.json" if args.download_only else "prepare-status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    status.update(data_dir=str(args.data_dir), upstream_commit=subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip())
    status.setdefault("tasks", {})
    status_lock = threading.Lock()
    prepare_lock = threading.Lock()

    def update(task, **fields):
        with status_lock:
            status["tasks"].setdefault(task, {}).update(fields, updated_at=timestamp())
            temp = status_file.with_suffix(".tmp")
            temp.write_text(json.dumps(status, indent=2) + "\n")
            temp.replace(status_file)

    def worker(task):
        started = time.monotonic()
        try:
            if args.prepare_only and args.wait_for_downloads:
                archive = args.data_dir / task / f"{task}.zip"
                while not archive.is_file():
                    update(task, state="waiting_for_download")
                    downloads = STATE / "download-status.json"
                    state = json.loads(downloads.read_text()) if downloads.exists() else {}
                    downloaded = state.get("tasks", {}).get(task, {})
                    if downloaded.get("state") == "failed":
                        raise RuntimeError(downloaded["error"])
                    if time.monotonic() - started > 10800:
                        raise TimeoutError(f"Download wait exceeded three hours: {task}")
                    time.sleep(10)
            if not args.prepare_only:
                update(task, state="downloading")
                print(f"[{timestamp()}] downloading {task}", flush=True)
                archive = download(task, args.data_dir, args.token_file)
                update(task, state="downloaded", archive_bytes=archive.stat().st_size, zip_checksum_verified=True)
            if not args.download_only:
                with prepare_lock:
                    update(task, state="preparing")
                    print(f"[{timestamp()}] preparing {task}", flush=True)
                    log = STATE / f"{task}.prepare.log"
                    env = os.environ.copy()
                    env.pop("KAGGLE_API_TOKEN", None)
                    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", TF_CPP_MIN_LOG_LEVEL="3", CUDA_VISIBLE_DEVICES="")
                    with log.open("a") as handle:
                        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--data-dir", str(args.data_dir), "--prepare-one", task],
                            stdout=handle, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
                    if result.returncode:
                        raise RuntimeError(f"Official preparation failed (exit {result.returncode}); see {log}")
                    update(task, state="prepared", public_private_checksums_verified=True, log=str(log))
            update(task, elapsed_seconds=round(time.monotonic()-started, 1), error=None)
            print(f"[{timestamp()}] finished {task}", flush=True)
            return True
        except Exception as error:
            reason = f"Kaggle transport error: {type(error).__name__}" if isinstance(error, requests.RequestException) else str(error)
            update(task, state="failed", error=reason)
            print(f"[{timestamp()}] FAILED {task}: {reason}", flush=True)
            return False

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, tasks))
    print(f"{sum(results)}/{len(tasks)} tasks succeeded. Status: {status_file}", flush=True)
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
