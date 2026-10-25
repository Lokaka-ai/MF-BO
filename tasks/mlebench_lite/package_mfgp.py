#!/usr/bin/env python3
"""Build a deterministic, source-only MF-GP bundle from a clean Git commit."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def build(root, output):
    root, output = Path(root), Path(output)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    if git("status", "--porcelain").strip():
        raise RuntimeError("Commit source changes before building an experiment bundle")
    commit = git("rev-parse", "HEAD").decode().strip()
    prefix = "mila-mfgp14-" + commit[:7]
    source = git("archive", "--format=tar", "HEAD")
    hashes = {}
    packed = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(source), mode="r:") as src,
        tarfile.open(fileobj=packed, mode="w:") as dst,
    ):
        for member in src.getmembers():
            if member.isfile():
                data = src.extractfile(member).read()
                hashes[member.name] = hashlib.sha256(data).hexdigest()
                stream = io.BytesIO(data)
            elif member.isdir():
                stream = None
            else:
                raise RuntimeError(f"Unexpected non-file source entry: {member.name}")
            member.name = prefix + "/" + member.name
            member.mtime = 0
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            dst.addfile(member, stream)
        for name, payload in [
            ("SOURCE_SHA256.json", hashes),
            ("BUNDLE_INFO.json", {"commit": commit, "mode": "mf_gp"}),
        ]:
            data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
            info = tarfile.TarInfo(prefix + "/" + name)
            info.size, info.mode = len(data), 0o644
            dst.addfile(info, io.BytesIO(data))
    output.mkdir(parents=True, exist_ok=True)
    path = output / (prefix + ".tar.gz")
    compressed = gzip.compress(packed.getvalue(), mtime=0)
    if path.exists() and path.read_bytes() != compressed:
        raise RuntimeError("Existing bundle differs; refusing to overwrite it")
    path.write_bytes(compressed)
    digest = hashlib.sha256(compressed).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return {
        "archive": str(path),
        "sha256": digest,
        "commit": commit,
        "source_files": len(hashes),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(ROOT, args.output_dir), indent=2))
