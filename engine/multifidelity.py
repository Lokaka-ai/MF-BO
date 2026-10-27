"""A frozen half-training view and a single, drained switch to full training."""

import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys


MODES = ("carry_over", "reset")
SWITCH_FRACTION = 0.40


def switch_due(elapsed_seconds, time_limit, budgeted_steps, step_limit):
    """Drain cheap work once either 40% budget threshold is reached."""
    return (elapsed_seconds >= SWITCH_FRACTION * float(time_limit)
            or budgeted_steps >= max(1, math.ceil(SWITCH_FRACTION * int(step_limit))))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_view(protocol_dir, workspace, seed):
    """Sample IDs uniformly once; preserve original CSV order and all V/test rows."""
    csv.field_size_limit(sys.maxsize)
    protocol_dir, workspace = Path(protocol_dir), Path(workspace)
    source = protocol_dir / "public"
    view = workspace / "input_half"
    manifest_path = workspace / "multifidelity_subset.json"
    with (source / "train.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError("Frozen T must contain sample_id")
        ids = [row["sample_id"] for row in reader]
    if len(ids) < 2 or len(ids) != len(set(ids)) or any(not key for key in ids):
        raise ValueError("Frozen T requires at least two unique, nonempty sample IDs")
    chosen = set(random.Random(seed).sample(ids, len(ids) // 2))
    training_ids = set(ids)
    expected = {"source_sha256": file_hash(source / "train.csv"), "seed": seed,
                "n_total": len(ids), "n_selected": len(chosen),
                "selected_ids_sha256": hashlib.sha256(json.dumps(sorted(chosen)).encode()).hexdigest()}
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen half-training sample changed on resume")
        if not (view / "train.csv").is_file() or file_hash(view / "train.csv") != saved.get("train_sha256"):
            raise ValueError("Frozen half-training CSV changed on resume")
        return view, saved
    if view.exists() or view.is_symlink():
        raise ValueError("Unrecorded half-training view exists")
    view.mkdir(parents=True)
    with (source / "train.csv").open(newline="", encoding="utf-8") as src, (view / "train.csv").open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["sample_id"] in chosen:
                writer.writerow(row)
    for name in ("validation.csv", "test.csv", "sample_submission.csv"):
        (view / name).symlink_to(source / name)
    assets_path = protocol_dir / "assets.csv"
    if assets_path.exists():
        with assets_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["sample_id"] not in chosen:
                    # Omitted T assets are hidden; full V/test assets remain visible.
                    if row["role"] == "target" or row["sample_id"] in training_ids:
                        continue
                relative = Path(row["public_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid frozen asset path")
                target = view / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source / relative)
    manifest = {**expected, "train_sha256": file_hash(view / "train.csv")}
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)
    return view, manifest


def bind_view(workspace, target, allowed_previous=()):
    destination = Path(workspace) / "input"
    target = Path(target).resolve()
    if destination.is_symlink():
        if destination.resolve() not in {Path(path).resolve() for path in allowed_previous} | {target}:
            raise ValueError("Workspace input is bound to an unrelated data view")
        temporary = destination.with_name("input.next")
        if temporary.exists() or temporary.is_symlink():
            raise ValueError("Stale input-view switch")
        temporary.symlink_to(target, target_is_directory=True)
        temporary.replace(destination)
    else:
        if destination.exists():
            if any(destination.iterdir()):
                raise ValueError("Workspace contains raw input data")
            destination.rmdir()
        destination.symlink_to(target, target_is_directory=True)


def full_node_ids(records):
    result = set()
    for row in records:
        snapshot = (row.metadata or {}).get("node_snapshot") or {}
        node = snapshot.get("node") or {}
        if node.get("training_fraction") == 1.0 and node.get("id") == row.child_node_id:
            result.add(node["id"])
    return result
