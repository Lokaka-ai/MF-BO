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


def cheap_execution_seconds_left(mode, phase, node, start_time, time_limit, now):
    """Bound cheap work that must finish before full-data work can begin.

    The staged modes drain their half-data workers at the 40% clock point.
    MF-GP is adaptive, so only its mandatory initial cheap drafts use this
    deadline; later controller-selected cheap actions retain their own guard.
    """
    fraction = getattr(node, "training_fraction", None)
    if (mode == "mf_gp" and fraction not in (0.1, 0.2, 0.5)) or (mode != "mf_gp" and fraction != 0.5):
        return None
    if mode in MODES and phase == "half":
        pass
    elif mode == "mf_gp" and (getattr(node, "phase1_selection_features", None) or {}).get("mf", {}).get("initial"):
        pass
    else:
        return None
    return max(0.0, SWITCH_FRACTION * float(time_limit) - (float(now) - float(start_time)))


def completed_full_winner(journal):
    """Use the same full-fidelity eligibility as each mode's finalizer."""
    state = journal.evaluation_state or {}
    mode = state.get("multifidelity_mode")
    if mode == "mf_gp":
        selector = (journal.parent_selector_state or {}).get("study7_state") or {}
        rows = [o for o in selector.get("observations", [])
                if o.get("valid") and o.get("fidelity") == 1.0
                and o.get("budget_eligible", True)]
        if not rows:
            return None
        best = max(rows, key=lambda o: o["utility"])
        return next((n for n in journal.nodes if n.id == best["program_id"]), None)
    if mode in MODES:
        eligible = [n for n in journal.nodes
                    if n.stage != "root" and n.is_valid is True
                    and n.is_buggy is False and n.training_fraction == 1.0
                    and n.evaluation_protocol_id == state.get("protocol_id")
                    and n.evaluation_result and n.evaluation_result.get("valid")
                    and n.metric and n.metric.value == n.evaluation_result.get("metric")]
        return max(eligible, key=lambda n: n.metric, default=None)
    return None


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


def prepare_view(protocol_dir, workspace, seed, fraction=0.5):
    """Sample IDs uniformly once; preserve original CSV order and all V/test rows."""
    from engine.mf_gp.settings import validate_low_fraction
    fraction = validate_low_fraction(fraction)
    csv.field_size_limit(sys.maxsize)
    protocol_dir, workspace = Path(protocol_dir), Path(workspace)
    source = protocol_dir / "public"
    view = workspace / ("input_half" if fraction == 0.5 else f"input_fraction_{fraction}")
    manifest_path = workspace / "multifidelity_subset.json"
    with (source / "train.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError("Frozen T must contain sample_id")
        ids = [row["sample_id"] for row in reader]
    if len(ids) < 2 or len(ids) != len(set(ids)) or any(not key for key in ids):
        raise ValueError("Frozen T requires at least two unique, nonempty sample IDs")
    # Preserve legacy 50% subset byte-for-byte. New fractions use prefixes of
    # one shuffled order, giving nested 10%/20% views without reselection.
    if fraction == 0.5:
        chosen = set(random.Random(seed).sample(ids, len(ids) // 2))
    else:
        count = math.floor(len(ids) * fraction)
        if count < 1:
            raise ValueError("Low-fidelity training subset would be empty")
        ordered = random.Random(seed).sample(ids, len(ids) // 2)
        chosen = set(ordered[:count])
    training_ids = set(ids)
    expected = {"source_sha256": file_hash(source / "train.csv"), "seed": seed,
                "n_total": len(ids), "n_selected": len(chosen),
                "selected_ids_sha256": hashlib.sha256(json.dumps(sorted(chosen)).encode()).hexdigest()}
    if fraction != 0.5:
        expected.update(training_fraction=fraction, sampling="legacy_half_prefix_v1")
    if manifest_path.exists():
        # A legacy 50% caller must also reject a new fraction manifest.
        if json.loads(manifest_path.read_text()).get("training_fraction", 0.5) != fraction:
            raise ValueError("Frozen training fraction changed on resume")
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
