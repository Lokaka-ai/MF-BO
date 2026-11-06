"""Reuse the baseline's frozen validation protocol without regenerating it."""

from pathlib import Path
import json


def register_source(source, tasks):
    source = Path(source).resolve()
    ids = {}
    for task in tasks:
        spec = json.loads((source / task / "protocol.json").read_text())
        if spec["task_id"] != task or not spec.get("protocol_id"):
            raise ValueError(f"Invalid baseline protocol: {task}")
        ids[task] = spec["protocol_id"]
    return {"directory": str(source), "protocol_ids": ids}


def attach_source(settings, campaign, task):
    saved = settings.get("protocol_source")
    if saved is None:
        return  # Standalone campaign; do not claim a matched comparison.
    source = Path(saved["directory"]) / task
    spec = json.loads((source / "protocol.json").read_text())
    if spec.get("protocol_id") != saved["protocol_ids"][task]:
        raise ValueError(f"Baseline protocol identity changed: {task}")
    destination = Path(campaign) / "protocols" / task
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"Campaign already points to another protocol: {task}")
    else:
        destination.symlink_to(source.resolve(), target_is_directory=True)
    # prepare_fixed_protocol and the worker both verify the actual frozen files,
    # seed and data source, not just this protocol-id field. Source is read-only.
