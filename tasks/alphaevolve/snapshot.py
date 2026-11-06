#!/usr/bin/env python3
"""Freeze the runnable source for a campaign without copying datasets or caches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def freeze(destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("agents", "config", "engine", "llm", "utils", "tools", "tasks/alphaevolve", "tasks/autoresearch", "tasks/mlebench_lite"):
        shutil.copytree(ROOT/name, destination/name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".cache"))
    for name in ("run.py", "__init__.py", "pyproject.toml"):
        shutil.copy2(ROOT/name, destination/name)
    (destination/".venv-mlebench").symlink_to(ROOT/".venv-mlebench", target_is_directory=True)
    (destination/"data").symlink_to(ROOT/"data", target_is_directory=True)
    (destination/".cache").symlink_to(ROOT/".cache", target_is_directory=True)
    hashes = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in destination.rglob("*") if p.is_file() and not p.is_symlink()}
    # pathlib does not recurse through directory symlinks here.
    (destination/"SOURCE_HASHES.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    freeze(p.parse_args().destination)
