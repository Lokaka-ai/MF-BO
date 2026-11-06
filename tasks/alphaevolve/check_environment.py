#!/usr/bin/env python3
"""Verify CPU dependencies, real memory embeddings and official constructions."""
import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tasks.alphaevolve.prepare import save, digest
from tasks.alphaevolve.registry import TASKS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    p.add_argument("--data-dir", type=Path, default=ROOT/"data/alphaevolve")
    p.add_argument("--output", type=Path, default=ROOT/".cache/alphaevolve/environment-ready.json")
    p.add_argument("--skip-embedding", action="store_true")
    args = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this resource check inside a Slurm CPU allocation")
    os.environ.update(CUDA_VISIBLE_DEVICES="", USE_TF="0")
    report = {"host": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "python": sys.version, "executable": sys.executable, "packages": {}, "tasks": {}, "passed": False}
    for distribution, module in {"numpy":"numpy", "scipy":"scipy", "sympy":"sympy", "torch":"torch", "faiss-cpu":"faiss",
                                  "sentence-transformers":"sentence_transformers", "omegaconf":"omegaconf", "dataclasses-json":"dataclasses_json"}.items():
        importlib.import_module(module)
        report["packages"][distribution] = importlib.metadata.version(distribution)
    import torch
    import numpy as np
    report["cuda_available"] = torch.cuda.is_available()
    if report["cuda_available"]:
        raise RuntimeError("CPU task unexpectedly sees CUDA")
    if not args.skip_embedding:
        from agents.memory.embedding_models import EmbeddingModel
        model = EmbeddingModel(model_name="BAAI/bge-base-en-v1.5", device="cpu")
        values = model.encode(["circle packing", "autocorrelation inequality"])
        assert values.shape == (2, 768) and np.isfinite(values).all()
        report["memory_embedding_shape"] = list(values.shape)
        del model
    manifest = json.loads((args.data_dir/"manifest.json").read_text())
    output_dir = args.output.parent/"reference_checks"
    output_dir.mkdir(parents=True, exist_ok=True)
    from tasks.alphaevolve.graders import score
    for task in args.tasks:
        started = time.monotonic()
        for name, expected in manifest["tasks"][task].items():
            if digest(args.data_dir/name) != expected:
                raise ValueError(f"Changed prepared asset: {name}")
        directory = args.data_dir/task/"prepared"
        initial_score = score(task, json.loads((directory/"public/initial.json").read_text()))
        result_path = output_dir/f"{task}.json"
        subprocess.run([sys.executable, str(Path(__file__).with_name("worker.py")), "grade", task,
                        str(directory/"private/reference.json"), str(result_path)], check=True, timeout=900)
        result = json.loads(result_path.read_text())
        expected = json.loads((directory/"private/expected.json").read_text())
        matched = result["valid"] and math.isclose(result["metric"], expected["metric"], rel_tol=0, abs_tol=expected["abs_tolerance"])
        report["tasks"][task] = {"reference": result, "expected": expected, "initial": initial_score,
                                 "passed": matched, "seconds": time.monotonic()-started}
        save(args.output, report)
        print(f"{'PASS' if matched else 'FAIL'} official reference: {task}", flush=True)
    report["passed"] = len(report["tasks"]) == len(args.tasks) and all(x["passed"] for x in report["tasks"].values())
    save(args.output, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
