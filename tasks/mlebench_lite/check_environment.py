#!/usr/bin/env python3
"""Check the installed Lite stack, official task registry, CUDA and memory model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/mlebench-lite/environment-ready.json")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/mlebench-lite/huggingface"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/mlebench-lite/matplotlib"))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["USE_TF"] = "0"
    packages = {"torch": "torch", "torchvision": "torchvision", "torchaudio": "torchaudio",
                "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
                "mlebench": "mlebench", "flask": "flask", "faiss-cpu": "faiss",
                "lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost",
                "timm": "timm", "transformers": "transformers", "sentence-transformers": "sentence_transformers",
                "librosa": "librosa", "soundfile": "soundfile", "opencv-python-headless": "cv2",
                "pydicom": "pydicom", "py7zr": "py7zr", "tensorflow": "tensorflow"}
    versions = {}
    for distribution, module in packages.items():
        importlib.import_module(module)
        versions[distribution] = version(distribution)
        print(f"OK {distribution} {versions[distribution]}", flush=True)
    from mlebench.registry import registry
    tasks = (ROOT / "config/mlebench_lite22.txt").read_text().split()
    assert set(tasks) == set(registry.get_lite_competition_ids())
    for task in tasks:
        registry.get_competition(task)
    import torch
    cuda = torch.cuda.is_available()
    if args.require_cuda and not cuda:
        raise RuntimeError("CUDA is required but unavailable")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
              "python": sys.version, "executable": sys.executable, "packages": versions,
              "official_lite_tasks": len(tasks), "cuda_available": cuda, "torch_cuda": torch.version.cuda}
    if cuda:
        a = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
        assert torch.allclose((a @ a).cpu(), a.cpu() @ a.cpu())
        report["gpu"] = torch.cuda.get_device_name(0)
    if args.embedding:
        from agents.memory.embedding_models import EmbeddingModel
        import numpy as np
        model = EmbeddingModel(model_name="BAAI/bge-base-en-v1.5", device="cuda" if cuda else "cpu")
        embeddings = model.encode(["leaf classification", "author identification"])
        assert embeddings.shape == (2, 768) and np.isfinite(embeddings).all()
        report["memory_embedding_shape"] = list(embeddings.shape)
    report["utilities"] = {name: shutil.which(name) for name in ("git", "unzip", "7z", "ffmpeg")}
    if not all(report["utilities"].values()):
        raise RuntimeError("Missing a required system utility")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Environment verified: {output}", flush=True)


if __name__ == "__main__":
    main()
