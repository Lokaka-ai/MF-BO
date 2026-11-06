#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$TASK_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
export AUTORESEARCH_CACHE_DIR="${AUTORESEARCH_CACHE_DIR:-$PROJECT_DIR/.cache/autoresearch}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv sync --frozen
uv run --frozen python -u "$TASK_DIR/repo/prepare.py" --num-shards 10 --download-workers 4
uv run --frozen python "$TASK_DIR/manifest.py"
uv run --frozen python -c 'import torch; assert torch.cuda.is_available(); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
uv run --frozen python -c 'from omegaconf import OmegaConf; from agents.memory.embedding_models import EmbeddingModel; c=OmegaConf.load("config/config.yaml"); m=EmbeddingModel(model_name=c.agent.memory_embedding_model_path,device=c.agent.memory_embedding_device); print("Global memory embedding:", m.encode(["nanoGPT architecture and optimizer"]).shape)'
