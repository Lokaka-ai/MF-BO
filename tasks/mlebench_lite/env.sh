# Source this file from bash; all large caches stay on scratch.
MLEBENCH_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$MLEBENCH_PROJECT_DIR/.venv-mlebench/bin/activate"
export MLEBENCH_DATA="${MLEBENCH_DATA:-$MLEBENCH_PROJECT_DIR/data/mlebench}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$MLEBENCH_PROJECT_DIR/.cache/uv-mlebench}"
export HF_HOME="${HF_HOME:-$MLEBENCH_PROJECT_DIR/.cache/mlebench-lite/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$MLEBENCH_PROJECT_DIR/.cache/mlebench-lite/torch}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$MLEBENCH_PROJECT_DIR/.cache/mlebench-lite/matplotlib}"
