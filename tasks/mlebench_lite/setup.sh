#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$TASK_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
MLEBENCH_COMMIT=507f92e1138bb6e40dac5c6ee7a6758e6424bf97
if [[ ! -d mle-bench/.git ]]; then
  git clone https://github.com/openai/mle-bench.git mle-bench
  git -C mle-bench checkout "$MLEBENCH_COMMIT"
fi
if [[ "$(git -C mle-bench rev-parse HEAD)" != "$MLEBENCH_COMMIT" ]]; then
  echo "mle-bench must be at $MLEBENCH_COMMIT; refusing to change an existing checkout." >&2
  exit 1
fi
git -C mle-bench lfs pull
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.cache/uv-mlebench}"
if [[ ! -x .venv-mlebench/bin/python ]]; then
  uv venv --python 3.12 .venv-mlebench
fi
.venv-mlebench/bin/python "$TASK_DIR/patch_upstream.py"
sync_options=(--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match)
if [[ -d "$PROJECT_DIR/.cache/mlebench-lite/wheels" ]]; then
  sync_options+=(--find-links "$PROJECT_DIR/.cache/mlebench-lite/wheels")
fi
uv pip sync --python .venv-mlebench/bin/python "$TASK_DIR/requirements.lock" "${sync_options[@]}"
uv pip check --python .venv-mlebench/bin/python
if [[ ! -x .venv-kaggle/bin/python ]]; then
  uv venv --python 3.12 .venv-kaggle
fi
uv pip sync --python .venv-kaggle/bin/python "$TASK_DIR/requirements-download.lock"
uv pip check --python .venv-kaggle/bin/python
echo "Activate with: source $TASK_DIR/env.sh"
echo "Prepare with: python tasks/mlebench_lite/prepare.py"
