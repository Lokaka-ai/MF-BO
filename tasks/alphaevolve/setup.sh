#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$TASK_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv-mlebench/bin/python ]]; then
  bash tasks/mlebench_lite/setup.sh
fi
source "$TASK_DIR/env.sh"
uv pip check --python .venv-mlebench/bin/python
python "$TASK_DIR/prepare.py" --data-dir "$ALPHAEVOLVE_DATA"
echo "Environment: .venv-mlebench (same locked dependencies as mlebench_lite; CUDA hidden)."
echo "Run CPU verification inside a Slurm allocation with tasks/alphaevolve/check_environment.py."
