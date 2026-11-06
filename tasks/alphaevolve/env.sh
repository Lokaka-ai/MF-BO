# Source from bash; inherit the installed, pinned mlebench_lite environment.
ALPHAEVOLVE_TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ALPHAEVOLVE_TASK_DIR/../mlebench_lite/env.sh"
export ALPHAEVOLVE_DATA="${ALPHAEVOLVE_DATA:-$MLEBENCH_PROJECT_DIR/data/alphaevolve}"
export CUDA_VISIBLE_DEVICES=""
export USE_TF=0
export MPLBACKEND=Agg
