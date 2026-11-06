#!/usr/bin/env bash
#SBATCH --job-name=alphaevolve-verify
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=.cache/alphaevolve/verify-%j.log
set -euo pipefail
source tasks/alphaevolve/env.sh
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -u tasks/alphaevolve/check_environment.py "$@"
