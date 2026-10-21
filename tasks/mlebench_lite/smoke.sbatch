#!/usr/bin/env bash
# Submit from the MLEvolve_BO directory.
#SBATCH --job-name=mlebench-lite-smoke
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=.cache/mlebench-lite/smoke-slurm-%j.log
set -euo pipefail
source tasks/mlebench_lite/env.sh
python -u tasks/mlebench_lite/smoke.py --gpu auto "$@"
