# Phase 1 Reproduction

以下命令均在独立副本根目录运行：

```bash
cd /gpfs/projects/AI4D/core-132/xixian/EvolveFold_markdown_protocol/MLEvolve_BO_Phase1
```

原仓库不作为工作目录，所有结果写入本副本的 `artifacts/`。

## 环境与测试

MLEvolve 自带 `.venv` 有项目依赖但没有 pytest；系统 pytest 可通过显式 PYTHONPATH 复用
该环境：

```bash
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages:$PWD" pytest -q
python -m compileall -q engine agents config experiments tests
git diff --check
```

检查默认配置与可导入性：

```bash
.venv/bin/python -c 'from config import _load_cfg; print(_load_cfg(use_cli_args=False).parent_selector.type)'
```

应输出 `legacy_original`。

## Synthetic benchmark

```bash
.venv/bin/python experiments/synthetic_parent_selection/run.py \
  --seeds 12 --steps 60 \
  --output-dir artifacts/synthetic_parent_selection
```

主要产物：`summary.csv/json`、`trajectories.csv`、
`prediction_diagnostics.csv`、`curves.json`、两个 PNG 和 `run_manifest.json`。

## 历史 transition evaluation

```bash
.venv/bin/python experiments/historical_transition_evaluation/run.py \
  --output-dir artifacts/historical_transition_evaluation
```

不传 journal 参数时只读发现项目目录中的 journal。也可在命令末尾显式列出 sealed journal
paths。产物包括 admitted manifest、重建的静态 transition、temporal predictions、全局/
operator/calibration/cost-validity CSV。

## Leaf 在线 pilot

运行前确认当前 allocation 与 GPU 空闲：

```bash
squeue -u "$USER" -o '%.18i %.12P %.20j %.8T %.10M %.6D %R'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

正式三方法运行：

```bash
bash experiments/leaf_online_pilot/launch.sh artifacts/leaf_online_pilot_final
tmux list-sessions | rg '^phase1_leaf_'
```

状态检查不需要进入会话：

```bash
for method in progressive_mcgs bayesian_child_ucb cost_aware_bayesian; do
  tail -40 "artifacts/leaf_online_pilot_final/$method/logs/workflow.log"
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

完成后汇总：

```bash
.venv/bin/python experiments/leaf_online_pilot/summarize.py \
  --input-dir artifacts/leaf_online_pilot_final \
  --output-dir artifacts/leaf_online_pilot_final
```

launcher 为每条流程建立独立 CODEX_HOME、workspace、log、format-server 端口和 GPU visibility。
它只复制已有 provider 配置，凭据内容不会写入本仓库或报告。复现者需具有相同 provider
访问权限；没有权限时 unit/synthetic/historical 仍可独立复现，但不能声称真实 pilot 完成。

## Controlled parent batch

先找到正式 Progressive pilot 的唯一 run root：

```bash
warmup_run="$(find artifacts/leaf_online_pilot_final/progressive_mcgs/runs \
  -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
```

冻结共同 snapshot 并拟合预测：

```bash
.venv/bin/python experiments/controlled_parent_batch/run.py prepare \
  --warmup-run "$warmup_run" \
  --output-dir artifacts/controlled_parent_batch \
  --parents 4
```

确认四张 GPU 空闲后并行执行四个独立 parent：

```bash
bash experiments/controlled_parent_batch/launch.sh artifacts/controlled_parent_batch
tmux list-sessions | rg '^phase1_controlled_'
```

完成后：

```bash
.venv/bin/python experiments/controlled_parent_batch/run.py summarize \
  --output-dir artifacts/controlled_parent_batch
```

`admitted_input/` 一次性复制共同 journal、config、transitions 与 Global Memory；四个 worker
只消费这份副本。`frozen_snapshot.json` 保存原 input path、一次 admission SHA、所有候选预测
和被抽取的四个 rank；每个 `candidates/<index>_<parent>/result.json` 保存真实
gain/cost/validity。

## 选择器启动参数

任意原 `run.py` 命令可追加：

```text
parent_selector.type=progressive_mcgs
parent_selector.type=bayesian_child_ucb
parent_selector.type=cost_aware_bayesian
```

其他模式为 `legacy_original`、`parent_metric_greedy`、`random_parent`、
`bayesian_gain_ucb`。生产默认 cold start 是：

```text
bo_parent_selector.min_valid_transitions=8
bo_parent_selector.min_total_transitions=10
bo_parent_selector.min_operator_observations=1
```

关键消融示例：

```text
bo_parent_selector.features.include_mcgs_statistics=false
bo_parent_selector.features.include_branch_history=false
bo_parent_selector.features.include_reference_summary=false
```

## 原仓库最终保护检查

```bash
git -C ../MLEvolve branch --show-current
git -C ../MLEvolve rev-parse HEAD
git -C ../MLEvolve status --short
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages:$PWD" \
  pytest -q tests/test_original_repo_untouched.py
```

期望依次仍为 `main`、`7d8403c899c40f01941c0429f1c4ef51e82ae41c`，以及 admission
时已有的 `llm/__init__.py` 修改和 `llm/codex.py` 未跟踪文件。本任务不执行 commit 或 push。
