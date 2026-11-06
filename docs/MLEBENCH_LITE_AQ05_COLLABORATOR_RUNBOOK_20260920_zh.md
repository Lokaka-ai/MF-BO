# 合作者运行手册：用 AQ05 跑 MLE-bench Lite

## 1. 目标和不可改变的 setting

你负责下列 7 个 MLE-bench Lite competition：

```text
dog-breed-identification
leaf-classification
nomad2018-predict-transparent-conductors
random-acts-of-pizza
siim-isic-melanoma-classification
tabular-playground-series-may-2022
text-normalization-challenge-russian-language
```

每个任务必须使用相同设置：

| 项目 | 值 |
|---|---|
| 方法 | AQ05 |
| AQ05 preset | `acq7_aq05_gp4_level15_rating_kg` |
| code 模型 | `muse-spark-1.3-contributor` |
| feedback / rating 模型 | `muse-spark-1.3-contributor` |
| provider | `codex://opencode-go` |
| GPU | 每个任务独占 1 张 A100；Mila 使用 `gpu:a100l:1` |
| agent budget | `43200` 秒（12 小时） |
| scheduler wall time | `13:00:00`，额外一小时不属于 agent 搜索预算 |
| steps / seed | 500 / 42 |
| initial drafts / workers | 3 / 3 |
| GPU 数 | 1 |
| candidate 执行上限 | `32400` 秒 |
| evaluation mode | `evaluation.enabled=false` |

关键区别：这是在 MLE-bench Lite competition 上使用 AQ05 selector，不是 nanoGPT/autoresearch
实验。正式命令中必须有
`experiment_preset=acq7_aq05_gp4_level15_rating_kg`，并且绝对不要添加
`task_preset=autoresearch`。

AQ05 preset 会固定 `parent_selector.type=acq7_gp4`、`finite_pool_kg` acquisition、level
Matérn-3/2 branch covariance、branch prior SD 1.5、rating 开启和 ancestor feature 关闭。
不要再用 CLI 覆盖这些 selector 字段。

## 2. 一次性环境准备

以下路径是当前共享仓库；若使用自己的 clone，只改 `PROJECT`，不要改实验 setting：

```bash
export PROJECT=/network/scratch/l/liux/EvolveFold/MLEvolve_BO
cd "$PROJECT"

bash tasks/mlebench_lite/setup.sh
source tasks/mlebench_lite/env.sh
```

配置自己的 OpenCode Go 凭据。凭据文件只放一行原始 key，不要把 key 写进命令、YAML、日志或
文档：

```bash
export OPENCODE_API_KEY_FILE=/absolute/path/to/your_opencode_key

# 与本轮设置对齐；这两个变量会由 --export=ALL 传入 Slurm job。
export MLEVOLVE_CODEX_REASONING_EFFORT=high
export MLEVOLVE_CODEX_TIMEOUT_SECONDS=1800

command -v codex
python tools/check_opencode.py
```

最后一条命令会发起一次真实 Muse 1.3 请求；成功时应出现
`"message": "OPENCODE_MUSE_OK"`。若 `codex` 不在 `PATH`，先设置：

```bash
export MLEVOLVE_REAL_CODEX_BIN=/absolute/path/to/codex
```

## 3. 准备自己负责的数据

准备程序会校验官方 archive MD5、public/private split 和 prepared checksums。先并行下载，再串行
prepare，尤其不要同时准备两份 SIIM：

```bash
cd "$PROJECT"
source tasks/mlebench_lite/env.sh

COLLAB_TASKS=(
  dog-breed-identification
  leaf-classification
  nomad2018-predict-transparent-conductors
  random-acts-of-pizza
  siim-isic-melanoma-classification
  tabular-playground-series-may-2022
  text-normalization-challenge-russian-language
)

python tasks/mlebench_lite/prepare.py \
  --download-only --workers 4 --tasks "${COLLAB_TASKS[@]}"

python tasks/mlebench_lite/prepare.py \
  --prepare-only --wait-for-downloads --workers 1 \
  --tasks "${COLLAB_TASKS[@]}"
```

默认数据根目录是 `$PROJECT/data/mlebench`，由 `source tasks/mlebench_lite/env.sh` 设置为
`$MLEBENCH_DATA`。`siim-isic-melanoma-classification` 约 116 GB，提交任务前确认 scratch
空间充足。

## 4. 正式提交：7 个独立的 AQ05 MLE-bench jobs

下面的命令会提交 7 个独立 Slurm jobs。每个 job 独占一张 A100，agent 搜索严格限制为 12
小时；Slurm 申请 13 小时只是给验证 server、submission fusion 和写盘留出余量。任务可能因为
fair-share 排队，不代表 setting 改变。

```bash
cd "$PROJECT"
source tasks/mlebench_lite/env.sh

export OPENCODE_API_KEY_FILE=/absolute/path/to/your_opencode_key
export MLEVOLVE_CODEX_REASONING_EFFORT=high
export MLEVOLVE_CODEX_TIMEOUT_SECONDS=1800

mkdir -p .cache/mlebench-lite/slurm-aq05

COLLAB_TASKS=(
  dog-breed-identification
  leaf-classification
  nomad2018-predict-transparent-conductors
  random-acts-of-pizza
  siim-isic-melanoma-classification
  tabular-playground-series-may-2022
  text-normalization-challenge-russian-language
)

for i in "${!COLLAB_TASKS[@]}"; do
  TASK="${COLLAB_TASKS[$i]}"
  SERVER_ID=$((240 + i))
  JOB_ID=$(sbatch --parsable \
    --job-name="aq05-mle-${i}" \
    --partition=long \
    --nodes=1 --ntasks=1 \
    --gres=gpu:a100l:1 \
    --cpus-per-task=21 \
    --mem=128G \
    --time=13:00:00 \
    --no-requeue \
    --output="$PROJECT/.cache/mlebench-lite/slurm-aq05/${TASK}-%j.log" \
    --export=ALL,PROJECT="$PROJECT",TASK="$TASK",SERVER_ID="$SERVER_ID" \
    --wrap='set -euo pipefail
cd "$PROJECT"
source tasks/mlebench_lite/env.sh
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in *A100*) ;; *) echo "Expected A100, got: $GPU_NAME" >&2; exit 2 ;; esac
export TIME_LIMIT_SECS=43200
export STEP_LIMIT=500
export CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
export MEMORY_INDEX=0
bash run_single_task.sh "$TASK" "$MLEBENCH_DATA" "$SERVER_ID" \
  experiment_preset=acq7_aq05_gp4_level15_rating_kg \
  log_dir="$PROJECT/runs_aq05_muse13" \
  workspace_dir="$PROJECT/runs_aq05_muse13" \
  preprocess_data=true \
  copy_data=false \
  use_grading_server=true \
  evaluation.enabled=false \
  exec.timeout=32400 \
  agent.steps=500 \
  agent.time_limit=43200 \
  agent.initial_drafts=3 \
  agent.seed=42 \
  agent.search.parallel_search_num=3 \
  agent.search.num_gpus=1 \
  agent.code.model=muse-spark-1.3-contributor \
  agent.code.base_url=codex://opencode-go \
  agent.feedback.model=muse-spark-1.3-contributor \
  agent.feedback.base_url=codex://opencode-go')
  printf '%s\t%s\n' "$TASK" "$JOB_ID" | tee -a .cache/mlebench-lite/slurm-aq05/submitted_jobs.tsv
done
```

`submitted_jobs.tsv` 是本轮任务到 job ID 的映射。不要重复执行整段提交命令；若不确定某次
`sbatch` 是否成功，先查 `submitted_jobs.tsv` 和 `squeue`，不要盲目重交。

如果本地 Slurm 对 A100 的 GRES 名称不是 `a100l`，只允许按管理员给出的等价 A100 资源名修改
`--gres`；不得换成其他 GPU 后仍将结果标为本轮 A100 setting。

## 5. 监控与 setting 核对

查看排队和运行状态：

```bash
squeue -u "$USER" -o '%.18i %.28j %.2t %.10M %.10l %.24R'
```

查看某项日志：

```bash
tail -f "$PROJECT/.cache/mlebench-lite/slurm-aq05/<TASK>-<JOB_ID>.log"
```

job 启动后，日志必须显示 A100。运行目录位于：

```text
$PROJECT/runs_aq05_muse13/<timestamp>_<competition-id>/
```

重点核对其 `logs/config.yaml`：

```yaml
agent:
  time_limit: 43200
  steps: 500
  seed: 42
  code:
    model: muse-spark-1.3-contributor
    base_url: codex://opencode-go
  feedback:
    model: muse-spark-1.3-contributor
    base_url: codex://opencode-go
parent_selector:
  type: acq7_gp4
study7:
  method_id: AQ05
  rating:
    enabled: true
acq7:
  enabled: true
  branch_kind: level_matern32
  branch_prior_std: 1.5
  branch_lengthscale: 1.0
  slope_prior_std: 0.0
  acquisition_type: finite_pool_kg
```

还应检查 `evaluation.enabled=false`、`parent_selector.decision_scope=all_normal_steps`、
`study7.ancestor.enabled=false`、`agent.search.num_gpus=1` 和
`agent.search.parallel_search_num=3`。任一关键 setting 不符时，先保存证据并停止该错误 job，
不要让它继续消耗 12 小时。

## 6. 完成后的 submission 与官方评分

`run_single_task.sh` 会在搜索结束后完成 legacy submission fusion，并把最终选择写到：

```text
<RUN_ROOT>/logs/final_submission.json
```

对每个 run 使用精确的 log 目录做 official grading：

```bash
source tasks/mlebench_lite/env.sh
python -m utils.grade_submission \
  --log-dir /absolute/path/to/RUN_ROOT/logs \
  --data-dir "$MLEBENCH_DATA" \
  | tee /absolute/path/to/RUN_ROOT/logs/official-grade.json
```

评分只能在完整搜索和 final submission 冻结后执行；private score 不得反馈给仍在运行的 search。
把 `submitted_jobs.tsv`、Slurm log、`logs/config.yaml`、`logs/final_submission.json` 和
`logs/official-grade.json` 一起交回。

## 7. release / 释放 GPU 命令

使用上述 `sbatch` 方式时，job 正常结束、报错退出或达到 `13:00:00` wall time 后，Slurm 会自动
释放 A100，无需额外命令。

需要提前停止某个错误 job 时，先从 `submitted_jobs.tsv` 或 `squeue` 核对精确 job ID，然后只
释放自己的目标 job：

```bash
scancel <JOB_ID>
squeue -j <JOB_ID>
```

第二条命令用于确认它已经离开队列。不要用模糊 job name 或批量命令取消其他人的任务。

如果改用交互式 `salloc`，任务结束后在 allocation shell 中执行：

```bash
exit
```

若交互 shell 已断开，可在登录节点用保存的 allocation ID 释放：

```bash
scancel <ALLOCATION_JOB_ID>
```

## 8. 失败与重跑规则

- `run_single_task.sh` 的 exit code `124` 表示 12 小时外层 timeout；先确认 fusion 和最终 submission
  是否已成功落盘，不能仅凭 `124` 判断任务无效。
- provider、凭据、数据缺失或错误 GPU 属于配置失败，不应记为一个正式 12 小时结果。
- OOM、节点故障或抢占要保留原 job 日志和已用时间；重跑使用新 job ID、新 run 目录，并明确标记
  哪一个是最终纳入结果的 run。
- 不要通过缩短预算、减少 workers、改 seed、换模型或关闭 rating 来临时修复；这些都会改变本轮
  setting。
- 不要让两个 competition 共享同一 GPU、workspace 或 grading-server ID。

完整分工见
[`MLEBENCH_LITE_AQ05_TASK_SPLIT_20260920_zh.md`](MLEBENCH_LITE_AQ05_TASK_SPLIT_20260920_zh.md)。
