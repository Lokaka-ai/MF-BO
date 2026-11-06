# 合作者运行手册：原版 MCGS 与 AQ05 的 MLE-bench Lite 对照

## 1. 实验目标与方法开关

沿用 AQ05 的 14 个任务、7/7 分工，在相同模型、任务数据和计算预算下，比较 AQ05 与原版
MLEvolve MCGS 的选择策略。每个 task 两种方法各运行一次，当前共同 seed 为 42。

原版方法通过现有超参数开启：

```text
experiment_preset=mlevolve_original
```

该 preset 位于 `config/experiments/mlevolve_original.yaml`，调用仓库原生
`select_with_soft_switch`（UCT 探索、按时间软切换到 Top-K 利用）并保留分支直接续接。
这是当前 MLE-bench 运行环境下的原版搜索策略对照，模型和资源采用本轮设置。
运行 competition 时不加载 `task_preset=autoresearch`。

| 方法配置 | AQ05 | 原版 MCGS |
|---|---|---|
| `experiment_preset` | `acq7_aq05_gp4_level15_rating_kg` | `mlevolve_original` |
| `parent_selector.type` | `acq7_gp4` | `legacy_original` |
| `parent_selector.decision_scope` | `all_normal_steps` | `root_only` |
| `parent_selector.include_root_draft` | false | false |
| 实际选择策略 | GP4 + level Matérn-3/2 + `finite_pool_kg` | 原生 UCT / Top-K soft-switch |
| `acq7.enabled` | true | false |
| `study7.rating.enabled` | true | false |
| `study7.ancestor.enabled` | false | false |
| `bo_parent_selector.agent_rating.enabled` | false | false |
| 分支续接 | 普通步骤重新比较合法父节点 | 原版直接续接 |

因此，这个实验比较的是两套完整选择策略，包含 AQ05 的 rating、GP/KG 与重新选择时机；
不能把结果差异完全归因于 KG 单独一个公式。原版 MCGS 的 BO 配置兼容字段可能仍出现在
最终 YAML 中，但不参与选父节点。仅修改 `acquisition.type` 或改成 `progressive_mcgs`
并不等同于这里指定的原版入口。

## 2. 任务分工保持一致

| # | 本人负责 | 合作者负责 |
|---:|---|---|
| 1 | `aerial-cactus-identification` | `dog-breed-identification` |
| 2 | `aptos2019-blindness-detection` | `leaf-classification` |
| 3 | `jigsaw-toxic-comment-classification-challenge` | `nomad2018-predict-transparent-conductors` |
| 4 | `mlsp-2013-birds` | `random-acts-of-pizza` |
| 5 | `new-york-city-taxi-fare-prediction` | `siim-isic-melanoma-classification` |
| 6 | `ranzcr-clip-catheter-line-classification` | `tabular-playground-series-may-2022` |
| 7 | `text-normalization-challenge-english-language` | `text-normalization-challenge-russian-language` |

仍剔除原来 8 个任务。准确 competition ID 和剔除清单见
[AQ05 分工文档](MLEBENCH_LITE_AQ05_TASK_SPLIT_20260920_zh.md)。AQ05 的完成状态不代表 MCGS 已运行。

## 3. 两组共同 setting

| 项目 | 共同设置 |
|---|---|
| code / feedback 模型 | `muse-spark-1.3-contributor` |
| provider | `codex://opencode-go` |
| reasoning / 单次模型调用超时 | `high` / 1800 秒 |
| GPU | 每个 task 独占 1 张 A100；Mila `gpu:a100l:1`，记录实际型号及显存 |
| agent 时间上限 | 43200 秒（12 小时） |
| Slurm 时间 | 13 小时，包含启动和收尾余量 |
| step 上限 / seed | 500 / 42 |
| 初始 drafts / search workers | 3 / 3，三个 worker 共用本任务的一张 GPU |
| 单 candidate 执行上限 | 32400 秒 |
| CPU / 主存 | 沿用 AQ05 合作者手册：21 CPU / 128 GiB；若已有实际 AQ05 run，则按该任务实配配对 |
| 外层固定评估 | `evaluation.enabled=false` |
| 数据 | 同一官方 split、相同 prepared public 数据；结束后用相同官方 grader 评分 |
| 数据处理 / grading server | `preprocess_data=true`、`copy_data=false`、`use_grading_server=true` |
| 其他搜索配置 | 同版本 `config/config.yaml`；保留 coldstart、memory、DEBUG、evolution、fusion、aggregation 默认设置 |

43200 秒和 500 steps 都是上限；达到 step 上限可能提前结束，需记录实际耗时，不能写成保证跑满
12 小时。两组都保留相同上限。额外的 Slurm 收尾时间不属于搜索预算，单独记录 fusion 用时。

比较时冻结相同代码 commit、依赖和非方法参数，以双方实际 `logs/config.yaml` 为准。
已有[本人 AQ05 运行记录](MLEBENCH_LITE_AQ05_OWNER_RUN_20260921_zh.md)使用过
4/6 CPU、24/32/64 GiB 等资源，并存在中断、重执行和实现修复；与下面 21 CPU / 128 GiB
的示例不能直接称为严格同资源对照。配对历史任务时记录这些差异，或在统一设置下重跑两组。

使用全新 MCGS run，避免继承 AQ05 的 checkpoint、搜索图、memory 或历史成绩。
相同 seed 不保证远端模型输出及初始草稿逐字相同，当前方案未额外引入共享草稿池。

## 4. 环境与数据准备

已有 AQ05 环境和准备好的数据可直接复用。首次部署：

```bash
export PROJECT=/network/scratch/l/liux/EvolveFold/MLEvolve_BO
cd "$PROJECT"
bash tasks/mlebench_lite/setup.sh
source tasks/mlebench_lite/env.sh
export OPENCODE_API_KEY_FILE=/absolute/path/to/your_opencode_key
export MLEVOLVE_CODEX_REASONING_EFFORT=high
export MLEVOLVE_CODEX_TIMEOUT_SECONDS=1800
command -v codex
python tools/check_opencode.py
```

模型检查会发起真实请求，预期返回 `OPENCODE_MUSE_OK`。凭据文件使用合作者自己的 key。
数据准备方法沿用 [AQ05 运行手册第 3 节](MLEBENCH_LITE_AQ05_COLLABORATOR_RUNBOOK_20260920_zh.md#3-准备自己负责的数据)，
准备自己的 7 项即可；SIIM 需要预留原始下载、解压数据与运行产物空间。

## 5. 合作者正式提交命令

下列 Bash 命令提交 7 个独立 MCGS jobs。方法以外的科学参数沿用 AQ05 合作者手册，
输出目录和端口编号单独设置，避免覆盖或碰撞。提交前确认上节环境已经准备完成。
若配对的 AQ05 实配不是 21 CPU / 128 GiB，应按对应任务调整请求并记录。

```bash
set -euo pipefail
export PROJECT=/network/scratch/l/liux/EvolveFold/MLEvolve_BO
cd "$PROJECT"
source tasks/mlebench_lite/env.sh
export OPENCODE_API_KEY_FILE=/absolute/path/to/your_opencode_key
export MLEVOLVE_CODEX_REASONING_EFFORT=high
export MLEVOLVE_CODEX_TIMEOUT_SECONDS=1800
mkdir -p .cache/mlebench-lite/slurm-mcgs

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
  SERVER_ID=$((340 + i))
  JOB_ID=$(sbatch --parsable \
    --job-name="mcgs-mle-${i}" \
    --partition=long --nodes=1 --ntasks=1 \
    --gres=gpu:a100l:1 --cpus-per-task=21 --mem=128G \
    --time=13:00:00 --no-requeue \
    --output="$PROJECT/.cache/mlebench-lite/slurm-mcgs/${TASK}-%j.log" \
    --export=ALL,PROJECT="$PROJECT",TASK="$TASK",SERVER_ID="$SERVER_ID" \
    --wrap='bash -c '\''
set -euo pipefail
cd "$PROJECT"
source tasks/mlebench_lite/env.sh
export MEMORY_INDEX=$(python -c "import torch; assert torch.cuda.device_count() == 1; p = torch.cuda.get_device_properties(0); assert \"A100\" in p.name; print(p.uuid)")
nvidia-smi --id="$MEMORY_INDEX" --query-gpu=uuid,name,memory.total --format=csv
export TIME_LIMIT_SECS=43200
export STEP_LIMIT=500
export CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
bash run_single_task.sh "$TASK" "$MLEBENCH_DATA" "$SERVER_ID" \
  experiment_preset=mlevolve_original \
  log_dir="$PROJECT/runs_mcgs_muse13" \
  workspace_dir="$PROJECT/runs_mcgs_muse13" \
  preprocess_data=true copy_data=false use_grading_server=true \
  evaluation.enabled=false exec.timeout=32400 \
  agent.steps=500 agent.time_limit=43200 agent.initial_drafts=3 agent.seed=42 \
  agent.search.parallel_search_num=3 agent.search.num_gpus=1 \
  agent.code.model=muse-spark-1.3-contributor \
  agent.code.base_url=codex://opencode-go \
  agent.feedback.model=muse-spark-1.3-contributor \
  agent.feedback.base_url=codex://opencode-go
'\''')
  printf '%s\t%s\n' "$TASK" "$JOB_ID" | tee -a .cache/mlebench-lite/slurm-mcgs/submitted_jobs.tsv
done
```

GPU UUID 从 Slurm 暴露的唯一 CUDA 设备读取，再交给原 launcher，避免把物理 GPU 0 当成分配的卡。
这是设备定位，不改变一任务一张 A100 的设置。上述 GPU 检查只在申请到的计算节点运行。
先确认分区和资源请求在目标集群可用；脚本不会自动降低资源。不要重复提交同一批任务。

如果手动运行单项，核心调用仍是
`bash run_single_task.sh "$TASK" "$MLEBENCH_DATA" "$SERVER_ID" experiment_preset=mlevolve_original ...`，
其余参数照上面保留。必须从基础配置加载 MCGS preset，不能拿 AQ05 已保存配置仅覆盖 selector。
后者会遗留 `acq7.enabled=true` / rating 并与原版路径冲突。

## 6. 核对、监控、评分和 release

启动后核对 `<RUN_ROOT>/logs/config.yaml`：

```yaml
parent_selector:
  type: legacy_original
  decision_scope: root_only
  include_root_draft: false
bo_parent_selector:
  model: conjugate_bayesian_linear
  agent_rating:
    enabled: false
study7:
  rating:
    enabled: false
  ancestor:
    enabled: false
acq7:
  enabled: false
```

同时核对第 3 节所有共同参数。日志中的 `legacy_original`、`direct_continuation` 或原生
exploration / exploitation 选择记录可用于确认执行路径；不要要求出现 AQ05 的 GP 拟合日志。

```bash
squeue -u "$USER" -o '%.18i %.28j %.2t %.10M %.10l %.24R'
tail -f "$PROJECT/.cache/mlebench-lite/slurm-mcgs/<TASK>-<JOB_ID>.log"
```

运行目录为 `$PROJECT/runs_mcgs_muse13/<timestamp>_<competition-id>/`。
确认搜索及 fusion 完成，冻结 `logs/final_submission.json` 指定的提交后，在评分端执行：

```bash
source tasks/mlebench_lite/env.sh
RUN_ROOT=/absolute/path/to/the/mcgs/run
python -m utils.grade_submission \
  --log-dir "$RUN_ROOT/logs" --data-dir "$MLEBENCH_DATA" \
  | tee "$RUN_ROOT/logs/official-grade.json"
```

沿用 AQ05 的最终提交选择规则，不根据 private score 重选候选或追加搜索。
`evaluation.enabled=false` 仍然需要最终官方评分。exit 124 表示外部时间上限，应另查提交和
fusion 是否完整；缺失、无效提交及其他失败需要如实保留。正式 benchmark 数据隔离应与 AQ05
一致：生成代码只访问 public 数据，private 留给可信验证/评分端；本提交命令自身不建立隔离。

job 结束由 Slurm 自动释放 GPU。提前释放时，先在本轮 `submitted_jobs.tsv` 核对具体 ID：

```bash
MCGS_JOB_ID=替换为本轮目标作业ID
scancel "$MCGS_JOB_ID"
squeue -j "$MCGS_JOB_ID"
sacct -j "$MCGS_JOB_ID" --format=JobID,State,ExitCode,Elapsed,AllocTRES
```

交互式 allocation 在其 shell 中 `exit` 即可退出；必要时用对应 allocation ID 执行
`scancel`。不要取消正在运行另一种方法的作业。

## 7. 返回结果与配对分析

每项交回 code commit、实际 config、依赖版本、GPU 型号、CPU/内存、全部 job ID、起止时间、
实际搜索/fusion 时间、退出原因、journal、最终提交及其 SHA256、官方 grading report。
同时记录有效/失败候选数、可获取的 LLM 用量和任何中断/重跑；MCGS 不继承 AQ05 完成状态。

汇总表建议每个 task 一行：

| task | seed | AQ05 score | MCGS score | 越大/越小越好 | AQ05 medal | MCGS medal | 资源/版本一致 | 运行偏差 |
|---|---:|---:|---:|---|---|---|---|---|
| 待填写 | 42 | — | — | 按官方 metric | — | — | 待核验 | 待填写 |

逐任务判断胜/平/负，再报告这 14 项上的有效提交率和获牌率。不同任务 metric 不直接平均；
筛选后的 14 项结果不能标为完整 22-task Lite 得分。当前单 seed 是一次配对比较，不能据此估计
多 seed 方差。已有 AQ05 中断或代码修订等偏差应随对照结果一起呈现。

配置来源：`config/experiments/mlevolve_original.yaml`、
`config/experiments/acq7_aq05_gp4_level15_rating_kg.yaml`、`engine/node_selection.py` 和
`engine/parent_selectors/runtime.py`。本手册仅提供运行说明，不代表已提交 MCGS 作业。
