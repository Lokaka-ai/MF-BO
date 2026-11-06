# nanoGPT 实验运行与协作交接

核对日期：2026-09-19。适用代码：EvolveFold 仓库内的 `MLEvolve_BO`。本文提供固定候选评估与完整 agent workflow 两条运行路径。

**建议首次使用时依次完成：准备环境和数据 → 跑一个参考候选确认 GPU 评估链路 → 检查 LLM 接口 → 跑小规模 workflow → 再做正式对照。** 本文编写过程中只做源码、配置及无训练的准备检查，没有启动 GPU 训练、LLM 请求或 Slurm 作业。

## 1. 我们运行的 nanoGPT 是什么

这是一个从随机初始化开始、以固定训练时间比较模型/优化器方案的语言模型任务。宿主固定数据、训练循环、计时和正式验证；agent 或实验者提供模型、优化器及允许调整的超参数。

完整 workflow 的主要过程是：

```text
初始化 draft / 选择父节点和算子
        → LLM 规划、生成或修改 candidate module、代码评审
        → 宿主拼接固定 train.py
        → CUDA 训练、因果性和损失检查、正式 BPB 验证
        → 结果评审 / 失败 DEBUG / 更新 memory 与搜索状态
        → BO、MCGS 或贪心方法选择下一次扩展
```

`improve` 修改父节点，`evolution` 利用分支历史，`fusion` 融合分支信息；失败可进入 DEBUG。共享 draft 池共享的是起始代码，各组仍独立训练、评估和修复。

### 固定评估协议

| 项目 | 当前设置 |
| --- | --- |
| 数据集 | `karpathy/climbmix-400b-shuffle` |
| 训练数据 | `shard_00000.parquet` 至 `shard_00009.parquet` |
| 验证数据 | 固定 `shard_06542.parquet` |
| Tokenizer | 仅训练数据训练的 rustbpe/BPE，词表 8192 |
| 上下文长度 | 2048 tokens |
| 单候选训练 | 300 个计时秒，从随机初始化开始 |
| 不计入上述 300 秒 | 前 11 个 step（含预热/编译）；最终验证、保存等另占时间 |
| 训练 seed | 固定为 42，写在宿主训练代码里 |
| 正式验证量 | 20,971,520 tokens |
| 主指标 | `val_bpb`，越低越好；BO utility 通常为 `-BPB` |
| 历史 study7 资源 | 每 workflow 1 张 A100-SXM4-80GB、4 CPU，2 个内部训练 worker 同卡运行 |
| 历史 study7 总预算 | 累计 active 21,600 秒，即 6 小时；不是每个候选 6 小时 |

BPB 是验证负对数似然换算为 bits 后除以对应文本字节数，零字节特殊 token 按固定 evaluator 的规则处理。它不是直接把 token 平均交叉熵改名为 BPB。

**计时务必分清三层：**候选的 300 秒训练计时；包含编译、验证、保存的候选总耗时；包含 LLM、训练、评审和失败的 workflow 总预算。固定时间训练意味着吞吐越高训练 token 越多，GPU 型号和同卡并发数都是实验设置的一部分。

任务文本 `TASK.md` 仍有“8 小时 campaign”的历史描述，基础 `config/config.yaml` 默认 `agent.time_limit=43200`（12 小时）。study7 实际通过命令显式设置 21600。**新实验每次都写出 `agent.time_limit`，最终以保存的 `logs/config.yaml` 和预算记录为准。**

## 2. 交接代码与环境

需要拿到本项目的 `MLEvolve_BO` 代码及 `uv.lock`。只有上游 MLEvolve 仓库不足以包含这里的 nanoGPT 宿主、BO 变体和研究运行工具。

本文核对时仓库 Git HEAD 为 `3121d1e9fc87cc7cc199f6d6e695cd520f0a34fb`，工作区同时存在未提交的 Branch7 开发文件；**单独记录 HEAD 无法唯一标定当前工作区**。正式实验应保存实际源码快照/哈希、有效配置和共享 draft 池。

若目标是复现旧 study7 的实现，应使用 `MLEvolve_BO/research/study7_20260918/source/` 的冻结代码及对应配置。这里的冻结 `adapter.py`、`worker.py`、三个训练/数据协议文件、launcher 和 `uv.lock` 在本文核对时与当前版本相同；搜索实现已有新增开发。本文不要求合作者复用原研究的运行目录或恢复旧任务。

已核对的运行环境：

| 依赖 | 版本/要求 |
| --- | --- |
| Python | 项目要求 3.12；已记录环境为 3.12.12 |
| PyTorch | `2.9.1+cu128`，由 uv 的 PyTorch CUDA 12.8 索引安装 |
| sentence-transformers | 2.6.1 |
| transformers | 4.57.1 |
| faiss-cpu | 1.11.0 |
| OmegaConf | 2.3.0 |
| 系统工具 | bash、uv、CUDA 可用的 NVIDIA 驱动、`nvidia-smi`、`taskset`、`flock` |
| 完整 workflow 额外需要 | 可执行的 Codex CLI、可用的 OpenCode Go 模型凭据、网络连接 |
| memory embedding | `BAAI/bge-base-en-v1.5`，默认放在 CUDA；它是检索模型，不是 nanoGPT 的预训练权重 |

以 `uv.lock` 为依赖依据，不要直接用当前 shell 的其他 Conda Python 运行。`setup.sh` 会 `uv sync --frozen`、准备数据与 tokenizer，并实际检查 CUDA 和 embedding，因此需要在已分配 GPU 的环境里执行。

先设定自己机器上的路径。本文之后的终端命令默认在同一会话中执行：

```bash
export NANOGPT_PROJECT=/absolute/path/EvolveFold/MLEvolve_BO
cd "$NANOGPT_PROJECT"
export AUTORESEARCH_CACHE_DIR="$NANOGPT_PROJECT/.cache/autoresearch"

# 若使用本交接包，设置为解压后脚本目录；仓库内默认如下。
export NANOGPT_HANDOFF="$NANOGPT_PROJECT/docs/nanogpt_handoff"

bash tasks/autoresearch/setup.sh
uv run --frozen python tasks/autoresearch/manifest.py --verify
```

默认数据和 tokenizer 共 13 个文件，本次 manifest 合计约 **1.01 GB**。Python/CUDA 环境、embedding、编译缓存和每个 trial 的 `checkpoint.pt` 另占空间；模型权重会随候选数累计。

可以从现有实验者处复制已准备的 `data/`、`tokenizer/` 和 `manifest.json` 到自己的缓存，然后运行 `manifest.py --verify`。这也避免重新训练 tokenizer 带来差异。当前协议要求缓存中正好是 10 个训练 shard 加固定验证 shard，随意增加或减少 shard 会校验失败。

初次准备需要联网。等数据和 embedding 都已缓存后，才可在正式 workflow 中设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；这两个变量只针对 Hugging Face，LLM 接口仍需网络。

## 3. 路径 A：只跑一个固定候选，不调用 LLM

适用于确认训练环境、手工改模型/优化器、复评已有 `candidate.py`。随文提供的 `prepare_trial.py` 只拼装文件，`single_trial.sh` 才执行正式训练和验证。

在**自己新分配的 GPU** 上指定 UUID。`run_autoresearch.sh --gpu` 接受物理 GPU 编号或 UUID；Slurm 中 CUDA 的逻辑编号 0 不一定是节点的物理编号 0，所以跨节点操作推荐 UUID。

```bash
cd "$NANOGPT_PROJECT"
export NANOGPT_GPU=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export NANOGPT_CPUS=4
export NANOGPT_TRIAL="$NANOGPT_PROJECT/runs/manual_trial_$(date +%Y%m%d_%H%M%S)"

# 默认从固定参考 train.py 提取可编辑 candidate body，不执行训练。
uv run --frozen python "$NANOGPT_HANDOFF/prepare_trial.py" \
  --project "$NANOGPT_PROJECT" --trial "$NANOGPT_TRIAL"

# 真正训练并产生正式 result.json 和 checkpoint.pt。
bash "$NANOGPT_HANDOFF/single_trial.sh"
```

从已有候选复评时，改用一个新的 trial 目录，并在准备命令增加：

```bash
uv run --frozen python "$NANOGPT_HANDOFF/prepare_trial.py" \
  --project "$NANOGPT_PROJECT" \
  --trial "$NANOGPT_TRIAL" \
  --candidate /absolute/path/to/candidate.py
```

后一个命令是前一个准备命令的替代，不要在已准备的同一目录上重复执行。每次试验使用新目录。修改模型时保存新的 candidate module，再重新组装；不要直接改生成后的 `train.py` 或固定 evaluator。

脚本会校验数据 manifest，使用新建的节点本地编译缓存，沿用同一 checkout 的 GPU 文件锁，运行正式 `worker.py` 并再次调用 `validate_result`。它不会申请 GPU，也不能代替 Slurm 资源分配；该文件锁只协调同一 checkout 的运行。

单候选结果在 `$NANOGPT_TRIAL/`，可读日志：

```bash
tail -n 40 "$NANOGPT_TRIAL/train.log"
uv run --frozen python -m json.tool "$NANOGPT_TRIAL/result.json"
```

重点核对 `valid=true`、`model_contract_check.passed=true`、`eval_tokens=20971520`、`training_seconds≥300`，以及 `val_bpb`、`num_steps`、`total_tokens`、`total_seconds`、`peak_vram_mb`。

这条路径是一个训练 worker 独占 GPU。study7 是两个 worker 同卡，二者吞吐条件不同；性能对照需要统一并发条件，不能只因同为 A100、同为 300 秒就当作完全相同 setting。

## 4. Candidate 可以改什么

候选模块可以定义模型结构、优化器和协议允许的超参数。宿主将其放到参考文件的 `# GPT Model` 与 `# IMMUTABLE HARNESS` 边界之间。已有完整接口见 `tasks/autoresearch/TASK.md`。

必须保留的主要接口：

- `GPTConfig` dataclass 接受 `sequence_len, vocab_size, n_layer, n_head, n_kv_head, n_embd, window_pattern`；新增字段应有默认值。
- `GPT(config)` 支持 meta-device 构造、`to_empty(device='cuda')`、`init_weights()`、`num_scaling_params()['total']`、`estimate_flops()`。所有参数和 buffer 都需正确初始化。
- `forward(idx, targets=None, reduction='mean')`：无 targets 返回 logits；训练返回实际交叉熵；`reduction='none'` 返回逐 token 损失。
- `setup_optimizer(...)` 保持规定参数接口。每个 optimizer group 包含 `lr`、`initial_lr`、`kind`；宿主用 `initial_lr` 更新学习率。
- 定义规定的架构、batch 和优化超参数，包括 `DEPTH`、`ASPECT_RATIO`、`HEAD_DIM`、`TOTAL_BATCH_SIZE`、`DEVICE_BATCH_SIZE` 等。

当前参考候选是 8 层、512 维、4 heads，词表 8192；Ampere 及更新设备使用 BF16，默认 microbatch=16 sequences、总 batch=65,536 tokens。它只是可运行起点，agent 生成的候选可以不同。

`DEVICE_BATCH_SIZE` 必须整除 10240，`TOTAL_BATCH_SIZE` 必须是 `DEVICE_BATCH_SIZE×2048` 的正整数倍，`HEAD_DIM` 必须为偶数。架构变化还要满足实际 attention 的 head/维度约束。OOM 时优先检查同卡两个 trial 的合计占用、模型大小和 microbatch。

保持数据、tokenizer、训练 seed、300 秒计时、训练循环、验证 token 数和 BPB 计算不变；不载入预训练权重，不训练验证集。代码需保持因果 next-token 预测。宿主会检查多个前缀切点的因果性，以及 loss 是否等于 logits 的逐 token 交叉熵。

**改变 `agent.seed` 不会改变训练 seed。** 若要研究不同训练 seed，需单独版本化宿主协议并新开实验；不能给原 study7 标签悄悄换训练 seed。普通 workflow 会清除 `EVOLVEFOLD_TRIAL_CONFIG`；不要把任意环境变量当作已登记的正式超参数接口，应保存 candidate 源码并核对实际 `result.json`。

配置描述也不替代代码检查。例如旧 E04 冠军虽然登记 `window_pattern=SSSL`，其 candidate 实际执行全局 causal attention；研究窗口注意力或 FLOPs 时应检查实现。

## 5. 路径 B：完整 agent workflow

### 5.1 配置 LLM

本任务 overlay 的 code/feedback 模型均为 `muse-spark-1.3-contributor`，`base_url=codex://opencode-go`。调用链是 MLEvolve → 仓库的 Codex wrapper → OpenCode Go。它需要可用的 Codex CLI，但不等于读取本次聊天会话。

```bash
cd "$NANOGPT_PROJECT"
command -v codex

# 文件内容只放自己的原始 API key，勿写进实验 YAML 或共享文档。
export OPENCODE_API_KEY_FILE=/absolute/path/to/your_opencode_key

# 若 codex 不在 PATH，指定真实可执行文件。
# export MLEVOLVE_REAL_CODEX_BIN=/absolute/path/to/codex

# 与历史 study7 campaign 的设置对齐。
export MLEVOLVE_CODEX_REASONING_EFFORT=high
export MLEVOLVE_CODEX_TIMEOUT_SECONDS=1800

# 此检查会发起一次真实模型请求；成功输出 OPENCODE_MUSE_OK。
uv run --frozen python tools/check_opencode.py
```

也支持 `OPENCODE_API_KEY` 环境变量。wrapper 有原作者本地 `../codex/opecode` 文件回退路径，合作者应显式设置自己的凭据位置。wrapper 不修改用户的 Codex home。项目 launcher 的未显式覆盖默认是 `medium/600`，因此正式复现应保留上面的 `high/1800` 设置。

memory 初始化失败会停止 nanoGPT workflow；应检查 embedding 是否缓存、是否可用及 CUDA 显存。临时改为 CPU 或关闭 memory 都属于实验设置变化，应单独记录。

### 5.2 小规模通路实验

下面会实际调用 LLM 并训练；最多 8 个原生 trial 步数，上限包括初始 drafts，且并不保证得到 8 个有效结果。2 小时是总预算上限，耗尽时可能尚未完成全部期望阶段。

```bash
cd "$NANOGPT_PROJECT"
export NANOGPT_RUNS="$NANOGPT_PROJECT/runs/collaborator_smoke"

bash run_autoresearch.sh \
  --gpu "$NANOGPT_GPU" --cpus 4 --steps 8 \
  --name colleague_smoke_E02 --runs-dir "$NANOGPT_RUNS" \
  experiment_preset=seven_e02_state4_ucb \
  agent.time_limit=7200 agent.seed=42 \
  agent.search.parallel_search_num=2
```

该命令自动添加 `task_preset=autoresearch`，验证数据并建立独立编译缓存。`--steps` 是 workflow trial 数量上限，不是一个 nanoGPT 的 optimizer step 数，也不是让每个候选只训练这么多步。

### 5.3 正式运行一组

```bash
cd "$NANOGPT_PROJECT"
export NANOGPT_RUNS="$NANOGPT_PROJECT/runs/collaborator_formal"

bash run_autoresearch.sh \
  --gpu "$NANOGPT_GPU" --cpus 4 --steps 500 \
  --name colleague_E02_seed42 --runs-dir "$NANOGPT_RUNS" \
  experiment_preset=seven_e02_state4_ucb \
  agent.time_limit=21600 agent.seed=42 \
  agent.search.parallel_search_num=2
```

实际新目录名为 `YYYYMMDD_HHMMSS_<name>`，其下是 `logs/` 和 `workspace/`。更换方法时改变 preset 和 name，保持数据、资源、并发、预算、LLM 和起始池一致。想做单 worker 运行可显式设 `agent.search.parallel_search_num=1`，同时在结果中标记与双 worker 的设置差异。

## 6. 方法名：旧 study7 与新 Branch7 分开

### 已完成 study7 的七组

| ID | `experiment_preset` | 方法 |
| --- | --- | --- |
| E01 | `seven_e01_gp6_original` | 原 GP6，普通动作使用 latent UCB |
| E02 | `seven_e02_state4_ucb` | 四维状态 GP |
| E03 | `seven_e03_state4_ancestorB_ucb` | State4 + ancestor B 核 |
| E04 | `seven_e04_state4_ratingmean_ucb` | State4 + 已完成节点评分的均值/协方差项 |
| E05 | `seven_e05_state4_ancestorB_ratingmean_ucb` | State4 + B + 评分项 |
| E06 | `seven_e06_mcgs_original` | 原生 MCGS/Top-K/continuation 路径 |
| E07 | `seven_e07_parent_greedy_managed` | managed 调度路径，按父节点质量贪心，无 GP |

E01–E05 这轮线上使用固定 `μ+1.5×latent_std`；此前报告里的 β 扫描和分支配额是离线分析。E02–E05 的 GP 标签是有效普通 proposal 相对于原父节点的标准化收益，DEBUG 修复归回原 proposal；失败、在途和预算外结果不会被强行当作质量 0。

study7 先消费 3 个初始 draft，shared pool 可有 5 份代码。GP/E07 的普通候选池不含 root draft，D4/D5 不会因增大 β 自动出现；MCGS 原生 draft 路径曾实际生成 D4/D5。

**固定方法有配置校验。** E02–E05/E07 的注册实现明确要求 fixed β=1.5 等方法身份；例如覆盖 `bo_parent_selector.acquisition.beta=10` 会被拒绝。研究 β 衰减、早期覆盖配额或改变 kernel，应新增版本化方法及对应选择/验证代码、日志字段，不能把未实现的参数放进命令行后当作已经生效。特别是离线报告中的 `initial_branch_quota` 是设计建议，不是现有通用 CLI 开关。

### 当前工作区另有 BR01–BR07

它们是独立的 branch random-effect 方法，不是把旧 E01–E07 重新命名。这里仅列已核对的配置入口，新增开发文件的正式版本需在交接时固定。

| ID | preset | 基础 GP / 组件 | branch prior std τ |
| --- | --- | --- | --- |
| BR01 | `branch7_br01_gp6_tau15` | GP6 | 1.5 |
| BR02 | `branch7_br02_gp4_tau15` | GP4 | 1.5 |
| BR03 | `branch7_br03_gp4_ancestorB_tau15` | GP4 + ancestor B | 1.5 |
| BR04 | `branch7_br04_gp4_rating_tau15` | GP4 + rating | 1.5 |
| BR05 | `branch7_br05_gp4_ancestorB_rating_tau15` | GP4 + B + rating | 1.5 |
| BR06 | `branch7_br06_gp6_rating_tau15` | GP6 + rating | 1.5 |
| BR07 | `branch7_br07_gp4_rating_tau20` | GP4 + rating | 2.0 |

**τ 是分支先验标准差，β 是 acquisition 系数。** Branch7 注册配置仍固定 β=1.5；`tau20` 不表示 β=2.0。这七个入口全部是 GP 方法；MCGS 对照仍使用 E06。

## 7. 共享起始 draft，控制方法间的起点差异

正式方法比较建议使用同一个已封存的 draft pool。生成池会调用 LLM 和代码评审，不训练 nanoGPT，也不按训练分数筛选草稿；因此池里仍可能有需 DEBUG 的候选。

```bash
cd "$NANOGPT_PROJECT"
export NANOGPT_POOL="$NANOGPT_PROJECT/runs/collaborator_pool_$(date +%Y%m%d_%H%M%S)"

uv run --frozen python tasks/autoresearch/prepare_draft_pool.py \
  --output "$NANOGPT_POOL"

bash run_autoresearch.sh \
  --gpu "$NANOGPT_GPU" --cpus 4 --steps 500 \
  --name colleague_E02_shared --runs-dir "$NANOGPT_PROJECT/runs/collaborator_shared" \
  experiment_preset=seven_e02_state4_ucb \
  agent.time_limit=21600 agent.seed=42 \
  agent.search.parallel_search_num=2 \
  "shared_draft_pool=$NANOGPT_POOL/manifest.json"
```

若复用现有池，复制整个 pool 目录并验证其 manifest/code 哈希。仅复制 `manifest.json` 不够，里面引用了草稿文件。精确复现旧池还应使用匹配的协议源码；路径无须与原作者相同，内容哈希应匹配。

每组独立执行相同起始代码，训练结果可能因吞吐和运行时变化而不同。生成/评审草稿的记录延迟会在消费时计入各组 active 预算。初始失败后的 DEBUG 也可能产生不同代码，因此评价 BO 组件时需单独报告初始化和修复阶段的差异。

## 8. Slurm 上提交一个新的独立 workflow

随文 `workflow.sbatch` 只启动一个新 workflow，不调用历史 campaign controller，不自动扩展资源或恢复旧研究。提交前完成环境/数据准备，指定自己的 partition/account 和资源。

```bash
export NANOGPT_PROJECT=/absolute/path/EvolveFold/MLEvolve_BO
export NANOGPT_HANDOFF="$NANOGPT_PROJECT/docs/nanogpt_handoff"
export NANOGPT_RUNS="$NANOGPT_PROJECT/runs/colleague_slurm"
export NANOGPT_PRESET=seven_e02_state4_ucb
export OPENCODE_API_KEY_FILE=/absolute/path/to/your_opencode_key
export NANOGPT_SECONDS=21600
export NANOGPT_WORKERS=2
export NANOGPT_WORKFLOW_SEED=42

# 已有共享池时设置；不使用共享池则省略。
# export NANOGPT_DRAFT_POOL=/absolute/path/to/pool/manifest.json

sbatch --partition=YOUR_PARTITION \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --gpus=a100:1 --mem=24G --time=08:00:00 \
  --job-name=colleague-nanogpt-E02 \
  --output=colleague-nanogpt-%j.out \
  "$NANOGPT_HANDOFF/workflow.sbatch"
```

`YOUR_PARTITION`、GPU 请求语法、账号和内存上限按本地集群调整。8 小时是作业申请示例，用于为 6 小时 active workflow 留出启动和结束空间，并非实际队列可用性保证。脚本从 Slurm 暴露给 CUDA 的唯一设备取得 UUID，再传给 launcher，避免把逻辑 GPU 0 当成物理 GPU 0。

一个 workflow 内部 2 workers 共享一张 GPU；若希望两个方法同时运行，应给每个 workflow 分配独立 GPU 和 CPU，不要直接在同卡启动两个完整 workflow 来充当对照。新实验使用新输出目录；原 `research/study7_20260918`、`research/branch7_20260919` 的 controller、allocation manifest 和状态文件是已有研究记录。

短时限/可抢占 partition 上的正式跨 allocation 运行还涉及 `segmented.py`、`segment_supervisor.py`、`study_worker.py` 和累计 active ledger。它不是简单重复提交上述脚本：重复提交会新开 workflow。需要续接时按独立 campaign 登记资源与检查点，保存排队不计时、失败/中断工作计时的原规则。

## 9. 结果、日志和恢复

普通 launcher 创建的核心目录如下；部分附加文件只在相应方法或 segmented 模式出现：

```text
runs/<时间戳_实验名>/
  logs/
    config.yaml                  实际合并后的配置
    autoresearch_protocol.json   数据、harness、worker 的协议哈希
    MLEvolve.log                 主运行日志
    MLEvolve.verbose.log         详细日志
    journal.json                 节点、图结构与可恢复状态
    best_solution.py             当前最好候选模块
    llm_calls.jsonl              LLM 调用和可获取的 token 用量
    shared_draft_usage.json      若使用共享起始池
    autoresearch_summary.json    正常 finalize 时写入，可能尚不存在
    phase1/
      selection_decisions.jsonl
      selection_snapshots.jsonl
      transitions.jsonl
      continuation_events.jsonl
      model_fits.jsonl
      study7_decisions.jsonl     对应方法的诊断
      study7_proposals.jsonl
      study7_dispatch.jsonl
  workspace/
    trials/<node_id>/
      candidate.py               agent/实验者提供的模块
      train.py                   固定宿主与模块拼装后的实际运行源码
      prepare.py, trial_config.py
      resources.json             workflow interpreter 的资源分配
      train.log
      result.json                完成正式验证后原子写入
      checkpoint.pt
```

看总体进度先看主日志和 journal，看某个候选失败原因看它自己的 `train.log`。没有 `result.json` 可能还在训练/验证，也可能已经失败，不能只据文件缺失判断。`autoresearch_summary.json` 缺失也不等于全部训练失败：进程可能未走到 finalize，历史分段研究还有各自的汇总和预算文件。

分析正式结果时要区分原始训练分数和预算内被接纳的结果。study7 的预算外完成结果可能有完整 `result.json`，却标记为 censored，不进入 best-so-far 或 GP 标签。DEBUG 不应当作同一个普通 proposal 的第二个独立收益标签；被回滚/中断的副本也不应重复计数。

MCGS 的回传 Q、UCT bonus 和日志里部分占位 `selected_score=0` 不等于 BPB 或 GP 方差。性能统一看正式 BPB；预测覆盖率分析需要明确原生预测和额外离线模拟的区别。

恢复一个已经停止的普通运行，传入**包含 logs/ 和 workspace/ 的 per-run 根目录**：

```bash
bash run_autoresearch.sh \
  --gpu "$NANOGPT_GPU" --cpus 4 --steps 500 \
  --resume /absolute/path/to/runs/TIMESTAMP_colleague_E02_seed42
```

这恢复 workflow 状态；新候选仍从随机初始化训练，不是用最佳 `checkpoint.pt` 继续预训练。数据/协议、方法身份和共享池必须匹配，且旧执行者必须已经退出。不要把“新方法实验”实现成恢复另一个方法的旧 run。

launcher 总会传入 `--steps`、CPU、name 等参数；如旧实验使用非 500 的步数上限，恢复时显式传回原上限。保存配置中的 worker 数通常保留，除非显式覆盖。查看 `logs/config.yaml` 和恢复日志确认最终值。

## 10. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 无 CUDA / worker 看到多张 GPU | 是否在分配好的计算节点，是否只暴露本 workflow 的一张卡；launcher 的物理编号与 CUDA 逻辑编号是否混用 |
| `GPU ... already has an autoresearch run` | 同一 checkout 的同一 GPU 已被另一运行持锁；使用自己分配的空闲资源 |
| manifest 缺失或不匹配 | 缓存路径、13 个文件是否齐全、是否混入其他 shard/tokenizer；使用匹配数据或单独新缓存 |
| provider/key/429/timeout | `OPENCODE_API_KEY_FILE`、Codex 路径、模型可用性和网络；先看 `check_opencode.py` 的结果，保留失败记录 |
| memory 初始化失败 | embedding 文件缓存、CUDA 显存和配置路径；不要静默关掉 memory 后与原方法直接对比 |
| OOM | 同卡两个 trial 的合计模型、optimizer 和 batch 占用；修改候选并记录设置 |
| GPU 长时间忙但未出现正式 BPB | 可能正在编译或做约 2,097 万 tokens 的验证；查看 trial 日志当前阶段 |
| `initial_lr` / `kind` 缺失 | 自定义 optimizer group 不满足宿主 API |
| batch 整除断言失败 | microbatch 是否整除 10240，总 batch 是否整除 microbatch×2048 |
| 因果性或 loss 检查失败 | future-token 泄漏、logits/loss 不一致、reduction='none' 实现错误 |
| 改 β 或 τ 后方法身份报错 | 注册方法锁定配置；新方法要单独实现/登记，不能只改命令行 |
| 同 seed、同代码分数不完全相同 | 固定时间导致实际训练步数不同；比较并发、吞吐、token 数和初始化/DEBUG |
| 磁盘快速增长 | 每个已完成 trial 都保存模型 checkpoint，另有编译缓存；交接分析通常优先保留日志、源码、配置、result 和必要模型 |

## 11. 交回实验结果时保留什么

每次新实验记录：源码 commit 与实际变更/源码哈希、preset、有效 `config.yaml`、数据 manifest、shared pool manifest/code、LLM 模型及 reasoning/timeout、workflow seed 与训练 seed、GPU 型号/CPU/worker 数、总预算与计时口径。

保留全量节点日志与结果，包含失败、DEBUG、在途/截尾和回滚信息；保留每个已评估候选的 `candidate.py`、实际 `train.py`、`result.json`、`train.log`。需要恢复工作流时还需原检查点、queue、memory、ledger 和完整 per-run 状态，仅有最佳模型权重不足以恢复搜索。

若修改训练协议、并发或资源，把它当作新的实验 setting 明确标记。模型优劣与 BO/MCGS 选点方法优劣分别分析，尤其要单列初始 draft 和 DEBUG 提供的起点。

改代码后的 CPU 检查入口示例（测试使用替代训练/LLM 服务，不代替真实 GPU 通路检查）：

```bash
uv run --frozen python -m pytest -q \
  tests/test_autoresearch.py tests/test_autoresearch_causality.py \
  tests/test_shared_drafts.py \
  tests/test_study7_math.py tests/test_study7_runtime.py
```

## 12. 源码与既有分析入口

路径均相对于 `MLEvolve_BO/`，除特别注明：

| 入口 | 用途 |
| --- | --- |
| `tasks/autoresearch/README.md` | 原任务说明 |
| `tasks/autoresearch/TASK.md`、`TASK_SERIAL.md` | 给 agent 的候选协议与资源约束 |
| `tasks/autoresearch/SOURCE.md` | 训练程序来源 |
| `tasks/autoresearch/repo/prepare.py` | 数据、tokenizer、固定正式 BPB |
| `tasks/autoresearch/repo/train.py` | 参考候选与固定训练循环 |
| `tasks/autoresearch/worker.py` | 因果性检查、正式 result 和 checkpoint |
| `tasks/autoresearch/adapter.py` | 候选拼装、执行、结果接纳 |
| `run_autoresearch.sh`、`tasks/autoresearch/launch.py` | 单 workflow 入口、GPU 锁、CPU 和编译缓存 |
| `config/config.yaml`、`tasks/autoresearch/config.yaml` | 基础配置与任务 overlay |
| `config/experiments/seven_*.yaml`、`config/study7.py` | 旧七组方法及固定身份校验 |
| `config/experiments/branch7_*.yaml`、`config/branch7.py` | 新 Branch7 的独立配置 |
| `tools/check_opencode.py`、`tools/codex_opencode.py` | LLM 通路检查和 wrapper |
| `tasks/autoresearch/study7_budget.py` | 预算内/外结果接纳 |
| `research/study7_20260918/source/` | 历史 study7 冻结源码；研究工件通常不在 Git 中 |
| `../exports/study7_analysis_20260919/REPORT.html` | 已有七组结果与初始化差异分析 |
| `../exports/study7_analysis_20260919/draft_mcgs_analysis/REPORT.html` | β 可达性、MCGS 与预测覆盖率 |
| `../exports/study7_analysis_20260919/branch_exploration/REPORT.html` | 离线分支配额/奖励/随机探索；不等于线上已实现 |

配套脚本都位于本文件旁的 `nanogpt_handoff/`：`prepare_trial.py` 只准备文件；`single_trial.sh` 执行一个固定候选；`workflow.sbatch` 提交后执行一个完整 workflow。交接包可单独阅读，但执行仍需项目代码、环境、数据和自己的计算/模型服务资源。
