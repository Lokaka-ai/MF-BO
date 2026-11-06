# MLEvolve_BO：实验 A / B 实现与运行

本次以用户上传的 `EvolveFold-xixian-dev (3).zip` 为基准。只修改 `MLEvolve_BO`，不改原始 `MLEvolve`、`MLEvolve_BO_agent` 或其他子项目。不申请计算资源，不调用真实 LLM，不启动 GPU 或 MLE-bench 实验。

## 两个原生配置

| 实验 | 配置文件 | 内部 selector ID |
|---|---|---|
| A：Exact GP7 + ancestry + UCB | `config/experiments/gp7_ancestry_ucb.yaml` | `gp7_ancestry_logei` |
| B：Laplacian anchor + bounded kernel uncertainty + LogEI | `config/experiments/laplacian_anchor_logei.yaml` | `laplacian_anchor_logei` |

这两个 YAML 是项目已有 `experiment_preset` 机制加载的覆盖配置，不是另一个框架的配置格式。原生 `_load_cfg`、runtime 路由、task/suite 脚本已经接入。它们在 selector 类型、`gp.max_optimizer_iterations` 和 acquisition 三个设置上不同。A 使用 `child_ucb`，B 使用 `logei`；内部 selector ID 保留历史值以兼容 checkpoint。不要混淆旧 `halfshrink_ucb` / `gp6_global_logei` 的历史命名与本轮 A/B。

两组均为 `all_normal_steps`，普通候选池禁用 root draft，使用预览出的合法 improve/evolution/fusion。初始 draft、必要的 feasibility draft、debug、aggregation 继续沿用原执行路径；root draft=false 不意味着禁止这些已有特殊路径。

## A 的实现

- 七维数值输入：`parent_gap, last_gain, recent_gain, stagnation, trials, valid_rate, absolute_parent`。
- `absolute_mode=auto`：最大化使用 parent utility；`exp_id=aptos2019-blindness-detection` 使用 `(u+1)/2`；最小化非负 loss 使用 `-log1p(loss/c0)`。`c0` 用冻结参考初始有效 seed 的绝对 utility 中位数、下限 `1e-6`，不使用未来观测最大值。
- `u(parent)` 是已知均值基线。训练目标为有效普通 proposal 的 `(u(child)-u(parent))/S`，不是原 MCGS reward 或后代最大分数。
- 核：`amplitude² * operation_corr * [0.7*Matern32(ell) + 0.3*ancestry(gamma=.5)]`。
- operation 同类相关性 1、异类 0.25；ancestry 使用归一化向量内积，排除虚拟 root，debug 修复不新增独立研究深度。
- 三个标量参数 `ell, amplitude, noise_std` 按原 GP6 规则做有界 MAP：起始 `1,1,.25`；边界 `[.25,4], [.1,5], [.05,2]`；至少 8 个有效普通样本后优化，每次最多 25 次迭代。不是另加价值神经网络。
- 输出 parent baseline + GP residual mean、latent SD、predictive SD。UCB 按 `mu + 1.5 * sigma_latent` 排序。不加成功率或成本权重。

## B 的实现

- 核使用与 A 相同的七维状态、operation、primary ancestry。
- 额外使用决策前已反馈的节点分数与 primary 边，构造无向单位边图；不读取未反馈 journal 节点分数。
- 解 `m = (I + 0.3 L)^(-1) u`，取候选 parent 的位置作为预测中心。**不改写真正的节点分数或 incumbent。**
- 每次决策将中心冻结在 `gp_graph_center`；提案实际完成后计算 `e_j = outcome_j - frozen_center_j`。后来的图不能重算历史中心。
- `tau = max(1e-6*S, min(RMS(e),5*S))`。约束探索尺度，不裁剪、删除或更改原始结果标签。
- 固定 `ell=1`、ancestry weight `.3`、decay `.5`、operation correlation `.25`、covariance ridge `.0625`。
- `q = max(1 - k @ solve(K + .0625 I, k),1e-12)`。
- `sigma_alloc=tau*sqrt(q)`；仅作预测分布诊断的 `sigma_pred=tau*sqrt(q+.0625)`。
- global LogEI 使用 `m` 和 `sigma_alloc`，`xi=0`。不加成功率或成本权重。
- 不拟合 GP 响应均值，不做边际似然优化，也不训练另一个 variance GP。配置中 `model: gp_matern32` 复用已有 GP runtime 生命周期；它不意味着 B 是标准 exact GP 的自洽后验。实际 selector 名称、日志及 `is_exact_gp_posterior=false` 区分了两者。

## 反馈、冷启动与恢复

两个方法都只拟合**已反馈、有效、已闭合的普通 proposal**。初始 draft 可以校准同一套冻结 `S`，但不作为质量模型训练样本。debug 成功继承原 proposal 的状态、parent、operation 和冻结中心，只增加一个有效训练观测；未解决链条不提前伪造质量标签。

保留现有初始 draft / feasibility / debug 流程，没有增加新的 warmup policy 或最小样本重选。没有有效普通反馈时，A 用 GP prior，B 用同样归一化 prior SD `S`；取得普通反馈后分别按各自公式更新。B 在 1–4 个样本阶段的公式是线上冷启动扩展，此前离线报告从至少 5 个样本开始评分，因此不能把这段称为已经离线验证的策略收益。A 仍到 8 个样本才开始调超参数。

`gp_primary_ancestry`、方向、图版本和中心冻结到 decision/transition。SearchNode 的特征字典类型更新为 `Any`，支持小列表和布尔值，不将 ancestry 当作数值 feature。断点保存冻结 utility scale、loss scale、参数和 selector 契约；恢复重建矩阵，不缓存另一个 GP 状态。拒绝 A/B 互相恢复、旧 GP6 日志自动迁移，以及已保存契约下偷偷改变参数。两组实验从新的 run 开始。

模型数值错误沿用合法候选池内的 parent-quality fallback，并记录 `fallback_reason`。配置错误尽量在初始化时拒绝；root draft、成本/成功率开关、错误 acquisition、B 启用 optimizer、B noise/ridge 不一致均不能悄悄启用。LogEI 模式的分数按 `log(EI/S)` 存储，与原始单位的 LogEI 只差同轮公共常数，不影响排序。

## 运行一个任务

在现有可运行 MLEvolve_BO 的环境中使用。两个 YAML 不更换 `config/config.yaml` 的 LLM/provider、CPU/GPU、budget、seed、prompts 或评估协议；原有 CLI 覆盖继续有效。原文件默认 code/feedback model 仍是其原值，没有替用户选择新模型。

```bash
cd MLEvolve_BO

# 示例值；与自己的 baseline 预算及已分配 CPU 数量保持一致。
export TIME_LIMIT_SECS=43200
export CPUS_PER_TASK=21
export MEMORY_INDEX=0
export START_CPU_ID=0
DATASET_ROOT=/path/to/mle-bench
TASK=aerial-cactus-identification

# 实验 A；其余 provider/model 覆盖可按原方式接在末尾。
bash scripts/run_selector_task.sh gp7_ancestry_ucb \
  "$TASK" "$DATASET_ROOT" 111 agent.seed=42

# 实验 B；同时运行时应在另一个已分配 worker/GPU 上执行，使用独立端口。
bash scripts/run_selector_task.sh laplacian_anchor_logei \
  "$TASK" "$DATASET_ROOT" 112 agent.seed=42
```

包装脚本沿用原来的数据布局 `${DATASET_ROOT}/${TASK}/prepared/public`、description 路径、grading server、timeout 和后处理，不新增 scheduler。输出目录分别为 `MLEvolve_BO/runs_gp7_ancestry_ucb/` 和 `MLEvolve_BO/runs_laplacian_anchor_logei/`。

已有自建调度器可继续调用原入口，仅替换 preset：

```bash
python run.py experiment_preset=gp7_ancestry_ucb \
  exp_id="$TASK" dataset_dir="$DATASET_ROOT" \
  data_dir="$DATASET_ROOT/$TASK/prepared/public" \
  desc_file="$DATASET_ROOT/$TASK/prepared/public/description.md" \
  exp_name="A_$TASK" log_dir=./runs_gp7_ancestry_ucb workspace_dir=./runs_gp7_ancestry_ucb \
  agent.seed=42
```

B 则替换 preset 和输出目录。直接调用 `run.py` 时保留自己的 grading server 和资源启动方式。提供 `exp_id`，以便 APTOS 的 absolute feature 正确选择 QWK 表示。不要把预设 YAML 当作 `run.py config=...` 的新参数：原仓库没有这样的入口。

## 跑完整的 22 个任务

```bash
# Worker A：
bash scripts/run_selector_suite.sh gp7_ancestry_ucb \
  "$DATASET_ROOT" 111 agent.seed=42

# Worker B：
bash scripts/run_selector_suite.sh laplacian_anchor_logei \
  "$DATASET_ROOT" 112 agent.seed=42
```

这两个 suite 使用未修改的 `config/mlebench_lite22.txt`，各自在单个 worker 内**顺序**跑 22 个任务，不代表 22 个并行任务。需要并行时，沿用现有调度器逐任务调用 task wrapper，并分配独立 GPU/CPU/端口。脚本不申请或改变 allocation。保留相同配对 seed；单次配对仍不足以消除 agent 生成随机性。

## 验证与边界

已执行：

- `python -m unittest discover -s tests_gp6 -v`：76 项通过，包含原 51 项和新增 25 项。覆盖核 PSD/归一化、七维输入、极端尺度、A 的矩阵公式、B 无优化器、debug 冻结归因、JSONL、合法池 fallback、配置、原生 runtime 类体、checkpoint 契约、snapshot 和真实 shell wrapper 路由。
- `python scripts/smoke_graph_selectors.py`：两组数值选择器通过；无 API/GPU。可用 `--output report.json` 保存输出。
- 与上一轮离线源代码对照：35 个 GP6 来源 run，每个取首个/最后一个可评估时点，共 69 个时点 × 2 方法 = **138 组核对**。七维矩阵、primary ancestry、均值/标准差、所选父节点一致，最大预测绝对差 `5.551115123125783e-16`。这是同一冻结尺度下的实现一致性核对，不是新在线性能实验。
- Python 编译与 shell `bash -n` 检查通过。

未完成：原生 `AgentSearch` 集成测试在本容器导入阶段因缺少原项目依赖 `dataclasses_json` 停止，尚未执行测试体；没有因此给完整 agent 测试报通过。六个原生测试已放在 `tests/test_graph_runtime.py`，在用户已有完整环境可执行：

```bash
python -m pytest tests/test_graph_runtime.py -q
```

轻量测试使用仓库原有的隔离 harness：生产 runtime 类体保持不变，但 agent/节点/部分持久化边界由测试替身提供，因此不是端到端 benchmark。没有新增生产依赖（使用现有 NumPy/SciPy/OmegaConf 等）。真实 provider、CUDA、任务数据和最终奖牌率需要用户实际运行确认。

完整日志在 `docs/ab_validation/`。数值参考核对可复现：

```bash
python tools/verify_graph_reference.py \
  --study-zip /path/to/MLEvolve_Variance_Focused_Offline_Study.zip \
  --output offline_reference_parity.json
```

这个工具只读取该研究包中的两份原始数值参考模块和重建数据，不运行其中的 agent/实验调度代码。

## 新增主要日志字段

`phase1/` 的已有 decisions/snapshots/model_fits/transitions 文件保持格式兼容并追加字段：

- `gp_feature_names` / `gp_feature_vector`（仅普通动作是 7 维）、`gp_primary_ancestry`、`gp_graph_center`、`gp_graph_revision`、节点/边数；
- frozen `utility_scale`、`absolute_loss_scale`、真实 incumbent；
- A 的实际 `length_scale/amplitude/noise_std` 与 MAP 状态；
- B 的 `rms_scale/scale_used/scale_ceiling_active/normalized_kernel_variance`；
- `posterior_mean_quality`、用于选择的 SD、predictive SD、`acquisition_kind`（A 为 `child_ucb`，B 为 `global_logei`）；
- `is_exact_gp_posterior` 和明确的 `uncertainty_used`。

既有 `posterior_*` 字段是共享日志接口；B 中它们表示固定中心/核分配不确定性，不可据字段名将 B 解释为完整贝叶斯后验。
