# 三组选择器：HalfShrink-UCB / GP6-GlobalLogEI / Codex Agent

基于本次上传的 `EvolveFold-xixian-dev (2)(2).zip`，仅修改 `MLEvolve_BO/`。本实现不启动集群任务，不改数据集、评估协议、生成模型、执行预算或 GPU 配额。

## 1. 本次可以直接选择的原生 preset

| 方法 | experiment_preset | 配置文件 | adaptive root draft |
|---|---|---|---|
| HalfShrink-UCB | `halfshrink_ucb` | `config/experiments/halfshrink_ucb.yaml` | false |
| GP6-GlobalLogEI | `gp6_global_logei` | `config/experiments/gp6_global_logei.yaml` | false |
| Codex/provider 自主选择 | `agent_selector` | `config/experiments/agent_selector.yaml` | false |

三个名字已注册到原生 config loader，无需使用上一版 `tonight.*` override。原 `gp6`、`gp6_draft` 及旧 BO/MCGS 路径保留。不要把上一版 patch 叠加到本版本上。

三个 preset 统一使用 `decision_scope: all_normal_steps`。配置校验会拒绝为这三个比较 preset 打开 root，避免误跑。建议对照同样关闭 root 的原 `gp6`。

**关闭的是普通选择阶段的虚拟 root draft 候选。** 初始化 draft、停滞或方向耗尽时追加 draft、全部失败时原有的可行性恢复 draft、debug 和原有 aggregation 规则保留；这不等于整个运行从来不生成 draft。没有新增或调大 H，也没有修改原硬预算、扩展限制与执行器。

### 动态尺度与异步补方向

- GP6/GP7 使用截至当前的全部有效提案结果动态计算尺度，debug 链合并为一个提案；每次拟合统一重算历史输入和标签，历史决策原始状态不变。`draft_anchor` 首次初始化后固定，checkpoint 中旧字段 `reference_frozen` 仅表示这个基准已初始化。
- `utility_scale = max(IQR(utility), 0.05 * median(abs(utility)), 1e-6)`；GP7 的 `absolute_loss_scale = max(median(abs(utility)), 1e-6)` 同步更新。GP7 draft 仍不直接产生普通训练标签，长期窗口标签仍等待窗口闭合。
- GP/agent selector 不再等待所有初始 draft 结束。已有合法结果即可继续；初始提交数、完成数和硬预算仍保留。
- `include_root_draft=false` 时，按主父链所属的 root draft 分别统计停滞。默认原始指标提升阈值为 `0.0001`，小幅进步可累积；允许 3 次进步不足，第 4 次申请追加方向。普通选择器切换不会触发停滞。
- 默认先生成 3 个方向，停滞时逐步补到最多 5 次 draft 尝试。同一时刻最多一个追加 draft（包括尚待 debug 的提案）；已有方向全部耗尽时也可追加。持续明显进步或预算不足时可以最终少于 5 个。
- 预留计入 draft 上限；提交时记录已消费反馈的位置，取消预留不消费停滞请求，恢复后不会重复消费。强制 draft 不增加 piBO 普通动作计数。`include_root_draft=true` 继续采用原有 root 参与打分的规则。

## 2. 方法实现

### A：HalfShrink-UCB

保留原 GP6 六维特征、操作类别、Matérn 3/2 混合核、局部 residual 标签及 GP 拟合。原 GP 输出还原到质量单位后的 `mu, sigma_latent, sigma_predictive`，父质量为 `b`：

```text
mu_A = b + 0.5 * (mu - b)
sigma_latent_A = 0.5 * sigma_latent
noise_var = max(sigma_predictive^2 - sigma_latent^2, 0)
sigma_predictive_A = sqrt(sigma_latent_A^2 + noise_var)
score_A = mu_A + 1.5 * sigma_latent_A
```

只收缩 residual 均值和 latent SD，不收缩观测噪声，不把训练标签除二，不更换 kernel。默认 beta=1.5。

### B：GP6-GlobalLogEI

完全保留原 GP6 后验，使用同一次决策可见的全局 incumbent `I`：

```text
EI = E[max(F - I, 0)], F ~ Normal(mu, sigma_latent^2)
score_B = log(EI / utility_scale)
```

在对数域稳定计算 EI，包含极小负尾处理；共同尺度不改变候选排序。没有成功率乘法、成本除法、额外 exploration margin 或长期价值标签。

**B 的 global 指 acquisition 比较全局 incumbent，并非训练“最终全局成绩”模型。** 外层 `bo_parent_selector.acquisition.type: child_ucb` 是现有 GP runtime 入口校验；真正的 B 排名由 `gp.variant.acquisition: logei` 决定。B 的 beta 仅影响兼容性诊断 UCB，不影响 LogEI 排名。

两组 BO 若遇到数值错误，沿用合法池内的父质量回退，并明确记录 `gp_variant_error`；不是回退到 MCGS。出现频繁回退的 run 不能作为正常算法结果解释。

### C：Codex + provider Agent selector

`bo_parent_selector.model: none`，不实例化/拟合 surrogate，不使用 MCGS/UCB/EI/TS 排名，没有隐藏的 BO 候选预筛选。通过仓库已有 `llm/codex.py` adapter 调用：

```yaml
agent_parent_selector:
  model: muse-spark-1.3-contributor
  base_url: codex://opencode-go
  reasoning_effort: medium
  timeout_seconds: 120
  max_attempts: 2
```

当前 selector 默认模型为 `muse-spark-1.3-contributor`，provider 为 `opencode-go`。实际环境需要已安装 `codex`，且现有 Codex 配置有对应 provider/auth。允许明确覆盖 `agent_parent_selector.model` 与 `agent_parent_selector.base_url`，不静默替换模型。

每个普通决策：

```text
原工作流生成全部合法 (parent, previewed operator, references) 候选
→ 读取各候选决策前的历史事实、plan、analysis、父代码节选、参考节点摘要
→ 给 Codex/provider 一次结构化选择请求
→ 返回一个候选编号及简短理由
→ 原执行器执行该候选，原 debug/反馈闭环继续
```

模型看到任务说明、已有父质量、六维推进历史、全局 incumbent、剩余预算、最近最多10个闭合提案与成本、父节点/参考节点文本。科学参数数值保留；SHA/UUID/绝对路径清洗，不向模型提供 MCGS Q、posterior、acquisition、private score 或未执行 child 内容。日志可以保留原节点ID用于关联，但 prompt 用局部 `n0/c0` 引用。

这是一组**选择器对照**，不是另造能自由修改代码/工具/工作流的 Manager。三组使用相同的原合法候选生成与 operation preview；Agent 从候选元组中自主选一个，不允许凭空新增不合法操作或提前生成方案。Agent 使用文本而 GP 两组不使用，因此不能称为相同输入下的纯决策公式消融。

默认总 prompt 上限60000字符，父代码1800字符、plan/analysis各900字符。超过上限先减少文本节选，但不会按已有分数丢掉候选；若连全部候选元数据都超限，明确报错。只有一个候选时直接选择并标记 `only_legal_option`，不浪费模型调用。

一次决策所有请求共用120秒总预算，最多初次+一次修复/重试；失败后该 run 明确终止，停止继续提交选择，不悄悄使用 BO/MCGS，也不无限重试。已在执行的工作可能由原 executor 收尾。选择耗时计入正常 wall-clock 预算，不额外赠送实验时间。

Agent prompt 位于 `agents/prompts/parent_selector.md`。选择调用使用临时工作目录和只读模式，关闭 shell/unified-exec/multi-agent/web/apps 并抑制项目文档加载；不把实验目录作为可浏览工作目录。完整 Codex/provider 行为仍需本地实际配置验证；本机只测了命令构造与假可执行文件，不声称已验证真实服务。

## 3. 使用方式

### 3.1 本地、不调用 provider 的检查

在修改后的 `MLEvolve_BO/` 目录：

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests_gp6 -v
OPENBLAS_NUM_THREADS=1 python scripts/smoke_three_selectors.py
```

第二条是小型合成候选的 selector-only smoke：A/B 走真实数值代码，Agent 返回测试回复；不运行 GPU、不访问数据集、不调用付费 API。

### 3.2 可选：一次真实 provider 选择检查

在你已经装好完整仓库依赖及 Codex/provider 的环境：

```bash
python scripts/smoke_three_selectors.py \
  --live-agent --output /tmp/mlevolve-selector-smoke
```

这会发出**一次真实模型请求**，产生相应 API 使用；没有第二次重试，不训练模型、不执行生成代码。只确认接口能返回合法候选，不代表完整 benchmark 已验证。

如需明确换模型/provider：

```bash
python scripts/smoke_three_selectors.py --live-agent \
  --model muse-spark-1.3-contributor --provider opencode-go \
  --output /tmp/mlevolve-selector-smoke
```

### 3.3 单任务：交给原调度器/已经分配的 worker

入口：

```bash
bash scripts/run_selector_task.sh MODE TASK DATASET_ROOT [SERVER_ID] [key=value ...]
```

必须明确传入旧 campaign 的资源环境变量；不要因为脚本示例改变对照条件。以下8小时/16CPU仅是**旧 campaign 恰好采用该预算时**的示例：

```bash
export TIME_LIMIT_SECS=28800
export CPUS_PER_TASK=16
export MEMORY_INDEX=0
export START_CPU_ID=0

DATASET_ROOT=/你的/mlebench数据目录
TASK=new-york-city-taxi-fare-prediction

# 在分别分配好资源的 worker 中运行对应一条，不要在同一张卡盲目叠三个任务。
bash scripts/run_selector_task.sh halfshrink_ucb "$TASK" "$DATASET_ROOT" 111
bash scripts/run_selector_task.sh gp6_global_logei "$TASK" "$DATASET_ROOT" 112
bash scripts/run_selector_task.sh agent_selector "$TASK" "$DATASET_ROOT" 113
```

`SERVER_ID` 继续由现有本地 grading server 使用；同一主机同时运行的 worker 需不同 ID。GPU、CPU绑定与 `run_single_task.sh` 原逻辑相同。上面三个命令依次执行本身是串行，不是三组并行调度。

**生成/反馈模型不会因为新 preset 自动更换。** 上传的 `MLEvolve_BO/config/config.yaml` 仍有原模型默认值；运行时必须把旧 campaign 的相同 `agent.code.*`、`agent.feedback.*`、seed、资源等 override 传给三组。

如果旧 campaign 本来就用同一个 Codex Muse provider 生成/反馈，可统一这样传（否则保留旧模型，不照抄更换）：

```bash
COMMON=(
  agent.code.model=muse-spark-1.2-contributor
  agent.code.base_url=codex://opencode-go
  agent.feedback.model=muse-spark-1.2-contributor
  agent.feedback.base_url=codex://opencode-go
)
bash scripts/run_selector_task.sh halfshrink_ucb "$TASK" "$DATASET_ROOT" 111 "${COMMON[@]}"
bash scripts/run_selector_task.sh gp6_global_logei "$TASK" "$DATASET_ROOT" 112 "${COMMON[@]}"
bash scripts/run_selector_task.sh agent_selector "$TASK" "$DATASET_ROOT" 113 "${COMMON[@]}"
```

Agent selector 的模型字段与这两类执行模型独立；改变前者不改变后者。

### 3.4 完整22任务

已复用 `config/mlebench_lite22.txt`：

```bash
bash scripts/run_selector_suite.sh halfshrink_ucb "$DATASET_ROOT" 111 "${COMMON[@]}"
bash scripts/run_selector_suite.sh gp6_global_logei "$DATASET_ROOT" 112 "${COMMON[@]}"
bash scripts/run_selector_suite.sh agent_selector "$DATASET_ROOT" 113 "${COMMON[@]}"
```

每条 suite 在对应 worker **串行**处理22任务，输出 task/exit_code TSV；没有新增 Slurm 或其它调度框架。要并行多个任务，交给现有调度器按 task 调用单任务入口。Agent suite 遇到非超时错误会停止，避免同一个 provider 错误重复22次。124只表示时间预算用完，不能据此声称任务成功或获得奖牌。

新输出分别在 `runs_halfshrink_ucb/`、`runs_gp6_global_logei/`、`runs_agent_selector/` 下，实际 run 子目录沿用原 loader。不要共用旧 journal 或跨 selector 恢复；同组恢复沿用原 `resume_run_dir`。三组配置的 root 都是false，不要额外传true。

## 4. 如何确认跑的确实是目标方法

查看实际 run 的 `logs/phase1/`（最终路径以原 loader 为准）：

- `selection_decisions.jsonl`：`selector_name` 分别为 `halfshrink_ucb`、`gp6_global_logei`、`agent_selector`。同时看 `selection_reason`，初始化/debug bypass 不等于模型选择。
- `selection_snapshots.jsonl`：BO 的均值、latent/predictive SD、实际 acquisition；Agent 标记 `acquisition_kind: agent_choice_indicator`。Agent 的0/1只是选中标识，不是 reward预测。
- `model_fits.jsonl`：A/B记录 `variant_config`、`acquisition_sigma: latent`、`observation_noise_shrunk: false`；Agent不生成拟合记录。
- `agent_selection_calls.jsonl`：Agent的prompt、raw response、合法候选映射、模型/provider、简短理由、次数和耗时。
- `transitions.jsonl`、continuation/decision ledger：复用现有追加日志与debug归因；`root_draft_state.json` 名称沿用原实现，不代表root开关开启。

BO fallback有明确原因；Agent失败明确抛 `AgentSelectionError`，不会产生“其实回退了但标为Agent”的结果。

## 5. 本次验证及边界

已执行：

1. **51项定向 unittest 全部通过**：原GP6 30项＋新增21项。覆盖收缩公式/噪声保留、EI/LogEI数值及单位变化、三个原生preset路由、root禁用、跨方法恢复拒绝、Agent合法选择/一次修复/失败上限、不输入Q或SHA、候选覆盖、日志关联、无surrogate的保存、失败向调度器传播。
2. 三组 selector-only 合成 smoke 全部通过，均无fallback/root候选。Agent使用假回复。
3. 真实 `llm/codex.py` 通过本地假Codex可执行文件完成subprocess调用、输出文件/schema解析和provider/model参数检查；这不是实际provider调用。
4. 修改的Python文件语法检查及新增shell入口 `bash -n`。

测试中runtime使用现有的“真实类体＋agent/journal/import边界替身”隔离框架，新增保存测试也用序列化替身；**不能把它说成完整AgentSearch/GPU端到端通过**。

当前容器缺少完整运行依赖（包括dataclasses_json等），也未进行真实Codex/provider调用，因此未运行真实LLM、完整原生agent集成或MLE-bench任务。以上代码和脚本已经完成；最终运行效果和环境兼容性需由实际环境的单次live smoke/任务日志验证。不声称任何奖牌率提升。

## 6. 改动范围

新增两个选择器模块、一个英文Agent prompt、三个独立preset、单任务/suite/smoke入口和定向测试；在现有runtime/config/transport/exception边界接入。原kernel、GP6训练器、候选生成、代码agent、数据评估、资源调度未重写；原有MCGS统计可继续为历史日志维护，但Agent不会读取它们做排名。

本次不新增普通文件SHA台账、历史全量重放或研究控制平台。实现完成后以这些相关检查收尾，遵循用户提供的bounded实施范围。
