# 通用节点潜力评级 + πBO：短期 / 固定 h 长期 GP

## 交付范围

基于 `EvolveFold_direct_long_A_h2(3).zip`，仅修改 `MLEvolve_BO`。
本次实现最近讨论的 **评级进入 acquisition 的 πBO 方案**：保留当前 GP7、operation、父节点质量基线、固定尺度、原短期 / 长期目标与调度规则，不额外加入 R2 核、评级第八维、线性模型或新 reviewer。

当前三个实验 preset：

| preset | 内部 selector ID | 预测目标 | acquisition | 评级 | 祖先核 | beta_prior |
|---|---|---|---|---|---|---|
| `gp7_long_horizon_ucb_pibo` | `gp7_long_horizon_logei` | 原固定 h=2 的预算内最好结果 | UCB | 开 | 开 | 25 |
| `gp7_short_logei_pibo` | `gp7_ancestry_logei` | 原一步有效 proposal 的结果 | LogEI | 开 | 开 | 25 |
| `gp7_short_ucb_pibo` | `gp7_ancestry_logei` | 原一步有效 proposal 的结果 | UCB | 开 | 开 | 15 |

全局 `agent_rating.enabled` 默认 false；这三个 preset 显式开启。内部 selector ID 保留历史值以兼容 checkpoint，实际 acquisition 由配置字段决定。
三者均为 `all_normal_steps`、`include_root_draft: false`；初始化、debug、aggregation、原有评估规则保持不变。没有新加 GPU/训练作业或更改资源预算。

## 1. 开关与公式

```yaml
bo_parent_selector:
  agent_rating:
    enabled: true
    beta_prior: 25.0
  gp:
    graph:
      use_ancestry_kernel: true
      graph_weight: 0.3
```

`acquisition.beta: 1.5` 是 UCB 的探索系数，在 LogEI 配置中只用于 UCB 诊断；它与 `agent_rating.beta_prior` 独立。上例的 `beta_prior=25` 适用于长期 UCB 和短期 LogEI，短期 UCB 的配置值为 15。

对当前候选已有的通用评分 g=0/1/2，固定权重 w=[0.5,1,2]：

```text
# LogEI
base_logei = 原 stable_log_ei 的输出（EI / 冻结尺度的 log）
prior_bonus = beta_prior / n * log(w[g])
selection_score = base_logei + prior_bonus

# UCB
base_ucb = mu + beta * sigma_latent
prior_bonus = utility_scale * beta_prior / n * log(w[g])
selection_score = base_ucb + prior_bonus
```

LogEI 的先验项为 `beta_prior*log(2)/n*(g-1)`；UCB 额外乘冻结的 `utility_scale`。缺失 / 非法评分为中性 `w=1`，不把缺失当作低潜力；低评分权重大于零，不硬删除任何候选。不改变 GP 的预测均值、latent / predictive 方差或训练标签。原 fallback 路径保持原样，不在故障回退里强行应用先验。

### n 的定义

`n` 是从初始化后开始的 **普通 BO 选择序号**，不是成熟长期标签数，也不是包含 draft/debug 的旧 `decision_index`。

实现中，当前候选冻结 `pibo_iteration = rating_bo_decision_count + 1`；在已有选择锁内，真正 reservation 一次 improve/evolution/fusion 时递增一次。纯粹预测 / 多次算候选 / 重试同一 reservation 不递增。draft、debug、aggregation 不递增；普通尝试之后失败仍计数。已发出后又取消的 reservation 不回滚序号，避免并行情况下重用编号。计数写入现有 ledger 并随 resume 恢复；不会在获得第 8 个长期标签时重置。

### 两个关闭方式不同

* `agent_rating.enabled=false`：不增加评级 prompt、上下文或 schema 字段，也不应用评级 prior。
* `agent_rating.enabled=true` 且 `beta_prior=0`：仍收集同一通用评分，但先验项为零，适合控制额外提示词影响的消融。

## 2. 实验后反馈：只多一个通用标量

复用现有 `result_parse_agent.run()` 的反馈 query，在构造好原 prompt / FunctionSpec 后做条件扩展；不恢复代码生成 session，不增加一轮 LLM 调用，也不增加新的代理角色。

唯一新增模型输出字段：

```json
{"potential_level": 2}
```

这是原结构化结果中的一个额外字段，不是取代原 `is_bug`、`summary`、`metric`、`lower_is_better` 或已有可选 `code_summary`。

等级：0=没有明确可行的剩余改善路径，1=有限局部改善，2=有具体证据支持显著可实现的剩余潜力。信息不足或执行无效用 null。短期与长期使用完全相同的通用评级 rubric，不输出 g1/gh、h 个分数、概率、精确分数预测或额外评级理由。

只有开启评级时，附加：任务说明与约束、刚完成节点 ID/stage、作为参考的父节点指标、此前已观察最好结果、指标方向、运行剩余尝试与时间、单次执行 timeout、配置 CPU 数、可用操作和 debug 上限。未知 GPU 资源不推测。代码和执行输出沿用原输入；固定评估路径保留程序权威结果。

事实总结仍只写观察事实 / 错误；主观未来判断仅放在 `potential_level`，不写入 summary / code_summary，不影响真实指标或有效性判断。没有要求额外未来方案或 potential_evidence。

### 归属与因果时间

`p -> c` 完成后，本次反馈产生的是 c 的评分，存为：

```text
c.agent_potential_level
c.agent_potential_assessed_at
```

它只在后续 c 成为候选父节点时进入候选快照。不会写回 p，也不会覆盖产生 c 时已经冻结的 `phase1_selection_features`。后续 GP 仍仅提取原七维数值输入，评分在同一 state_features 里是 acquisition 元数据，不是新增回归坐标。

畸形值（例如 bool、字符串、越界整数）按缺失处理，不为评级追加重试，也不将原本成功的实验判失败。当前 node 最终被主程序判无效时，清除其有效评级。成功 debug 修复产生的新节点单独评级，原 proposal 输入不变。

### 固定评估分支

有效固定评估结果仍使用原有那次 LLM summary query，附加同一字段；分数只来自程序。原本在无效固定评估结果处提前返回、不调用 LLM 的路径不变，评级保持 null。没有为这个特殊失败分支另外引入调用。

## 3. 祖先核开关

`gp.graph.use_ancestry_kernel` 缺省 true。

* true：原 `a² k_op [0.7 k_Matern(x,x') + 0.3 k_ancestor]`（比例仍由 graph_weight 控制）。
* false：有效 graph_weight=0，核成为 `a² k_op k_Matern(x,x')`，状态项权重归一到 1，而不是保留 0.7。

关闭会同时影响 MAP 拟合和预测，保留七维输入、操作相关性和父节点均值基线。不会删除主父链，因为原历史摘要、长期窗口、debug 归因仍需要它。已有旧 checkpoint 缺少这个字段时按 true 解释。

## 4. 启动

在原环境里按基线设置 `TIME_LIMIT_SECS`、`CPUS_PER_TASK` 等资源参数。现有脚本的允许列表支持这三个 preset，仍然不申请集群资源。

单个任务（替换数据路径；其他资源沿用基线）：

```bash
cd EvolveFold-xixian-dev/MLEvolve_BO
TASK=detecting-insults-in-social-commentary
DATASET_ROOT=/path/to/mlebench

bash scripts/run_selector_task.sh gp7_long_horizon_ucb_pibo "$TASK" "$DATASET_ROOT" 111
bash scripts/run_selector_task.sh gp7_short_logei_pibo "$TASK" "$DATASET_ROOT" 111
bash scripts/run_selector_task.sh gp7_short_ucb_pibo "$TASK" "$DATASET_ROOT" 111
```

22 个任务，现有 suite 为顺序执行，不是并行调度器：

```bash
bash scripts/run_selector_suite.sh gp7_long_horizon_ucb_pibo "$DATASET_ROOT" 111
bash scripts/run_selector_suite.sh gp7_short_logei_pibo "$DATASET_ROOT" 111
bash scripts/run_selector_suite.sh gp7_short_ucb_pibo "$DATASET_ROOT" 111
```

也可以沿用已有直接启动命令，仅添加：

```text
experiment_preset=gp7_long_horizon_ucb_pibo
# 或
experiment_preset=gp7_short_logei_pibo
# 或
experiment_preset=gp7_short_ucb_pibo
```

消融覆盖项（作为已有启动命令的额外参数）：

```text
bo_parent_selector.agent_rating.enabled=false
bo_parent_selector.agent_rating.beta_prior=0
bo_parent_selector.gp.graph.use_ancestry_kernel=false
```

祖先核和评级开关彼此独立。用独立运行目录进行消融，勿修改同一已启动运行的 horizon / selector / 评级 / beta / 祖先设置后强行 resume。新 preset 的 resume 检查会对齐 CLI 最终覆盖项，运行时 contract 还会再次拒绝混用实验定义。

## 5. 日志与恢复

保留原有 journal、transition 和 selection snapshot 格式，追加可空节点评级字段。对真正应用先验的普通候选，快照增加：

```text
agent_potential_level
agent_potential_assessed_at     # 位于 feature_vector
pibo_iteration
base_logei / base_ucb        # 按 acquisition 类型记录
rating_prior_log_weight
rating_prior_bonus
rating_prior_beta
rating_prior_applied
acquisition_kind = pibo_global_logei / pibo_child_ucb
acquisition_score             # 加权后的最终分数
```

可以直接检查 `acquisition_score = base_logei + rating_prior_bonus`（LogEI）或 `base_ucb + rating_prior_bonus`（UCB），也能区分 LLM 评级、GP 预测和最终先验影响。特殊路径 / fallback 的 `rating_prior_applied` 为 false。

现有节点快照和 Journal 的 dataclass 字段持久化评级；现有 `root_draft_state.json` 持久化 `rating_bo_decision_count`。未启用评级的旧 checkpoint 缺字段时使用关闭评级 / 开启祖先核的旧语义；不允许把有评级和无评级运行直接接成同一实验。

## 6. 验证与真实限制

本地执行：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:tests_gp6 \
  python -m unittest discover -s tests_gp6 -q
```

结果：**119 tests passed**（原有 100 项，加 19 项新增检查）。原测试涉及 GP 数值、短期 / 长期标签、debug 归因、配置和隔离运行时；新增检查覆盖：

* 先验公式、正权重、缺失评分、beta=0、当前 GP 输出保持不变；
* 祖先关闭与 graph_weight=0 的训练 / 预测等价；
* 关闭开关时普通 / 固定评估 feedback prompt 和 schema 保持原样；
* 开启时短长期仅增加同一字段，补足上下文，事实与主观评级分开；
* 单次 query、非法评分无额外重试、固定评估分数不被 LLM 替换；
* 当前节点归属、候选快照冻结、原七维输入与选择日志；
* 普通计数不随 debug / 重复 reservation / 预测递增，状态恢复继续计数；
* 新配置的加载、CLI 覆盖、resume 拒绝混用定义、真实 shell wrapper 参数传递；
* 扩展 schema 通过项目 Codex strict schema 正规化函数和 JSON Schema 验证。

生产数值模块直接运行；结果解析函数和运行时使用现有隔离测试方式，传输、节点/指标类型、Journal 序列化与不相关 HTTP 服务使用测试替身。不是完整原生 AgentSearch / 真实模型供应商集成测试。当前环境缺少 dataclasses-json、部分 LLM SDK 等完整依赖，且无法联网安装，因此没有运行真实 LLM、GPU、数据集训练或奖牌率实验。不能声称本次证明了在线性能。

历史离线分析使用的是 g1/g2，本次按要求改为单一通用评分，因此 beta=25 是配置默认，不是新通用评分已重新通过性能验证。没有更改已有验证协议或修复此前任务日志中的评分缺陷。

变动仅在 MLEvolve_BO；所有原实验 preset 与其他平台目录保持原样。
