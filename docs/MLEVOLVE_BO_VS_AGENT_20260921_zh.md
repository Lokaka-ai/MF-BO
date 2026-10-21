2026-09-21 当前 MLEvolve_BO 与 MLEvolve_BO_agent 源码对照

两个目录已经形成不同方向的架构分支。MLEvolve_BO_agent 侧重候选工作区、持久会话、角色协作和同候选内修复；MLEvolve_BO 后续侧重 BO 选择器实验、独立任务评分、数学任务和集群运行支持。双方保留大量共同基础，也各有对方没有的功能。

基线记录表明 agent 目录于 2026-09-02 从当时的 BO 目录复制。工作区 git 历史中，agent 最近的相关提交为 2026-09-09 的生物/扰动实验扩展；BO 后续继续增加图 GP、Study7、Branch7、Acq7 和 AlphaEvolve 等模块。依据：[agent 基线记录](/network/scratch/l/liux/EvolveFold/MLEvolve_BO_agent/ORIGINAL_REPO_BASELINE.json:1)。这里比较的是两个当前工作树，包含尚未提交的修改，不能把所有差异解释成单向“从 agent 修改成 BO”。

统计只覆盖 git 跟踪及未被忽略的文件；不计被忽略的 runs、模型、数据缓存和虚拟环境。本报告加入前的文件统计为：

| 统计 | 数量 |
|---|---:|
| BO 文件 | 378 |
| agent 文件 | 287 |
| 同路径文件 | 133 |
| 同路径且完全相同 | 76 |
| 同路径但内容不同 | 57 |
| 仅 BO 有 | 245 |
| 仅 agent 有 | 154 |

这些数量包含代码、配置、测试、文档和资源。只看非测试 Python 文件：同路径 77 个，其中 39 个相同、38 个不同；BO 独有 83 个，agent 独有 67 个。共同文本文件按 agent → BO 比较增加 2044 行、删除 1840 行；该行数不包含两边独有文件，不代表全部开发工作量。

| 方面 | MLEvolve_BO | MLEvolve_BO_agent 默认 stateful 路径 |
|---|---|---|
| LLM 调用 | 按阶段构造 prompt，Codex 路径使用 ephemeral 调用，返回计划/代码/评价 | 按 role + candidate 保存会话，后续 resume；同候选可在工作区读写、执行、修复 |
| 生成流程 | Planner、Coder、代码审查、执行、结果总结，保留 stepwise/diff 等路径 | Draft、Experiment、Evolution/Fusion、Aggregation、Review 五类职责；普通 improve 直接交给 Experiment |
| Debug | 存在独立 debug SearchNode | 修复记为同 candidate 下的 attempts，默认最多 6 次；不新建 debug 科学节点 |
| 检查 | 当前数学任务仍有模型代码审查、模型执行总结，数学有效性和分数由 host 验证 | PRE/POST 检查使用确定性规则；当前实现正常情况不调用 Review LLM，仅部分 POST 非信任异常调用 |
| 节点准入 | 保留原生成/执行/反馈生命周期及失败、debug 节点；有效评分决定候选可用性 | 冻结版本正式执行，只有 FINALIZED / POST VALID 候选进入科学 Journal；受管理失败决策仍闭合记账 |
| 选择算法 | GP6/GP7、图与长时域变体、Study7、Branch7、Acq7、rating 和 KG 等 | GP6/gp6_draft 已迁入，也保留基础 BO/可选线性策略；没有 Acq7/AQ05 等后续模块 |
| 演化/融合操作 | 保留独立 evolution/fusion 选择与运行方式 | 先选择 evolve_fuse 宏操作，由角色决定实际 evolution、fusion 或回退 improve，分别记 requested/actual |
| 任务 | AlphaEvolve 15 数学题、autoresearch/nanoGPT、MLE-bench Lite 接入与运行脚本 | 生物预测、扰动研究、相关资源和研究控制器；缺少当前 AlphaEvolve/autoresearch 任务接入 |
| 恢复和记录 | 数学构造哈希、有效预算、任务检查点、共享 draft、集群分段续跑及模型调用 token 记录 | candidate contract、代码 revision、attempt、冻结执行、角色会话和 PRE/POST 文件恢复 |

持久会话的实现见 [codex_session.py](/network/scratch/l/liux/EvolveFold/MLEvolve_BO_agent/llm/codex_session.py:484)；BO 的独立调用见 [llm/codex.py](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/llm/codex.py:146)。agent 的 stateful 会话以具体角色和候选为作用域，不是整棵搜索图共用一个无限延长的会话。

agent 流水线见 [stateful_pipeline.py](/network/scratch/l/liux/EvolveFold/MLEvolve_BO_agent/engine/stateful_pipeline.py:85)，节点准入见 [_admit_stateful_node](/network/scratch/l/liux/EvolveFold/MLEvolve_BO_agent/engine/agent_search.py:348)。该路径明确忽略旧的 use_stepwise_generation 和 fusion_vs_evolution_prob，因此同名配置字段存在并不代表生效机制相同。旧 legacy 路径仍保留。

关于 Review，以 [实际分支条件](/network/scratch/l/liux/EvolveFold/MLEvolve_BO_agent/agents/review_agent.py:406) 为准：PRE、POST 无失败以及硬失败直接由控制器规则产生结论；只有部分 POST 非硬失败进入模型 Review。旧实施文档中描述的更宽泛 PRE/POST 角色调用不能当作当前每轮都调用两次模型的依据。

共同基础仍然明显：gp_selector.py、gp_surrogate.py、acquisition.py、基础 selectors.py、记忆 embedding/retriever、diff 应用器及多个 LLM provider 文件逐字节相同。GP6 的主要数值实现共享，但 agent 增加了 evolve_fuse 操作类别、特征/恢复 schema 和候选准入适配，所以相同 GP6 名称仍不等于端到端实验完全相同。

GP6 迁移的边界见 [迁移记录](/network/scratch/l/liux/EvolveFold/MLEvolve_BO_agent/MIGRATION_REPORT_ZH.md:1)。agent/config/experiments 中只有 gp6.yaml、gp6_draft.yaml；BO 中有 36 份 YAML 实验预设。AQ05 实际定义见 [AQ05 预设](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/config/experiments/acq7_aq05_gp4_level15_rating_kg.yaml:1)。

模型和默认策略还需要区分“仓库默认配置”与“当前数学运行配置”：

| 配置层级 | 模型 | 选择策略 |
|---|---|---|
| BO 仓库基础默认 | gemini-3-pro-preview | bnvs_ts + conjugate_bayesian_linear |
| agent 仓库基础默认 | muse-spark-1.2-contributor | bayesian_child_ucb + bayesian_ridge；runtime=stateful_codex |
| 当前暂停的数学实验 | muse-spark-1.3-contributor / high | AQ05 acq7_gp4 + finite_pool_kg + rating |

当前数学运行另覆盖为 4 CPU、24 GiB、0 GPU、CPU embedding，并关闭不适用的 ML coldstart。详见 [实际运行配置](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/docs/ALPHAEVOLVE15_AQ05_CPU_RUN_20260921_zh.md:9)；不能仅用两个根 config.yaml 推断这轮任务配置。

与当前讨论的构造继承有关：agent 有更完整的候选文件、合同、父代码快照和角色读文件能力，但它没有接入这 15 道数学题的 construction.json 协议。不能由“有状态 agent”直接推断它已实现 AlphaEvolve 的最优构造传递。若迁移这轮实验，必须显式接入数学适配器、评分器、构造输入、AQ05 操作/反馈和累计预算；不能只把目录或模型名称替换。

关于 token：持久会话、文件引用以及正常路径的确定性 Review 提供了减少重复 prompt 或审查调用的机会；Experiment 内部也可以发生多轮工具交互和修复。仅凭静态结构无法断定总 token 一定更少，且一次角色 turn 不能直接等同于一次底层模型调用。此次未运行对照实验。

本次只做静态文件差异、配置、调用路径与历史核对，没有启动模型或实验，也没有运行测试。完整不同文件及各自独有文件清单、文件哈希证据见 [JSON](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/reports/mlevolve_bo_vs_agent_20260921.json)。

主要共同路径差异如下；完整独有文件列表保存在 JSON：

| 同路径文件 | BO 行数 | agent 行数 | agent → BO 增加 | agent → BO 删除 |
|---|---:|---:|---:|---:|
| `engine/parent_selectors/runtime.py` | 1810 | 1676 | 415 | 281 |
| `engine/agent_search.py` | 662 | 801 | 116 | 255 |
| `agents/aggregation_agent.py` | 190 | 121 | 179 | 110 |
| `agents/draft_agent.py` | 201 | 104 | 192 | 95 |
| `config/__init__.py` | 514 | 523 | 113 | 122 |
| `run.py` | 311 | 230 | 99 | 18 |
| `engine/operator_preview.py` | 238 | 332 | 1 | 95 |
| `agents/result_parse_agent.py` | 540 | 453 | 89 | 2 |
| `engine/search_node.py` | 571 | 577 | 41 | 47 |
| `config/config.yaml` | 236 | 257 | 30 | 51 |
| `llm/codex.py` | 322 | 274 | 62 | 14 |
| `engine/conditions.py` | 114 | 53 | 68 | 7 |
| `agents/coder/stepwise_coder.py` | 550 | 507 | 45 | 2 |
| `agents/memory/global_memory.py` | 377 | 419 | 2 | 44 |
| `engine/solution_manager.py` | 249 | 267 | 9 | 27 |
| `agents/debug_agent.py` | 328 | 304 | 26 | 2 |
| `engine/executor.py` | 398 | 388 | 19 | 9 |
| `llm/__init__.py` | 141 | 157 | 5 | 21 |
| `agents/code_review_agent.py` | 184 | 161 | 24 | 1 |
| `engine/parent_selectors/transition_store.py` | 123 | 144 | 2 | 23 |
| `engine/parent_selectors/features.py` | 587 | 599 | 5 | 17 |
| `agents/prompts/impl_guideline.py` | 125 | 104 | 21 | 0 |
| `engine/parent_selectors/gp_features.py` | 97 | 96 | 10 | 9 |
| `engine/parent_selectors/surrogate.py` | 758 | 767 | 4 | 13 |
| `agents/prompts/shared.py` | 75 | 65 | 12 | 2 |
| `engine/parent_selectors/base.py` | 120 | 128 | 3 | 11 |
| `engine/parent_selectors/proposal_feedback.py` | 85 | 80 | 8 | 3 |
| `agents/fusion_agent.py` | 513 | 507 | 8 | 2 |
| `agents/prompts/__init__.py` | 20 | 27 | 1 | 8 |
| `agents/improve_agent.py` | 340 | 337 | 5 | 2 |
| `agents/planner/base_planner.py` | 438 | 432 | 6 | 0 |
| `agents/coder/diff_coder/diff_generate.py` | 174 | 169 | 5 | 0 |
| `agents/evolution_agent.py` | 322 | 319 | 4 | 1 |
| `engine/execution.py` | 45 | 45 | 2 | 2 |
| `engine/evaluation.py` | 211 | 210 | 2 | 1 |
| `config/experiments/gp6.yaml` | 19 | 17 | 2 | 0 |
| `config/experiments/gp6_draft.yaml` | 19 | 17 | 2 | 0 |
| `engine/validation/format_client.py` | 88 | 88 | 1 | 1 |
| `engine/validation/quality_check.py` | 383 | 381 | 2 | 0 |
