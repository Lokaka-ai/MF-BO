# Phase 1 Bayesian Parent-Operator Selector

## 范围与不变量

Phase 1 只替换 MLEvolve 在回到 virtual root 时的 parent selection。原有
Draft、Improve、Evolution、Fusion、Aggregation、Debug Agent，Prompt，Memory，
Code Review，执行器，result parse，submission validation，reward 与
backpropagation 均沿用原路径。完成节点后的直接 continuation 也保持不变；这类步骤
会记录一个 singleton decision，但不会被描述成 BO 在全图候选中作出的选择。

选择动作定义为合法的 `(parent, existing-next-operator)`。BO 不生成 plan，不修改
child，也不能发明 operator。`legacy_original` 是默认配置，因此原启动命令不显式
选择 Phase 1 模式时仍调用原 soft-switch selector。

## 数据流

一次正常的全局选择按以下顺序进行：

1. 根据原 MLEvolve 的 regular 与 Top-K child limit 枚举合法 parent。
2. 对每个 parent 无副作用地预览原规则会执行的 next operator 和 Fusion references。
3. 在 child 生成前提取 feature，并把完整候选 snapshot 追加到 JSONL。
4. 冷启动不足时调用原 Progressive MCGS；否则拟合 gain、cost 与 validity 模型。
5. acquisition 选出一个 parent；原 Agent 用同一份 operator preview 生成 child。
6. 原 Code Review、执行、解析、验证、评估与 reward backprop 正常运行。
7. 评分完成后记录连续 gain、生成成本、执行/评分成本以及原 ordinal reward。

实现入口是 `engine/parent_selectors/runtime.py`。接口类型位于
`engine/parent_selectors/base.py`：

- `ExpansionOption` 保存 parent id、operator、reference ids、selection-time features
  以及仅供当前进程执行的 node/preview 引用；
- `SelectionDecision` 保存 selector、winner、score、model version、fallback、id 和时间；
- `SelectorResult` 同时保存 winner 与全部当前候选 score；
- `ParentSelector` Protocol 只负责选择，不负责生成或执行 child。

## 合法候选集合

正常 BO 候选必须同时满足：

- 不是 virtual root；
- 非 terminal、非 locked；
- `is_buggy is False` 且不是显式 invalid；
- 有有限的 parent metric；
- 尚未达到 regular child limit；或者该节点属于原流程当前 Top-K，且尚未达到
  `topk_max_improves`。

Top-K overflow 只有在原 soft switch 已进入可能采样 Top-K 的时间区间才开放。若
parent 只能通过该路径扩展，preview 会携带 `topk_triggered=true`；真正执行前才设置
原 `_topk_triggered` 标志，使 child 的 `from_topk` 与原 Agent 的宽松 stagnation
threshold 保持一致。Progressive adapter 与 BO 从同一 normal option builder 接收
候选；原 selector 返回 root、Debug 恢复或其他特殊节点时走明确的 bypass。

## Operator preview

`engine/operator_preview.py` 把原 `_run_single_step()` 中的 operator 判断抽成纯预览：

- root：Draft，或在原条件已使 root 被选中时 Aggregation；
- failed/invalid parent：Debug；
- 未完成、`is_buggy=None` 的 parent：保持原 no-op，而不是擅自 Debug；
- 正常 parent：Improve，或按原 stagnation/time/feature flag 进入 Evolution/Fusion；
- Evolution trajectory 不足或 Fusion references 为空时，preview 直接变成实际会执行的
  Improve；
- Fusion reference 仍由原 `fusion_agent` 选出，最多复用原来的前五个。

Fusion/Evolution 的随机分支通过保存并恢复 Python RNG state 进行 peek；预览不推进
RNG。winner 确定后只消费原流程会消费的一次 draw。执行后断言 node stage 与 preview
一致。Draft、Debug、Aggregation、no-op 均标记为 selector bypass。

## Feature schema

固定 schema version 为 `phase1-parent-operator-v1`。特征只使用 child 生成前可见状态，
包括：

- parent utility、global/branch best gap、depth、stage；
- visits、MCGS Q、children count、branch age/stagnation；
- 最近五个 child 的 gain mean/std/max 与 valid rate、last improvement；
- local-best、global-top-K、improve count、failure depth；
- code length、parent last cost、elapsed/remaining budget、Diff/Stepwise flags；
- operator one-hot；
- Fusion reference 数量、质量差、分支多样性、depth 与 Top-K 比例；
- 同 operator 历史 generation/execution cost 和 observation count。

`include_mcgs_statistics=false` 会只归零 visits/Q；
`include_branch_history=false` 会归零分支历史组；
`include_reference_summary=false` 会归零 reference summary。它们不删除列，因此训练与
恢复始终使用固定维度。未生成的 child code/plan/metric/log、后续分支状态、最终 best
和测试集分数从未进入 feature。

## Gain posterior

连续目标统一为 utility 方向：maximize metric 使用 `metric`，minimize metric 使用
`-metric`，因此

```text
improvement = child_utility - parent_utility
```

模型是 scikit-learn `BayesianRidge`。输入由 `StandardScaler` 标准化，gain target 也
按训练均值/标准差标准化；预测均值和标准差再变换回原 gain 尺度。模型返回每个
option 的 `mean_gain` 与 `std_gain`，而不是只复制 parent score。每次 refit 记录随机
model version、训练 transition ids、feature schema、时间和训练 RMSE 等诊断。

失败、invalid、无合法 metric、Draft 或 Debug 的无父分数 transition 不进入连续 gain
训练；它们的 `improvement` 为 null，不会把原 `-1` reward 当作回归目标。

## Cost 与 validity

cost model 使用全部有非负 total wall time 的 transition，包括失败样本。目标是
`log1p(generation + execution)`，模型同样为标准化输入的 Bayesian Ridge。数据少于
两条时按同 operator median、全局 median、最后才按配置 default cost 回退；预测经
`expm1` 后截到严格正数，EI/cost 不会除零。

validity 是按 operator 的 Beta(1,1)-Bernoulli 后验均值。它不替代 Debug 或原 reward；
只在配置开启，或 cost-aware EI/cost 模式中乘到 acquisition 上。

## Acquisition

实现的 acquisition 为：

- Child-UCB：`parent_utility + mean_gain + beta * std_gain`；
- Gain-UCB：`mean_gain + beta * std_gain`，用于消融；
- posterior mean：`parent_utility + mean_gain`，用于去掉 uncertainty；
- cost-aware：相对当前 global-best child utility 的
  `p_valid * expected_improvement / predicted_cost ** cost_power`。

`beta` 配置化，Phase 1 当前只实现并显式校验 `beta_schedule=fixed`。每个候选的 parent
utility、mean/std、child UCB、cost、p_valid、最终 acquisition、beta 与 schedule 都写入
selection snapshot。未知 model、fallback 或 beta schedule 会直接报配置错误，而不会
静默换成另一种算法。

## Cold start、fallback 与恢复

默认在 total transition < 10、valid continuous-gain transition < 8，或任一当前
operator 的 valid observations < 1 时调用原 Progressive MCGS。warmup 消耗正常 step、
wall time 和 LLM 调用，其 transition 随后进入模型。feature 非有限、拟合异常、预测异常
或 acquisition 非有限同样 fallback。每次 fallback 写入 `selector_fallbacks.jsonl`，不会
终止 workflow。

JSONL 是恢复边界：重启后 `TransitionStore` 顺序读取已完成记录，跳过并警告单条损坏
记录，然后重新拟合；模型不依赖隐藏内存对象。Phase 1 不把文件 SHA 当作执行许可或新
scientific round。

## 与 Progressive MCGS 的精确差异

`ProgressiveMCGSSelector` 直接调用现有 `select_with_soft_switch()`，保留 UCT、visit/Q、
随时间变化的 exploration weight、inverse-rank Top-K 抽样以及原 RNG；它不是近似重写。
原 reward 仍沿 parent edge 回传，BO 模式同样继续维护这些统计量。

差异仅在正常全局 parent decision：MCGS 用访问/Q/时间调度直接选择，BO 把这些量当作
普通 selection-time feature，并从历史 parent→child transition 学习下一次 gain 分布。
直接 continuation 与特殊路径在两个系统中完全相同，因此在线结果只能解释为
“Agent + selector”系统比较，不能当作固定候选池上的纯 selector 因果效应。

支持模式：`legacy_original`、`progressive_mcgs`、`parent_metric_greedy`、
`random_parent`、`bayesian_gain_ucb`、`bayesian_child_ucb`、
`cost_aware_bayesian`（兼容别名 `cost_aware_bayesian_selector`）。
