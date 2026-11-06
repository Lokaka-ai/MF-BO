# BNVS 与 Latent Child-UCB：设计、离线证据和实验决策

> Runtime decision update (2026-09-04): the local default is now BNVS H=1
> with the exact 10-feature schema and Thompson Sampling from the first normal
> decision. It has no MCGS warm-up or hard observation threshold. The history
> thresholds below are offline diagnostic slices, not activation requirements.

## 结论先行

这批 42 个历史 run 支持继续研究 **Bayesian Node Value Search（BNVS）**，但离线日志不能证明任一反事实 selector 的最终 online 分数。按照当前实验决策，runtime 默认改为从第一个 normal decision 启用 BNVS H=1 + exact-10 + Thompson Sampling；较大的早期 posterior 尾部作为必须在线监控的风险，而不是硬 cold-start 的理由。

离线证据最明确地支持两项改变：

1. 把预测目标从 `child - parent` 改成 `child - pre_action_global_best`，并把无效结果记为 `-1`；
2. 用 10 个与计算分配直接相关的特征代替当前 15 维 local-gain 特征。

以下内容暂不进入核心版本：

- active-node 和图距离两个特征；
- H=2 延迟回传；
- root restart；
- Validity head、Cost head、预训练或跨任务迁移；
- MCGS 分数融合；
- 高 beta 的 Latent Child-UCB。

Latent Child-UCB 可以保留为 acquisition ablation。若必须在线试，`beta=0.10` 是当前数据下唯一适合作为低风险起点的值；`beta=0.25` 只能作为压力测试，不能从本次离线重放推断它会提高最终分数。

## 1. 要回答的研究问题

核心问题不是“下一步相对 parent 能涨多少”，而是：

> 在有限实验预算下，下一次计算应该分配给哪个可扩展节点，才最可能改善最终 incumbent？

一个候选集合同时表达 continue 和 switch：

- 选择当前活跃节点：continue；
- 选择祖先节点：回退后重新探索；
- 选择其他分支节点：switch；
- 选择 root：启动新方向。

第一版只让模型选择 parent。每个 parent 的下一 operator 和 reference 继续由 MLEvolve 的既有规则预览并固定。

## 2. 对当前实现的一个重要修正认识

“模型只选节点、operator 由固定规则决定”并不是一次完整重写。当前 runtime 已经对每个 parent 调用 `preview_next_operator`，并只为该 parent 构造一个 `ExpansionOption`。因此目前真正需要研究的变量是：

| 组件 | 当前 reward-only BO | BNVS 候选设计 |
|---|---|---|
| 决策对象 | parent，加上固定预览 operator | 相同 |
| 训练标签 | 有效 child 的 `child - parent` | `clip((child - incumbent) / scale, -1, 1)` |
| 无效 transition | 只记录，不训练 | 训练标签固定为 `-1` |
| 输入 | 15 维，包含 `mcgs_q` 和 reference 信息 | 10 维计算价值特征，不含 `mcgs_q` |
| acquisition | gain-trust 收缩后的 joint TS | 默认 full Thompson Sampling；Mean-Greedy 为消融 |
| 冷启动 | 10 个有效 transition 前 Progressive MCGS | 无；从 Bayesian prior 直接开始，不把 MCGS 分数混入 acquisition |

这意味着 BNVS 的科学贡献应表述为 **value target + compact computation context + unified node allocation**，而不是“第一次把 operator 从 BO 中移除”。

## 3. 核心标签

执行前 incumbent 为 `u*_t`，本次生成 child 为 `c_t`，run-local robust scale 为 `s_t`：

```text
y_t = clip((u(c_t) - u*_t) / s_t, -1, 1)
```

若生成、执行、评估失败或 child utility 缺失：

```text
y_t = -1
```

`s_t` 使用历史 utility IQR、有效 local gain 的 median absolute magnitude 和 `1e-6` 三者的最大值。scale 只使用决策前历史，避免 future leakage。

这个标签直接区分两类 local improvement：

- 一个低质量 parent 大幅改善，但 child 仍远低于 incumbent；
- 一个接近 incumbent 的 parent 小幅改善并形成新 best。

当前 local-gain 标签会更偏好第一类；BNVS 标签与最终搜索目标更一致。

## 4. 第一版输入

离线可精确重建的 10 维版本如下：

| Feature | 变换 | 含义 |
|---|---|---|
| `incumbent_gap_scaled` | `(global_best - parent) / scale`，限制到 `[-10, 10]` | parent 离 incumbent 多远 |
| `branch_backed_gap_scaled` | `(global_best - branch_best) / scale` | 分支历史最好结果是否仍有竞争力 |
| `expansion_count_log` | `log1p(visits + children)` | 该区域已投入多少计算 |
| `depth_squashed` | `depth / (depth + 5)` | 搜索深度 |
| `stagnation_log` | `log1p(branch_stagnation)` | 分支持续未改善程度 |
| `recent_gain_scaled` | `recent_gain_mean / scale`，限制到 `[-10, 10]` | 近期趋势 |
| `recent_valid_rate` | `[0, 1]` | 该区域近期可执行性，不单独建立 Validity head |
| `remaining_budget_fraction` | step 与 time 剩余比例的较小值 | 当前搜索阶段 |
| `operator__evolution` | indicator | 固定规则预览的 operator context |
| `operator__fusion` | indicator | `improve` 为 reference level |

12 维 active-aware 消融再加入：

- `is_active_leaf`；
- `distance_to_active_squashed`。

历史日志没有显式记录所有决策时刻的 active node。离线评估只能利用 continuation option 的顺序和已观察 parent graph 重建，所以 10 维版本是更可靠的核心比较，12 维版本只应视为近似消融。

## 5. 模型与 acquisition

值模型仍为带 Normal-Inverse-Gamma 先验的共轭 Bayesian linear regression：

```text
y = [1, x]^T w + epsilon
```

先验保持简单固定：普通系数 precision `4.0`、intercept precision `0.25`、inverse-gamma shape/scale `2.0/1.0`。不在 run 内搜索超参数，不使用预训练。

需要区分两个 acquisition：

### Vanilla Thompson Sampling

```text
w_sample ~ posterior(w)
score(node) = x(node)^T w_sample
```

它最符合原始 BNVS 叙述，但本次重放显示：在 5 条 observation 后，候选集合内最大绝对 sampled score 的 P99 为约 `5.49` 个标签尺度；选择结果也只在约 `22%–25%` 的决策中落在最高 parent-utility 节点。这是一次非常大的策略分布迁移，不能直接视为安全默认值。

### Latent Child-UCB

在当前 local-gain 模型上测试的分数是：

```text
score(node) = parent_utility
            + gain_trust * (posterior_mean_gain + beta * latent_std_gain)
```

其中 `latent_std_gain` 只包含系数后验的不确定性，不包含 observation noise、当前 predictive `2x` multiplier 或 uncertainty floor；最后 10% step/time budget 将 beta 线性退火到 0。没有 gain clipping，也没有 MCGS 融合。

它是确定性的 optimism policy：同一个高不确定候选会每次获得正 bonus；TS 则只会偶尔把它采样成最优。当前后验尾部尚不稳定，因此两者不能互换理解。

## 6. 离线评估协议

- 数据：MLEbench-lite 21 个任务，Legacy Progressive MCGS 与旧 BO 各一个 run，共 42 个 run；
- 预序评估：每个预测只使用同一 run 中更早的 transition；
- candidate replay：只重放日志中存在的多候选决策；
- factual metric：只评价历史上实际执行的 action；
- 未选 action 没有 outcome，因此不能从 replay 推断 counterfactual regret 或最终 private-score uplift；
- root 在这些日志中不是普通候选，因此无法离线评估 root restart；
- 所有专项重放的异常数均为 0。

## 7. Latent Child-UCB 专项结果

共得到 644 个 factual 预测和 472 个多候选重放决策。`beta=0` 是 posterior-mean child value。

| Beta | 与当前 TS 一致 | 与 posterior mean 一致 | 选最高 parent | P90 parent sacrifice / scale | P99 已选 UCB bonus / scale | P99 候选集最大 bonus / scale |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.850 | 0.977 | 0.801 | 0.018 | 0.263 | 0.850 |
| 0.25 | 0.845 | 0.953 | 0.788 | 0.019 | 1.582 | 2.124 |
| 0.50 | 0.843 | 0.922 | 0.797 | 0.015 | 3.165 | 4.249 |
| 1.00 | 0.790 | 0.845 | 0.758 | 0.061 | 6.329 | 8.498 |
| 1.50 | 0.758 | 0.792 | 0.739 | 0.086 | 9.494 | 12.747 |

当前 production TS（temperature `0.25`）在同一 472 个决策上：与 posterior mean 一致率 `0.850`、选择最高 parent 比例 `0.771`、P90 parent sacrifice `0.018`，P99 已选绝对 gain contribution 为 `1.504` 个 scale。

解释：

- `beta=0.10` 几乎是 posterior-mean policy，只改变约 2.3% 的选择；它比当前 TS 更保守，不是更强探索；
- `beta=0.25` 开始产生可观察的 UCB 影响，但 P99 最大候选 bonus 已达 `2.124` 个 scale；
- `beta>=0.50` 的 optimism tail 快速放大，不适合作为当前小样本模型的默认值；
- latent std 的 median/P90/P99 为 `0.311/1.404/15.959` 个 run-local scale，说明少量特征方向仍严重欠约束；
- 10–14 条有效 transition 时 latent std 的 P99 达 `65.866`，因此“已经过 10 条 warm-up”不等于每个候选方向的不确定性都已可靠。

`realized_gain <= latent_upper` 的比例从 beta `0/0.10/0.25/0.50/1.0/1.5` 的 `0.537/0.573/0.621/0.691/0.801/0.893` 上升，但这不是置信区间 coverage：upper 只针对 latent mean，而 realized transition 还包含 observation noise。

**专项决策：不把 Latent Child-UCB 替换为 production acquisition。** 如要在线做 acquisition ablation，先试 `beta=0.10`；`0.25` 仅作上界压力测试；排除 `>=0.50`。

## 8. BNVS 标签与特征的同目标比较

以下模型全部评价同一个 H=1 incumbent-relative 标签，并只看至少已有 10 个有效 local-gain observation 的时刻，因此数值可以直接比较：

| Predictor | N | MAE | Clipped-prediction MAE | Sign accuracy | Pooled Spearman | Invalid MAE |
|---|---:|---:|---:|---:|---:|---:|
| 当前 local-gain raw child estimate | 846 | 0.996 | 0.373 | 0.779 | 0.368 | 1.221 |
| 当前 gain-trust mean acquisition | 846 | 0.813 | 0.308 | 0.838 | 0.440 | 1.380 |
| incumbent target + 当前 15 features | 846 | 0.371 | 0.352 | 0.874 | 0.358 | 0.557 |
| BNVS H=1 + 精确 10 features | 846 | **0.353** | 0.341 | **0.883** | **0.390** | 0.510 |
| BNVS H=1 + active-aware 12 features | 846 | 0.355 | 0.342 | 0.878 | 0.386 | **0.509** |

最重要的消融结论：

- 仅把标签换成 incumbent-relative，MAE 就从当前 acquisition 的 `0.813` 降到 `0.371`；
- 再从当前 15 features 换成精确 10 features，MAE 从 `0.371` 降到 `0.353`，约再改善 4.9%；
- active-aware 12 features 没有稳定超过 10 features；
- 新模型对“产生新 incumbent”的 recall 仍低，说明低 MAE 很大程度来自正确识别大量非正值/无效结果，不能只凭 MAE 宣称搜索会更强。

## 9. observation 数量敏感性

BNVS 会用无效 transition 的 `-1` 标签训练。下表按决策前已有的
**normal observations 总数**分层，用于诊断小样本表现；它不定义启动门槛。

| 最少历史 normal observations | 10-feature MAE | Sign accuracy | Pooled Spearman | Invalid MAE |
|---:|---:|---:|---:|---:|
| 3 | 0.379 | 0.842 | 0.402 | 0.542 |
| 5 | 0.368 | 0.854 | 0.406 | 0.531 |
| 10 | 0.357 | 0.882 | 0.387 | 0.514 |
| 20 | 0.339 | 0.897 | 0.423 | 0.499 |
| 30 | 0.337 | 0.916 | 0.419 | 0.491 |

这些 threshold 行的后段样本不同，不能把 `0.379 -> 0.337` 全部解释为学习曲线。它们可以说明：

- 3–5 条 observation 时均值预测没有崩溃；
- 随数据增加，invalid outcome 的拟合逐渐改善；
- 但 candidate-level vanilla TS 在 5 条后仍有大尾部，因此“均值可用”不等于“full posterior draw 可安全接管”。

当前 runtime 从第一个 normal decision 直接使用 Bayesian prior 和 BNVS
acquisition。零数据阶段是 prior-driven selection；随着每条 normal outcome
出现立即更新 posterior。MCGS 保留为独立 baseline 和异常恢复路径，不作为
BNVS acquisition 的初始化或组成部分。

## 10. H=1 与 H=2

在 eventual H=2 标签上评价：

| Model | N | MAE | Sign accuracy | Pooled Spearman | Median per-run Spearman |
|---|---:|---:|---:|---:|---:|
| H=1 active-aware model | 846 | **0.365** | **0.857** | **0.371** | **0.323** |
| delayed H=2 active-aware model | 846 | 0.378 | 0.825 | 0.361 | 0.320 |

H=2 没有改善，且会让同一历史 observation 的标签随未来搜索路径变化。核心版本保留 H=1；只有在新的在线日志显示 Evolution/Fusion 明显需要 stepping-stone credit 时再恢复 H=2 消融。

## 11. 与旧 BO 和 MCGS 的关系

历史实际结果只能比较当时运行过的 Legacy MCGS 和旧 BO，不能代表新 BO、Latent Child-UCB 或 BNVS：

| 历史 outcome | Legacy Progressive MCGS | 旧 BO |
|---|---:|---:|
| private-score wins | 8/21 | 13/21 |
| any medal | 16/21 | 14/21 |
| gold | 10/21 | 12/21 |
| candidate launches | 877 | 930 |
| valid yield | 0.716 | 0.718 |

13–8 的 score win sign-test `p=0.383`，单 seed 不足以证明旧 BO 优于 MCGS。作为信号代理，MCGS-Q 与下一次 realized gain 的 pooled Spearman 为 `-0.033`，而当前 compact reward model 为 `0.238`；但 MCGS 完整策略还包含 UCT、tree traversal 和 progressive Top-K，不能用单个 Q 值替代完整 MCGS 基线。

因此在线实验必须保留完整 Progressive MCGS，而不是只和 `mcgs_q` proxy 比较。

## 12. 推荐的在线实验矩阵

### 第一阶段：安全 pilot

选择 5 个覆盖不同 valid rate、搜索深度和历史 BO/MCGS 胜负模式的任务，每个方法 3 个 paired seeds：

1. Legacy Progressive MCGS；
2. Mean-Greedy：BNVS H=1 value model + posterior mean；
3. BNVS-TS：BNVS H=1 value model + Thompson Sampling；
4. BNVS-TS-local：相同模型、特征和 TS，但 target 改为 parent-relative gain。

### 第二阶段：完整比较

安全 pilot 没有显著 invalid-rate、极端 parent sacrifice 或早期退化后，再运行 21 个任务、至少 3 个 paired seeds。核心统计单位是 task，不是 transition。

### Primary outcomes

- normalized private score；
- task-level win/tie/loss 与 paired bootstrap interval；
- medal/gold 仅作辅助离散指标；
- best-so-far curve AUC；
- time/steps to best。

### Search-behavior outcomes

- continue、ancestor switch、cross-branch switch、root restart 比例；
- 被选 parent 的 utility rank、depth、visit count；
- 连续 branch length；
- switch 后多少次实验出现新 incumbent；
- TS 与 posterior mean 分歧率，以及分歧后的 factual outcome；
- invalid rate 与失败类型。

### Surrogate outcomes

- H=1 target MAE、Spearman、新-incumbent precision/recall；
- posterior predictive calibration；
- coefficient posterior 和跨 task 的符号稳定性；
- sampled score、parent sacrifice 和候选最大 uncertainty 的尾部分位数。

## 13. 实现前必须补齐的日志

新在线版本应显式记录：

- `active_node_id`；
- 每个候选相对 active 的关系：current/ancestor/other/root；
- graph distance；
- 决策前 incumbent、reward scale 和 target normalization state；
- candidate sampled score、posterior mean 与 latent std；
- fixed preview operator；
- root option 是否可用和其生成语义；
- 若将来测试 H=2，记录 credit update 的来源 transition 和可用时间。

当前日志不能支持 root restart 的公平离线重放。核心 v1 应先排除 root；root 必须在明确候选语义和完成上述日志后作为独立 ablation 加入。

## 14. 最终版本选择

当前默认版本固定为：

1. **target**：H=1 incumbent-relative、invalid=`-1`；
2. **features**：精确 10 维；
3. **activation**：从第一个 normal decision 使用 prior，无硬 threshold；
4. **acquisition**：full Thompson Sampling；
5. active-aware、H=2、Latent Child-UCB 和 root restart 均后置。

也就是说，Latent Child-UCB 的测试结果不是“UCB 一定无效”，而是：

> 当前小样本后验的某些方向仍过宽，UCB 会把这种不确定性每轮都当成正价值；相比之下，改对标签和特征的离线收益更明确，也更值得先投入在线预算。
