# Phase 1 Transition Data Contract

## TransitionRecord

每个已完成的 parent→child 扩展追加一条 `transitions.jsonl`。记录是 immutable dataclass，
字段如下：

| 字段 | 含义 |
|---|---|
| `transition_id` | 本转移唯一 id |
| `decision_id` | 连接生成前 selection snapshot 的 id |
| `parent_node_id`, `child_node_id` | 原 journal node ids |
| `operator` | 实际执行且经断言验证的 operator |
| `reference_node_ids` | 原 Fusion policy 预览并复用的 references |
| `parent_metric_raw`, `child_metric_raw` | 未改变方向的原 metric；不可用时 null |
| `parent_utility`, `child_utility` | 越大越好的统一 utility；不可用时 null |
| `improvement` | `child_utility - parent_utility`；非连续有效转移为 null |
| `valid` | child 非 buggy、非显式 invalid 且有 metric |
| `generation_wall_time_sec` | 原 Agent 生成和原 Code Review 的 wall time |
| `execution_wall_time_sec` | 执行、解析、submission validation 和评分 wall time |
| `total_wall_time_sec` | 上述两部分之和 |
| `gpu_time_sec` | 当前执行器没有可靠观测，保持 null |
| `reward` | 原 MLEvolve ordinal reward，仅供审计/原 backprop，不作 gain target |
| `selector_name` | 产生该 decision 的模式；direct continuation 也保留运行模式名 |
| `selection_features` | child 生成前冻结的固定 schema feature vector |
| `timestamp` | UTC completion timestamp |

minimize metric 的 utility 是 `-raw_metric`。例如 parent log loss 0.40、child 0.35 时，
utility 从 -0.40 变为 -0.35，连续 improvement 为 +0.05。

## 失败与特殊转移

- buggy、invalid 或 metric-less child：`valid=false`、`improvement=null`；仍进入 cost 与
  validity 数据，并继续原 Debug、Memory、reward/backprop 路径。
- Draft、无父 metric 的 Debug、root/特殊恢复：不虚构 parent score，也不写 delta=0。
- 原 `reward=-1` 与连续 gain 是两个独立信号；前者绝不进入 Bayesian gain regression。
- child 生成过程在产生 node 之前抛异常时没有“已完成父子转移”，因此不会伪造 child id
  或 target；原异常回滚逻辑仍执行。

## Selection-time snapshot

每次 decision 在调用任何生成 LLM 前写入：

```text
<run>/logs/phase1/selection_snapshots.jsonl
```

每个当前 option 一行，至少包含 decision/parent/operator/reference ids、raw parent/global/
branch-best metric、depth、visits、Q、children、stagnation、recent gain summary、success rate、
elapsed/remaining budget、完整 feature map、posterior mean/std、parent utility、child UCB、
predicted cost、p_valid、acquisition、beta/schedule、selected flag 与 timestamp。完整 winner
decision 单独写入 `selection_decisions.jsonl`；两处都记录真实 selector decision wall time。

snapshot 中的 feature dict 被复制到 frozen dataclass 和 JSON；随后 child 的生成、metric
变化或 branch 更新不会修改已写行。测试会在写 snapshot 后构造并改变一个 future child，
并要求 JSONL 字节级语义不变。`decision_id` 把 winner snapshot、child metadata 与最终
transition 连接起来。

## 模型拟合与 fallback 日志

- `model_fits.jsonl`：model version、训练 transition ids、schema version、fit time、gain
  wall time、training RMSE/target scale、cost count/fitted flag、validity counts；
- `selector_fallbacks.jsonl`：完整 decision 与 cold-start/model-error reason；
- `selection_decisions.jsonl`：每次 global、direct 或特殊路径 decision；
- `selection_snapshots.jsonl`：每个 decision 的全部候选，而不仅是 winner。

JSON 编码禁止 NaN/Infinity；不可用诊断写 null。append 在进程内加锁并 flush，保证单行
完成后才进入内存 store。恢复时只使用已经落盘的 transition，不从最终 journal 反推
selection-time visits、Q 或 stagnation。

## Feature schema 与消融

schema version：`phase1-parent-operator-v1`。列顺序定义在
`engine/parent_selectors/features.py::FEATURE_NAMES`，vectorization 必须逐列存在且全部有限。
operator/stage 用固定 one-hot；关闭消融组时归零指定列而不改变维度。模型 fit 日志会保存
完整列名，避免配置变化后把旧 scaler 错套到新列。

## 历史数据的边界

旧 journal 没有 Phase 1 snapshot。历史评估只承认 parent/child link、raw metric、stage/
operator、depth、code length、legacy execution time 等静态字段；不会伪造 selection-time
visits/Q、stagnation、Fusion reference、generation time 或完整 feature。历史数据按 run 的
时间顺序拆分，来自同一 run 的相邻节点不会随机散到 train/test 两侧。

新的在线和 controlled 实验只消费本实现生成的完整 snapshot/transition。外部 warmup
作为受控批次输入时，在 admission 时记录 journal 与 transition SHA 一次；该 SHA 仅证明
输入未变化，不代表科学批准、执行许可或新的 attempt。
