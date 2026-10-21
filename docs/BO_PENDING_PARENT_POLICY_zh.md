# BO 并行父节点选择

`parent_selector.prefer_unoccupied_parents: true` 默认开启。适用于本目录注册的 BO 选择器，包括 Bayesian UCB、cost-aware BO、Reward-only TS、BNVS、GP 和 AQ 系列。原版 MCGS、随机和非贝叶斯分数贪心等对照选择器保持原有策略。

## 调度语义

1. 每次选点读取所有尚未结束的决策，按 `parent_node_id` 识别占用。预留、生成、等待执行和正式执行都计入；无需等到代码生成完成才占用。
2. 在当前合法候选池中优先排除这些父节点。同一父节点的不同操作一起排除，同分支其他节点仍然可选。
3. 若存在未占用候选，在这个候选池上重新计算原采集函数。AQ04 使用剩余候选的完整联合后验进行一次 TS；AQ05 保持 KG，UCB 等也保持各自算法。不会先在完整池采样再截取排名。
4. 若过滤后为空，使用过滤前的合法候选池重新选择，允许重复父节点以利用空闲资源。此回退不补回原有合法性规则已经剔除的候选。
5. 普通任务完成、失败或取消后释放其占用。若回退导致同一父节点有多个任务，只有这些任务全部结束后，该父节点才成为未占用候选。
6. 初始化 draft 共用的虚拟根不受此去重限制。draft/aggregation 继续遵守原有的预算和触发规则。

过滤、采集与预留在同一选择锁内完成。全局入口、continuation、旧 BO 冷启动和模型错误回退均处理占用。只开启旧的 research `parallel_pending_exclusion` 并不能覆盖以上路径；新默认策略对 BO 优先生效。

## 生命周期与恢复

已有 managed 决策直接读取持久化 pending 账本，因此生成中的任务和恢复后的冻结候选仍占用父节点。旧 BO 的非 managed 路径按 decision ID 维护轻量占用表；恢复冻结候选时从原选择快照重建占用。失败与取消不会制造 GP 质量标签，同候选的 bounded 修复期间保持占用。

不合并并行候选，不修改评分、数据边界、修复次数、GP 核或训练目标。待完成任务不产生虚构质量观测。

## 配置与诊断

关闭此功能用于对照：

```bash
python run.py parent_selector.prefer_unoccupied_parents=false ...
```

选点日志 `<log_dir>/phase1/selection_decisions.jsonl` 增加：

- `prefer_unoccupied_parents`：此次运行是否应用该策略。
- `pending_parent_ids`：本次决策之前仍被其他任务占用的父节点。
- `pending_parent_reuse`：本次是否选择了已占用父节点。

AQ04/AQ05 的 `study7_decisions.jsonl` 和联合后验 NPZ 保存实际参与计算的候选池。新默认会改变搜索轨迹；旧配置缺少这个字段时也使用默认 true，保存的显式 false 则保留。实验对比应记录该开关。

## 有界验证

`tests/test_pending_parents.py` 覆盖默认值、各旧 BO 路由、并发原子预留、同分支保留、按父节点跨操作过滤、全部占用回退、多任务占用计数、完成/失败/取消释放和关闭开关。AQ 集成测试覆盖真实选择器在缩小候选池上的计算、单候选不消耗 TS 随机抽样、冻结候选恢复和失败闭合。模型调用及长训练使用现有明确的测试替身，不启动真实模型服务或训练任务。
