# 直接长期 A：固定 h=2 的 GP7 + Ancestry + UCB

## 交付范围

基于本对话上传的 `EvolveFold-xixian-dev (2)(3).zip`。只修改 `MLEvolve_BO`，新增一个实验 preset：

```text
config/experiments/gp7_long_horizon_ucb.yaml
```

preset 名称：`gp7_long_horizon_ucb`。内部 selector ID 仍为 `gp7_long_horizon_logei`，用于兼容已有 checkpoint；实际 acquisition 由配置中的 `child_ucb` 决定。

原 `gp7_ancestry_ucb`、`laplacian_anchor_logei` 和其他 preset 均保留；没有修改默认配置、agent prompt、LLM/provider、原评估协议、debug 上限、预算、并行度、操作预览、候选合法性或执行流程。没有实现 S10、G25、目标对数变换、额外均值收缩、Count baseline 或多步模拟。

## 方法

固定 h=2，输入仍然是七维：

`parent_gap, last_gain, recent_gain, stagnation, trials, valid_rate, absolute_parent`。

operation 类别和主祖先链独立进入现有核。h 只进入配置、日志和恢复契约，**不进入 GP 特征**。历史回看窗口仍为 5，与预测预算 h 无关。

```text
k = amplitude² × operation_corr × [0.7 × Matern32(7D; ell) + 0.3 × ancestry(gamma=0.5)]
```

第 j 个完整方向窗口的训练目标为：

```text
Y_j = max(起点父节点 utility, h 次尝试中所有有效结果的 utility)
r_j = (Y_j - 起点父节点 utility) / S
```

S 沿用 A 的初始有效 seed 校准和冻结规则。父节点分数始终作为均值基线：

```text
C = K + noise_std² I + jitter I
mu(p) = u(p) + S k_p^T C^{-1} r
sigma_latent²(p) = S² [k(p,p) - k_p^T C^{-1} k_p]
sigma_predictive²(p) = sigma_latent²(p) + S² noise_std²
```

继续用既有的有界 MAP 拟合 ell、amplitude、noise_std；满 8 个成熟、互不重叠的训练窗口才启动超参数优化。此前不是关闭 GP：仍使用先验超参数做后验更新；零窗口时为父节点均值 + 非零先验标准差。

当前 preset 按 `mu + beta * sigma_latent` 排序，`beta=1.5`。不使用 predictive SD 排序，不加入有效率或成本权重。

## 在线窗口的精确定义

本实现与离线主测试一样，使用**实际策略下的方向预算窗口**，不是新增固定继续策略或强制串行 rollout。

1. 每个普通 proposal 可以成为窗口起点，冻结该决策原有的七维状态、operation、祖先链和父节点分数。
2. 窗口最初只允许起点父节点。按现有 managed ledger 的决策编号，依次计入从允许父节点发出的普通 improve/evolution/fusion proposal；将这些 proposal 新产生的节点加入允许集合。
3. 起点父节点既有的其他子分支不会自动加入窗口。其他方向的实验、fusion 参考分支的成绩、全局 aggregation/draft 不会获得免费的窗口收益。一次 fusion 自身的有效输出仍按其主父节点归因。
4. 一次普通 proposal 加上允许的 debug 修复算一个窗口预算单位。debug 的原始实际执行消耗仍受原调度器预算约束；这里没有改变原有全局 step/time 计数。
5. 最终失败的 proposal 消耗一次窗口机会；它不产生有效分数。完整 h 次尝试都没有改善时，标签为零收益，而不是 -1。
6. 尚未闭合的修复链，或方向内更早但仍在途的普通尝试，会阻塞窗口，不能跳过它取后面两个成功结果。
7. 仅预约但未提交且后来取消的决策不消耗预算；预约期间只作为防止并发错序的临时阻塞。
8. 允许暂停后恢复和并行实验。并行时按**决策顺序**选窗口成员，不按结果完成顺序挑选。窗口只有在全部 h 个成员最终闭合后可用。
9. 完整窗口按可用反馈顺序、再按起点决策编号排序，贪心保留成员互不重叠的窗口训练。同一次 proposal 不重复贡献为多个独立训练样本。
10. 未完成窗口保持截断，不伪造目标；结束时不把不足 h 的短窗口转为完整窗口。

因此 h=2 **不等于强制连续执行两步**，也不等于两个 GPU 小时。原来的 `all_normal_steps` 每次重新比较继续／切换仍然保留。该模型学习的是当前实际 agent/调度行为下的固定预算结果，不是最优未来策略的 Q*。

## 最小接入方式

修改了 6 个既有文件：

- `engine/parent_selectors/gp_graph_features.py`：注册新 selector；数值特征和核公式不改。
- `engine/parent_selectors/gp_graph_surrogate.py`：只在新模式替换训练行来源，复用原 A 的拟合／预测／acquisition；保存固定 h 契约。
- `engine/parent_selectors/runtime.py`：传入在途决策、记录顺序编号、添加目标日志及恢复契约；不改候选／执行调度。
- `config/__init__.py`：preset 和恢复路由。
- 两个 `scripts/run_selector_*.sh`：允许新 preset 名称。

主要新增文件：

- `engine/parent_selectors/gp_long_horizon.py`：纯标签构造器。用内存中的训练适配行表示窗口，不覆写原 transitions。
- `config/experiments/gp7_long_horizon_ucb.yaml`：唯一新实验配置。
- `tests_gp6/test_long_horizon.py`：24 个新标签／GP／runtime／恢复与路由测试。
- `tests/test_long_horizon_runtime.py`：3 个在完整项目依赖下运行的 AgentSearch 集成测试。

## 配置

新 preset 除 selector 名称与以下新增段落外，和当前 A 完全一致：

```yaml
parent_selector:
  type: gp7_long_horizon_logei  # 兼容 checkpoint 的内部 ID
  decision_scope: all_normal_steps
  include_root_draft: false
bo_parent_selector:
  acquisition:
    type: child_ucb
    beta: 1.5
  gp:
    long_horizon:
      horizon: 2
```

这是摘要；运行使用完整 YAML，不要将摘要覆盖为完整 preset。没有新增根 draft 选择，原初始化／必要可行性 draft、debug、aggregation 仍走原路径。

## 启动

解压完整代码包，进入 `EvolveFold-xixian-dev/MLEvolve_BO`。沿用之前实验的环境变量与分配，包括 `TIME_LIMIT_SECS`、`CPUS_PER_TASK`、`MEMORY_INDEX`、provider／model 和其他 CLI overrides。wrapper 要求前两个变量已设置。

单任务：

```bash
bash scripts/run_selector_task.sh \
  gp7_long_horizon_ucb \
  <TASK_ID> \
  <DATASET_ROOT> \
  <SERVER_ID>
```

整个原有 22-task 清单：

```bash
bash scripts/run_selector_suite.sh \
  gp7_long_horizon_ucb \
  <DATASET_ROOT> \
  <SERVER_ID>
```

`SERVER_ID` 沿用原约定，默认 111。suite 是**顺序任务 runner**，不是集群调度器，也不是把 22 个任务同时提交。wrapper 原样保留 `run_single_task.sh` 的 grading server 和任务启动行为。

也可以在原 `python run.py ...` 命令上只替换／追加：

```text
experiment_preset=gp7_long_horizon_ucb
```

原有任务路径、预算、LLM 设置参数照旧传递。wrapper 默认将这一组结果隔离在 `runs_gp7_long_horizon_ucb`。

## 恢复与日志

请从新 run 开始；不要将旧 A/B 的 checkpoint 切换成 Long A。新模式在原恢复链中保存固定 h、标签契约、核参数及冻结尺度，拒绝在恢复时更换 h。窗口由原始 transitions 与现有在途决策 ledger 重建，没有另加一个持久化数据库。

主要查看 `phase1/model_fits.jsonl` 中的 `fit_diagnostics`（文件名由现有 SelectionLogger 决定）：

- `target = fixed_h_best_utility_parent_residual`
- `horizon = 2`，`horizon_is_feature = false`
- `training_count`：实际成熟且去重后的窗口数，不是单步实验数。
- `complete_window_count`、`incomplete_window_count`、`blocked_window_count`、`pending_ordinary_count`、`overlap_excluded_count`。
- `training_windows`：起点、成员、成熟反馈、最好结果、收益及失败次数。

原始普通／debug transition 的目标状态标为 `deferred_long_horizon_target`；完整长期标签在后续 model-fit 日志出现，避免把一次完成误报为长期标签已成熟。

选择 snapshot 仍有 7 项 `gp_feature_vector`，另外记录 `long_horizon`、`prediction_target=fixed_h_best_utility`。这些元数据不进入核距离。

## 验证与未验证范围

已实际执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m unittest discover -s tests_gp6 -v
```

结果：**100 tests，全部通过**；其中新文件包含 24 tests。包括原 A/B、GP6 等该目录原有相关回归，以及本次以下路径：标签延迟／失败／debug／并发顺序／窗口去重，七维 exact GP 后验数值公式，真实 LogEI 排序，生产 runtime 类的记录—拟合—选择—JSONL 恢复，preset 和真实 shell wrapper 的 22 个任务路由。

runtime 测试复用项目已有的隔离式生产类 harness，替代不相关的 import 边界和 journal I/O；**不是完整 AgentSearch 集成通过的声明**。shell 测试仅替换外部任务 runner，不实际执行 benchmark。

完整 AgentSearch 集成测试的收集在本容器被 `ModuleNotFoundError: dataclasses_json` 阻塞；安装缺失依赖的尝试因网络 DNS 不可用失败。因此该文件的 3 个 native integration tests 未执行。在原项目完整环境下运行：

```bash
python -m pytest -q tests/test_long_horizon_runtime.py tests/test_graph_runtime.py
```

本次没有真实 LLM 调用、GPU 训练、MLE-bench campaign 或资源申请，未提供在线性能结论。较少的完整窗口可能使模型较长时间处于先验／小样本阶段；保留此行为，不通过强制推进或伪造短窗口改变实验定义。

另外，普通 GP 对非负 best-of-h 收益是高斯近似，没有额外施加非负／指标上下界约束；保持直接长期 A，不加入后处理变体。
