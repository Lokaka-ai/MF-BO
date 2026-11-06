# Legacy 模式自动集成与旧 Base 一致性核查

核查对象：当前 `MLEvolve_BO`，历史 Base 提交 `d6d28567518e9b7179fa341f414c9f35a7e1dc8e`，以及 2026-09-16 分析目录中保存的历史配置与集成脚本。本文讨论 `evaluation.enabled=false` 的新运行，不把固定评估 checkpoint 改成 legacy checkpoint。

## 本次修改

`run.py` 正常完成搜索后，在 legacy 模式中保存运行状态，并直接调用现有 `utils/submission_fusion_utils.py::ensemble()`。固定评估模式仍使用原来的 `finalize_submission()`。

新增入口传入本次运行的准确 `log_dir` 和 `workspace_dir`，支持自定义目录、分开的日志与工作目录及恢复运行。旧的 `--exp_name` CLI 继续可用；新增 `--run_info` CLI 供外层超时后处理使用。

`run_single_task.sh` 在 legacy 正常结束时不重复集成；超时或异常退出时，仍按原有流程在外部补做集成，但使用 run-info 中的准确目录。原有中断退出行为保留。

保留的旧规则包括：

- 使用 `top_solution`，最多 6 个有效候选；小候选集遍历全部规模，大候选集使用 `[1, 2, 3, 4, 6]`。
- 保留原指标权重、预测格式识别、ID 对齐、平均／分类投票／文本投票／逐元素融合、概率归一化及输出格式。
- 保留候选累计执行时间 9 小时限制、跳过任务规则、无 Top-K 时的 best-submission 回退。
- 继续生成原来的 `ensembles_csv/top{k}ens-total_run_time{...}h.csv` 文件族，不引入新的验证择优规则，不重新训练模型。

已用 AST 对照归档旧脚本：从候选选择到保存的 `ensemble()` 主体一致，`EnsembleConfig` 及 14 个辅助函数一致。只有入口目录定位与 CLI 分派发生变化。

## 最终用于评分的文件已接入旧 Base 收集规则

旧脚本生成多份 ensemble 文件，不把其中一份覆盖成 `best_submission`。历史 campaign 的最终收集规则是：

1. 对 ensemble 文件按 `(mtime_ns, absolute_path)` 倒序遍历。
2. 选择首个与保存的 best submission 在 CSV 表头、行数、列宽和首列有序值上匹配的文件。
3. 没有合格 ensemble 时才回退 best submission。

该规则见分析目录 `evidence/historical_pipeline_pinned.py` 的 `latest_structurally_valid_ensemble_else_best_v1`。现由 `utils/grade_submission.py::finalize_legacy_submission()` 实现：集成结束后自动写入 `logs/final_submission.json`，仅记录最终 CSV 路径、选择类型和规则名称。找到首个结构匹配的候选后立即停止扫描；参考文件无法解析时直接采用原来的 best 回退。跳过集成、部分集成失败和 shell 超时后处理也走此出口。保留原 `best_submission`，避免改变恢复运行和再次集成时的参考文件。

评分时，在安装了官方 mle-bench 且拥有评分数据的环境，从 `MLEvolve_BO` 目录运行：

```bash
python -m utils.grade_submission --log-dir /absolute/path/to/run/logs
```

如评分机器的数据位置不同，附加 `--data-dir /absolute/path/to/mle-bench/data`。该入口读取保存的 `config.yaml`：`evaluation.enabled=false` 使用已记录的旧 Base 选择；旧运行尚无记录时先按同一规则生成记录。`true` 使用已有的固定评估最终节点记录。评分只对记录的路径调用一次官方 `grade_csv()`，评分失败或提交无效都不会换另一个候选重试。完全没有提交时返回 `report: null`。

按 bounded 规则，删除了整文件 SHA、全候选审计、评分前的文件冻结校验和重复的规则名称校验。记录指向 CSV 的当前内容；文件缺失由评分读取直接报错。固定评估的离线评分也只需已导出的提交，不再依赖原始节点快照。CSV 首列的摘要仍用于判断有序值是否相同，这是历史选择条件的流式实现，不是文件冻结或执行许可。最终记录继续原子写入，避免超时中断留下半份 JSON。

`run.py` 的自动收尾只导出选择记录，不调用私有测试评分。已有外部 campaign 若仍硬编码 `best_submission`，必须改为读取 `final_submission.json` 的 `submission` 字段，或调用上述入口；归档 evidence 脚本保持原样。

### 官方仓库核对

已核对 [InternScience/MLEvolve 提交 `9c5c8a3b23f0361708b59a401452dddc00f97189`](https://github.com/InternScience/MLEvolve/tree/9c5c8a3b23f0361708b59a401452dddc00f97189)（2026-09-04）：

- 官方 `run_single_task.sh` 在搜索后执行集成脚本；`engine/validation/format_server.py` 调用的是 `validate_submission()`，只检查格式。
- 官方公开运行流程生成多份 ensemble CSV，没有给出唯一最终 CSV 的私有评分收集器。因此“最新结构匹配 ensemble，否则 best”的规则来自之前实际 Base campaign，不能称为官方原版明确规定的评分选择规则。
- 当前 `_ensemble_legacy()` 从候选选择到保存的主体与该官方 `ensemble()` 对应部分 AST 一致，集成参数与其余辅助函数一致。已有的 `detect_format()` 元数据识别修复与官方不同：数字预测旁的短文本列作为元数据保留；这个修复已存在于历史 Base，本次保留。
- `engine/validation/` 的 `__init__.py`、`format_client.py`、`format_server.py` 和 `quality_check.py` 均与该官方提交 AST 一致。
- 关闭固定评估后，执行评分继续使用原结果解析、指标方向判断和提交检查；PiBO 的潜力评级扩展仍属于算法差异，本次没有改动该评分分支。

## 已对齐的部分

关闭固定评估后，数据准备进入原始公开训练数据分支，候选不受新协议的固定 T/V 视图约束；提示词走 legacy 分支，执行结果走旧评分解析与提交检查。候选仍可以自行留出或交叉验证，这与旧流程的自由度一致。

历史 Base 的 `use_evolution`、`use_fusion`、`use_aggregation`、`use_diff_mode`、`use_stepwise_generation`、`use_global_memory` 均为 `true`。这些开关不能为了“回到旧协议”而关闭。

## 仍不能宣称只剩 BO 与 all_normal_steps 两个差异

以下是现有代码中确认存在、且不受 `evaluation.enabled` 控制的差异。本次没有把它们作为集成接入的一部分修改。

### 1. GP 的额外 draft 补充策略

`engine/parent_selectors/runtime.py::_replenishment_result()` 在 GP 开启、`include_root_draft=false` 时仍可强制生成新 draft。某一方向连续无足够改善的次数超过 `max_improve_failure`，或候选耗尽时，会触发补充；受总 draft 数和预算限制。

按当前／历史共同的 `initial_drafts=3`、`num_drafts=5`、`max_improve_failure=3`，这一路径可以在四次停滞反馈后补第 4、5 个 draft。旧 Base 不走这段 GP 专用逻辑。不能把 `include_root_draft=false` 理解成“只有三个初始 draft，之后永不增加”，也不能把强制补充视为 acquisition 排名本身。

### 2. GP 改变有效节点的继续扩展与回传规则

`engine/evaluation.py::check_improvement()` 对 GP／agent selector 的有效节点提前返回，将 `continue_improve` 置为 `true`，从而跳过旧的无改善次数与时间阶段强制回传启发式。硬限制仍存在，但候选状态的延续规则与 Base 不同。

如果实验允许变化的严格范围仅是“相同候选集合上换 BO 排名，以及扩大选择时机”，这项差异也要另做统一或消融。

### 3. 决策管理和预算计数不同

GP 即使禁用 root-draft 选择，仍会启用 `managed_decisions`。其预算按已提交／预留尝试计数，保存 pending 决策，并检查是否还有派发预算；旧 Base 在 runner 中主要按 journal 已记录节点计进度。生成失败、待执行任务及时间耗尽时，两边不保证完全等价。

直接运行 `run.py` 也不具备外层脚本的 `timeout` 硬截止和格式检查服务启动。内层停止派发不等于正在执行的训练立即结束。对比实验应使用相同的外层运行环境和硬截止；进程被直接强杀后，无法依赖该进程自己执行收尾。

### 4. PiBO 额外改变执行复核提示词

`agents/potential_rating.py::extend_review()` 在 PiBO 中向原来的复核请求增加 `potential_level` 字段及上下文；关闭该功能时返回原提示词和 schema。这没有增加独立的 LLM 调用，但 PiBO 的反馈提示词与 Base 不逐字相同。可以将其视为 PiBO 算法组成，但需要明确记录。

### 5. 仓库默认参数不等于历史实验参数

下表来自当前基础配置和已保存的历史 Base 实际运行配置；实验预设主要设置选择器，不会自动补齐历史 campaign 的全部覆盖参数。

| 参数 | 当前基础默认 | 历史 Base |
|---|---|---|
| 代码／反馈模型 | `gemini-3-pro-preview` | `muse-spark-1.3-contributor` |
| 总搜索时间 | 43,200 秒（12h） | 28,800 秒（8h） |
| 单次执行超时 | 32,400 秒（9h） | 21,600 秒（6h） |
| CPU 请求 | 21 | 16 |

共同参数包括：500 步上限、seed 42、3 个初始 draft、3 路并行和 1 GPU。仍需在最终展开配置及运行环境中统一模型供应方与 reasoning、预训练资源、数据版本、环境版本和 global-memory 初始状态。代码与反馈模型同名，也不能消除跨日期 API 状态或生成随机性。

## 验证范围

2026-09-17 bounded 精简后，`test_grade_submission.py`、`test_legacy_finalization.py`、`test_submission_fusion_utils.py`，以及固定评估测试中的 shell 分派和最终节点导出检查，共 **43 项通过**。覆盖首个合格候选即停止扫描、旧选择与回退规则、评分只使用记录路径、固定模式离线评分不依赖节点快照，以及原有融合、runner、resume 和超时出口。官方评分调用使用本地替身，未进行 GPU 训练或重新测定奖牌率。

以下为精简前的验证记录，保留用于说明已有检查的范围及未解决的固定评估准备耗时问题：

新增检查覆盖旧目录入口与准确目录入口的字节级输出一致性、数值及文本预测、ID 重排、6 候选规模、累计时间限制、跳过任务、best-only 回退、带空格的独立目录、CLI 后处理、Base／GP7 直接运行与恢复运行、GP 搜索时间预算耗尽后的自动集成，以及 shell 的正常／超时／异常／中断分派。

验证结果：`test_legacy_finalization.py`、`test_submission_fusion_utils.py`、`test_fixed_evaluation.py` 的相关回归共 64 项通过；随后加入预算耗尽场景，包含该场景的 3 项 runner／resume 检查全部通过。Shell 语法检查及 `git diff --check` 通过。

这些是本地回归与源码一致性检查，没有进行远程 GPU 训练或重新测定奖牌率。

2026-09-17 最终评分补丁的验证：

- `test_grade_submission.py`、`test_legacy_finalization.py`（追加部分失败用例前）和 `test_submission_fusion_utils.py`：33 项通过。
- 追加的部分集成失败用例，以及固定评估最终节点导出、真实 CPU runner、shell 正常／异常／超时分支：10 项通过，1 项失败。
- 失败项为已有 `test_full_runner_real_cpu_and_fatal_evaluator_propagation[True]` 的 25 秒时限断言，首次运行实际约 109.6 秒。预期 `EvaluationError` 已正常抛出；单独复查仍在同一时限断言失败。30 秒时的线程栈位于 `FixedEvaluation.prepare_or_load()` 等待准备子进程，尚未进入这次修改的 legacy 收尾路径。未放宽断言或修改固定协议准备代码，不能将此轮回归报告为全部通过。
- 对本轮临时目录内 159 份 CSV 调用归档 Base 的原 `signature()` 和新实现，全部结果一致；包括无法解析及结构不匹配的文件。
- Python 语法、空白检查通过；官方集成主体及验证模块的 AST 对照结果见上文。
