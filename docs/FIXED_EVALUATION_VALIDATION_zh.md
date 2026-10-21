# 固定评估本地验收记录

日期：2026-09-14。工作树：`/Users/liuxixian/.codex/worktrees/d290/EvolveFold`。所有修改保留未提交状态，供主任务审查合入；范围仅为 `MLEvolve_BO`。

## 结果

| 验证 | 结果 | 实际边界 |
| --- | --- | --- |
| `tests/test_fixed_evaluation.py` | **48 passed，177.72 秒** | 新固定评估的准备、执行、评分、恢复、runner 与 helper |
| 相关既有 runtime/resume/submission/graph/tests_gp6 | **144 passed，28.66 秒** | 本次改动影响的原有运行与 GP 回归 |
| 修改目录 Python compileall | 通过 | Python 语法/可编译性 |
| 单任务、GP6 与 selector task/suite shell 的 `bash -n` | 通过 | shell 语法 |
| `git diff --check`（允许原有 CRLF） | 通过 | 已跟踪 diff 无新增空白错误；不重写原 CRLF 文件 |

最后的候选协议文案统一补明 metadata 原样保留，未改执行逻辑；随后完成 compileall、shell 与 diff 检查，没有因纯文案修改重复整套测试。

本地使用隔离测试环境 `/tmp/mlevolve-fixed-eval-d290/bin/python`。完整输出分别保留在 `/tmp/mlevolve-fixed-evaluation-final-tests.txt` 和 `/tmp/mlevolve-fixed-evaluation-final-regressions.txt`。复现命令在 `MLEvolve_BO` 下运行：

```bash
PYTHONPATH=.:tests /tmp/mlevolve-fixed-eval-d290/bin/python -m pytest -q \
  tests/test_fixed_evaluation.py --tb=short

PYTHONPATH=.:tests /tmp/mlevolve-fixed-eval-d290/bin/python -m pytest -q \
  tests/test_gp6_runtime.py tests/test_root_draft_runtime.py \
  tests/test_resume_state.py tests/test_resume_config.py tests/test_resume_memory.py \
  tests/test_submission_fusion_utils.py tests/test_graph_runtime.py tests_gp6 --tb=short

/tmp/mlevolve-fixed-eval-d290/bin/python -m compileall -q \
  agents engine config utils tests/test_fixed_evaluation.py run.py __init__.py
bash -n run_single_task.sh scripts/run_gp6_task.sh scripts/run_gp6_suite.sh \
  scripts/run_selector_task.sh scripts/run_selector_suite.sh
git -c core.whitespace=cr-at-eol diff --check
```

## 主任务复核与合入

主任务已逐项审查实现及实际测试输出，将审查发现的问题交回同一实现任务修复。2026-09-14 已将审查后的 24 个既有文件修改和 7 个新文件整合到 `/Users/liuxixian/Documents/EvolveFold`，全部位于 `MLEvolve_BO`，未提交或推送。

除核对上述 48 + 144 项结果外，主任务独立检查了七组指标数值，并在当前仓库路径复跑完整 CPU 生成/执行/评分/GP/恢复链路：

```bash
PYTHONPATH=.:tests /tmp/mlevolve-fixed-eval-d290/bin/python -m pytest -q \
  tests/test_fixed_evaluation.py::test_cpu_candidate_generation_execution_scoring_gp_and_resume --tb=short
```

结果为 **1 passed，12.37 秒**；当前工作树的 shell 语法和允许原有 CRLF 的 diff 检查也通过。该复跑验证了合入后的实际路径，不重复计入上面的 48 项覆盖。真实数据、外部模型和 GPU 的未验证边界保持下文所述。

## 哪些路径是真实执行

| 路径 | 实际执行内容 | 受控部分 |
| --- | --- | --- |
| 一次性准备 | `llm.query`、HTTP 请求、FunctionSpec、prompt 编译、回复解析、生成程序子进程、admission、协议落盘与复用 | localhost HTTP server 返回固定结构化回复，不请求付费模型 |
| 二分类纵向链路 | 正常 candidate code parser → draft/deferred 与 improve → Interpreter → CPU LogisticRegression 训练 → T 内校准 → V/test 推理 → 文件 scorer → valid transition → 六维 GP fit → journal 恢复 | 代码生成/代码审查/运行总结的模型回复受控 |
| debug | 首次训练保存 logits，但故意删除 V 预测；正常判失败。debug 从父缓存重新输出，不重训；首个有效结果归并到原 draft 并累计成本 | 修复代码由测试指定 |
| evolution/fusion | 实际 operator 分发、生成后解析、Interpreter、固定 scorer、transition | 测试指定 operator preview 与模型回复；不宣称自然语言模型自主选择该动作 |
| stepwise/diff/review | 各子步骤及 merge prompt 的真实编译、diff 解析应用、共享协议/helper 可见性 | 没有用远端模型完成整套 stepwise 多轮训练 |
| 完整 `run.run` | 初始 deferred draft、等待 worker、继续搜索、持久化、最终最佳提交；公共 scorer 故障传播与等待 worker 退出 | task/config 使用合成 fixture；候选模型回复受控 |
| 库入口 | `Experiment.run` 经真实 Interpreter/scorer 导出结果 | CPU 合成任务、受控模型回复 |
| shell | 真正执行 `run_single_task.sh` 的 run-info 判断、退出码与后处理分支 | 服务探测、主 python 命令和外层 timeout 使用本地可执行 stub；没有启动 GPU 服务 |

## 对应风险的验证证据

- 主分数可信度：候选真实打印 `Final Validation Score: 999.0`，总结模型也返回 999，但节点与 GP 使用宿主分数。合法 AUC 0/1、真实劣质 log loss 大于 3 均写入有效 transition。
- 完整对齐：行重排合法；缺行、重复/额外 ID、错误目标、NaN、越界、缺文件拒绝。多目标、所有 token 和不同大小图像的展开像素 fixture 经实际 admission 和 scorer，删除一条评分行会拒绝。
- 指标定义：独立数组区分 macro/micro AUC、RMSE/MSE、逐目标 RMSLE 平均；验证 class 顺序、数值/字符串标签、quadratic kappa 离散标签和文本 exact match。
- 原始模板：plain CSV 及 ZIP archive member 独立核对全部 test IDs/targets。ZIP + metadata 组合 fixture 保留空 Date、字面 NA，拒绝 V 元数据改变、公开模板元数据改变或缩减 test IDs。
- 数据边界：拒绝 group 重叠、重复原始 source_id、V 标签列、错误 metric 方向；opaque V 文件以硬链接/复制暴露，T-only 目标资产隔离，拒绝跨分区重复源资产。测试不等于 OS 隔离证明。
- 校准：网格最优温度返回单个 float，±1000 稳定转换；非法网格/形状和 `clip_eps=1e-30` 拒绝。实际 V/test 调用同一个 helper。
- 准备失败：两次受控失败后停止，修复请求包含真实第一次程序错误。API deadline 在本地 server 已收到请求、仍延迟应答时终止准备；日志保存、公开输入为空、没有候选评分回退。
- 并发与故障：三路真实候选使用不同 attempt 目录；新 attempt 不复用旧预测。删除公共真值文件会抛出公共评估错误，保留候选预测，在写 transition 前停止。
- 恢复与预算：普通恢复和不同工作目录的新方法组复用同一协议，不再次请求准备 API；共同准备成本被扣除。拒绝无协议历史、固定/旧模式切换和数据变化。
- 最终提交：即使节点原提交后来被改写，最终导出仍来自评分时宿主保存的快照。shell 的空/无效 run-info 不触发旧 fusion；固定与 legacy 分支各有对应测试。

## 实际变更分组

新增：

- `agents/evaluation_setup_agent.py`：一次性准备 agent、任务/数据 prompt、结构化回复、有界修复和共享协议锁。
- `engine/fixed_evaluation.py`：协议 admission、完整对齐与指标、公共视图、宿主输出目录、评分、恢复校验、最终选择。
- `utils/evaluation_helpers.py`：binary sigmoid/clipping 与温度网格 helper。
- `config/fixed_evaluation_metrics.json`：22 项任务的独立指标/方向/ID/元数据约束。
- `tests/test_fixed_evaluation.py`：上述 48 项本地机制与集成验证。
- 两份固定评估中文文档。

既有文件：

- config 定义/YAML、SearchNode/Journal 增加配置与持久化元数据。
- AgentSearch、executor、execution、result parser 在共同执行/评分入口接入。
- draft/debug/improve/evolution/fusion/aggregation、stepwise/diff/shared guidelines/code review 统一协议说明。
- run、库入口、单任务 shell 和旧 submission fusion 工具接入最终选择与错误传播；runtime 只加恢复契约和等待超时。
- README 增加运行与验收文档入口。

没有改 BO 核函数、特征、acquisition 或模型默认值，没有修改 `MLEvolve_BO` 以外的工程。

## 尚未验证的条件

没有运行真实 22 项数据集、GPU 训练、远端 SSH 或付费 LLM，也没有修改历史成绩。22 项在实现上具备统一启动路径、指标/格式约束和通用原生数据适配接口，但真实数据准备成功率、模型生成质量、GPU 性能和最终榜单成绩仍须独立实验验证。

同 UID 路径隔离是合作式限制，无法防止恶意候选直接读取宿主绝对路径。大规模像素 CSV、archive 解码与准备时间需要在实际数据上测量；task adapter 语义仍需审查。准备代码/原始数据变化不会自动迁移已有搜索；公共评估故障后保留材料并停止，不能把修过的 scorer 分数混入旧 GP 历史。

启动、跨方法共享及恢复命令见 [固定评估说明](FIXED_EVALUATION_PROTOCOL_zh.md)。
