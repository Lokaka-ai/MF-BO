# AlphaEvolve 15 项任务接入与 CPU 实测（2026-09-20）

15/15 项任务已完成真实 LLM 生成的有效 draft 和一次有效 improve；所有选定的 30 个节点通过数学约束检查，导出的最佳构造通过独立复评分。任务沿用 MLEvolve_BO 的原生 AgentSearch、GP6、全局记忆、分步生成、diff 改进、代码审查和结果审查。

这是两节点流程验证，未运行完整 12 小时性能实验，也不据此宣称复现论文得分。首轮 14/15 项通过；第三自相关变体重跑一次后通过，原始失败记录保留在汇总报告中。

## 与 mlebench_lite 的配置对应

| 项目 | 正式运行默认 | 本次短验证 |
|---|---|---|
| 每任务 agent 时间预算 | 43,200 秒 / 12 小时 | 同样 43,200 秒，两个节点后停止 |
| 搜索设置 | 仓库现有 `gp6.yaml` | 同一预设 |
| 总步数 / 初始 draft / 并发 worker | 500 / 3 / 3，继承基础设置 | 2 / 1 / 1 |
| GPU | 0；隐藏 CUDA | 0；CPU Slurm 节点 |
| CPU 核心 | 21，三个 worker 各 7 核 | 两个圆填充任务各 3 核，其余各 1 核 |
| 候选执行超时 | 32,400 秒，继承基础设置 | 每个候选 120 秒，传给 solve 的搜索预算 110 秒 |
| 全局记忆 | 开启，BGE 在 CPU 上运行 | 同样开启，实际检索 |
| 分步生成 / diff / review / evolution / fusion / aggregation | 保留原生设置 | 保留；两节点只触发 draft / improve 路径 |
| code / feedback 模型 | 继承基础配置，可显式覆盖 | `muse-spark-1.3-contributor`，`opencode-go` 后端 |
| 数学领域适配 | `solve(problem, initial, time_budget, seed)` 与宿主数学评分 | 同一协议 |
| ML 预训练模型 coldstart | 关闭，数学构造无对应的预训练模型库 | 同样关闭 |

GP6 保持 Matérn-3/2、child UCB、固定 beta=1.5、每次观测重新拟合、BO 选择中不增加 root draft。每项实测都核对了 120 个保存的配置字段，并检查 `selection_decisions.jsonl` 中实际使用 `gp6-*` 模型选择 draft 为 improve 的父节点，`fallback_used=false`。同时检查两个候选的 CPU 亲和性和空 `CUDA_VISIBLE_DEVICES` 记录。

## 数据、公式和环境

已准备 [15 项数学任务数据](../data/alphaevolve) 和 [任务顺序](../config/alphaevolve15.txt)。这些任务使用数学问题定义与初始构造，没有监督学习的 train/test 数据集。公开目录包含问题和简单可行初始构造；官方结果放在 private 目录，仅用于验证评分器，不放入 agent 上下文。

题目对应 [MLEvolve §4.1 / Table 2](https://arxiv.org/html/2606.06473#S4)，公式及参考构造来自 [Google DeepMind 官方 notebook](https://github.com/google-deepmind/alphaevolve_results/blob/4226acbf237ff9ad10ba7673a2af127a2d8a5971/mathematical_results.ipynb)，固定 commit 为 `4226acbf237ff9ad10ba7673a2af127a2d8a5971`，下载后校验 SHA256 并生成逐文件哈希。

复用 `.venv-mlebench`：Python 3.12.12、NumPy 2.2.6、SciPy 1.16.3、SymPy 1.14.0、Torch 2.9.1+cu128、faiss-cpu 1.11.0。Torch 虽为 CUDA 构建，本次只在 CPU 上执行。`uv pip check` 通过；真实 BGE CPU embedding 输出形状为 `[2, 768]`，没有用占位 embedding。

环境验证作业 `10870504` 在 `cn-m004` 通过全部 15 个简单初始构造和 15 个官方参考构造。官方结果复算与论文表格在 1e-9 内一致，唯一例外是 sums/differences 1：官方构造重算为 `1.1479888965092757`，论文表格为 `1.1479889651`，差约 6.86e-8；没有为匹配表格修改公式。严格差值审计另存为 JSON。

第三自相关与其变体分别使用 `abs(max(convolution))` 和 `max(abs(convolution))`；第二自相关使用分段线性自卷积平方的精确积分；max/min 使用距离比的平方。两项 sums/differences 对应官方两组参考构造，使用同一个原始不等式，没有另造大小奖励。作者未公开的数学任务 prompt、初始化和单次构造搜索预算不声称逐字复现，完整说明见 [来源与复现范围](../tasks/alphaevolve/SOURCE.md)。

## 实测结果

下表是最后一次明确运行的结果，不按分数挑选重跑。得分下降或不变仍可满足流程验证；最佳构造按原生目标方向保留。精确值、节点 ID、代码、输入和输出哈希均在逐项报告中。

| 数组索引 | 任务及实测记录 | 目标 | Draft | Improve | 保留最佳值 | Slurm Job ID |
|---|---|---|---:|---:|---:|---|
| 0 | [hexagon_packing](../runs/alphaevolve_cpu_validation_20260920/tasks/hexagon_packing/report.json) | 最小化 | 4.27995374538 | 4.06627945789 | 4.06627945789 | 10870580 |
| 1 | [circle_packing_square](../runs/alphaevolve_smoke_10870538/circle_packing_square/report.json) | 最大化 | 2.16666404067 | 2.52594702703 | 2.52594702703 | 10870539 |
| 2 | [circle_packing_rectangle](../runs/alphaevolve_smoke_10870538/circle_packing_rectangle/report.json) | 最大化 | 2.31589432183 | 2.34027522742 | 2.34027522742 | 10870538 |
| 3 | [heilbronn_convex](../runs/alphaevolve_cpu_validation_20260920/tasks/heilbronn_convex/report.json) | 最大化 | 0.0176221498995 | 0.0219087283362 | 0.0219087283362 | 10870581 |
| 4 | [heilbronn_triangle](../runs/alphaevolve_cpu_validation_20260920/tasks/heilbronn_triangle/report.json) | 最大化 | 0.030650933415 | 0.0244596409527 | 0.030650933415 | 10870637 |
| 5 | [kissing_number_d11](../runs/alphaevolve_cpu_validation_20260920/tasks/kissing_number_d11/report.json) | 最大化 | 109 | 109 | 109 | 10870638 |
| 6 | [sums_differences_1](../runs/alphaevolve_cpu_validation_20260920/tasks/sums_differences_1/report.json) | 最大化 | 1.10268493213 | 1.09752835073 | 1.10268493213 | 10870645 |
| 7 | [sums_differences_2](../runs/alphaevolve_cpu_validation_20260920/tasks/sums_differences_2/report.json) | 最大化 | 1.11150774058 | 1.10980107833 | 1.11150774058 | 10870687 |
| 8 | [uncertainty_inequality](../runs/alphaevolve_cpu_validation_20260920/tasks/uncertainty_inequality/report.json) | 最小化 | 0.352100793346 | 0.352099104416 | 0.352099104416 | 10870718 |
| 9 | [first_autocorrelation](../runs/alphaevolve_cpu_validation_20260920/tasks/first_autocorrelation/report.json) | 最小化 | 2 | 2 | 2 | 10870719 |
| 10 | [third_autocorrelation_variant](../runs/alphaevolve_cpu_validation_20260920/retry1/third_autocorrelation_variant/report.json) | 最小化 | 1.65433842097 | 1.65123606081 | 1.65123606081 | 10870811 |
| 11 | [third_autocorrelation](../runs/alphaevolve_cpu_validation_20260920/tasks/third_autocorrelation/report.json) | 最小化 | 1.47605649661 | 1.64940414204 | 1.47605649661 | 10870721 |
| 12 | [second_autocorrelation](../runs/alphaevolve_cpu_validation_20260920/tasks/second_autocorrelation/report.json) | 最大化 | 0.806828012967 | 0.808537408591 | 0.808537408591 | 10870722 |
| 13 | [max_min_ratio](../runs/alphaevolve_cpu_validation_20260920/tasks/max_min_ratio/report.json) | 最小化 | 13.0190353529 | 13 | 13 | 10870723 |
| 14 | [minimum_overlap](../runs/alphaevolve_cpu_validation_20260920/tasks/minimum_overlap/report.json) | 最小化 | 0.382268073543 | 0.381611823484 | 0.381611823484 | 10870577 |

短测试以 CPU 数组作业并行运行：`10870538`（两个圆填充）、`10870577`（其余 13 项）、`10870811`（变体重跑）。申请 8–12 GiB 内存；待运行任务曾按实际用量降低内存申请以适应集群配额。所有正常测试均未请求 GPU。

第三自相关变体首次 draft 错用了不存在的 `np.finite`，原生系统随后生成了有效 debug 节点；该次运行按要求判为失败，没有将 debug 计作 improve。相同配置下重新生成 draft，完成一次有效 improve 后通过。没有手工替换 agent 生成的候选，也没有注入官方答案。

两个圆填充先验证接入；其余任务及重跑使用 [225 个源码文件的固定快照](../runs/alphaevolve_cpu_validation_20260920/source/SOURCE_HASHES.json)，运行期间文件哈希未变。快照后增加的汇总配置审计和运行轨迹审计对全部任务统一执行。

## 验证证据

- [15 项完整汇总 JSON，含所有尝试、配置审计及 GP6/CPU 轨迹审计](../reports/alphaevolve_cpu_validation_20260920.json)
- [环境与全部官方构造验证](../.cache/alphaevolve/environment-ready.json)、[更严格的表格差值审计](../.cache/alphaevolve/reference-table-audit.json)
- [测试记录](../.cache/alphaevolve/unit-checks.json)、[Slurm 资源与退出状态](../.cache/alphaevolve/slurm-accounting.tsv)

相关测试共 134 个不同用例最终通过：主回归首次 125 通过、1 项原有 25 秒耗时断言失败（预期错误正确抛出，实际耗时 30.7 秒），该用例在 CPU 计算节点单独复查通过，未修改断言；另外 8 个配置和两节点流程契约检查通过。Python AST、shell 语法和 `git diff --check` 检查通过。

## 运行命令

从 `MLEvolve_BO` 目录执行：

```bash
# 环境、数学输入和官方参考构造
bash tasks/alphaevolve/setup.sh
sbatch tasks/alphaevolve/verify.sbatch

# 15 个任务各生成 draft，再执行一次 improve
sbatch tasks/alphaevolve/smoke.sbatch \
  --model muse-spark-1.3-contributor --provider opencode-go

# 正式 12 小时 / CPU / GP6 默认任务
sbatch tasks/alphaevolve/run.sbatch \
  --model muse-spark-1.3-contributor --provider opencode-go
```

模型参数是本次使用的可用后端；省略时继承 mlebench_lite 的基础模型配置。只跑一个任务可在 `sbatch` 加 `--array=<索引>`。正式启动命令已准备，本次仅实际执行用户要求的短验证。完整接口说明见 [任务 README](../tasks/alphaevolve/README.md)。
