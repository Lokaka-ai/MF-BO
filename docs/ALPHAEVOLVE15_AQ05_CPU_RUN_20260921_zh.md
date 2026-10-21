# AlphaEvolve 15 项：AQ05 / CPU / 12 小时

**2026-09-21 15:00 UTC 已按用户要求暂停全部 15 项并释放 CPU 分配。**
等待用户明确要求继续后再续跑；暂停期间不计入累计 12 小时预算。
检查点、剩余预算和恢复注意事项见 [暂停记录](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/pauses/20260921T145824Z/README.md)。

本轮初始 Slurm 数组为 **10875299（0–14，最大并行 15）**，使用 `long-cpu` 的纯 CPU 节点。
后续续跑作业由运行目录中的 `recovery-map.json` 关联到同一实验，`poll_status.py` 会查询它们的实时状态。
2026-09-21 07:37 UTC 核对时，15 项均为 RUNNING，分布于 cn-f004、cn-h002、cn-h003、cn-h004。
每项申请 **4 核 CPU、24 GiB、0 GPU**。Slurm 时限 13 小时包含启动和结束评分；
agent 有效预算为 **43200 秒（12 小时）**，初始采用 AQ05 原有的 500-step 上限。
用户已明确要求累计跑满 12 小时，并允许必要时提高步数上限。
提前触及上限的任务将保留已有搜索状态，提高 `agent.steps` 后续跑剩余有效预算。
当前实验已暂停，尚未完成每项累计 12 小时。
启动复核已通过：15/15 的实际配置匹配，CPU 亲和性互不重叠，
所有任务均已记录成功的 Muse 1.3 / high reasoning 调用，凭据路径均为 `opencode_mila`。

配置来自正在运行的 `aq05_lite7_20260921T0600Z/code` 独立源码快照：

- preset：`acq7_aq05_gp4_level15_rating_kg`，finite-pool KG、rating 开启。
- code / feedback：`muse-spark-1.3-contributor`，`codex://opencode-go`，high reasoning。
- seed 42；3 个初始 draft；3 个 worker；单候选执行上限 32400 秒。
- 保留全局记忆、stepwise、diff、代码审查和结果审查。
- 凭据仅从 `codex/opencode_mila` 读取，不将 key 写入命令或报告。

数学领域适配沿用既有 AlphaEvolve 接口和独立验证器：隐藏 CUDA，记忆 embedding 使用 CPU，
关闭不适用的 ML pretrained coldstart。新快照另修正了 rating 资源上下文，
准确说明 CPU 数学构造协议；补丁仅存在于本轮快照中。

本轮的 [运行目录](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/README.md) 包含：

- [manifest](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/manifest.json) 和
  [Slurm 提交回执](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/submission.json)。
- [启动检查](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/preflight/passed.json)：
  15 项各对照 7 份原实验配置、每份 149 字段；84 项相关测试；真实 BGE CPU embedding；
  全部数学参考构造；新 key 的真实后端请求；真实 AQ05 draft → improve → 独立复评分。
- `tasks/<task>/`：每项的 CPU 资源、命令、原始日志、配置、journal、候选构造和独立最终评分。
- `startup-verification.json`：正式任务的实际配置、独立 CPU 分配和成功模型调用核验。
- `status.json`：Slurm 状态与任务进度；`completion-audit.json` 仅在预算或原有步数上限满足、
  最终构造独立评分通过后标记通过。
- 完整 12 小时与最终构造的联合核验记录在 `final-audits/<task>/completion-audit.json`，
  对应字段为 `full_duration_and_result_verified`。

启动验证中的两次检查脚本问题和对应自动取消的未启动依赖数组保留在 `preflight-attempts/`：
一次是无关 AQ09 研究测试的快照外依赖，另一次是将 AQ05 正常的 `single_candidate`
决策标记误认成应为 `model`。后者保留已成功的原生试跑，以关联的 GP/KG 决策和哈希重新核验；
没有重新生成候选来挑选更好的结果。最终验证作业 **10875290** 成功退出。

本轮 prepared 数据、embedding 缓存、代码和输出使用独立目录，依赖环境只读复用。
启动时原 mlebench-lite 的 10874910、10874911、10857987 均在运行；
其后续分配续租由原实验自己的 controller 管理。
本轮只读监控 tmux `aq05_alpha15_cpu_20260921` 已随本次暂停停止；原实验的 controller 未作更改。

```bash
cd /network/scratch/l/liux/EvolveFold/MLEvolve_BO/runs/aq05_alphaevolve15_cpu_20260921T071425Z
python poll_status.py
```

发生抢占或节点故障时，必须先确认该具体作业已终止，并保留原 journal、决策记录和已消耗预算；
不能将中断静默重置为一项新的 12 小时实验。

2026-09-21 10:30 UTC 前，`kissing_number_d11`、`second_autocorrelation`、
`minimum_overlap` 因框架保存搜索树时的并发复制或递归深度异常退出。
已保存完整原始存档，按原生累计预算准备续跑；其余 12 项保持运行。
具体错误、源代码最小修复、CPU 预检和恢复回执见
[恢复记录](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/recovery/README.md)。
恢复暂停时间不计入实验预算；最终完成核验使用 AQ05 原生 transition 计数，
其中包含生成失败但已经记录的步骤。

恢复预检作业 **10876021** 已完成：22 项测试和 3 份真实存档的加载、全局记忆恢复、
完整及筛选 journal 再保存全部通过。续跑数组 **10876042（0–2）** 于 11:01:03 UTC 启动，
分别保留 148、108、169 个已完成节点及 149、189、174 条 transition。
[正式续跑验证](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/recovery/startup-verification.json)
确认每项 4 CPU、0 GPU、原会话、原工作目录、原累计预算及实际成功的 high-reasoning 请求。
目前实验仍在运行，最终结果尚待预算和独立数学评分审计。

11:15 UTC，`heilbronn_triangle` 因 HTTP 429 导致必需的执行评审失败而停止。
同一 key 的真实请求再次通过后，作业 **10876128** 完成真实存档恢复预检，
续跑作业 **10876129** 保留 25 个已完成节点、原工作目录、会话和配置，继续剩余 30178 秒。
中断的步骤按原生恢复规则记为失败；没有跳过评审或更换算法。
[本次恢复验证](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/recovery/round-02/startup-verification.json)
同时核对了全部 15 项的 CPU 亲和性互不重叠，并确认续跑后的模型请求成功。

11:47 UTC，另外 5 项因 HTTP 429 中断的任务通过作业 **10876263** 的存档恢复预检后，
由数组 **10876280（0–4）** 继续各自剩余预算。此次独立快照为同一个模型请求增加
有上限的退避重试，所有重试共用原有的 1800 秒调用预算；模型、key、会话、算法和必需评审保持原配置。
5 项新增重试测试通过，既有测试与修复前的结果一致。
[正式续跑验证](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/recovery/round-03/startup-verification.json)
确认全部存档节点保留、149 字段配置对照通过、15 项 CPU 分配互不重叠，
并记录到实际 HTTP 429 请求经退避后成功返回。11:53 UTC 时全部 15 项均在运行。
每项是否累计达到 12 小时将单独核验；达到原有 500-step 上限本身不代表已经满足完整时长。
用户确认的停止策略记录在
[停止策略](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/stop-policy-question.json)：
优先满足每项累计 43200 秒，必要时提高步数上限；恢复暂停时间不计入有效预算。

12:03–12:05 UTC，仍使用旧请求处理代码的 `heilbronn_triangle` 与 `kissing_number_d11`
也因 HTTP 429 导致必需评审失败而终止。两份存档分别保留 44、207 个节点，
经 CPU 预检 **10876497**、**10876537** 后，由 **10876498**、**10876538** 续跑
各自剩余的 27687、29427 秒，使用同一份已验证的退避重试代码。
[heilbronn 启动核验](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/recovery/round-04/startup-verification.json)
与 [kissing 启动核验](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/recovery/round-05/startup-verification.json)
均已通过：原节点保留、149 字段配置匹配、实际模型请求成功、全部 15 项 CPU 分配互不重叠。
12:18 UTC 时，15 项正式作业均在运行；累计 12 小时和最终独立评分仍待完成。

最终核验进一步要求导出的最佳节点具有唯一对应的有效 transition、明确的预算内标记，
以及不超过 43200 秒的累计完成时间；缺失记录或字段不会被视为通过。
CPU 验证作业 **10876761** 复现了原核验在缺少记录时的误放行，修正后的 10 项回归测试通过，
并核对了当时全部 15 项任务的 1207 条有效候选时间记录。
[验证回执](../runs/aq05_alphaevolve15_cpu_20260921T071425Z/audit-verification/passed.json)
只证明这项核验逻辑，完整时长和最终构造的独立评分仍须在各任务结束后执行。
