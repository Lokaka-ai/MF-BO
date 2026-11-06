# 本人七项 MCGS 正式运行记录

按 [MCGS 合作者手册](MLEBENCH_LITE_MCGS_COLLABORATOR_RUNBOOK_20260921_zh.md) 启动本人负责的七项 competition。使用 `MLEvolve_BO`、全新的输出目录及 `experiment_preset=mlevolve_original`，未加载 AQ05 的检查点、搜索图、memory 或候选。提交时间为 2026-09-22 13:16 UTC（Toronto 09:16）。

运行根目录：`runs/mcgs_lite7_20260922T131459Z/`。

| 任务 | 分区 | 首批 allocation | 每项 CPU / 内存 / GPU |
|---|---|---|---|
| APTOS | unkillable | 10881709，复用原交互 allocation | 6 / 32 GiB / 1 A100 |
| RANZCR | main | 10890735 | 4 / 24 GiB / 1 A100 |
| Cactus | main | 10890735 | 4 / 24 GiB / 1 A100 |
| Taxi | short-unkillable | 10890736 | 4 / 64 GiB / 1 A100 |
| Jigsaw | short-unkillable | 10890736 | 4 / 64 GiB / 1 A100 |
| Birds | short-unkillable | 10890736 | 4 / 64 GiB / 1 A100 |
| English | short-unkillable | 10890736 | 4 / 64 GiB / 1 A100 |

资源与此前本人非 agent AQ05 对应任务的初始实配配对。当前分区用户配额为 main 2 GPU / 8 CPU / 48 GiB，unkillable 1 GPU / 6 CPU / 32 GiB，short-unkillable 最多 4 GPU；main 申请 13 小时，short 每段申请 3 小时。原交互 allocation 的 shell 保留，控制器只管理自己的 step。

科学设置：原生 UCT / Top-K soft-switch 和分支直接续接，`legacy_original / root_only / include_root_draft=false`；AQ、rating、ancestor 均关闭。code 和 feedback 使用 `muse-spark-1.3-contributor`、`codex://opencode-go`、high reasoning；按照本手册使用单次调用超时 1800 秒。每项 43200 秒累计搜索预算、500 个节点上限、seed 42、3 drafts、3 workers、候选累计执行上限 32400 秒，`evaluation.enabled=false`。使用相同 prepared public 数据，结束后执行原有 submission fusion 和官方 grading。

## MCGS 短租期续跑

用户明确选择补齐并验证续跑后七项并行。本次将已有原生 competition 检查点扩展到 MCGS；这是显式环境开关 `MLEVOLVE_COMPETITION_CHECKPOINT=1` 下的记录和恢复功能。全局选点仍直接调用 `_legacy_result` / `select_with_soft_switch`，续接仍为 `direct_continuation`，不使用 BO 的选点入口或占用父节点筛选。

检查点记录已完成节点、尚未完成的决策、生成代码、Python/NumPy RNG、节点计数与锁、搜索统计、原始数据预览、已完成但尚未派发的续接节点，以及累计运行和候选执行耗时。中断的生成继续原决策身份，已生成代码用原 node ID 和代码恢复执行，不重复增加子节点计数，不把租期中断生成虚假失败节点。

short allocation 结束前约 10 分钟保存并退出。控制器确认旧 allocation 已实际释放后再提交下一段；排队和重载不会重置 12 小时预算。此机制不能通用恢复候选自己训练中的模型/优化器状态，可能重执行同一候选代码，该开销保留在累计成本里。

搜索达到预算后才进入融合和官方评分。MCGS 仍按原生已完成节点计数，单纯生成异常只保留日志，不因新增记录机制多消耗科学 step，也不添加科学节点。

## 验证与追溯

- 58 项相关测试分批通过，覆盖原生 MCGS、AQ05 检查点、pending-parent 和恢复状态。
- 对相同种子和模拟模型/执行输出，检查点开关前后搜索树、节点统计及 RNG 一致，覆盖早期探索及后期 soft-switch。
- 三个并行槽位的中断代码恢复、反复中断时决策/计数不重复、原分支续接、累计预算保留均通过测试。
- 独立子进程验证 MCGS/AQ05 在租期结束和预算耗尽时先保存再 exit 75，暂停时不进行最终评分。
- 真实 Muse 请求返回 `OPENCODE_MUSE_OK`；原 unkillable allocation 上 A100 CUDA 运算通过。
- 七项预备配置与数据路径全部通过检查，依赖版本保存在 `preflight.json`。

`manifest.json`、`code/`、`source.patch`、`source-sha256.json` 定义本次冻结代码和配置。MCGS 的这次检查点扩展属于相对历史 AQ05 的实现差异，历史 AQ05 还有其自身已记录的修复和中断，不能只凭相同参数宣称完全相同运行条件。

启动核验已完成，证据保存在 `startup-verification.json`：七项均为 running，实际配置符合 MCGS 设置，使用七张不同的 A100，控制器存活。核验时六项已进入原生选点及首批草稿生成，Taxi 正在处理输入预览；尚未据此宣称完成候选评估。用户要求启动后停止人工监控，后续由后台控制器处理续租。

查看 `summary.json`、`queue.json` 和 `tasks/<competition>/status.json` 获取最新调度状态。实际配置在 `runs/<timestamp>_<competition>/logs/config.yaml`，各次 Slurm 命令、资源、搜索日志和退出记录保存在 `tasks/<competition>/attempt-*/`。

不要重复执行 `init` 或重复提交同一任务。原控制器 `controller-launch.json` 和 `controller-process.json` 记录进程身份；自动续租将继续运行，无需人工持续监控。正在运行的 AlphaEvolve 作业未作任何修改或取消。

状态和日志已接入 [MCGS 共享 Space](https://huggingface.co/spaces/jzshared/mlebench-mcgs-12h-20260922)，负责人筛选 `liux`。独立 CPU 同步作业、日志范围、服务有效期及恢复命令见 [HF 同步记录](HF_MCGS_OWNER_SYNC_20260922_zh.md)。
