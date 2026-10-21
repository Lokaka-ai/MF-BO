# AQ05：本人负责的 7 项正式运行记录

本轮使用 `MLEBENCH_LITE_AQ05_TASK_SPLIT_20260920_zh.md` 的 competition 设置：
AQ05 preset `acq7_aq05_gp4_level15_rating_kg`、code/feedback 模型
`muse-spark-1.3-contributor`、provider `codex://opencode-go`、seed 42、3 drafts、
3 workers、每项独占 1 张 A100、43200 秒有效预算、500 steps、候选执行上限
32400 秒、`evaluation.enabled=false`。不加载 autoresearch task preset。

## 最终结果

2026-09-21 14:39（Toronto），本人负责的七项全部达到完整搜索预算并完成官方评分。
七份提交均有效，逐项 12 项完成检查全部通过，包括冻结后的评分顺序和文件 SHA256。

| Competition | 有效搜索秒数 | 完成 / 自检有效候选 | 官方分数 | 官方报告比较 |
|---|---:|---:|---:|---|
| aerial-cactus-identification | 43201.240729 | 68 / 37 | 1.00000 | 达到金牌阈值 |
| aptos2019-blindness-detection | 43202.210144 | 116 / 59 | 0.91155 | 高于中位数，无奖牌 |
| jigsaw-toxic-comment-classification-challenge | 43200.231231 | 5 / 3 | 0.98616 | 高于中位数，无奖牌 |
| mlsp-2013-birds | 43206.753728 | 272 / 225 | 0.88146 | 达到铜牌阈值 |
| new-york-city-taxi-fare-prediction | 43200.836224 | 19 / 14 | 8.98329（越低越好） | 未达到中位数水平 |
| ranzcr-clip-catheter-line-classification | 43201.191414 | 25 / 10 | 0.92956 | 未达到中位数水平 |
| text-normalization-challenge-english-language | 43201.262765 | 26 / 17 | 0.99047 | 达到铜牌阈值 |

最终提交均按预定的 `latest_structurally_valid_ensemble_else_best_v1` 规则固定，
没有根据官方 private score 重选候选或追加预算。原生搜索均以 exit 75、
`budget_complete` 保存退出；这与 Slurm step 的最终退出状态分别记录。
最后的 Birds step `10880077.2` 为 `COMPLETED / 0:0`，short allocation 于
14:39:46 释放。全部本轮新申请的 main、short allocation 已结束；用户原有
unkillable `10857987` 继续保留。控制器于 14:40:04 正常结束调度。

结果状态为 **已完成，存在已记录的运行偏差**。短租期会重执行被中断的候选代码，
没有通用模型／优化器状态恢复；另有初次恢复数据预览变化、检查点线程故障、选择器
运行时修复和 Slurm OOM。APTOS、Cactus、RANZCR 合计保留的 5 个框架关闭影响
的无效结果可能改变后续选择。本轮不能视为无中断对照，官方有效评分不消除这些偏差。

完整登记和验收证据：

- [`final-results.csv`](../runs/aq05_lite7_20260921T0600Z/final-results.csv)：每项负责人、全部 job ID、
  基线 commit、实际 GPU、完整配置、时间、退出状态、最终提交、SHA256 和评分报告路径。
- [`completion-audit.json`](../runs/aq05_lite7_20260921T0600Z/completion-audit.json)：
  七项预算、正式配置、提交有效性及中断结果核验；逐项明细在同目录 `completion-checks/`。
- [`requirements-audit.json`](../runs/aq05_lite7_20260921T0600Z/requirements-audit.json)：
  本人任务范围、Lite 归属、AQ05 设置、各租期资源和 1+2+4 分配记录。
- [`source-provenance-audit.json`](../runs/aq05_lite7_20260921T0600Z/source-provenance-audit.json)：
  372 个冻结文件及已记录修订的 SHA256 链条。
- [`final-resource-audit.json`](../runs/aq05_lite7_20260921T0600Z/final-resource-audit.json)：
  最终 Slurm 释放状态与原 unkillable allocation 保留证明。

以下按发生顺序保留启动、故障、修复与续跑记录；其中“当前”状态指对应记录时刻。

## 资源与第一批作业

| Competition | 分区 | Slurm job | 节点 | 每项 CPU / 内存 |
|---|---|---|---|---|
| aptos2019-blindness-detection | unkillable | 10857987，复用已有 allocation | cn-g018 | 6 / 32 GiB |
| ranzcr-clip-catheter-line-classification | main | 10874910 | cn-g027 | 4 / 24 GiB |
| aerial-cactus-identification | main | 10874910 | cn-g027 | 4 / 24 GiB |
| new-york-city-taxi-fare-prediction | short-unkillable | 10874911 | cn-g023 | 4 / 64 GiB |
| jigsaw-toxic-comment-classification-challenge | short-unkillable | 10874911 | cn-g023 | 4 / 64 GiB |
| mlsp-2013-birds | short-unkillable | 10874911 | cn-g023 | 4 / 64 GiB |
| text-normalization-challenge-english-language | short-unkillable | 10874911 | cn-g023 | 4 / 64 GiB |

实际 GPU 均为 `NVIDIA A100-SXM4-80GB`。main 总申请为 2 GPU / 8 CPU / 48 GiB，
short-unkillable 总申请为 4 GPU / 16 CPU / 256 GiB，与 `gpu_.md` 及现场 QOS
限制一致。每项通过独立 Slurm step 分配一张 GPU；不同 step 中可见设备编号均为
0 是 cgroup 内编号，实际隔离按 GPU UUID 验证。

## 运行目录和监控

所有本轮状态与输出位于：

```text
/network/scratch/l/liux/EvolveFold/MLEvolve_BO/runs/aq05_lite7_20260921T0600Z/
```

- `manifest.json`：负责人、任务、正式设置、源码 commit 和资源分配。
- `code/`、`source.patch`、`source-status.txt`、`source-sha256.json`：冻结的初始运行源码；
  调度脚本的后续修正另记在 `operations-revision-*.json`，恢复修复的完整前后版本、
  patch 和 SHA256 保存在 `source-revisions/`。
- `queue.json`：各分区当前 job ID、后续续跑作业及 Slurm 终态证明。
- `summary.json`、`tasks/<competition>/status.json`：控制器记录的任务与租期状态；
  其中累计有效时间主要在交接和结束时更新。
- `latest-verified-progress.json`：从实际 Slurm 状态、周期 heartbeat 和 atomic ledger
  读取的最新进度，包含当前累计有效时间、完成／有效候选数和检查点更新时间。
- `tasks/<competition>/attempt-*/`：各次租期的命令、GPU/CPU、search log、退出记录。
- `startup-verification.json`、`gpu-isolation.json`：7 项实际配置和独占 GPU 验证。
- `runs/<timestamp>_<competition>/logs/`：完整 `config.yaml`、journal、checkpoint、
  搜索结束后的 `final_submission.json` 和 `official-grade.json`。
- `verify_handoff.py --job <旧 job ID>`：用归档核对交接连续性，并重新查询 Slurm
  释放状态，结果写入该交接目录的 `continuity-verification.json`。
- `audit_completion.py`：仅在 7 项均进入终态后生成 `completion-audit.json`，核对预算、
  正式配置、提交文件 hash 与官方评分有效性。两脚本使用 `.venv-mlebench/bin/python`。

控制器在登录节点的 tmux session `aq05_lite7_20260921` 中运行，只负责 Slurm
调度及状态查询；训练、搜索、fusion 和评分全部在 allocation 内执行。

```bash
cd /network/scratch/l/liux/EvolveFold/MLEvolve_BO
squeue -u "$USER" -o '%.18i %.30j %.18P %.2t %.12M %.12l %R'
cat runs/aq05_lite7_20260921T0600Z/summary.json
tail -n 30 runs/aq05_lite7_20260921T0600Z/controller.log
```

不要重复执行 `init` 或另交同一任务。若控制器停止，先核对其记录的 PID、tmux、
`squeue`、`sacct` 和具体 step，再用本轮 `controller.sh` 恢复调度；原作业仍在运行时
继续使用原 job/step。未知的 `sbatch` 提交结果会保留 `submit-intent-*.json` 并暂停重交，
避免重复申请。

## 短分区续跑与预算

short-unkillable 每次申请 3 小时，在结束前约 10 分钟保存并退出。控制器只有确认
上一 allocation 已离开队列且 `sacct` 给出终态后，才签发释放记录并提交下一次申请。
每个任务始终复用同一个 run 目录；排队、租期交接和环境重载不重新获得 12 小时预算。

续跑保存已完成 observations、pending decision identity、selector RNG、父节点计数和
已完成候选执行输出。正常租期交接中，被中断的候选不作为失败观测加入 GP；已生成的相同代码会重执行，
累计执行时间仍受 32400 秒上限约束。管理器无法通用恢复候选自己训练中的模型／优化器
状态，因此短分区存在实际的中断开销，不能声称等同于连续 12 小时训练。该开销不被
抹掉，也不通过重新开始任务或缩短正式预算来隐藏。

异常退出有不同情况：本轮保留了 5 个受框架关闭影响的原生无效结果，详见下文
“对此前退出结果归因的修正”。这些结果可能影响后续选择，不能套用上述正常交接说明。

当有效预算达到 43200 秒或达到正式 500-step 上限，才执行原有 legacy fusion，固定
最终 submission 后调用官方评分。private score 不提供给运行中的搜索。未到预算或
step 上限的意外退出会记录失败，不自动记为完整正式结果。

## 已完成的启动检查

- 真实 Muse 1.3 API 请求成功，返回 `OPENCODE_MUSE_OK`。
- 7 项 prepared/public 与 private 数据存在，沿用官方校验通过的数据；官方 benchmark
  commit 为 `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`。
- 已在分配的 A100 上验证 CUDA 运算、依赖和 BGE memory embedding。
- 7 份实际 `logs/config.yaml` 的正式参数均已逐项校验。
- 7 份实际运行配置均通过恢复命令的 `_load_cfg` / `prep_cfg` 加载检查，记录在
  `resume-config-preflight.json`；检查没有启动恢复搜索或重启现有进程。
- checkpoint、competition rating context、续跑时间记账与 Slurm 释放判断共 21 项测试通过。
  其中新增的突然中断测试复现了较新 decision ledger 的父节点计数被旧 journal 覆盖的问题；
  修复后恢复时以实际接纳的 ledger 为准。该修复只在后续进程恢复时生效，不中断当前搜索。

启动期间 main 的较大内存申请被 QOS 拒绝，未产生 job ID，随后改为规则中的 48 GiB。
最初 APTOS bootstrap step `10857987.18` 因控制器退出而终止，尚未产生 run-info、
journal 或任何候选；日志保留在 `attempt-01-10857987/`。修复后使用独立会话启动新的
step，正式 APTOS run 记录在 `attempt-02-10857987/`，不会把 bootstrap 当成正式结果。

## 首次短队列交接

2026-09-21 04:50:22–04:50:27（Toronto），四项短队列搜索均以 exit 75 保存，
reason 为 `allocation_boundary`。旧 allocation `10874911` 于 04:50:50 结束；
控制器确认 Slurm 终态后才提交 `10875653`，新 allocation 于 04:50:58 在
`cn-g023` 启动，四项搜索于 04:52 左右恢复。

| Competition | 暂停时有效秒数 | 已完成 observations |
|---|---:|---:|
| new-york-city-taxi-fare-prediction | 10115.7445 | 12 |
| jigsaw-toxic-comment-classification-challenge | 10133.5120 | 2 |
| mlsp-2013-birds | 10097.1283 | 82 |
| text-normalization-challenge-english-language | 10132.0989 | 21 |

暂停／恢复快照保存在 `handoff-audit/10874911/`，核对结果在
`first-handoff-verification.json`。四项 observations、pending decision identity、
Python／NumPy RNG、父节点计数、已完成执行缓存均保持一致；恢复命令使用向上取整的
已用秒数，未重置 43200 秒总预算。11 个已生成但中断的候选均用原代码 hash 进入第二次
执行，保留此前累计执行耗时；另一个生成中的 decision 保留原身份继续生成。
恢复后的正式配置再次通过校验，四个搜索进程分别使用四个不同 GPU UUID。

Slurm 将 English 的旧 step `10874911.3` 记为 `OUT_OF_MEMORY`，因此旧 batch
终态为 `FAILED`。该状态如实保留；step 结束时的 OOM 日志不足以确定具体事件发生时间。
四项搜索实际均已成功保存退出，English 的 21 条 observations 和 3 个 pending decisions
在恢复核对中完整保留，没有据此重置任务、改写候选结果或调整正式参数。

首次交接核对没有覆盖 `data_preview`。第二段发现旧实现没有保存该字段，部分后续提示
重新扫描了包含训练产物的 workspace；Taxi 还出现扫描临时 `runfile_1.py` 时的
`FileNotFoundError`。这项恢复偏差与额外耗时如实保留，不声称本轮等同于连续运行。

## 第二段恢复故障与提前交接

2026-09-21 05:55 左右（Toronto），Birds 的周期检查点线程因并发添加节点属性触发
`RuntimeError: dictionary changed size during iteration` 而退出，搜索本身继续运行，
atomic decision ledger 仍有更新。修复在深复制前固定属性列表，并让周期保存异常后重试；
同时保存原始 `data_preview`，旧检查点从最初 draft 的已记录提示恢复原文。
这两项故障均已复现，21 项相关测试通过，四项真实检查点的原始预览迁移也通过。
完整源码前后版本和 SHA256 见 `source-revisions/02-snapshot-and-preview-recovery/`。

为载入修复，06:15:50 向三项正常保存线程发出暂停请求，三项均 exit 75。
Birds 于 06:16:02 收到 SIGINT，06:16:06 完成同步 journal 保存；其工作线程尚未退出，
随后仅终止所属 step `10875653.2`。旧 allocation 于 06:17:13 完全结束后才准许恢复。

SIGINT 停止 Birds 训练时，两个 worker 随后将被终止的执行写成失败观测。同步保存的
journal 在这两条提交之前，含 123 条原有 observations。恢复使用该检查点，并保留随后
完成生成的同一 pending decision 的原代码、身份和已记账父节点计数；仅将两个明确由
本次 SIGINT 造成的执行恢复为 interrupted，原执行耗时 497.6212 秒和 317.0345 秒保留。
两条终止观测及原文件均归档，没有删除正常候选失败结果。审计、释放证明和恢复脚本见
`handoff-audit/10875653/stop-recovery.json` 及同目录文件。以后保存线程异常时应在确认
最新 atomic ledger 后直接终止所属 step，避免 SIGINT 的 legacy 结果解析副作用。

| Competition | 交接前累计有效秒数 | 恢复 observations |
|---|---:|---:|
| new-york-city-taxi-fare-prediction | 15132.0921 | 15 |
| jigsaw-toxic-comment-classification-challenge | 15146.8348 | 2 |
| mlsp-2013-birds | 15195.5992 | 123 |
| text-normalization-challenge-english-language | 15148.9992 | 23 |

Birds 时间保守计至 allocation 实际结束；其他三项使用成功保存的精确时钟，均不重置预算。
这次提前交接增加了训练中断和重执行成本。Slurm 另将 Taxi 的旧 step `10875653.0`
记为 `OUT_OF_MEMORY`，搜索保存退出仍为 75；两种记录均保留，不能仅凭 step 结束日志
推断 OOM 的具体发生时间。第三段 `10875950` 于 06:21:59 提交，继续沿用相同正式设置。
main 和已有 unkillable 上的三项未在这次交接中重启。

第三段 allocation 于 06:22:06 启动，四项于 06:23 左右恢复。
`second-handoff-verification.json` 已核对四项原有 observations、pending 身份及代码、
Python／NumPy RNG、父节点计数、累计时间和已完成执行缓存。12 个中断候选保留代码
hash 与此前执行耗时；四份原始数据预览 hash 均匹配初始 draft。实际正式配置再次通过，
四个搜索 PID 使用四个不同 GPU UUID，周期检查点已恢复更新。

Birds 本次恢复的三个候选随后均以原 decision ID、node ID 和代码 hash 完成，且得到
有效结果；原中断耗时继续计入候选累计执行时间。实际后续记录见
`handoff-audit/10875653/post-recovery-outcomes.json`。

## 第三段 Birds 的 aggregation 修复记录故障

2026-09-21 08:48:39（Toronto），Birds 在选取下一个父节点时触发
`Study7Error: Legal study7 parent has no committed scientific outcome`。
一个失败的 aggregation 经 debug 成功修复，但科学状态重建只覆盖直接成功的
aggregation，漏掉了这条修复链。搜索于 08:48:51 左右退出，所属 step
`10875950.2` 的 Slurm 终态为 `FAILED / 1:0`，结束时间 08:49:00。
195 条已完成 observations 和原 journal 均完整保存，pending 为空。

修复让科学节点重建折叠 aggregation 的 debug 链，保留最初 aggregation 的身份、
显式来源和第一次成功修复的结果。普通 GP 训练目标沿用原有接纳规则。新增测试在
旧代码上复现同一错误；修复后 134 项相关测试通过。对 Birds 真实保存状态的离线
核验确认只补回遗漏节点，旧科学节点、冻结参考、分支映射、GP 训练目标和 selector
其他状态（包括 RNG）均保持一致。代码前后版本、patch 和 SHA256 保存在
`source-revisions/03-aggregation-debug-scientific-state/`；修复由后续搜索进程载入。

09:04:04 已重新查询 step 终态，在原节点确认搜索 PID 与所属进程组均不存在，
并取得空闲 lease lock 后，将 Birds 标记为等待下一次 allocation 的 paused 状态。
累计有效时间保守计至已确认的搜索进程退出，为 `23946.3421` 秒，比最后同步 journal
额外保留 `12.7465` 秒。未改写原 journal、观测 ledger 或候选执行结果。
控制器仍等待整个 `10875950` 真正释放后，才签发 allocation 释放证明并提交下一段；
其余六项继续运行。修复与等待期间不增加搜索预算，也不重新获得 12 小时。

原始文件归档、真实状态前后核验、进程释放证明及排队操作记录见
`recovery-audit/20260921T124839Z-birds-selector/`。这次实现故障和额外停顿计入本轮
方法说明；恢复后仍需核对实际配置、状态连续性、剩余预算及最终结果。

## 第三次短队列交接与 Birds 恢复

2026-09-21 09:12 左右（Toronto），Taxi、Jigsaw、English 均按租期边界保存，
三个搜索 step 的退出码为 75。旧 allocation `10875950` 于 09:12:26 结束；
其 `FAILED / 1:0` 包含前述 Birds selector 错误。此次三个正常暂停 step 均没有
Slurm `OUT_OF_MEMORY` 终态。控制器于 09:12:32 确认释放，09:12:35 提交第四段
`10878325`，新 allocation 于 09:13:46 在 `cn-g023` 启动，四项于 09:15 左右恢复。

| Competition | 恢复命令累计秒数（向上取整） | 保留 observations |
|---|---:|---:|
| new-york-city-taxi-fare-prediction | 25280 | 17 |
| jigsaw-toxic-comment-classification-challenge | 25291 | 4 |
| mlsp-2013-birds | 23947 | 195 |
| text-normalization-challenge-english-language | 25297 | 24 |

`handoff-audit/10875950/continuity-verification.json` 的四项严格核对均通过，覆盖
原有观测、pending 身份及代码、Python／NumPy RNG、计数、父节点计数、原始数据预览、
累计预算、已完成执行缓存和真实 allocation 释放证明。7 个中断执行记录的代码 hash
和既有耗时均保留。Birds 在故障后没有 pending 候选，直接从保留的完整历史继续。

`selector-state-verification.json` 另核对了 AQ05 自身的 RNG、已接纳抽样计数、契约、
科学状态、模型参数和分支调度状态。唯一的序列化差异是三项 GP 的 `training_hash`
缓存在加载后暂为 null；原生 `State4GP.load_state_dict` 将旧值放入
`_restore_training_hash`，供重新构建数值分解时使用。这项缓存差异明确记录，其他
序列化 selector 字段一致。`resume-resources.json` 从实际进程核实四项分别使用四个
不同 GPU UUID；第四段仍为 4 × A100 80GB、16 CPU、256 GiB、3 小时及 nvlink。

09:20 的实际运行核验确认 Birds 的修复节点已进入科学状态，原先 195 条观测仍是完整
前缀，并已完成 3 条新观测。记录见
`recovery-audit/20260921T124839Z-birds-selector/post-resume-verification.json`。
本段下一次计划保存边界为 12:03:46，排队和恢复启动时间继续排除在有效预算之外。

最终预算、submission 和官方分数需要在全部运行结束后复核。

## APTOS 同类选择器故障与原节点恢复

2026-09-21 10:00:42（Toronto），APTOS 长驻搜索进程触发与 Birds 相同的
aggregation 经 debug 修复后缺少科学节点的问题。该进程早于源码修订 03 启动，
尚未载入修复。原搜索退出码为 1，所属 step `10857987.20` 于 10:01:26 结束；
Slurm 同时记录 `OUT_OF_MEMORY / 0:125` 和 29 次 oom_kill 事件。保留原生结果及
Slurm 记录，不能据最终 step 日志推断各次 OOM 的发生时刻。后续复核发现两个结果
与框架退出中断相符，不能归因为真实候选代码失败，详见文末归因修正。

现场归档包含 81 个文件。最后周期 journal 有 68 条观测，而较新的原子 ledger
已有 69 条、pending 为空，69 个候选执行缓存均已完成。恢复直接由原生逻辑接纳
较新的 ledger，保留最后一条原生结果；未改写历史观测或训练输出。已在原节点核实
旧 PID 和所属进程组消失、旧 step 终止、lease lock 可获取。原用户 allocation
仍在运行，因此不签发整个 allocation 的释放证明。

对真实保存状态的离线核验通过：现有修订 03 只补回遗漏科学节点，原观测、分支、
冻结参考、GP 训练目标及 selector RNG 保持一致。本次没有新增搜索算法修改。
累计时间保守补计至已确认的搜索退出，得到 `28752.7256` 秒，比最后 heartbeat
多保留 `2.8112` 秒，恢复命令向上取整为 `28753` 秒。

控制器于 10:08:24 在同一 allocation 提交新 step `10857987.30`，10:09:27 捕获
第一份恢复检查点。连续性核验确认 69 条原观测、70 个完整节点及关系、Python／NumPy
RNG、计数、69 个已完成执行缓存、原始数据预览和正式配置均保留。AQ05 selector
仍仅有前述 GP `training_hash` 加载缓存差异。实际运行中已重新接纳新决策，遗漏
科学节点与离线核验一致；10:12 的进程核验确认新 PID `3567879` 使用原 GPU UUID
`GPU-4db69579-3309-8f45-1a1e-b1387d0105c1`、6 CPU。其余六项在此期间继续运行。

完整原始现场、恢复准入、首份恢复快照和验证结果保存在
`recovery-audit/20260921T140119Z-aptos-selector/`，其中
`continuity-verification.json` 与 `resume-resources.json` 记录上述证据。
10:17:51，恢复后的首个新候选完成并通过原生自检；10:19 再次核对全部 69 条旧观测
仍为完整前缀。新候选身份、代码 hash 和完成时间见 `post-recovery-outcomes.json`。
此处的有效性为候选自检结果，最终官方评分仍在搜索冻结后进行。
这次实现故障、停顿和 Slurm OOM 记录继续纳入最终方法说明；修复与启动时间不增加
有效搜索预算。七项重新全部运行，最终提交及官方评分仍待正式搜索结束后核验。

## Cactus 同类选择器故障与 main 原配额恢复

2026-09-21 11:21:36（Toronto），Cactus 的旧长驻进程触发同一 aggregation 经
debug 修复后缺失科学节点的问题。该进程同样早于源码修订 03 启动。异常处理终止
当时正在执行的训练子进程，再等待工作线程退出；搜索于 11:25:22 退出，退出码 1。
原 step `10874910.1` 于 11:25:29 结束，
Slurm 为 `FAILED / 1:0`。此 step 没有 `OUT_OF_MEMORY` 终态。

现场保存了 67 个原始文件。定期 journal 含 52 条观测，最新原子 ledger 含 53 条，
pending 为空，53 个候选执行缓存均完成；最后的 TypeError 结果完整保留，另一个
较早失败与框架退出中断相符，详见文末归因修正。既有修订 03
在该真实状态上的 9 项离线核验通过，只补回修复后的聚合节点，旧科学节点、分支、
冻结参考、普通 GP 训练目标和其他 selector 状态保持一致，没有新增算法修改。

11:31:42 的恢复准入重新核对旧 step 终态、原节点 PID 和进程组均不存在、lease lock
空闲、所有归档 hash 未变。累计时间补计至实际搜索退出为 `33815.5398` 秒，比最后
heartbeat 多保留 `12.2809` 秒；恢复命令向上取整为 `33816` 秒。没有修改 journal、
ledger、观测、候选代码或训练输出，也没有为仍在运行的整个 main allocation 签发
释放证明。

11:31:54 仅在原 allocation 启动 Cactus 新 step `10874910.12`，使用同一张 A100、
4 CPU、24 GiB。11:32:39 捕获首份恢复检查点；连续性核验全部 23 项通过，包含
53 条观测、54 个完整节点和关系、Python／NumPy RNG、计数、执行缓存、原始数据预览、
正式配置及累计预算。AQ05 selector 仍只有已说明的 GP `training_hash` 加载缓存差异。
实际新搜索 PID 为 `3984840`，GPU UUID 为
`GPU-55f5b871-9efa-6f0a-1feb-8bbed6f3dea7`。同配额 RANZCR 原 PID `3680014`
继续运行，使用另一张 GPU；其余五项没有重启。

原 batch 的 group 进程只等待最初的两个 srun，因此新增独立监督脚本管理 Cactus
替代 step 的生命周期。它会在原 RANZCR 搜索剩余 300 秒、接近步数上限或离开
running 状态时，为 Cactus 请求原生保存，避免原 batch 随 RANZCR 退出而直接终止
替代 step。若仍需续跑，由原控制器在实际 allocation 释放后按保存状态继续，不重置
预算。这一额外交接及其成本需要在最终方法说明中保留。

证据保存在 `recovery-audit/20260921T152136Z-cactus-selector/`，包括现场归档、
离线核验、准入记录、启动命令、首份恢复快照、`continuity-verification.json` 和
`resume-resources.json`。11:33 的核验中七项均已继续运行，新的决策已接纳；最终
submission 及官方评分仍须等待搜索冻结。此次实现故障及停顿计入最终方法说明。

## 第四次短队列交接

2026-09-21 12:03:46（Toronto）到达第四段保存边界。四项均写出原生暂停检查点，
搜索退出码均为 75；Taxi、Jigsaw、Birds、English 的暂停归档分别在 12:03:55、
12:03:59、12:04:28、12:04:01 完成。旧 allocation `10878325` 于 12:05:10
以 `COMPLETED / 0:0` 结束。四个搜索 step 的 Slurm 状态均为 `FAILED / 75:0`，
对应约定的暂停返回码；此次没有 step 的 `OUT_OF_MEMORY` 终态。

控制器于 12:05:22 确认整个 allocation 释放，12:05:29 提交第五段 `10880077`。
新 allocation 于 12:06:37 在 `cn-g023` 启动；四项首份恢复快照在 12:07:24–53
捕获。恢复命令继续使用原 run 目录和累计时间：

| Competition | 累计秒数（向上取整） | 保留 observations | 保留 pending |
|---|---:|---:|---:|
| new-york-city-taxi-fare-prediction | 35421 | 19 | 3 |
| jigsaw-toxic-comment-classification-challenge | 35432 | 5 | 3 |
| mlsp-2013-birds | 34088 | 263 | 3 |
| text-normalization-challenge-english-language | 35437 | 25 | 3 |

12:08 的严格核验全部通过：旧观测、pending 决策和代码、Python／NumPy RNG、
计数及父节点计数、原始数据预览、正式配置和累计预算一致；已完成执行缓存未变。
11 个中断执行的代码 hash 与已有耗时保留，另保留 Birds 的一个待完成生成。
AQ05 自身的随机状态、契约、分支调度、科学状态和模型参数核验也全部通过，只有
前述 GP `training_hash` 加载缓存差异。中断候选继续受累计 32400 秒执行上限约束。

实际节点进程核验确认四项使用四张不同的 A100 80GB，仍为每项 4 CPU、64 GiB，
第五段合计 16 CPU、256 GiB、3 小时及 nvlink。四项新搜索 PID 分别为 Taxi `21791`、
Jigsaw `21790`、Birds `21794`、English `21798`。下一个 allocation 保存边界为
14:56:37，硬结束时间 15:06:37；正式搜索仍按各自累计预算或步数上限先到者冻结。

完整归档和证据位于 `handoff-audit/10878325/`，包含
`slurm-accounting.json`、`continuity-verification.json`、
`selector-state-verification.json` 与 `resume-resources.json`。排队与恢复启动时间
不计入有效搜索预算；候选训练内部状态的恢复限制继续适用。七项均继续运行，最终
submission 与官方评分仍待搜索冻结后核验。

## RANZCR 同类故障、main 交接和续跑

2026-09-21 12:44:17（Toronto），RANZCR 最后一个尚未载入修订 03 的长驻进程
触发相同选择器错误。已完成的 aggregation debug 候选 `d253e88d…` 保留有效自检
结果；异常处理调用 `terminate_all_subprocesses()`，另两个正在训练的候选于错误后
2.76 秒及 5.27 秒结束，被原生解析器写成空诊断 RuntimeError。它们属于框架中断
影响的结果，不能解释为已经证实的候选代码错误。最后 journal 已有其中一个结果，
较新 ledger 包含全部 24 条观测及两个中断结果；没有可完整还原到中断前的检查点。
因此保留原生历史和成本，单独标记其影响，不构造较早搜索状态或重置预算。

RANZCR 原搜索于 12:45:21 退出，原 allocation `10874910` 于 12:45:32 结束。
Cactus 的独立监督脚本在此之前成功请求原生暂停；62 条观测、3 个 pending、累计
`38184.0660` 秒得到保存。控制器核实 allocation 释放后，于 12:45:36 提交
`10880542`，12:45:37 在 `cn-g027` 启动，仍为 main 2 A100 / 8 CPU / 48 GiB。
Cactus 在新 step `.0` 续跑，累计时间向上取整为 `38185` 秒，PID `4029834`。
12:46:27 的首份恢复归档经严格核验：观测、pending 决策和代码、RNG、计数、预览、
配置与执行成本全部保留；AQ05 selector 仍只有已说明的 GP 加载缓存差异。
证据位于 `handoff-audit/10874910/`。

RANZCR 原现场 57 个文件及追加日志均已归档。修订 03 在真实保存状态上的 9 项
离线核验通过；累计时间补计至实际退出为 `38642.1942` 秒，保留最后 heartbeat 后
的 `15.4796` 秒，续跑使用 `38643` 秒。12:58:11 仅在新 main allocation 的空闲
GPU 上提交 step `.1`，仍为 1 A100 / 4 CPU / 24 GiB。12:59:24 捕获新 PID
`4035857` 的首份恢复检查点：23 项连续性检查全部通过，24 条观测、25 个完整节点、
执行缓存、随机状态、配置和原始数据预览均保留，已接纳新的 AQ05 决策。
本次使用已有修订，没有新增搜索算法变更。

新的独立监督脚本会在 Cactus 原租期接近搜索上限或结束时，为 RANZCR 请求原生
保存，避免共享 batch 退出造成直接终止。原控制器继续管理实际释放后的续跑。
完整证据位于 `recovery-audit/20260921T164417Z-ranzcr-selector/`，包括
`shutdown-outcome-deviation.json`、`queued-recovery.json`、
`continuity-verification.json` 及启动记录。13:00 时七项均继续运行；最终提交和
官方评分仍须等待各自搜索冻结。

## 对此前退出结果归因的修正

13:01 对已归档执行缓存、异常时间和退出代码路径进行复核。APTOS 有两个候选
`0c50c8f1…`、`1a431220…` 在选择器报错后约 2.4 秒终止；Cactus 的
`fb27329f…` 在约 1.77 秒后终止。三者均为空诊断 RuntimeError，与框架异常处理
主动终止子进程相符。APTOS 同时有 Slurm OOM 记录，现存材料不能精确区分每个
子进程的信号来源。此前将这些结果统称为“真实候选失败”的表述不准确，现已修正。
Cactus 最后一条 `0b6e12f7…` 则有明确的 `BCEWithLogitsLoss(label_smoothing=…)`
TypeError 堆栈，应与上述中断结果分别解释。

APTOS、Cactus 已在这些原生结果之后继续搜索，复核保留全部历史和后续决策，
没有回滚或追加预算。连同 RANZCR 两项，本轮保留了 5 个受到框架退出影响的
原生无效结果；其对后续选择的影响也是实验偏差，不能把这些运行视作无中断对照。
正式配置不变不等于实验没有运行偏差。复核证据见
`infrastructure-outcome-review.json` 及各恢复目录的
`shutdown-outcome-retrospective.json`；最终结果说明必须保留这一限制。

13:14 的逐项要求复核已保存至 `requirements-audit.json`。检查直接读取任务分工、
七项实际配置、各租期命令及 Slurm allocation/step 账目：本人七项范围、Lite 归属、
AQ05 配置、原 unkillable 复用、1+2+4 分配、每项单 GPU 及各段无重叠均通过。
`audit_requirements.py` 可在最终阶段重查；当前 `completion_verified=false`，尚需
搜索预算结束、冻结提交和官方评分证据。`audit_completion.py` 的最终登记还包含
负责人、代码基线 commit、完整配置路径及 hash、实际 GPU、各阶段时间和退出码。

## 首两项完成与 RANZCR 第二次 main 交接

14:05:11（Toronto），Cactus 累计有效时间达到 42911 秒，独立监督脚本按约定为
RANZCR 请求保存。RANZCR 原生搜索退出 75，step `10880542.1` 于 14:05:18
结束；25 条观测、3 个中断候选及其代码和成本保留，累计 `42591.108584` 秒，
剩余约 609 秒。检查点及 9 项暂停核验见 `handoff-audit/10880542/`。

14:10，Cactus 与 APTOS 均达到预算并完成官方评分；核验结果如下：

| 任务 | 有效搜索秒数 | 完成 / 自检有效候选 | 官方分数 | 官方有效提交 |
|---|---:|---:|---:|---|
| aerial-cactus-identification | 43201.240729 | 68 / 37 | 1.00000 | 是 |
| aptos2019-blindness-detection | 43202.210144 | 116 / 59 | 0.91155 | 是 |

两项均以原生 `budget_complete` 保存结束，搜索退出码为 75。最终提交使用预定的
`latest_structurally_valid_ensemble_else_best_v1` 规则，官方评分发生在搜索冻结后。
提交路径、官方报告和完成记录的 SHA256 均已核对；逐项证据位于
`completion-checks/<competition>.json`。Cactus 的官方报告标记达到金牌阈值，
APTOS 高于中位数、未达到奖牌阈值。先前实现故障和中断偏差仍须与分数一起说明。

Cactus 的 step `10880542.0` 在结束时另外报告 1 次 `oom_kill`，Slurm 终态为
`OUT_OF_MEMORY / 0:125`，使整个 allocation 为 `FAILED / 1:0`。不能从这一汇总
日志确定 OOM 发生时刻或对应候选；其原生预算完成、最终有效评分和文件哈希已独立
核验。APTOS 的最终 step `10857987.30` 为 `COMPLETED / 0:0`；用户原有
unkillable allocation 继续保留，控制器不再向其派发本轮任务。

旧 main allocation 于 14:10:14 实际释放。控制器于 14:10:29 核实后提交
`10881363`，14:10:46 在 `cn-g018` 启动，仍为 main 2 A100 / 8 CPU / 48 GiB。
RANZCR 使用新 PID `3725367`，14:11:29 捕获首份恢复快照，累计预算从 `42592`
秒续算。严格连续性和 AQ05 selector 核验全部通过，包括 25 条观测、3 个 pending、
RNG、计数、原始数据预览、候选代码及累计执行成本。GPU 实测为
`GPU-e6324f00-b5ee-9530-5de1-f4c60916c5df`。证据见同目录的
`continuity-verification.json`、`selector-state-verification.json`、
`resume-resources.json` 与 `slurm-accounting.json`。其余五项仍待最终完成和验收。

14:17，Taxi、Jigsaw 和 English 也达到完整预算并完成官方评分：

| 任务 | 有效搜索秒数 | 完成 / 自检有效候选 | 官方分数 | 报告比较结果 |
|---|---:|---:|---:|---|
| new-york-city-taxi-fare-prediction | 43200.836224 | 19 / 14 | 8.98329（越低越好） | 未达到中位数水平 |
| jigsaw-toxic-comment-classification-challenge | 43200.231231 | 5 / 3 | 0.98616 | 高于中位数，未达到奖牌阈值 |
| text-normalization-challenge-english-language | 43201.262765 | 26 / 17 | 0.99047 | 达到铜牌阈值 |

三项提交均被官方报告判为有效，逐项预算、配置、冻结后的评分顺序、提交路径与
SHA256 核验全部通过，结果保存在 `completion-checks/<competition>.json`。
最终三个 Slurm step 均为 `COMPLETED / 0:0`。它们的历史短租期重执行成本、预览
偏差和修复说明继续保留；最终候选仍按原先固定规则选择。当前剩余 RANZCR 与 Birds。

14:21，RANZCR 在累计 `43201.191414` 秒后结束，保留 25 条结果、10 条自检有效
结果。官方有效分数为 `0.92956`，未达到中位数或奖牌阈值；冻结提交、评分顺序和
文件 SHA256 的逐项核验通过。最终 step `10881363.0` 和 main allocation
`10881363` 均为 `COMPLETED / 0:0`，allocation 于 14:21:49 释放。两个已标记的
框架中断结果及其后续影响继续保留在方法说明中。当前仅 Birds 尚在搜索。

14:33 的源码追溯核验逐一检查了初始清单中的 372 个文件。6 个文件的最终内容与
已记录修订一致，其余文件与初始冻结版本一致；修订前后 SHA256 链条全部匹配，
没有缺失或额外源码文件。证据为 `source-provenance-audit.json`，可用
`audit_source_provenance.py` 重查。基线 commit、初始未提交改动和后续修订共同
定义本轮实际代码，不能仅凭基线 commit 声称全过程使用同一版本。
