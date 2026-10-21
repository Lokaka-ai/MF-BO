# 本人 MCGS 七项状态与日志同步

- Space：https://huggingface.co/spaces/jzshared/mlebench-mcgs-12h-20260922
- 公开 Bucket：https://huggingface.co/buckets/jzshared/autoresearch-runs
- experiment ID：`mlebench-mcgs-12h-20260922`
- contributor：`liux`；method：`MCGS`；seed：42。
- 稳定 run ID：`liux.mcgs-20260922T131459Z.<competition>`，Slurm 续租只更新 attempt ID。
- 训练来源：`runs/mcgs_lite7_20260922T131459Z`。

复用 Space 已有的汇总服务，未另起 coordinator。七项新记录与原有 `andy` 七项并存，当前共 14 项。页面计数已支持自动发现的新数据集，运行详情提供该运行的 Logs 下载链接。现有曲线规则保持原样。

## 上传内容和状态口径

每项写入 `experiments/mlebench-mcgs-12h-20260922/runs/<run_id>/live.json`。累计预算来自训练 journal、heartbeat 和 supervisor 状态中的实际累计时钟；同步不增加搜索耗时。运行中的来源时间取当前搜索进程的心跳，上传时间单列，旧心跳不会因为再次上传变成新心跳。

验证指标只取 journal 中已完成、非 buggy、有限数值的候选，最大化/最小化方向取候选自身记录。尚无候选时保留空曲线。最终官方分数单独从 `official-grade.json` 读取。

`artifacts/` 已上传搜索 stdout/stderr、MLEvolve 普通与详细日志、LLM 调用和选点日志、journal、配置、状态、资源及设置核验信息。后续新租期日志按 attempt 独立保留，融合及评分日志出现后也会上传。未上传输入数据集、训练模型二进制或凭据。日志中的常见凭据格式及提供的 HF token 会被脱敏。日志为上传时的文件快照；活跃日志后续继续更新，不宣称它们已经包含未来输出。

## 后台服务

独立目录：`/network/scratch/l/liux/EvolveFold/.staging/hf-mcgs-liux-20260922/`。

- 首轮一次性验证作业：`10890861`，成功上传七项及对应日志。
- 后台 CPU 服务首作业：`10890903`，分区 `long-cpu`，1 CPU / 2 GiB / 无 GPU，已于 Toronto 2026-09-22 09:36:51 在 `cn-f004` 启动，已连续多轮七项全部上传成功。
- 原 48 小时申请 `10890874` 因排队已取消，只取消了本次新建的同步作业。
- 目标上传周期 60 秒；Space 再经现有 coordinator 刷新及托管缓存传播，不能保证秒级到达。
- CPU 租期最多 1 小时，允许 Slurm 按至少 10 分钟的空闲窗口调短。服务按实际 allocation 结束时间提前保存并提交 `afterany` 后继作业。后继排队期间可能出现同步间隔。
- 统一服务有效期至 Toronto **2026-09-24 09:35:26**（UTC 13:35:26），以 `service-policy.json` 的 `expires_epoch` 为准。到期停止续接；七项全部完成并成功上传最终日志时提前退出。此期限仅影响上传，不影响训练。
- `service-status.json` 记录最后一轮上传是否七项均成功；`service-jobs.jsonl` 记录后继作业；`uploader-<job>.log` 为同步日志。
- 读取根目录 `hf_token`，token 只进入进程环境，不写入命令参数或公开产物。

仅在确认旧同步作业已停止、没有等待中的后继作业后恢复：

```bash
sbatch /network/scratch/l/liux/EvolveFold/.staging/hf-mcgs-liux-20260922/submit.sh
```

保留同一 `cache/`、run ID 和服务目录。若已超过服务有效期，恢复前需明确延长 `service-policy.json` 的 `expires_epoch`。本地 writer 锁也会阻止同一服务目录同时上传。

## 核验

七项真实源记录均通过快照 schema 检查。首次发布成功后，已匿名读取 Bucket 中的 `live.json` 和完整搜索日志；匿名访问 Space 的 HTML、JavaScript、Plotly 和数据均成功，汇总数据包含 `liux` 的七项且无无效记录。前端改动通过 JavaScript 编译检查；未进行浏览器视觉验收。

另已验证 Space 于 UTC 13:38:42 发布的后续数据：七项上传时间和来源心跳均较首轮前进，日志路径正确。证据保存在服务目录的 `first-space-verification.json`、`later-space-verification.json`、`frontend-deployment.json` 及同步日志中。CPU 服务的完成退出、续接和到期逻辑已通过隔离模拟检查。整个共享过程不修改或重启 MCGS 与 AlphaEvolve 训练。
