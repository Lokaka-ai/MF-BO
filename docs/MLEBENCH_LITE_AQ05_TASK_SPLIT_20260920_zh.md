# MLE-bench Lite：AQ05 12 小时任务分工

## 1. 本轮口径

本轮在 MLE-bench Lite 的 competition 任务上运行 AQ05。这里的 AQ05 是
`config/experiments/acq7_aq05_gp4_level15_rating_kg.yaml` 对应的 selector preset；运行时不加载
`task_preset=autoresearch`。

统一设置如下：

| 项目 | 固定设置 |
|---|---|
| benchmark | MLE-bench Lite |
| 方法 | AQ05 |
| experiment preset | `acq7_aq05_gp4_level15_rating_kg` |
| code / feedback 模型 | `muse-spark-1.3-contributor` |
| 模型 provider | `codex://opencode-go` |
| GPU | 每个任务独占 1 张 A100；Mila 请求为 `gpu:a100l:1` |
| agent 有效预算 | 每个任务 12 小时，即 `43200` 秒 |
| Slurm wall time | 建议 13 小时，为 server 启动、fusion 和结果落盘预留时间 |
| step 上限 | 500 |
| seed | 42 |
| 初始 drafts | 3 |
| search workers | 3 |
| evaluation | `evaluation.enabled=false`，搜索结束后再做 official grading |

AQ05 的关键 selector 设置是：GP4、level Matérn-3/2 branch covariance、branch prior SD 1.5、
branch lengthscale 1.0、slope prior SD 0、agent rating 开启、ancestor feature 关闭，并用
`finite_pool_kg` acquisition。不得把它写成其他 Acq7 方法、普通 GP7、原始 MCGS 或 autoresearch task。

## 2. 本轮剔除的 8 个任务

| 展示名称 | competition ID | 处理 |
|---|---|---|
| Denoising Dirty Documents | `denoising-dirty-documents` | 剔除 |
| Detecting Insults | `detecting-insults-in-social-commentary` | 剔除 |
| Dogs vs Cats | `dogs-vs-cats-redux-kernels-edition` | 剔除 |
| Histopathologic Cancer Detection | `histopathologic-cancer-detection` | 剔除 |
| Plant Pathology 2020 | `plant-pathology-2020-fgvc7` | 剔除 |
| Spooky Author Identification | `spooky-author-identification` | 剔除 |
| Tabular Playground Dec 2021 | `tabular-playground-series-dec-2021` | 剔除 |
| Whale Challenge | `the-icml-2013-whale-challenge-right-whale-redux` | 剔除 |

官方 Lite split 共 22 项；剔除以上 8 项后剩 14 项，双方各运行 7 项。

## 3. 任务分工

### 我负责的 7 项

| # | competition ID | 类型 | 状态 |
|---:|---|---|---|
| 1 | `aerial-cactus-identification` | Image Classification | AQ05 已完成：43201.24 秒，官方有效分数 1.00000；保留 selector 中断及 OOM 说明 |
| 2 | `aptos2019-blindness-detection` | Image Classification | AQ05 已完成：43202.21 秒，官方有效分数 0.91155；保留 selector 中断及 OOM 说明 |
| 3 | `jigsaw-toxic-comment-classification-challenge` | Text Classification | AQ05 已完成：43200.23 秒，官方有效分数 0.98616 |
| 4 | `mlsp-2013-birds` | Audio Classification | AQ05 已完成：43206.75 秒，官方有效分数 0.88146（达到铜牌阈值）；保留恢复偏差说明 |
| 5 | `new-york-city-taxi-fare-prediction` | Tabular | AQ05 已完成：43200.84 秒，官方有效分数 8.98329（越低越好） |
| 6 | `ranzcr-clip-catheter-line-classification` | Image Classification | AQ05 已完成：43201.19 秒，官方有效分数 0.92956；保留两个框架中断结果偏差 |
| 7 | `text-normalization-challenge-english-language` | Seq-to-Seq | AQ05 已完成：43201.26 秒，官方有效分数 0.99047（达到铜牌阈值） |

本轮启动记录、资源分配和续跑说明见
[`MLEBENCH_LITE_AQ05_OWNER_RUN_20260921_zh.md`](MLEBENCH_LITE_AQ05_OWNER_RUN_20260921_zh.md)。
七项均已达到完整预算并通过官方提交有效性核验。完整登记见
[`final-results.csv`](../runs/aq05_lite7_20260921T0600Z/final-results.csv)。
本轮包含短租期重执行、实现修复及 OOM 等运行偏差，并保留了 5 个受框架关闭影响的
原生无效结果，不能作为无中断对照运行。

### 合作者负责的 7 项

| # | competition ID | 类型 | 状态 |
|---:|---|---|---|
| 1 | `dog-breed-identification` | Image Classification | 待运行 |
| 2 | `leaf-classification` | Image Classification | 待运行 |
| 3 | `nomad2018-predict-transparent-conductors` | Tabular | 待运行 |
| 4 | `random-acts-of-pizza` | Text Classification | 待运行 |
| 5 | `siim-isic-melanoma-classification` | Image Classification | 待运行 |
| 6 | `tabular-playground-series-may-2022` | Tabular | 待运行 |
| 7 | `text-normalization-challenge-russian-language` | Seq-to-Seq | 待运行 |

两边都是 7 项且各有 3 个图像任务。`siim-isic-melanoma-classification` 的数据约 116 GB，
明显大于其他任务；另一组集中放置 APTOS、RANZCR 和 NYC Taxi，作为数据准备与运行负载上的补偿。

## 4. 结果登记要求

每项任务至少登记以下信息：

- competition ID、负责人和 Slurm job ID；
- 代码 commit、完整 `logs/config.yaml` 和实际 GPU 名称；
- 开始/结束时间、exit status，以及是否完整用满 `43200` 秒 agent budget；
- `logs/final_submission.json` 指向的最终 submission；
- official grading report；
- 若失败，保留原始日志并注明失败阶段，不把缩短预算或更换 setting 的重跑混入本轮结果。

合作者的环境准备、正式提交、检查、评分和资源 release 命令见
[`MLEBENCH_LITE_AQ05_COLLABORATOR_RUNBOOK_20260920_zh.md`](MLEBENCH_LITE_AQ05_COLLABORATOR_RUNBOOK_20260920_zh.md)。
