2026-09-21 AlphaEvolve 数学任务阶段性结果对照

当前 AQ05 运行对比 MLEvolve 论文：0 项明确更好、1 项数值基本持平、14 项落后。对比原版 AlphaEvolve：1 项更好、1 项数值基本持平、13 项落后。共有 7 项与 MLEvolve 的相对差距在 1% 以内。

本次运行目录：`/network/scratch/l/liux/EvolveFold/MLEvolve_BO/runs/aq05_alphaevolve15_cpu_20260921T071425Z`。15 项均处于用户暂停状态，每项已消耗 6.1747–7.34 小时有效预算，计划预算为每项 12 小时。暂停及累计预算依据 [resume-manifest.json](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/runs/aq05_alphaevolve15_cpu_20260921T071425Z/pauses/20260921T145824Z/resume-manifest.json)。

两列文献数值采用 [MLEvolve 论文 Table 2](https://arxiv.org/html/2606.06473#S4) 的 MLEvolve 与原版 AlphaEvolve 列；原版问题定义和构造见 [AlphaEvolve 原文](https://arxiv.org/html/2506.13131) 与 [官方 notebook](https://github.com/google-deepmind/alphaevolve_results/blob/4226acbf237ff9ad10ba7673a2af127a2d8a5971/mathematical_results.ipynb)。本表中的“原版”不包括 AlphaEvolve-v2。

↑ 越大越好；↓ 越小越好。“落后”列正数表示当前较差，负数表示当前较好。最大化任务用 (参考−当前)/|参考|，最小化任务用 (当前−参考)/|参考|。不同任务的原始数值不作平均。绝对差小于等于 1e-9 标为“约持平”，仅用于避免把数值误差当作突破。

| 任务 | 当前最佳有效值 | MLEvolve | 落后 MLEvolve | 原版 AlphaEvolve | 落后原版 |
|---|---:|---:|---:|---:|---:|
| 六边形装填 ↓ | 3.9610906414183 | 3.9284759302 | +0.8302% | 3.930092 | +0.7888% |
| 正方形内圆装填 ↑ | 2.4569481257651 | 2.6359830395 | +6.7920% | 2.6358627564 | +6.7877% |
| 矩形内圆装填 ↑ | 2.3621813886278 | 2.3658323759 | +0.1543% | 2.3658321334 | +0.1543% |
| Heilbronn（凸域） ↑ | 0.025647183385408 | 0.0309372079 | +17.0992% | 0.0309368890 | +17.0984% |
| Heilbronn（三角形） ↑ | 0.036129619715235 | 0.03652988988003020 | +1.0957% | 0.03652988988003016 | +1.0957% |
| 11 维接吻数 ↑ | 350 | 592 | +40.8784% | 593 | +40.9781% |
| 和差集 1 ↑ | 1.1084865780949 | 1.1901774219 | +6.8638% | 1.1479889651 | +3.4410% |
| 和差集 2 ↑ | 1.1203628464153 | 1.1585457700 | +3.2958% | 1.1584172816 | +3.2850% |
| 不确定性不等式（Hermite） ↓ | 0.35209910441599 | 0.3520991044160 | 约持平 | 0.3520991044225 | 约持平 |
| 第一自相关 ↓ | 2 | 1.5028628749 | +33.0793% | 1.5052939684 | +32.8644% |
| 第三自相关变体 ↓ | 1.536309740736 | 1.4587698922 | +5.3154% | 1.4687620697 | +4.5990% |
| 第三自相关 ↓ | 1.4610782330052 | 1.4548507482 | +0.4280% | 1.4556427954 | +0.3734% |
| 第二自相关 ↑ | 0.90223123648534 | 0.9054217971 | +0.3524% | 0.8962799442 | -0.6640% |
| 最大/最小平方距离比 ↓ | 12.890494218675 | 12.8892299077 | +0.0098% | 12.88926611203 | +0.0095% |
| 最小重叠 ↓ | 0.38170857382426 | 0.3808968496 | +0.2131% | 0.3809230351 | +0.2062% |

第二自相关当前为 0.9022312364853353，比原版表列值高约 0.6640%，但仍比 MLEvolve 低约 0.3524%。这说明超过了该原版参考构造，并不代表超过后续方法。

不确定性不等式当前为 0.3520991044159866，与 MLEvolve 的差只有约 1.3e-14，与原版表列值的差约 6.5e-12，应视作数值基本持平。这里只比较三系数 Hermite 问题，不把它解释为追平官方 notebook 中更强的 Laguerre 界。

目前最大差距是 11 维接吻数（350 对 592，落后 40.88%）、第一自相关（2.0 对 1.5028628749，落后 33.08%）及凸域 Heilbronn（落后 17.10%）。优先排查这些任务的搜索进展更有价值；这些差距本身不能证明评分器有错误，也不能确定原因是 CPU。

当前结果逐项来自 canonical transitions.jsonl 中已提交、valid=true、budget_eligible=true、censored=false 且在实际累计预算内完成的最佳候选。15 项均核对了已有 host 评分的有效性、任务名、分数和方向，以及构造 SHA256、候选代码与记录的一致性、节点状态和协议标识。此次只检查已有记录及文件，没有重新执行候选或重新评分。

有两处导出文件不能直接作为本次对照依据：

- 第三自相关的旧 alphaevolve_summary.json 为 1.461492815955847；最新已提交有效候选为 1.4610782330052148，完成于 2026-09-21 14:59:00 UTC。表中采用后者。
- 不确定性不等式缺少 alphaevolve_summary.json；其已提交候选及对应 host result.json、构造和代码核对通过，故采用已提交记录。

参考构造的既有验算记录见 [reference-table-audit.json](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/.cache/alphaevolve/reference-table-audit.json)。和差集 1 的官方构造重算为 1.1479888965092757，论文表列为 1.1479889651，相差约 6.86e-8；表中保留论文值，这一细小差异不改变当前比较结论。

当前运行使用 Muse Spark、高推理配置和 AQ05 GP/KG+rating，每项申请 4 CPU、24 GiB、0 GPU。MLEvolve 论文实验设置使用 Gemini-3.1-Pro-preview，温度 1.0，最多 500 次扩展及 12 小时；论文描述的计算环境为 21 vCPU、234 GB 内存、1 张 H200。该配置描述不等于证明这些数学任务都需要 GPU。见 [MLEvolve 实验设置](https://arxiv.org/html/2606.06473#S4)。

模型、搜索方法、算力、已用时长和初始化条件均未控制一致，因此目前只能比较产出的数学目标值，不能据此把差距归因于某一种算法或硬件。任务定义映射及复现边界见 [SOURCE.md](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/tasks/alphaevolve/SOURCE.md)；本地实现并不声称复原作者未公开的数学任务提示词、初始化及逐次构造搜索预算。

完整精度、逐项候选 ID、记录路径、文件哈希及核对结果保存在 [JSON](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/reports/alphaevolve_result_comparison_20260921.json)；可用于表格分析的数值保存在 [CSV](/network/scratch/l/liux/EvolveFold/MLEvolve_BO/reports/alphaevolve_result_comparison_20260921.csv)。

| 任务 ID | 已用有效小时 | 最佳候选 ID |
|---|---:|---|
| `hexagon_packing` | 7.3375 | `8a90a2e6a90646139489e246d40e225b` |
| `circle_packing_square` | 7.3372 | `5e8156f8629e4e7aa456852e0d54ddee` |
| `circle_packing_rectangle` | 6.8681 | `f8ed28b39ee84fdbb89d39f84346404c` |
| `heilbronn_convex` | 7.3372 | `67a67c37ab5044b69e7408b617116213` |
| `heilbronn_triangle` | 7.1278 | `4ba4f033f66b4879b40e3b5711040aa0` |
| `kissing_number_d11` | 6.5514 | `3ab2c01860864d06a34a10b7e20ac5c0` |
| `sums_differences_1` | 7.3375 | `8b10fe4329f247dcbc59113c0d27e09e` |
| `sums_differences_2` | 6.8469 | `6f47597fcccf4b06a638e4871bce79ae` |
| `uncertainty_inequality` | 7.3375 | `ef9f8d6fffa1465c8d3337d9a933e0e9` |
| `first_autocorrelation` | 7.3375 | `bdbb934f4faf4a299b8702fe0ab68a47` |
| `third_autocorrelation_variant` | 7.3378 | `a93ca12b70e54f009edcbe830f34d933` |
| `third_autocorrelation` | 7.0825 | `21e54ea96b92471b9aa89f8f4b7dcc97` |
| `second_autocorrelation` | 6.1747 | `a3d6fa13d2654dbb80db33e870ec2de6` |
| `max_min_ratio` | 7.3400 | `9048de79bf764bb19d666a1077b24478` |
| `minimum_overlap` | 6.3400 | `357532cca6b14e2cacf9b44916a1254e` |
