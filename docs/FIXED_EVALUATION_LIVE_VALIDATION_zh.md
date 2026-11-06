# MLEvolve BO 准备协议真实集成测试（2026-09-14）

22/22 个 mlebench-lite 任务完成：Muse 编写/DEBUG 准备程序并实际执行、一次有效基准、一次有效后续实验、宿主完整评分、GP7 反馈入库与拟合、完整 test 提交文件生成及校验。每个任务的两次有效实验使用同一固定协议和全部 V 评分单元。

模型为 `muse-spark-1.3-contributor`，通过生产 Codex transport 连接 OpenCode Go；使用用户授权的 opencode 凭据，报告不保存凭据。准备、候选编写、DEBUG、review 与结果摘要均使用真实模型调用。并行节点为 trixie-cn103、cn113、cn123、cn125、cn128。

配置：seed=42，`gp7_ancestry_logei`，每次候选执行上限 300 秒，准备总 deadline 900 秒，scorer 上限 300 秒；搜索在获得基准和一次后续成功实验后停止。失败由 DEBUG 修复后重跑。训练可在 T 内取小样本，V/test 必须完整推断。表内记录固定外层 V 的宿主分数；本次范围是短程集成验证。

| 任务 | 指标 | V 评分单元 | 基准 | 后续实验 | GP 有效反馈 |
| --- | --- | ---: | ---: | ---: | ---: |
| aerial-cactus-identification | roc_auc ↑ | 2835 | 0.999446042 | 0.998586909 | 1 |
| aptos2019-blindness-detection | quadratic_kappa ↑ | 659 | 0.566709031 | 0.32755832 | 1 |
| denoising-dirty-documents | rmse ↓ | 4604040 | 0.0568115339 | 0.0504055565 | 1 |
| detecting-insults-in-social-commentary | roc_auc ↑ | 790 | 0.909614122 | 0.901223317 | 1 |
| dog-breed-identification | log_loss ↓ | 1840 | 6.55181549 | 6.06896219 | 1 |
| dogs-vs-cats-redux-kernels-edition | log_loss ↓ | 4500 | 0.610183006 | 0.590159537 | 1 |
| histopathologic-cancer-detection | roc_auc ↑ | 34893 | 0.882453013 | 0.88024808 | 1 |
| jigsaw-toxic-comment-classification-challenge | roc_auc ↑ | 23936 | 0.978111761 | 0.974743376 | 1 |
| leaf-classification | log_loss ↓ | 198 | 0.144344782 | 0.0536205624 | 1 |
| mlsp-2013-birds | roc_auc ↑ | 1254 | 0.698790192 | 0.605188588 | 1 |
| new-york-city-taxi-fare-prediction | rmse ↓ | 10000 | 4.95479726 | 5.2110727 | 1 |
| nomad2018-predict-transparent-conductors | rmsle ↓ | 432 | 0.063376628 | 0.0590372468 | 1 |
| plant-pathology-2020-fgvc7 | roc_auc ↑ | 328 | 0.722468356 | 0.722567577 | 1 |
| random-acts-of-pizza | roc_auc ↑ | 576 | 0.658747719 | 0.658085563 | 1 |
| ranzcr-clip-catheter-line-classification | roc_auc ↑ | 5587 | 0.634836371 | 0.716045392 | 1 |
| siim-isic-melanoma-classification | roc_auc ↑ | 5991 | 0.809187413 | 0.857350098 | 1 |
| spooky-author-identification | log_loss ↓ | 3525 | 0.408700187 | 0.373396298 | 1 |
| tabular-playground-series-dec-2021 | accuracy ↑ | 720000 | 0.954545833 | 0.954673611 | 1 |
| tabular-playground-series-may-2022 | roc_auc ↑ | 160000 | 0.898514337 | 0.969009271 | 1 |
| text-normalization-challenge-english-language | accuracy ↑ | 40019 | 0.979509733 | 0.979959519 | 1 |
| text-normalization-challenge-russian-language | accuracy ↑ | 21163 | 0.919104097 | 0.942919246 | 1 |
| the-icml-2013-whale-challenge-right-whale-redux | roc_auc ↑ | 14294 | 0.810582507 | 0.810730288 | 1 |

每个任务提供 1 条普通后续实验的有效 GP 观测；初始 draft 用于建立基准。GP7 输入维度为 7，实际执行后验拟合，少量观测下超参数保持 `fixed_prior`。后续实验分数可以下降；测试要求的是正确执行、评分和反馈入库。

本次保留的实现修改：

- 准备生成后实际运行；失败把完整代码、traceback 和程序输出交给 DEBUG，修复后继续执行与 admission。默认尝试次数 0 表示仅受总 deadline 限制，修复不重置时间预算。
- 独立准备脚本先创建工作目录，Path 类型与生产 launcher 保持一致。
- 明确 split 枚举、公开资产目录由宿主创建、`@prepared/path` 对应 `output_dir/path`；错误信息直接显示解析路径和预期目标列。
- 提交行由 ID 与原样元数据共同识别，支持 Insults 重复 Comment、不同 Date 的合法完整模板。
- 归一化多类 AUC 的准备自检使用总和为 1 的概率。
- 原生媒体准备需真实验证读取方式，AIFF 使用已验证的解码器；候选不能吞掉解码错误并替换为零特征。Whale 最早两次全零特征实验已作废，表内只保留修复后的真实音频实验。
- 明确二分类温度 helper 输入为一维；多标签逐目标拟合或展开 T 数据拟合共享温度。
- 默认准备时限由 600 秒调至实测使用的 900 秒，覆盖 GPFS 上 NYC Taxi（约 732 秒）及 Histopathology（约 686 秒）的准备。

RANZCR 使用 MLE-bench 官方 [CLASSES](https://github.com/openai/mle-bench/blob/main/mlebench/competitions/ranzcr-clip-catheter-line-classification/classes.py) 的全部 9 个目标，与本地模板一致；原始 Kaggle 描述有 11 个标签，全部原始监督标签仍从 V/test 特征排除。

相关验证：准备失败→DEBUG→实际重跑、语法/运行错误、尝试次数上限及总 deadline 测试已通过；本次新增重复 ID + 元数据的 2 项测试、多类 AUC 自检的 1 项测试通过。没有运行与改动无关的全仓回归。

默认 `evaluation.enabled=false`；正式使用此协议需设置 `evaluation.enabled=true`。

清理状态：五个节点上的 `/dev/shm/mlevolve-protocol-live-20260914` 和 `/tmp/mlevolve-protocol-live-code-20260914` 已删除，相关临时进程及 tmux 会话已清理；本报告保留汇总结果。
