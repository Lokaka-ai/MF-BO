# 固定外层评估与二分类温度校准

本实现只作用于 `MLEvolve_BO`。启用 `evaluation.enabled=true` 后，搜索启动前由一次性 `EvaluationSetupAgent` 生成任务适配程序；宿主核验并冻结外层训练集 T、外层验证集 V、官方指标和完整预测接口。所有候选通过同一程序评分，主分数不再取自 stdout、候选 JSON 或 LLM 返回值。默认开关为 false，原实验模型配置不变。

准备 agent 生成程序后，宿主立即实际执行 `prepare` 并运行原有协议校验和 smoke score。解析、执行或协议校验失败时，调用 `debug_agent.repair_evaluation_setup`，将上一版完整源码、异常和程序输出交给 DEBUG agent；每次修复后重新执行，直到通过才发布协议并开始搜索。DEBUG 沿用 `agent.code` 的模型配置和结构化接口，不创建 SearchNode 或 GP 观测；成功后继续按原机制复用协议。

## 执行路径与审查入口

1. `config.prep_agent_workspace` 创建空工作目录。固定模式跳过原始数据复制、全量解压和旧 preview 的合并逻辑。
2. `AgentSearch` 在创建虚拟 root、初始化 selector 和提交 draft 前调用 `FixedEvaluation.prepare_or_load`。准备不是 SearchNode，不占 draft 数量，不向 GP 写观测。
3. `agents/evaluation_setup_agent.py:build_prompt` 使用现有 prompt compiler、`FunctionSpec` 和 `llm.query`。准备模型沿用 `agent.code` 配置。完整 prompt 包含 Introduction、任务描述、受限原始文件预览、官方格式约束、科学划分、程序接口、监督边界、响应格式与有界修复说明。
4. 准备回复必须通过结构化接口提交 `supported`、`reasoning`、完整 `prepare(data_dir, output_dir, seed)` Python 程序。可返回原始 Python 或一个 Python code fence。宿主附加 prepare/score 命令入口；不接受另一套候选自报评分协议。
5. 宿主 admission 核验实际成员、原始身份、分组或时间边界、V 真值映射、完整官方模板、目标次序和指标方向，然后用实际预测接口执行一次 smoke score。通过后原子发布协议目录。
6. 即时执行与初始 deferred draft 都先分配该节点的新 attempt 目录，再运行候选。公共 `result_parse_agent.run` 在 improvement、top-K、backprop 和 transition 前核验 V/test 预测并调用宿主 scorer。LLM 可总结运行情况或报告实际代码错误，不能提供备用主分数。
7. draft、debug、improve、evolution、fusion、aggregation 的共享指南、stepwise 子步骤/合并、diff coder 与 code review 使用相同协议说明。已失败 proposal 的第一次有效 debug 仍归并到原动作，成本按原机制累计。
8. 结束时只导出当前协议下、已通过评估的最佳节点。宿主保存评分时的提交快照，最终选择不依赖后来可能被覆盖的节点输出。该结果同步写到 `best_submission/submission.csv` 和既有兼容位置 `ensembles_csv/top1ens-total_run_time0.0h.csv`。未经固定 V 评分的末尾 fusion 不会替换它。

核心代码为 `engine/fixed_evaluation.py`、`agents/evaluation_setup_agent.py`、`utils/evaluation_helpers.py`；生命周期接入位于 `engine/agent_search.py`、`engine/executor.py`、`engine/execution.py`、`agents/result_parse_agent.py`、`run.py`、库入口 `__init__.py` 和 `run_single_task.sh`。

`engine/parent_selectors/runtime.py` 仅增加协议恢复契约和可带超时的状态等待；GP 特征、核、目标和 acquisition 公式没有修改。`run.py` 同时保留并发结束时已完成节点的后续调度，避免一个等待 worker 报 exhausted 时丢掉另一个 worker 刚返回的 continuation。

## 数据与预测协议

协议目录包含以下宿主侧文件：

| 文件 | 内容与边界 |
| --- | --- |
| `evaluate.py` | 生成的准备程序及宿主附加的命令入口 |
| `protocol.json` | 指标、ID/目标/元数据列、划分依据与 seed、任务说明、数据身份、版本、协议 ID、准备成本 |
| `members.csv` | 每个原始样本的唯一 `source_id`、候选可见 `sample_id`、T/V/test 分区，以及需要时的 group/time |
| `truth.csv` | V 的评分 ID、真值、元数据和映射到原生样本的 `__sample_id`；不在公开目录中 |
| `assets.csv`（可选） | 单个原生文件的来源、公开相对路径、sample_id、feature/target 角色；不公开来源映射 |
| `public/train.csv` | 仅 T 的特征与标签 |
| `public/validation.csv` | 仅 V 特征，含 opaque sample_id |
| `public/test.csv` | 所有官方 test 的特征 |
| `public/sample_submission.csv` | 全量官方 test 评分 ID、全部目标列与必要元数据 |

工作目录 `input` 只指向 `public`；候选 preview 仅展示这四张表的有限行和公开协议。T 内可训练、交叉验证、early stopping、OOF、stacking 和拟合校准/集成权重，V 答案只由宿主 scorer 读取。旧候选若看过 V，不得当作可复用训练 checkpoint。

`source_id` 必须标识重命名前的真实样本，例如原始表名加原始行键或原始文件名；给同一原始样本加不同分区前缀不构成合法身份。宿主拒绝重复原始身份、分组跨 T/V，以及时间划分中 `max(T) >= min(V)`。Whale 等任务的准备 prompt 明确要求查看真实录音日期，不能一概随机切分。

原生图像、音频与目标图通过逐文件 asset 清单暴露，禁止整个原始目录或 archive 作为公开 symlink。clean image、mask 等目标资产仅允许 T。V 文件使用 opaque 公开路径和硬链接/复制，不使用能通过 readlink 暴露 `cat.*`、`dog.*`、Whale `_0/_1` 等标签的 symlink。宿主还核验同一原始文件身份不能跨分区。

准备程序需为每种原生媒体格式实际解码一个文件，并在公开说明中写出已验证的读取方法。AIFF 不能用 `scipy.io.wavfile` 读取；本次 Python 3.11 环境可用 `aifc` 读取未压缩 AIFF。候选必须暴露读取/解码异常，不能吞掉错误并用全零特征代替。解包资产的 `@prepared/path` 指向 `output_dir/path`；公开 asset 目录由宿主创建。

候选必须产出：

```python
validation_path = Path(os.environ['MLEV_EVALUATION_OUTPUT_DIR']) / 'validation.csv'
test_path = Path('submission/submission.csv')
```

两份文件的列集合都必须恰为 `id_columns + prediction_columns + metadata_columns`。宿主按 ID 与元数据组成的完整行键对齐，所以行重排合法；缺失、重复、额外的行键或空 ID、目标列错误、数值 NaN/Inf、概率越界、多类概率不归一化均拒绝。Insults 的 Comment 可重复，需结合 Date 才能唯一识别行；`Date` 必须原样保留（允许空串），不进入指标。字符串读取保留空串、字面 `NA` 和 Unicode，文本任务按原样精确匹配。

`official_sample_submission` 必须指向原始公开官方 CSV，或原始 ZIP 中的 `archive_member`。宿主独立读取它，比对全部 test ID、目标列及顺序、元数据，防止适配程序与缩减后的模板自洽但遗漏任务目标。

原生输出通过 CSV 保留任务全部评分单元：Denoising 按全部像素 ID 展开；Birds 按样本/物种 ID 展开；Text Normalization 保留所有 token 并按 sentence 分组；多目标指标保留所有目标及官方次序。宿主可核验声明的完整成员和模板；适配程序是否忠实表达原始像素/物种/分组语义，仍需结合任务说明和准备代码审查，不能仅凭 JSON schema 推断。

## 22 项任务的指标约束

`config/fixed_evaluation_metrics.json` 记录 `config/mlebench_lite22.txt` 的全部 22 项任务的指标、方向、ID 和元数据格式。数据适配由准备 agent 根据实际文件完成，host 不会以另一种简单任务作为降级路径。

| 任务 | 固定主指标 |
| --- | --- |
| aerial-cactus-identification | 单目标 ROC AUC，最大化 |
| aptos2019-blindness-detection | Quadratic weighted kappa，最大化；预测须为合法离散标签 |
| denoising-dirty-documents | 全像素 RMSE，最小化 |
| detecting-insults-in-social-commentary | 单目标 ROC AUC；Comment ID 与 Date 元数据原样保留 |
| dog-breed-identification | 全类别 log loss，最小化 |
| dogs-vs-cats-redux-kernels-edition | 二分类 log loss，最小化 |
| histopathologic-cancer-detection | 单目标 ROC AUC，最大化 |
| jigsaw-toxic-comment-classification-challenge | 全目标 macro ROC AUC，最大化 |
| leaf-classification | 全类别 log loss，最小化 |
| mlsp-2013-birds | 官方 Id/Probability 长表所有样本/物种的整体 AUC，最大化 |
| new-york-city-taxi-fare-prediction | RMSE，最小化 |
| nomad2018-predict-transparent-conductors | 两目标各自 RMSLE 的平均，最小化；使用原始非负尺度 |
| plant-pathology-2020-fgvc7 | 全目标 macro ROC AUC，最大化 |
| random-acts-of-pizza | 单目标 ROC AUC，最大化 |
| ranzcr-clip-catheter-line-classification | MLE-bench 模板全部 9 个目标的 macro ROC AUC，最大化 |
| siim-isic-melanoma-classification | 单目标 ROC AUC，最大化 |
| spooky-author-identification | 全类别 log loss，最小化 |
| tabular-playground-series-dec-2021 | 类别 accuracy，最大化 |
| tabular-playground-series-may-2022 | 单目标 ROC AUC，最大化 |
| text-normalization-challenge-english-language | 全 token 精确匹配 accuracy，最大化 |
| text-normalization-challenge-russian-language | 全 token 精确匹配 accuracy，最大化 |
| the-icml-2013-whale-challenge-right-whale-redux | 单目标 ROC AUC，最大化 |

RANZCR 原始 Kaggle 描述列出 11 个标签，MLE-bench 的 [CLASSES](https://github.com/openai/mle-bench/blob/main/mlebench/competitions/ranzcr-clip-catheter-line-classification/classes.py) 与本地准备模板使用 9 个。评分保留模板全部 9 个目标；公开 V/test 特征仍需排除全部 11 个原始监督标签及标注。

定义核对来源为 OpenAI [MLE-bench competitions 的官方实现](https://github.com/openai/mle-bench/tree/main/mlebench/competitions)，核对日期 2026-09-14。需要特别区分的实现包括 [Birds](https://github.com/openai/mle-bench/blob/main/mlebench/competitions/mlsp-2013-birds/grade.py)、[NOMAD](https://github.com/openai/mle-bench/blob/main/mlebench/competitions/nomad2018-predict-transparent-conductors/grade.py)、[Denoising](https://github.com/openai/mle-bench/blob/main/mlebench/competitions/denoising-dirty-documents/grade.py) 和 [Text Normalization](https://github.com/openai/mle-bench/blob/main/mlebench/competitions/text-normalization-challenge-english-language/grade.py)。多类概率列按协议标签次序显式重排后交给 sklearn，不假定原表列已按标签字典序排序。

## 温度校准与无需重训的排错

候选运行时可直接从宿主提供的 `evaluation_helpers` 导入：

```python
from evaluation_helpers import binary_probabilities, fit_binary_temperature

# calibration_logits/calibration_labels 只来自 T 内部的校准划分。
temperature = fit_binary_temperature(
    calibration_logits, calibration_labels,
    temperatures=[1.0, 2.0, 3.0], clip_eps=1e-6,
)
validation_probabilities = binary_probabilities(validation_logits, temperature, 1e-6)
test_probabilities = binary_probabilities(test_logits, temperature, 1e-6)
```

`fit_binary_temperature` 仅返回网格上最小 binary log loss 对应的一个 Python float 温度，避免把 `(loss, temperature)` 对调。输入为一维有限 binary log-odds；双列 logits 须先转成正负类 logit 差。网格须非空且每项有限正数。稳定 sigmoid 处理 ±1000，`clip_eps` 须至少为 float64 epsilon 且小于 0.5，避免极小 epsilon 使上界舍入为 1。V/test 必须共享模型、推理、集成顺序、温度及 clipping。

按需把 `calibration_cache.npz` 放在该次 `MLEV_EVALUATION_OUTPUT_DIR`，记录 `protocol_id`、`node_id`、T 内校准 logits/labels、V/test logits 与对应 IDs。debug prompt 提供父节点的实际目录；校验协议与模型来源一致后，可读取父缓存重新求温度并输出到新 attempt，而不重训。测试包含一次真实训练失败于缺预测、随后完全依靠已存 logits 修复的路径。缓存不包含 V 答案，不要求所有任务保存所有张量，也不自动迁移历史已泄漏 V 的模型。

## 准备预算、并发、失败与恢复

默认配置：

```yaml
evaluation:
  enabled: false
  protocol_dir: null
  setup_timeout_seconds: 900
  setup_max_attempts: 0  # 0：修复直到成功或总准备时间耗尽
  score_timeout_seconds: 120
```

准备总 deadline 为 `min(setup_timeout_seconds, agent.time_limit)`，覆盖 Python 启动、原始数据 inventory、准备/DEBUG 的 API 等待与重试、生成程序的执行和 admission；修复不会重置 deadline。父进程以独立 process group 启动准备，到期终止整组。默认 `setup_max_attempts=0` 不设次数上限，在现有总预算内修复到通过；显式正整数仍限制总尝试次数（包含首次准备）。修复 prompt 包含上一份完整程序、实际异常和程序输出。失败保留 `logs/evaluation_setup/` 中的 prompt、response、prepare log、错误及总失败摘要，不启动候选、不暴露原始输入、不回退旧评分。

同一协议目录通过文件锁只准备一次。成功记录 `setup_cost_seconds`；每个新方法组从任务预算扣除同一准备成本，实际载入更慢时扣除实际耗时。日志独立记录 `load_wall_seconds`。恢复从累计活动耗时继续，只额外计本次载入；不会再次扣一次完整原始准备成本，也不会在首个 step 重置搜索起点。

各节点及重试使用宿主分配的 UUID attempt 目录和 `MLEV_EVALUATION_PROTOCOL_ID`、`MLEV_NODE_ID` 环境变量。旧的同节点 test 文件在开始时移除；评分仅接受宿主刚登记的 attempt。三个并发候选不会共写一份 V 预测。

两类失败有不同含义：

- 候选执行错误、缺少预测或违反预测格式：进入已有 buggy/debug 流程，没有有效 GP 质量标签。真正有效的 AUC 0/1 和很差的 log loss（包括大于 3）保留其真实数值。
- 公共 scorer 导入、读取真值、程序崩溃或超时：抛出 `EvaluationError`，保留预测与日志，停止该搜索后续评分和入库，唤醒等待 worker。不会把所有候选记为训练失败。修复不能把已混入错误分数的搜索直接接续；故障标记阻止自动恢复。

正常恢复需要相同数据路径/身份、task、seed、协议目录/ID。数据身份检查使用路径、大小、mtime；协议内容在 admission/载入时核验，未在每个候选重复扫描全数据。journal、最新 decision ledger、恢复节点及有效 transition 的协议必须一致。旧无协议 checkpoint、固定/旧模式切换、缺失协议后重建或新搜索混用旧日志都会拒绝。

`run.py` 尽早写出本次解析后的精确 log/workspace 路径至 `run_result_file`。shell 依据它导出本次最佳节点；文件缺失或无效（例如 import/config 启动失败）时跳过所有后处理，保留主程序退出状态。固定模式的零步 cleanup 保留协议与排错材料。直接调用旧 submission fusion 工具读取到固定模式时，也转入已评估最佳节点导出。

## 启动与复用

以下命令在 `MLEvolve_BO` 目录执行，数据需保持现有 `<DATASET_ROOT>/<task>/prepared/public` 布局，使用现有服务、依赖和已配置的模型访问方式。示例预算和 seed 仅为显式参数，不改变仓库默认值。已完成的 22-task 真实短程验证见 [集成测试报告](FIXED_EVALUATION_LIVE_VALIDATION_zh.md)。

单任务：

```bash
export TIME_LIMIT_SECS=28800
export CPUS_PER_TASK=21
bash scripts/run_gp6_task.sh gp6 dogs-vs-cats-redux-kernels-edition /absolute/mlebench-data 111 \
  evaluation.enabled=true \
  'evaluation.protocol_dir=/absolute/fixed-protocols/{task_id}/seed-{seed}' \
  agent.seed=42 agent.initial_drafts=3
```

完整 22-task、比较不同 BO 方法时共享对应协议：

```bash
export TIME_LIMIT_SECS=28800
export CPUS_PER_TASK=21
DATASET_ROOT=/absolute/mlebench-data

bash scripts/run_gp6_suite.sh gp6 "$DATASET_ROOT" 111 \
  evaluation.enabled=true \
  'evaluation.protocol_dir=/absolute/fixed-protocols/{task_id}/seed-{seed}' \
  agent.seed=42 agent.initial_drafts=3

bash scripts/run_gp6_suite.sh gp6_draft "$DATASET_ROOT" 111 \
  evaluation.enabled=true \
  'evaluation.protocol_dir=/absolute/fixed-protocols/{task_id}/seed-{seed}' \
  agent.seed=42 agent.initial_drafts=3

# 同样适用于已有 selector suite；比较组使用同一数据、预算、seed 与协议模板。
bash scripts/run_selector_suite.sh gp6_global_logei "$DATASET_ROOT" 111 \
  evaluation.enabled=true \
  'evaluation.protocol_dir=/absolute/fixed-protocols/{task_id}/seed-{seed}' \
  agent.seed=42 agent.initial_drafts=3
```

suite 仍按现有列表逐任务调用。`{task_id}` 和 `{seed}` 由宿主在准备前展开，例如 `.../dogs-vs-cats-redux-kernels-edition/seed-42`；单引号让模板作为同一个 CLI 参数传递。不同任务不会共用单一协议目录，同一 task/seed 的不同方法共用一份协议。更换数据版本应使用新的协议根目录；若省略 `protocol_dir`，默认落在各自 run 的日志下，不构成跨方法共享。

正常恢复示例：

```bash
python run.py \
  resume_run_dir=/absolute/run-directory \
  evaluation.enabled=true \
  agent.steps=500
```

保存的 config 自动提供原协议目录等参数；不要改 task、seed 或协议位置。`agent.steps` 是总步数上限。未完成的候选执行按现有恢复机制重新执行并得到新 attempt；准备不重新生成。

库入口支持 `Experiment(data_dir, goal, eval, evaluation={"enabled": True, "protocol_dir": "/absolute/protocol"})`。自行传入执行 callback 时，固定模式会调用原有三个参数加 `evaluation_context=...`；标准 `Interpreter.run` 已接入该环境与超时协议。

## 验证范围

测试文件为 `tests/test_fixed_evaluation.py`。准备 agent 使用本地 HTTP server 返回结构化模型回复，经过真实 `llm.query`、HTTP、prompt 编译和 response parser；候选模型回复受控，但通过真实代码 parser、操作分发、子进程 Interpreter、CPU sklearn 训练、helper、宿主 scorer 和 GP 入库。它证明生产调用链已贯通，不是外部大模型生成质量或真实 22 数据集成绩的证明。

测试覆盖完整 runner 成功/公共故障退出、库入口、deferred/即时执行、debug 缓存修复、improve/evolution/fusion、stepwise/diff/review 指令、3 路并发、恢复不重复调用 API、严格格式/完整目标、独立指标数学、官方 ZIP 模板与元数据、真实 API 等待 deadline、固定最佳提交和 shell 启动失败。API deadline case 断言本地 server 确实已收到请求。

运行命令（测试环境需安装仓库依赖与 pytest）：

```bash
PYTHONPATH=.:tests python -m pytest -q tests/test_fixed_evaluation.py --tb=short

PYTHONPATH=.:tests python -m pytest -q \
  tests/test_gp6_runtime.py tests/test_root_draft_runtime.py \
  tests/test_resume_state.py tests/test_resume_config.py tests/test_resume_memory.py \
  tests/test_submission_fusion_utils.py tests/test_graph_runtime.py tests_gp6 --tb=short
```

本地验收结果另见 `FIXED_EVALUATION_VALIDATION_zh.md`。没有执行远端 SSH/GPU、真实付费 LLM 请求或完整 22-task 实验，没有改历史结果或提交 Git commit。

同 UID 的目录组织属于合作式边界，不是 OS 强隔离；恶意代码仍可能绕过相对路径。模型是否实际以合法数据训练、V/test 是否使用相同模型，也仍依赖生成指令和 code review。大规模像素展开、原生 archive 解码耗时及新数据格式是否能在准备预算内完成，需要真实数据运行验证；准备失败将明确停止，不能以代理指标降级。
