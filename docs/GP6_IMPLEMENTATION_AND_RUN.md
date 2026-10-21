# GP6：Matérn 3/2 BO 实现与两组实验运行说明

## 1. 交付范围与入口

基于本次上传的 `MLEvolve_BO(1).zip` 修改，只修改独立的 `MLEvolve_BO/` 文件夹。不是基于旧的整个 EvolveFold 仓库重新开发。新功能为：一个共享的 Matérn 3/2 混合 GP、六个推进状态特征、普通/draft 提案的 debug 修复归因、统一的逐步决策边界，以及可选的新 draft 候选。没有加入 Q、TD target、BNVS 的截断标签或成本感知 EI；没有增加预训练/LLM 特征模型。

**必须使用下面的 preset 或专用脚本。** 为兼容已有工作流，原 `config/config.yaml` 的默认 selector 仍为 BNVS；仅运行 `python run.py` 不会自动切换到本次 GP。两套 preset 继承同一份基础配置，内容只在一个字段上不同：

| 项目 | `gp6` | `gp6_draft` |
|---|---|---|
| selector | `gp_child_ucb` | 同左 |
| surrogate | `gp_matern32` | 同左 |
| 决策范围 | `all_normal_steps` | 同左 |
| 数值特征数 | 6，操作单独作为类别 | 同左 |
| acquisition | 质量 UCB，beta=1.5，latent std | 同左 |
| `include_root_draft` | `false` | `true` |
| 原基础配置的初始 draft / 总 cap | 3 / 5 | 3 / 5 |
| 原基础配置的 seed / 并发 | 42 / 3 | 42 / 3 |

两个实验共用初始化、预算、pending、反馈、合法性和恢复逻辑。`gp6` 禁止的是“新 draft 与普通候选的自适应竞争”，不是取消初始化 draft；若初始化后仍无有效 incumbent，原有可行性恢复 draft 规则在两组中一致保留，并受总 cap 限制。总 cap 是上限，不强制补满。

## 2. 运行两组 22 任务

将压缩包中的 `MLEvolve_BO/` 放回原工程位置，使用原来能够运行该工程的 Python 环境、数据目录、LLM 接入与机器资源。没有新增 GP 框架依赖；使用原有 NumPy/SciPy。不要用示例占位 API 配置替代你此前实际使用的配置。

包内 `config/mlebench_lite22.txt` 是从提供的运行包中提取的相同 22 个任务，不是另换一个任务集合。套件脚本顺序执行，不自动提交集群作业、不申请 GPU，也没有在交付过程中运行过这些实验。

之前对比的运行预算为每任务 8 小时。基础项目仍保留原来的 12 小时默认，因此要显式指定 28800：

```bash
cd /path/to/EvolveFold/MLEvolve_BO

# 实验 A：Matérn 3/2 混合 GP + 六维推进特征，不将新 draft 纳入竞争
TIME_LIMIT_SECS=28800 bash scripts/run_gp6_suite.sh \
  gp6 /path/to/mlebench-datasets 111

# 实验 B：与 A 完全相同，只增加新 draft 候选
TIME_LIMIT_SECS=28800 bash scripts/run_gp6_suite.sh \
  gp6_draft /path/to/mlebench-datasets 111
```

这两个命令顺序运行时可以使用同一个 server ID。若在同一台机器上并行启动两组，请自行分配不同的 server ID、GPU 与 CPU 集合，避免两组争用资源。保持两组 LLM、温度、评估方式、seed、并发数和时间预算相同；命令没有擅自更换你原来的 LLM。

单任务入口：

```bash
TIME_LIMIT_SECS=28800 bash scripts/run_gp6_task.sh \
  gp6 leaf-classification /path/to/mlebench-datasets 111

TIME_LIMIT_SECS=28800 bash scripts/run_gp6_task.sh \
  gp6_draft leaf-classification /path/to/mlebench-datasets 111
```

需要接入现有批量启动器时，只需在其原有 `python run.py` 参数后增加相应 preset。示例：

```bash
python run.py experiment_preset=gp6 \
  exp_id=leaf-classification dataset_dir=/path/to/mlebench-datasets \
  data_dir=/path/to/mlebench-datasets/leaf-classification/prepared/public \
  desc_file=/path/to/mlebench-datasets/leaf-classification/prepared/public/description.md \
  exp_name=leaf-classification agent.time_limit=28800 agent.steps=500 \
  log_dir=./runs_gp6 workspace_dir=./runs_gp6
```

另一组改为 `experiment_preset=gp6_draft`，并将输出目录改为 `runs_gp6_draft`。其他 CLI override 在 preset 后合并，适合沿用此前的 LLM/资源配置；两组使用相同的 override，不要额外改变 beta、cap 或 seed。密钥仍使用你原有的凭据配置方式，不应写进公开运行记录。

脚本支持 `MEMORY_INDEX`、`START_CPU_ID`、`CPUS_PER_TASK`、`TIME_LIMIT_SECS`、`STEP_LIMIT` 环境变量。原 `run_single_task.sh` 的主进程退出码现在不再被可选的 submission fusion 退出码覆盖；fusion 也会读取本组输出目录。每组套件写 `suite_<timestamp>.tsv`，记录每个任务的退出码。124 表示 shell 时间预算到达，不代表一定成功产出了有效最终结果。

## 3. 六个特征的实际语义

模型真正接收的是 `gp_feature_vector` 的 6 个数值，加一个单独的操作类别。历史日志为了兼容旧工具仍可能含旧的诊断字段，这不意味着 GP 又训练了 49 维。

| 顺序 / 名称 | 定义 |
|---|---|
| `parent_gap` | 当前 incumbent utility 减去父节点 utility；draft 使用冻结的 draft 质量先验作为起点。 |
| `last_gain` | 沿主推进历史最近一次有效的局部增益；debug 链折叠为原提案。 |
| `recent_gain` | 最近 5 个闭合提案中有效局部增益的均值；没有观测时为先验填充值 0。 |
| `stagnation` | 连续闭合但没有超过原改进阈值的提案数，包括最终失败；有效 seed 重置起点。 |
| `trials` | 该方向已闭合提案数，包括 seed，以及从其主祖先尝试的替代提案。 |
| `valid_rate` | 修复后的提案有效率，Laplace 平滑为 `(success+1)/(closed+2)`；没有历史时为 0.5。 |

叶节点继承主父链的推进历史，不再因为没有直接 children 就丢失此前的改善信息。不会把 fusion 参考分支的整个历史当成当前方向自己的历史，也不计入其他独立分支的后代。没有闭合的修复链不计为失败 trials。

draft 本身尚未生成具体方法，因此其 last/recent gain 和 stagnation 为 0，trials/valid_rate 总结此前已闭合 draft；没有增加 LLM 对尚不存在方案的潜力评分。

前三个指标量使用同一冻结尺度做 `tanh` 变换，两个次数量使用 `log1p(c)/(1+log1p(c))`，有效率位于 `[0,1]`。不对小样本每列重新估计极小标准差，不把未来结果放入决策特征。输入有界不等于标签截断。

## 4. GP、均值函数与 UCB

核为：

```text
k((z,o),(z',o'))
  = amplitude² × Matern32(z,z'; length_scale)
    × [rho + (1-rho) × 1(o=o')]

Matern32 = (1 + sqrt(3)r) exp(-sqrt(3)r)
r = ||z-z'|| / length_scale
```

操作为 `improve / evolution / fusion / draft`。同类操作共享更多信息，跨操作保留 rho 比例的信息共享；不是把操作编号当成连续距离。

默认 length_scale=1、amplitude=1、noise_std=0.25、rho=0.25。在少于 8 个有效提案时使用固定先验参数。之后仅拟合三个共享标量：长度尺度、幅度、噪声；rho 固定。单次有界 MAP 优化最多 25 次迭代，有 log-normal 先验约束，无多次随机重启；不使用每特征自由 ARD，不按操作另训练四个 GP。精确 Cholesky 求解只允许一次必要的 jitter 增强尝试。

预测对象仍是连续产出质量：

```text
U_child = m0(action) + scale × GP(state, operation)

normal m0 = parent utility
draft  m0 = 初始可见有效 draft utility 的中位数
```

utility 统一为越大越好。初始 draft 终态被观察后再进行普通 GP 决策。首次有有效数据拟合时冻结 draft_anchor 和尺度（优先使用初始化 draft；尺度为 IQR、绝对指标中位数的 5%、1e-6 三者最大值）；以后不随任务进度反复改变标签尺度。恢复时复用同一尺度与均值基线。

GP 训练连续残差，不截断成 `[-1,1]`，没有 bootstrap target。最后的 acquisition 是：

```text
posterior_mean_quality + 1.5 × posterior_std_latent
```

同时记录 predictive_std，它还包含观测噪声，但本次 UCB 不把这部分噪声直接作为知识不足奖励。没有 EI、成本除法、成功率乘法或 beta 自动调度。beta 可显式设置为 0 做均值消融，但本次两组默认均为 1.5。

模型异常时只在当前冻结的合法候选池内做基线质量回退，并写 fallback 原因，不跨越 draft cap 或 pending 限制。不要将大量 fallback 的运行解释为 GP 正常运行。

## 5. 反馈归因与执行控制

新 GP 使用统一的 `effective_proposals()`：普通提案和 draft 初次失败、随后 debug 成功，使用原提案当时的特征和操作，配上修复后的第一个有效结果；累计整条链的生成和执行成本。原始 transitions 追加日志不改写，重复 terminal 不重复训练，debug 不另外变成一个 GP 自由决策动作。

最终失败没有被随意赋予一个跨任务通用的连续 utility。它影响历史 trials/stagnation/valid_rate，但当前 GP 质量回归仍只训练有效结果，属于“有效结果条件下的质量模型”，不是已经完整求解失败概率的模型。悬而未决、可能继续 debug 的反馈是 censored，而非负分标签。

两组 GP 都使用同一 managed decision 生命周期：当前节点、其他合法节点，以及开启时的唯一虚拟 root 候选一起排序。draft 在提交前预留名额，pending 会阻止同类 root 重复候选；未提交的取消可释放，已提交不得偷偷退回 cap。预算、锁、扩展上限、debug 上限保留。已有操作 preview、LLM prompt、评估协议和代码执行器不改成新研究系统。

GP 模式下，有效节点不再被旧的连续无提升/时钟式强制回传软规则提前终止；是否继续由下一次统一排名决定。这个变化在两组完全相同。初始化短暂等待终态用于形成质量基线，不是在整个任务中引入全局 wave barrier。没有“每个方向必须固定连续 H 步”的新规则。

因此，**实验 A 相对历史 BO 是完整升级包，不是纯 kernel 单因素消融**：还包括反馈、特征语义、不确定性和继续控制修复。实验 B 相对 A 才是仅 draft 开关的配对消融。

## 6. 旧线性 BO 与兼容性

旧 selector 和原有 BNVS 路径保留。旧 BayesianRidge 在拟合后用完整精度矩阵重建 `sigma_`，包括 n<p 时的零空间方差；不依赖升级整个项目到新 sklearn，均值拟合保持原样。这项修正保留旧 BO 作为可用对照，但新 GP 不调用 BayesianRidge。

新增 `gp6_progress_v1` schema。不要把旧 BO/BNVS 运行目录作为新 GP 实验继续跑，也不要用一组 preset 恢复另一组；显式 preset 与保存的组别冲突会拒绝，而非静默覆盖。新实验使用全新输出目录。恢复相同组别可使用原 `resume_run_dir=/absolute/path/to/run` 接口；冻结参考写入现有 ledger，GP 从已接纳反馈确定性重拟合，不额外维护另一套模型矩阵缓存。

## 7. 日志与结果检查

专用脚本输出：

```text
runs_gp6/<timestamp>_<task>/logs/
runs_gp6_draft/<timestamp>_<task>/logs/
```

各运行 `logs/phase1/` 中：

- `selection_decisions.jsonl`：selector、动作、fallback。
- `selection_snapshots.jsonl`：六维向量、操作类别、mean quality、latent std、predictive std、UCB、候选资格与 root 信息。
- `model_fits.jsonl`：训练提案/真实修复结果 id、三个超参数、尺度、anchor、修复条数、闭合失败/未闭合条数、拟合状态。
- `transitions.jsonl`：不可变原始反馈，普通/draft/debug 都保留。
- `continuation_events.jsonl`：继续、切换、新方向。
- `root_draft_state.json`：两组共用的预算/pending/恢复 ledger，文件名不代表 root 开关必然开启。

正常 GP 决策应显示 `selector_name=gp_child_ucb`、schema=`gp6_progress_v1`、六维向量及 `uncertainty_used=latent`；实际有效反馈后应看到 model fit，而不是持续 fallback。

## 8. 验证边界

本地通过 30 项测试：17 项真实数值/特征/反馈/选择器测试，11 项生产 Runtime 类方法的隔离执行测试，2 项 preset 加载与跨组恢复拒绝测试。隔离 Runtime 测试使用生产方法，但 agent、持久化和无关依赖由轻量替身提供；不是完整真实 agent 端到端成功。

Python 编译检查、三个 shell 启动脚本语法检查通过。新增 `tests/test_gp6_runtime.py` 使用原工程 native agent fixture，覆盖调度、生成、debug、恢复等；本环境在收集测试前被缺少 `dataclasses_json` 阻断，未执行这些 native 测试。没有把依赖缺失伪装成通过，也没有调用真实 LLM、GPU 或运行 2×22 benchmark。

在你原来可运行该项目的环境中，相关测试命令为：

```bash
python -m unittest discover -s tests_gp6 -v
python -m pytest -q tests/test_gp6_runtime.py
```

完整状态和实际输出见 `docs/GP6_TEST_REPORT.md` 与 `docs/gp6_test_results/`。代码已实现；最终集成运行和 benchmark 性能尚未在本环境验证，不能预先保证奖牌率提高。
