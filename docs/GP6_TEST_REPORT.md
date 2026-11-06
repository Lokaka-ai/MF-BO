# GP6 相关测试报告

## 实际通过

```text
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests_gp6 -v
Ran 30 tests
OK
```

17 项核心测试直接运行新生产模块，检验混合核对照/正定性、Cholesky 后验数值、latent 与 predictive 方差、MAP 和冻结参考恢复、低样本 BayesianRidge 完整协方差、普通及 draft debug 归因、前缀可见性、推进历史继承、归一化、UCB/beta=0、合法池 fallback 和两组配置匹配。

另外 11 项测试隔离执行生产 Runtime 类，覆盖 root on/off、pending 排重、预留与取消、draft cap/总预算、未终态不作样本、terminal 幂等、初始 draft 的连续目标、两组构造和六维日志。它们借助 AST 加载未改写的生产 Runtime 方法以避开缺失的重量级 agent 依赖，agent/序列化/无关模块使用替身，不能算 native 全链路测试。

另有 2 项配置测试执行生产 `_load_cfg`，验证两组 preset、显式 override 及不能静默跨组恢复。

实际逐测试输出在 `gp6_test_results/core_tests.log`。

以下命令退出码为 0：

```bash
python -m compileall -q engine config run.py tests_gp6 tests/test_gp6_runtime.py
bash -n run_single_task.sh scripts/run_gp6_task.sh scripts/run_gp6_suite.sh
```

## 尚未通过执行验证的部分

新增 native 集成用例 `tests/test_gp6_runtime.py`：

```text
python -m pytest -q tests/test_gp6_runtime.py
ImportError while loading tests/conftest.py
ModuleNotFoundError: No module named 'dataclasses_json'
```

原 conftest 导入失败后，没有继续执行 native 用例。真实收集错误保存在 `gp6_test_results/native_collection.log`。当前环境也缺少部分其他原项目依赖；不能将安装一个包视为已经验证整个 agent。没有为了这个边界修改生产 import 或绕过正式序列化实现。

没有执行真实 LLM/API、外部训练代码、GPU、grading server 或 44 个 MLEbench-lite 实验；没有 benchmark 提升结果。对并发恢复的端到端结论仍需原可运行环境中的 native 集成和实际运行支持。

## 环境

Python 3.13.5

- numpy: 2.3.5
- scipy: 1.17.0
- scikit-learn: 1.8.0
- omegaconf: 2.3.0

GP 只使用原有 NumPy/SciPy，没有新增 gpytorch/botorch 依赖；requirements 中旧 sklearn pin 保留，局部修正旧 Ridge 的完整协方差，不依赖本机 sklearn 版本的内部修复。
