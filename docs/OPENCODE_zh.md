# 原生 OpenCode 后端

两个项目的新运行默认使用 `opencode/muse-spark-1.3-contributor-free`，经官方
`opencode run` 调用。配置中的 `opencode://opencode` 是本地后端选择标记，不是 HTTP 地址。
`codex://opencode-go` 及已有 Codex checkpoint 仍可使用原来的后端。

```yaml
agent:
  code:
    model: opencode/muse-spark-1.3-contributor-free
    base_url: opencode://opencode
    api_key: ""
  feedback:
    model: opencode/muse-spark-1.3-contributor-free
    base_url: opencode://opencode
    api_key: ""
```

`MLEvolve_BO_agent` 的默认运行模式为 `agent.runtime.mode=stateful_opencode`；
仍支持 `stateful_codex` 和 `legacy_stateless`。不带 runtime 字段的旧配置仍恢复为
`legacy_stateless`。任务启动器的显式覆盖可用
`--model opencode/muse-spark-1.3-contributor-free --provider opencode`；其他原有任务参数不变。

## Cluster 上的使用

本次验证的 Mila CLI 为 OpenCode 1.18.32，安装位置 `~/.opencode/bin/opencode`，
`~/.local/bin/opencode` 已链接到它。

```bash
opencode --version
# 尚未登录时运行，选择 OpenCode Zen 并输入自己的 key：
opencode auth login

# 在 MLEvolve_BO 中，使用项目现有 Python 环境：
.venv/bin/python tools/check_opencode.py

# 在 MLEvolve_BO_agent 中，可复用同一环境：
../MLEvolve_BO/.venv/bin/python tools/check_opencode.py
```

凭据可使用已有 OpenCode 登录、`OPENCODE_API_KEY` 或 `OPENCODE_API_KEY_FILE`。
显式配置的 `agent.code.api_key` / `agent.feedback.api_key` 优先；空字符串使用已有凭据。
仅在没有这些凭据时兼容本工作区的 `../codex/opecode`。不要把 key 提交到配置文件。
凭据经子进程环境传递，不写进命令参数、usage 报告或会话配置。

| 配置 | 作用 |
| --- | --- |
| `MLEVOLVE_OPENCODE_BIN` | 覆盖官方 CLI 可执行文件路径 |
| `MLEVOLVE_OPENCODE_TIMEOUT_SECONDS` | 单次 CLI 超时；普通调用默认 1800 秒，有状态角色默认 7200 秒 |
| `MLEVOLVE_OPENCODE_REASONING_EFFORT` | 普通文本后端的 thinking 覆盖值 |
| `agent.reasoning_effort.<role>` | agent 版各角色的 thinking，默认均为 `high` |

原有 `MLEVOLVE_CODEX_TIMEOUT_SECONDS`、`MLEVOLVE_CODEX_REASONING_EFFORT` 作为兼容回退，
因此已有启动脚本设置的 600/1800 秒和 medium/high 不会失效；新的 OpenCode 环境变量优先。
`MLEVOLVE_CODEX_BIN` 只影响 Codex 后端，不能拿 Codex 包装脚本充当 OpenCode CLI。

## Thinking 默认设置

| 路径 | 默认值 |
| --- | --- |
| BO 普通 `generate()` | `high` |
| BO 普通 `query()` / feedback | `high` |
| BO agent parent selector | `medium` |
| BO autoresearch launcher | 原脚本指定的 `medium`；campaign 的 `high` 覆盖仍有效 |
| agent 版 Draft / Experiment / Evolution-Fusion / Aggregation / Review / Rating | 全部 `high` |
| agent 版 research Manager | `high` |

这些值传给 OpenCode 的 `--variant`；显式角色配置不被环境变量覆盖。

## 调用、权限和恢复

普通生成和选点调用在临时目录中执行，不执行模型请求的工具。使用 `ask` 保留原生工具
声明，非交互 CLI 会拒绝权限请求，且不会传 `--auto`。不要把全部工具设成 `deny`：
本次实测这会触发 Zen 的 `OpenCode's free tier can only be used from within OpenCode`。

agent 版各角色独立保存 OpenCode 数据库和 `ses_...` 会话 ID，续接使用 `--session`。
沿用原来的会话锁、恢复包、失败次数上限和截止时间；不能用原 Codex thread ID 续接
OpenCode，切换已有运行的 CLI 后端时应创建新运行。

Manager/Rating 只读；其他角色的文件编辑限定在自己的工作目录；只有 Experiment 可以
执行 shell 检查。OpenCode 的工具权限不等同于 Codex 的操作系统 sandbox，Experiment
的 shell 仍依赖任务所在的 Slurm/容器环境进行资源及系统隔离。

`--format json` 输出的是 CLI 事件流。适配器提取最后一条 assistant 消息，遇到错误事件、
截断或空返回会失败；需要 JSON 时附加 schema 指令并用 `jsonschema` 校验实际结果。
超时会终止 CLI 及其工具进程组。普通调用的 token/cost 记录在 `logs/llm_calls.jsonl`；
有状态角色记录在 `agent_sessions/<role>/<context>/turns/<turn>/usage.json`。

## 验证

本次实测与回归结果见 [验证报告](OPENCODE_TEST_REPORT_20260924_zh.md)。

`tools/check_opencode.py` 通过真实免费模型检查文本与结构化 JSON；在 agent 版还检查
文件读取、修改、Python 执行、会话续接和跨轮记忆。结果写入 `.cache/opencode-check.json`。
测试仅操作合成临时文件，不启动 GPU 训练或完整 MLE-bench 实验。

离线回归覆盖 CLI 协议、退出码为 0 的错误事件、JSON schema、超时、凭据处理、thinking、
角色权限、会话恢复和候选准入。详见 `tests/test_opencode*.py`。

官方说明：[CLI](https://opencode.ai/docs/cli/)、[权限](https://opencode.ai/docs/permissions/)、
[配置](https://opencode.ai/docs/config/)、[Zen](https://opencode.ai/docs/zen/)。
