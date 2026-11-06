# OpenCode 适配验证

验证环境：Mila cluster，Python 3.12.12，官方 OpenCode CLI 1.18.32。
日期：2026-09-23 Toronto / 2026-09-24 UTC。

两个项目均使用原生 `opencode run`，默认模型为
`opencode/muse-spark-1.3-contributor-free`，配置入口为 `opencode://opencode`。
BO 的普通 generate/query 分别保持 high/low；parent selector 保持 medium；
agent 版各角色和 Manager 保持 high。原有 campaign 环境变量覆盖继续有效。

## 真实模型验证

| 项目 | 已通过的检查 | 返回费用 |
| --- | --- | --- |
| MLEvolve_BO | 公共 generate 文本、query 结构化 JSON；确认 high/low | 两次调用均为 0 |
| MLEvolve_BO_agent | 文本、JSON、原生 read/edit/bash、文件修改、同一会话续接、跨轮记忆、只读 Rating | 五次调用均为 0 |

agent 会话数据库和测试文件实际放在 `/network/scratch` 共享文件系统。
记忆检查的第二轮仅要求返回上一轮标记；预期标记没有写入第二轮 prompt/schema，
由 Python 在返回后独立比较。文件内容也由宿主检查。

原始结果：

- [BO 真实调用](../../MLEvolve_BO/.cache/opencode-check.json)
- [agent 共享目录真实调用](../../MLEvolve_BO_agent/.cache/opencode-sharedfs-check.json)

## 回归验证

以下计数按运行批次列出，批次之间有重复用例：

| 批次 | 结果 |
| --- | --- |
| BO：原生传输、旧 Codex 重试、autoresearch、恢复配置、parent selector | 78 passed |
| BO：原生 medium 选点与 Codex 重试补测 | 8 passed |
| agent：原有会话、角色、候选准入、Acq7、GP6 执行/恢复 | 79 passed |
| agent：原生传输/会话、配置、AlphaEvolve、MLE-bench Lite、异步默认值、campaign | 126 passed |

后续 agent research 扩展回归已按用户要求停止，不计作完整通过；以上已完成的验证结果保留。

离线测试显式注入模拟 CLI，不依赖真实模型答案。原生 OpenCode 的候选纵向测试
覆盖 Draft → Experiment → 修复续接 → 正式执行 → 准入，不增加虚假的质量观测。
历史 `reference_settings.json` 保持原始记录；新运行模型默认值单独验证。

回归同时修复了已有 Codex generate 忽略重试参数的问题，恢复已有测试要求的最多
三次尝试；新 OpenCode generate 使用相同上限。旧 Codex 会话继续使用原后端。

未启动完整 GPU 训练或完整模型驱动的 MLE-bench 搜索。OpenCode 工具权限不等同于
Codex 的操作系统 sandbox；具体权限、凭据和运行方法见 [使用说明](OPENCODE_zh.md)。
