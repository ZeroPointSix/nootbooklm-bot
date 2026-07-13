# NotebookLM Slack Agent

这是一个基于 Slack Bolt 的个人研究 Agent。当前主线把 NotebookLM 收敛成仓库内置 tool provider：Slack bot、登录服务和 `/notebook status` 都只依赖同一个 `NotebookToolProvider` 接口，而不是让模型直接面对外部 MCP 的认证/清理工具。

注意：NotebookLM 没有用于本场景的正式公开 API/OAuth。请使用专用 Google 账号，并把 Storage State 当作完整凭据保护。

## 能力

- Slack Assistant、私聊和 mention 自然语言研究
- 内置 NotebookLM tool provider：`list_tools()`、`call_tool()`、`health()`、`reconnect()`
- 默认 `NOTEBOOKLM_BACKEND=local`，MCP 仅作为显式兼容后端
- `/notebook login`、`status`、`logout`、`login cancel`
- 一次性短时效登录链接、token 摘要、单活跃会话、跨进程 SQLite 状态
- Storage State 格式验证、0600 权限、provider health 探针、原子替换和失败回滚
- 兼容 MCP backend 时过滤 `setup_auth`、`re_auth`、`cleanup_data` 等会破坏认证态的工具
- 独立 Bot/Auth 容器与持久化 Profile Volume

## 快速开始

要求 Python 3.11+。创建虚拟环境，安装 requirements.txt，复制 .env.sample 为 .env，并至少配置 Slack Bot token、Slack App token、OpenAI API key。把 manifest.json 导入 Slack App 后运行 python app.py。

默认后端是内置 provider：

```env
NOTEBOOKLM_BACKEND=local
NOTEBOOKLM_PROFILE_PATH=.notebooklm/profiles/default/storage_state.json
```

如需临时回退到旧 MCP 后端，显式设置：

```env
NOTEBOOKLM_BACKEND=mcp
NOTEBOOKLM_MCP_TRANSPORT=http
NOTEBOOKLM_MCP_URL=http://notebooklm-mcp:8080/mcp
```

## 登录服务

登录不是 Google OAuth。`/notebook login` 创建一次性链接；Auth 服务消费链接后，跳转到运维方提供的隔离浏览器/noVNC Worker。Worker 完成 Google 页面登录后，把 Playwright Storage State 通过仅限内网的完成接口交给 Auth 服务。应用绝不创建用户名、密码或验证码表单。

生产环境必须配置 AUTH_BASE_URL、AUTH_BROWSER_VIEWER_URL、至少 32 字符的 AUTH_INTERNAL_TOKEN、共享的 AUTH_SESSION_DB_PATH 和 NOTEBOOKLM_PROFILE_PATH。

登录完成后不再用 MCP `tools/list` 证明成功，而是调用 provider health。`/notebook status` 也走同一套健全逻辑，返回后端、摘要和 profile/storage/google/notebooklm origin 等分层检查结果。

反向代理必须满足：

- 对 /auth/notebooklm/* 禁用访问日志或脱敏路径，并强制 HTTPS；
- 禁止缓存并保持 Referrer-Policy: no-referrer；
- 只允许 Browser Worker 网络访问 /internal/*；
- 不把 Auth 服务的 127.0.0.1:8080 直接暴露公网；
- 登录结束后销毁临时浏览器 Profile 和 Viewer 会话。

ExternalBrowserWorker 是明确的部署边界。本仓库不内置永久公开的 noVNC，因为浏览器隔离、TLS、进程清理和网络 ACL 必须由部署环境保证。

## 部署与验证

推荐依次执行：

1. ruff check .
2. ruff format --check .
3. pytest
4. docker compose config
5. docker build -t notebooklm-slack-agent:test .
6. docker compose up -d
7. curl http://127.0.0.1:8080/healthz

测试覆盖动态发现、通用调用、未知工具、畸形参数、工具循环上限、上游宕机、敏感字段脱敏、token 重放、过期、非法状态转换、跨进程会话、Profile 权限、认证失败回滚和内置 provider health。

真实 Slack/Google/NotebookLM 端到端需要有效 Slack App、专用 Google 账号和 Browser Worker。Mock 通过不代表这些外部依赖已经通过。

## 安全边界

- .env、.notebooklm、Cookie、Storage State 和 token 不得提交 Git。
- 模型只能看到 provider 暴露的脱敏结果，不能读取 Profile 文件。
- 工具名称必须来自当次 `list_tools()`；未知工具不会被转发。
- HTTP/stdio 错误被归一化，不向 Slack 返回内部堆栈或端点。
- 共享账号适合个人 Bot；多用户隔离不是当前版本的安全承诺。
