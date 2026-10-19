# 本机 ChatGPT 套餐网关（显式开发选项）

这不是把 ChatGPT Pro 换成一个 OpenAI API Key，也不把 Pro 套餐变为无限量 API。
本机 CLIProxyAPI 保存经浏览器授权的 OAuth 凭据，LoreGuard 只保存一个独立的固定
**本机客户端密钥**。网关刷新上游凭据时，通常不必修改 LoreGuard 中的客户端密钥；
撤销授权、refresh token 失效、计划/工作区不符合条件时，仍须重新授权。

适用范围：用户本人、本机自用、管理员明确启用的开发实例。本轮不提供公网代理、
多人共享 Pro 凭据、现有临时分享站点集成或 Docker 网关发布。账户隔离、CSRF、
账户密钥静态加密与运行配置冻结继续由 LoreGuard 原有机制负责。

## 1. 固定版本、运行目录和安全边界

脚本固定下载 [CLIProxyAPI v8.0.33](https://github.com/router-for-me/CLIProxyAPI/releases/tag/v8.0.33)：

- 源码：`2422c6b0f5fe16495320fa29456493874c1977a0`。
- Windows x64 ZIP SHA-256：`6a1e04025710f01c528b88c4ae26ea1a332aabc27c7387ae3527107b9096f8a5`。
- 发布方 `checksums.txt` SHA-256：`a69d87de882f6b8bb84117558dda23c99515ee1f9f38e82095e7e5ae5239bcfc`。
- 这提供下载完整性/版本固定，不等同于第三方软件的独立安全审计或 OpenAI 官方背书。

运行 PowerShell，切换到 LoreGuard 仓库：

```powershell
.\scripts\local-pro-gateway.ps1 -Action Setup
.\scripts\local-pro-gateway.ps1 -Action Start
.\scripts\local-pro-gateway.ps1 -Action Status
```

当前机器的生成目录为：

```text
C:\Users\dell\Desktop\dev\tools\LoreGuard-Pro-Gateway\
  bin\8.0.33\cli-proxy-api.exe
  downloads\                         # 固定发行包及校验文件
  installation.json                   # 归属、版本、SHA-256，不含密钥
  config.yaml                         # 私有真实配置，包含本机客户端密钥
  client-api-key.txt                  # 在 LoreGuard 表单使用的固定密钥
  auth\                              # 本实例的 OAuth 凭据，不复用 .codex/auth.json
  loreguard-data\                    # 独立数据库、auth-secret、账户密钥主密钥
  logs\                              # 私有进程输出，可能含授权链接，不要上传
  process.json / oauth-process.json   # PID、创建时间和路径，不含凭据
  loreguard-process.json              # 独立应用父进程/真实 Python worker 的归属
```

目录位于仓库外，脚本不提交任何上述内容。Setup 在生成密钥前关闭 ACL 继承，
仅允许当前 Windows 账户、SYSTEM、Administrators；配置、日志与凭据也继承这一
受限权限。它们是受权限保护的本地文件，不声称是磁盘加密文件。不要把目录放进
Git、网盘共享、公开支持包或截图中。源代码中没有真实客户端密钥或 OAuth token。

Setup 重复执行只校验已有的自有安装并保留密钥。脚本拒绝不属于本实例的已有目录、
修改后的配置/程序、junction/symlink、非受限 ACL；不会覆盖或删除陌生目录。
若首次下载中断，保留失败目录核查，勿直接删除整个 tools、Desktop 或仓库。

实际生成配置的主要约束：

```yaml
config-version: 8
server:
  host: "127.0.0.1"
  port: 8317
management:
  allow-remote: false
  secret-key: ""
  disable-control-panel: true
access:
  api-keys:
    - "<由脚本生成，不在文档展示>"
oauth:
  auth-dir: "<本实例隔离 auth 目录>"
```

脚本另外关闭发现广播、插件、pprof、请求正文日志及管理面板更新，限制请求体为
4 MiB，并关闭网关层额外请求重试，避免与 LoreGuard 的有界重试叠乘。启动使用
`--local-model` 固定本版本内置模型目录，不为无凭据预检下载动态模型目录。
没有管理密钥时，两套 management/config 路由返回 404。服务只监听 IPv4 loopback，
不监听 `0.0.0.0`，不要为方便访问而修改为全网卡绑定。

## 2. 先准备，再由本人做浏览器授权

尚无 OAuth 凭据时可启动服务做协议预检，但不能调用模型。不要把空模型目录或一次
普通 Codex 登录当作 Pro 接入成功。这个版本需要两个**不同的**授权步骤：

### 第一步：生成此网关独有的 Codex 凭据

```powershell
.\scripts\local-pro-gateway.ps1 -Action CodexLogin
```

脚本自动停止自己管理的网关，隐藏启动授权进程，第一步使用该固定版本要求的
`http://localhost:1455/auth/callback`，不传端口 override。该版本的 legacy Codex
登录 URL 写死 1455，而 override 只改监听端口；将两者一起改成 1457 会造成“浏览器
登录完成但凭据未保存”。用户只需在自动打开的官方浏览器页面完成本人登录，不用复制 key、授权链接、
浏览器回调或 token 给任何人。确认浏览器域名是 OpenAI 官方域名。

用 Status 查看 `OAuthRunning`；授权成功后，`auth` 目录会生成 JSON 文件。此时
只是普通 Codex 凭据，还没有 SIWC 套餐使用授权，Start 会拒绝它，防止误走不同的
backend-api 路径。可在资源管理器中查看文件名，**不需要打开文件正文**。

### 第二步：给刚才的文件绑定 SIWC 套餐权限

```powershell
.\scripts\local-pro-gateway.ps1 -Action SiwcBind -CredentialFileName '<刚生成的文件名.json>'
```

只接受本实例 `auth` 目录内的 JSON 文件名，不接受外部路径。这一步会再次打开
浏览器，要求同一个 ChatGPT 账户授权通过该第三方本机网关使用套餐。用户本人检查
权限并决定同意；此操作与 LoreGuard 的账户登录相互独立。

SIWC 第二步明确使用 `http://127.0.0.1:1457/auth/callback`。该路径会依据
override 同时生成匹配的 redirect URI 和 listener，不与第一步的硬编码机制混同。

**固定版本的临时 OAuth 回调限制：** 其共享 `NewOAuthServer` 实际监听 `:1455` /
`:1457`（全接口），不是只绑定 loopback，虽然浏览器 redirect URI 是上述本地地址。
这个第三方实现限制不适用于本项目配置的 8317/8787（两者仍严格 127.0.0.1），也不应
被描述为“整个授权进程完全只有 loopback listener”。仅在用户控制的可信本机网络进行
短时授权，不为此开放系统防火墙或建立公网隧道；授权结束即关闭回调进程。需要严格
loopback-only OAuth 的环境应先对上游回调实现做独立修补/验收，不直接部署此二进制。

CLIProxyAPI 绑定源码保存 `use_siwc=true` 与独立 `siwc` 数据。启动脚本要求该段有
access/refresh token、正式签发的 `client_id` 及 `chatgpt.tokens.use.direct` scope；
不会把旧 Codex 登录或身份 scope 误判为可使用套餐。用户不同意套餐权限时，保持未连接。

```powershell
.\scripts\local-pro-gateway.ps1 -Action Status
.\scripts\local-pro-gateway.ps1 -Action Start
.\scripts\local-pro-gateway.ps1 -Action Models
```

授权进程最多等待约五分钟。超时、浏览器未打开或端口冲突时，不要反复并行启动。
Status 可查看进程状态；必要时 Stop 仅停止脚本核验归属的网关及授权进程，再开始
新的授权。不要把私有 OAuth 日志或其中的 URL 发到聊天、issue 或 CI。

最后一条只输出模型 ID，不输出客户端密钥。该网关的模型列表是路由目录，不能证明
账户有某个模型的权限；最终以该账户的一次完整成功推理为准。不要猜填模型名，
也不要为了尝试列表中全部模型而运行大规模评测。

## 3. 独立本机 LoreGuard 实例

脚本已经提供隔离实例，可直接使用仓库已有 `.venv` 与构建后的前端：

```powershell
.\scripts\local-pro-gateway.ps1 -Action StartLoreGuard
# 在浏览器打开 http://127.0.0.1:8787/
.\scripts\local-pro-gateway.ps1 -Action StopLoreGuard
```

仅使用 `scripts.local_pro_gateway_app:create_app` 工厂，关闭 `.env`、普通环境变量和
file-secret 来源；即使当前 shell 中有既存 `OPENAI_API_KEY`、`DATABASE_URL` 或
`AUTH_MODE` 也不会继承。它只读取启动器注入的非密钥 runtime 目录并核验固定路径与
安装归属，先注入隔离 Settings，再首次导入真实应用。独立数据保存在上述
`loreguard-data`：SQLite、独立 session HMAC secret、独立账户密钥主密钥；未注入全局
模型 key，只开启主抽取，embedding 和实验 Agent 默认保持关闭。账户仍需通过原页面
注册/登录并自行保存模型配置；未登录的信息页与空工作区仍可浏览。

启动器不改变已有分享实例，应用仅监听 `127.0.0.1:8787`，关闭代理头信任和访问日志，
TrustedHost 只接受 `127.0.0.1`/`localhost`。8787 被占用时拒绝，不杀陌生进程。
Windows venv 的 launcher 与真实 Python worker 分别记录并核验，StopLoreGuard 先停
受控 worker 再停 launcher，避免孤儿监听；不会停止网关 OAuth 或其他 Python 服务。

不要修改正在通过公网隧道分享的实例。给 API **及处理同一任务的 worker** 显式设置：

```dotenv
DEPLOYMENT_ENVIRONMENT=local
AUTH_MODE=required
LOCAL_MODEL_GATEWAY_ENABLED=true
LOCAL_MODEL_GATEWAY_ORIGINS=http://127.0.0.1:8317
```

这是追加到独立本机启动配置的无密钥片段，不是完整的生产 `.env`。数据库、
`AUTH_SECRET_KEY`、账户密钥 keyring/持久化位置、服务端口均使用独立实例自己的
设置；不要复用、复制或打印已有分享服务的密钥。API 绑定 `127.0.0.1`，不得把含本机
Pro 网关能力的实例挂到现有公网隧道。普通账户登录和私人数据隔离不关闭。

在该本机实例的“模型设置”中填写：

| 表单字段 | 本机设置 |
| --- | --- |
| Base URL | `http://127.0.0.1:8317/v1` |
| API Key | 从受限 `client-api-key.txt` 在本机复制；不在聊天或截图展示 |
| 模型 | 授权后 Models 输出中、且实际烟测成功的模型 ID |

保存仍创建不可变 revision，账户 key 仍加密；已排队运行仍绑定创建时的配置，不借用
其他账户或全局 key。关闭本地网关开关/撤销 origin 后，后续实际 transport 也会拒绝，
不是只在前端隐藏选项。账户连接测试采用本地网关专用的有界等待，不循环昂贵请求。

`LOCAL_MODEL_GATEWAY_ENABLED` 默认关闭。非 local 部署不得开启；普通 HTTPS
Provider 的精确 origin allowlist、主机限制和禁止重定向继续保留，不因这一功能而
允许任意 HTTP、局域网 IP 或云元数据地址。前端只接受后端明确返回的本地 origin。

### Docker 边界

容器里的 `127.0.0.1` 是容器自己，不能直接访问这个 Windows loopback 网关。
虽然服务端策略可显式允许 `http://host.docker.internal:8317`，当前网关只绑定
Windows `127.0.0.1`，**不保证 Docker Desktop 能访问它**。本轮只实测本机直跑；
没有修改监听到 `0.0.0.0`、开放防火墙端口或建立转发。需要 Docker 时，必须先设计
宿主机—容器的最小安全转发并单独验收，不能把配置允许当作网络已经通了。

## 4. 已处理的兼容性与没有承诺的能力

链路为 LoreGuard Chat Completions → 第三方网关转换 → 公开 Responses API 的 SIWC
流 → 网关聚合 Chat Completions。固定本地 key 只验证调用网关的客户端，不是 OpenAI
API Key，也不替代浏览器授权。

- v8.0.33 的 `json_object` 不能假定被转换为强制 JSON。LoreGuard 的显式本地 profile
  改用可转换的 JSON schema 外壳，仍校验最终 JSON、业务 Pydantic schema、证据和工具合同；
  外壳正确不等于业务内容正确，非法结果仍失败/有界重试，不伪装为成功。
- `temperature`、`max_tokens` 及 DeepSeek `thinking` 扩展不在此 profile 中发送；SIWC
  不支持温度与输出 token 上限参数，因此不能声称固定温度可复现或网关执行硬 token 上限。
  LoreGuard 的请求期限、响应体限制、工具步数和已有应用级预算保护仍有效，但无法等同
  于上游服务端的输出 token 硬限额。
- 网关会调整工具参数，不能保证保留调用方的 `parallel_tool_calls=false`。LoreGuard
  本地 profile 在工具调用返回后拒收多个同批工具调用，不并行或自动执行它们；工具名、
  参数、证据范围与步数仍受本项目合同约束。这是安全边界，不是多工具能力声明。
- 不增加托管搜索、图像生成、文件搜索、Code Interpreter、浏览器执行或任意 shell 工具。
  RAG/embedding 服务仍是原有独立配置，不通过 ChatGPT 套餐隐式接入。
- 本机网关连接测试允许首次 OAuth 刷新/慢推理，仍有超时、限频和单次尝试，不能保证
  所有模型、长篇故事或完整昂贵评测都满足 SLA。
- 上游套餐资格、可用模型、应用限额、地区和工作区政策以服务端为准。Pro 不等于
  “无限量、无限并发、保证全部 API 功能”；遇到额度/权限错误应显示实际失败，不切换
  到未获授权的 API 账单或其他账户。

## 5. 本阶段验证记录

2026-10-19，本机 Windows x64 完成（不包含用户 OAuth 或真实模型推理）：

1. 固定 checksum 文件和 ZIP 摘要匹配；exe help 显示版本 `8.0.33`、commit `2422c6b0`，
   并确认 `--codex-login`、`--siwc-bind`、`--oauth-callback-port`、`--local-model` 参数。
2. 生成目录/凭据目录/日志目录关闭 ACL 继承，仅当前账户、SYSTEM、Administrators。
3. Setup 重复校验保留固定 key；Start 重复使用自己的进程；Stop 两次安全完成，再启动成功。
4. 监听地址仅 `127.0.0.1:8317`；未带 key 的 `/v1/models` 为 401；带本地 key 为 200、
   `data=[]`，准确标示尚无授权账户，而不是模型调用成功。
5. `/v0/management/config` 和 `/v8/management/config` 均为 404。
6. auth 目录为空时一次极小合成 Chat Completions 预检返回 HTTP 400、
   `model_not_found`，没有上游凭据可调用；检查后仍为零凭据。
7. 工厂自动测试 `tests/test_local_pro_gateway_app.py` 3 项通过：全新子进程忽略冲突
   环境/合成 `.env`，保持 required/空全局 key/独立库、本地网关 flag，拒绝陌生 Host，
   私人 API 未登录为 401，重复初始化拒绝。第一次相对 `--basetemp` 被既有测试夹具的
   cwd 重定向导致临时目录创建错误；改为 Desktop artifacts 下绝对路径后通过。
8. 真实独立实例验证：`/health` 200、首页静态 200、`/api/v1/auth/me` 401、陌生 Host
   400，仅监听 `127.0.0.1:8787`；没有创建账户或调用业务/模型接口。
9. 追加 Windows 子进程生命周期验证：只停止校验过的自有 Python worker/launcher，
   8787 监听消失。首次重启因重复 Set-Acl 重写受限目录时缺少 SeSecurityPrivilege
   被拒绝，改为先核验已经正确的 protected DACL、不再重写 owner/audit 后恢复；
   实际 worker 与 8787 listener 的 PID 一致，重启后 health/首页/未登录/Host 边界正确。
   没有触及 1457 或 OAuth 进程。
10. 首次 Codex OAuth 端口 override 1457 与固定 1455 redirect URI 不一致，用户在浏览器
    完成操作后未保存凭据；固定源码核对后已将两种授权端口契约分开。此失败保留记录，
    不能以浏览器显示成功替代凭据落盘及 SIWC 权限验证。

浏览器授权第一步已由集成方发起，仍须用户本人完成；这不代表 SIWC 绑定或真实推理成功。

项目自身的配置保存、运行冻结/worker、连接测试、payload/tool 合同、SSRF 回归与前端
构建以本阶段整体验收记录为准。此处不把这些网关预检替代为真实 Pro 调用。
用户完成两次必要授权后，才可用一个很小的原创合成输入进行真实端到端烟测；
烟测前仍不宣称 Pro 套餐路径已经跑通。

## 6. 停止、续期与排障

```powershell
.\scripts\local-pro-gateway.ps1 -Action Status
.\scripts\local-pro-gateway.ps1 -Action Stop
```

Stop 不删除凭据或固定 key，不停止陌生的 8317/1457 进程。PID、创建时间、可执行文件
路径和包含本配置的命令行必须全部匹配才控制进程，避免 PID 复用误杀。
独立应用另用 `StopLoreGuard` 停止，Stop 不会顺带停止它。

临时网络/503 错误不是凭据失效，不要直接删除 token。上游确认授权撤销或 refresh
token 失效时，重新走浏览器授权；网关的 OAuth 自动刷新与固定本地 key 无关。
若需要完全撤销访问，还应在 ChatGPT 设置里断开该第三方应用，并移除 LoreGuard
保存的本地模型配置；仅停止进程不等于撤销上游授权。

## 官方与固定版本参考

- [OpenAI：本地开源应用注册与登录](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [OpenAI：模型目录与推理完成](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [OpenAI：SIWC 预览限制](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [OpenAI：账户、刷新与凭据保护](https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions)
- [OpenAI：错误与恢复](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
- [CLIProxyAPI v8 配置模板](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.33/config.example.yaml)
- [CLIProxyAPI：SIWC 绑定](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.33/internal/cmd/siwc_bind.go)
- [CLIProxyAPI：Chat Completions 转换](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.33/internal/translator/codex/openai/chat-completions/codex_openai_request.go)
- [CLIProxyAPI：SIWC 请求规则](https://github.com/router-for-me/CLIProxyAPI/blob/v8.0.33/internal/runtime/executor/codex_executor_request.go)
