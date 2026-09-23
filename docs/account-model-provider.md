# 账户级模型 Provider（BYOK V1）

## 范围

V1 允许每个账户保存一份 OpenAI-compatible **聊天** Provider：HTTPS Base URL、
模型名与 API Key。它用于该账户发起的文档上下文推断和分析运行中的主抽取、
角色一致性、问题证据复核与 Evidence Investigator。Embedding/RAG 凭据仍是独立的
部署配置；V1 不支持多 Provider、团队共享密钥或按项目覆盖。

接口为：

- `GET /api/v1/account/model-provider`：返回安全配置状态；
- `PUT /api/v1/account/model-provider`：创建或替换不可变 revision；
- `DELETE /api/v1/account/model-provider`：清除当前绑定并擦除该账户全部历史密文；
- `POST /api/v1/account/model-provider/test`：执行一次受限连接/JSON 合同检查。

所有写请求都要求 CSRF token 和 `expected_revision`；冲突返回 `409`。响应均带
`Cache-Control: no-store`。API Key 只显示固定掩码，不返回长度、首尾字符、上游正文、
Prompt 或响应头。连接测试每账户限频，持久化内容只有安全分类、耗时、JSON 合同与
Token 计数。

## 运行绑定

创建分析运行时，服务在同一数据库事务中冻结 `provider_config_id` 与不含密钥、
不含 Base URL 的 provider identity。Celery 消息仍然只携带 `run_id`；worker 根据
运行记录核验账户归属和 identity，再解密精确 revision，构造仅供本次运行使用的
Settings。替换设置不会改变已排队运行；显式 retry/recheck 会绑定操作者当时的当前
revision。

删除会擦除当前和历史 revision 的 ciphertext、nonce、key ID，并使后续每次物理
HTTP 尝试前的数据库授权检查失败。尚未发出请求的旧排队运行随后会安全降级/失败，
绝不会借用另一个账户或服务端的全局密钥。已经发出的上游请求无法被数据库删除动作
撤回；这是 V1 的并发边界，而不是对在途请求的绝对密码学吊销。

`AUTH_MODE=required` 下，没有账户配置时只能走确定性能力，不能借用
`OPENAI_API_KEY`。`AUTH_MODE=anonymous` 保留原有本地单用户服务端 Provider 兼容，
但其模型名和 endpoint 指纹同样在运行创建时冻结；运行期间配置漂移会 fail closed。

## 静态加密与主密钥

API Key 使用 AES-256-GCM 加密。AAD 绑定账户 ID、配置 ID、revision、endpoint
SHA-256 与模型名，数据库字段被替换时无法通过认证解密。该主密钥必须与
`AUTH_SECRET_KEY` 独立，系统会拒绝复用相同密钥材料。

配置项：

- `ACCOUNT_MODEL_ACTIVE_KEY_ID`：新密文使用的 key ID；
- `ACCOUNT_MODEL_KEYRING_JSON`：`{"version":1,"keys":{"key-id":"base64-32-bytes"}}`；
- `ACCOUNT_MODEL_KEYRING_FILE`：上述 JSON 的受限普通文件（与 JSON 二选一）；
- `ACCOUNT_MODEL_LOCAL_KEY_PATH`：仅本地模式无显式 keyring 时的持久化位置；
- `ACCOUNT_MODEL_ALLOWED_ORIGINS`：逗号分隔的精确 HTTPS origin。

生产模式没有自动生成或 `AUTH_SECRET_KEY` 派生回退，缺少显式 keyring 会拒绝启动。
本地非 Docker 模式会以排他创建方式生成 Git 忽略的独立文件；丢失该文件会导致已有
账户密钥不可恢复。开发 Compose 的 API 与 worker 共享 `account_model_keys` volume，
避免两个容器各自生成不同主密钥。生产 Compose 把同一份宿主机 keyring 文件作为
只读 secret 挂载到两者，不把主密钥 JSON 放入容器环境。

轮换步骤：

1. 将新 key ID 与旧 key ID 同时加入 keyring，先部署到全部 API/worker；
2. 把 `ACCOUNT_MODEL_ACTIVE_KEY_ID` 切到新 ID；
3. 用户下一次保存/替换配置时会使用新 active key 加密（保留 API Key 时也会先用旧
   key 解密，再用新 key 加密）；
4. 旧 key 必须保留到所有引用旧 revision 的队列任务结束，或相关账户执行删除后，
   才能从 keyring 移除。

V1 没有批量在线 rewrap 作业；不要在仍有历史密文/排队运行时直接移除旧 key。

## SSRF 边界

保存时和最终 HTTP transport 前都会重新校验 endpoint。只接受管理员 allowlist 中的
精确 HTTPS origin；拒绝 HTTP、userinfo、query、fragment、IP literal、localhost、
`.local`、单标签主机、路径穿越与重定向。请求边缘使用从部署 Settings 冻结的 origin
集合，不会因为 run-local `OPENAI_BASE_URL` 被替换而自动信任新 origin；HTTP 客户端
继续设置 `follow_redirects=False`。

增加新中转时，应先由部署管理员把它的**仅 origin**（例如
`https://api.example.com`，不能带 `/v1`）加入 `ACCOUNT_MODEL_ALLOWED_ORIGINS`，再允许
用户保存完整 Base URL。
