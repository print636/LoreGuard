# 隔离 PostgreSQL 备份恢复演练

状态：工具与集中单元测试已完成；本机真实演练因 Docker Engine 不可达而未执行成功。
日期：2026-10-02。

## 解决什么问题

数据库备份成功不等于可恢复。本工具用本项目的当前 Alembic 迁移，在独立测试容器里
创建合成账户与故事关系数据，再真实执行 `pg_dump`／`pg_restore`，检查恢复后的资料、
冻结输入、反馈、迁移版本和数据库约束是否保留。

它是发布前的可重复练习，不是生产备份服务：不接受数据库地址，不读取 `.env`、
用户数据库、真实账户、剧情文件、API Key 或人工评测材料，不调用任何模型。

## 准备与运行

需要通过本机 socket／named pipe 访问的 Docker Linux Engine、本项目 Python 环境，以及已有的
`pgvector/pgvector:pg16` 镜像。工具不会安装、重装 Docker 或自动下载镜像。
镜像尚不存在时，由操作者确认网络与镜像来源后自行拉取；不因演练变更现有 Compose 服务。

在仓库目录执行：

```powershell
.venv/Scripts/python.exe scripts/backup_restore_rehearsal.py
```

退出码 `0` 表示演练验证和自建容器清理均完成；非零表示未完成，查看输出的
`report.json` 路径与固定 `failure_code`。终端不打印底层异常、密码或数据库连接串。

产物保留在 Git 忽略的 `.artifacts/backup-restore-rehearsal/loreguard-rehearsal-<唯一ID>/`：

- `synthetic.pgcustom`：本次合成数据库的 PostgreSQL custom-format dump，成功 dump 后才存在。
- `report.json`：成功／失败阶段、dump 摘要、恢复检查与清理情况；不包含剧情正文或密钥。

该合成 dump 不加密，不能当成真实私人数据库的存储范例。生产备份须另行加密、访问控制
和异机保留，且恢复使用匹配的 PostgreSQL／扩展版本和受控操作者。

## 实际执行顺序

1. 验证 Docker 与本地镜像，不启动用户现有 Compose 服务。
2. 创建随机 `loreguard-rehearsal-` 名称及拥有标签的独立容器，仅映射随机环回端口。
3. 在新源库 `loreguard_rehearsal_source` 中确认没有用户表，再通过显式 URL 升级当前 Alembic head。
4. 插入两个合成账户、工作区、项目，文稿两版本、冻结旧稿运行、问题和反馈，以及假会话和假密文配置。
5. 计算全表行数与 JSON 内容指纹、列结构、索引、约束、触发器和扩展版本指纹。
6. `pg_dump --format=custom --no-owner --no-acl` 后新建空目标库
   `loreguard_rehearsal_restored`，确认无用户表；`pg_restore --exit-on-error` 恢复。
7. 验证源与恢复指纹一致、旧运行原文和哈希没有被当前新稿替换、工作区过滤查询隔离、
   单独内存密钥能够解开恢复的合成密文。
8. 在可回滚事务中故意违反文档活动版本唯一、文档版本唯一、反馈外键、成员唯一、
   项目工作区外键及 Provider revision 检查，要求数据库拒绝；再次比较指纹确认没有留下探测写入。
9. 确认自建容器的完整 ID、名称、目的标签和随机拥有标签后，仅删除该容器及其匿名卷。
   演练产物保留，不删除宿主机目录。

每次运行都新建唯一容器与两个专用数据库，不提供选择既有容器或生产数据库的参数。
镜像标签仅用于发现已存在镜像，随后校验完整 `sha256` image ID，并按该不可变 ID
创建容器，避免运行期间标签变化使记录与实际镜像不符。
工具会检查当前 Docker context 的有效传输，拒绝 SSH、TCP 及远程 named pipe context。
端口映射是 Docker 主机的环回地址，不是远程 CLI 客户端的环回；远程 context 不在支持范围。
不会执行 `docker system prune`、通用 `compose down`、`DROP DATABASE` 或对已有目标使用 `--clean`。
如果创建请求结果不明确或拥有身份无法验证，宁可留下安全失败记录，也不会猜测目标删除。

## 密钥与生产落地边界

账户 Provider 的数据库密文需要独立的 AES-256-GCM keyring 才能解密。
**只有数据库 dump，不能恢复用户保存的 API Key。** 本演练只用一次性的合成 AEAD 信封，
密钥留在进程内存、不会写入 dump、报告或文件；它不是应用 Provider 信封格式或真实配置。
演练结束后不保留这把假密钥，因而产物中的假密文也无法凭 dump 自行解密。

真实部署须分别受控备份数据库、Provider keyring 与必要的认证／服务配置，制定
加密、异机留存、保留周期、轮换和恢复授权。keyring 不应与公开代码同存，也不应
不加保护地随数据库放在同一目录。遗失 AUTH_SECRET_KEY 会影响既有会话验签；更换该密钥
意味着旧会话需重新登录，不能把合成假会话恢复当成可继续使用真实会话的证明。

本工具只验证干净新库从现有迁移链升级后备份恢复，不验证真实旧版／生产规模数据迁移、
迁移降级回滚、账户密码并发安全、HTTP 端点权限或公网高可用。工作区检查是数据库归属
过滤查询，不能替代项目现有跨租户 API 回归。空库演练通过不表示可以直接开放公网；
域名、HTTPS、主机安全、真实备份恢复和运营策略仍由部署者决定。

单元测试仅验证保护与编排，不能被称为真实 PostgreSQL 演练成绩。Docker 不可达或镜像
未准备时报告必须如实保留失败原因，不将 Mock 结果当作恢复成功。

## 当前验证记录

- 集中单元测试一次运行，25 项通过，2.48 秒；覆盖拥有身份与环回端口校验、
  数据库与产物范围限制、密码仅经进程环境传递、错误脱敏、Mock dump／restore 的
  二进制传输，以及失败后只清理本次容器的流程。
- 补齐不可变镜像绑定和远程 Docker context 拒绝后，受影响验证与运行目录合同
  合并执行一次，53 项通过，26.59 秒，其中本工具 32 项；没有重复本机真实入口。
- 本机真实入口执行一次，在 `docker_preflight` 返回 `docker_unavailable`。
  没有创建容器、读写数据库或生成 dump，也没有进行恢复与约束验证。
- 未因环境失败重新安装 Docker、修改用户设置或接触正在使用的服务。

因此当前只能确认工具保护和编排合同通过，不能宣称本机 PostgreSQL 备份恢复已通过。
后续在可用 Docker Engine 与本地镜像环境执行同一入口，才能取得真实恢复报告。

## CI 真实执行入口

GitHub Actions 增加独立的 `backup-restore-rehearsal` job，在 Ubuntu／Python 3.12
环境安装现有依赖、准备 pgvector PostgreSQL 16 镜像后运行同一脚本。该 job 不借用其他
service 数据库，不读取生产 secret，不上传 dump。终端只输出固定失败阶段或通过后的
迁移 head、表数量、约束探测数量、dump hash／大小和清理状态等有限元数据。
CI 是否实际通过以对应提交的运行结果为准；新增 job 本身不是恢复成功证明。
