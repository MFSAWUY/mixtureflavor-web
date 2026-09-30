# 同机部署、备份与恢复

本包已完成本机实现。**当前电脑未安装 Docker，也没有可用服务器与域名；Docker 镜像构建、云端容量和跨地域访问尚待目标机器验证。** 不把本地性能当作云端验收结果。

## 1. 本地 Docker 演练

准备 Docker Engine + Compose 插件（或 Docker Desktop）及 Python 3。进入解压后的 `mixtureflavor-web` 目录：

```sh
python scripts/init_env.py
docker compose config --quiet
docker compose up -d --build
docker compose ps
```

打开 http://127.0.0.1:8080。`.env` 由脚本创建，默认监听回环地址，包含独立随机管理员密码。脚本拒绝覆盖已有 `.env`，不会重置已有登录凭据。`.env.example` 是变量说明，不能直接作为有效密码配置。

检查健康状态与启动失败原因：

```sh
curl --fail http://127.0.0.1:8080/api/health
docker compose logs --tail=100 api
docker compose logs --tail=100 web
```

后端未通过权重校验和预热时，健康检查不会通过；Compose 等待后端健康后启动网站入口。这个依赖行为使用 [Compose 的 `service_healthy`](https://docs.docker.com/compose/how-tos/startup-order/)。

## 2. 将同一部署包放到 Linux 服务器

规划对应腾讯云 Lighthouse 香港、Linux x86-64、2 核 4 GB、约 60 GB 磁盘；先用 CPU。购买规格、地域和费用需以购买时页面为准，本次没有购买操作。若学校已提供同等资源，也可使用。

在服务器安装官方 Docker Engine 和 Compose，上传并解压发布包。建议使用专门的部署目录，例如 `/opt/mixtureflavor-web`，不要把原训练数据、环境缓存、用户数据库或本机密码一起上传。

有域名后，将该域名的 DNS A 记录指向服务器 IPv4；仅在配置有效 IPv6 时添加 AAAA。云安全组和系统防火墙放行 TCP 80 / 443，SSH 仅按管理需求开放。后端 8000 不映射到公网。

在尚未创建 `.env` 的服务器目录运行（替换为自己实际拥有的域名）：

```sh
python3 scripts/init_env.py --domain your-domain.com
docker compose config --quiet
docker compose up -d --build
```

若已有 `.env`，保留密码和数据库路径，只修改以下四项：

```dotenv
SITE_ADDRESS=your-domain.com
BIND_ADDRESS=0.0.0.0
HTTP_PORT=80
HTTPS_PORT=443
```

执行 `docker compose up -d --force-recreate web` 应用变更。域名、DNS 和端口满足条件时，Caddy 会管理证书和 HTTPS，参见 [Caddy 自动 HTTPS](https://caddyserver.com/docs/automatic-https)。线上管理员使用 HTTPS 登录。

前端由 Caddy 提供，`/api/*`、`/v1/*`、`/v2/*` 保留原路径转发到后端。配置使用 [Caddy `handle`](https://caddyserver.com/docs/caddyfile/directives/handle)，不使用会删除路径前缀的 `handle_path`。后端只运行一个 Uvicorn worker，避免复制模型、重复调度备份及多进程重启状态冲突。

服务重启策略为 `unless-stopped`。数据库命名卷 `flavor-data`、证书卷 `caddy-data` 和配置卷 `caddy-config` 独立于容器。正常更新使用 `docker compose up -d --build`；**不要在保留数据时运行 `docker compose down -v`**，该命令会删除卷。

## 3. 管理记录与每天备份

管理员凭据保存在服务器 `.env`。网页「管理入口」提供查看、CSV / JSON 全部导出、单条删除和立即备份。输入、结果、时间、版本、状态和 UUID 记录编号都会保存，上传预览不保存。没有访客注册，也不向访客公开其他人的记录。

数据库默认位于容器 `/data/submissions.sqlite3`。后端运行期间每隔 24 小时使用 SQLite online backup API 创建完整快照，校验完整性后原子替换临时文件，保留最近 7 份（包括手动备份）。重启后重新计算 24 小时周期；长期停机不会产生新快照。已删记录可能仍在旧快照中，直至轮换到期。

在管理页面点击「立即备份」后，可把快照复制到宿主机：

```sh
mkdir -p private-backups
docker compose cp api:/data/backups/. ./private-backups/
```

同盘备份不能应对整台服务器丢失。定期将 `private-backups` 另存到由管理员控制的其他设备或私有对象存储；备份同样包含用户输入与结果。不要放进前端目录、公开 Git 仓库或发布包。

## 4. 恢复到新数据库

恢复脚本只接受**不存在的新目标文件**，不会覆盖当前数据库。先查看备份文件名，在下列命令中替换 `submissions-TIMESTAMP.sqlite3`：

```sh
docker compose stop api
docker compose run --rm --no-deps api python scripts/restore_database.py /data/backups/submissions-TIMESTAMP.sqlite3 /data/restored.sqlite3
```

确认脚本报告完整性检查通过并返回记录数。修改 `.env` 中 `FLAVOR_DATABASE=/data/restored.sqlite3`，然后：

```sh
docker compose up -d api
docker compose up -d web
```

在后台核对记录数量、旧记录结果和模型版本。原数据库仍在原路径，必要时停止后端并把 `FLAVOR_DATABASE` 改回，可回退恢复。恢复到旧备份会使备份之后的提交不可见，也可能重新出现备份中尚未删除的记录，应由管理员检查。

非 Docker 本地恢复：停止启动终端，运行 `python scripts/restore_database.py runtime/backups/实际文件名.sqlite3 runtime/restored.sqlite3`，再在 `runtime/local.env` 加入 `FLAVOR_DATABASE=新文件的绝对路径` 并重启。

## 5. 上线验收

先用页面实际提交单分子、批量和五组分混合物，再检查管理员记录、数据与图表下载。重启容器后确认记录仍在，执行一次手动备份与新数据库恢复演练。恢复演练完成前不要删除原数据库。

在安装测试依赖的环境中，可对已经部署的目标运行：

```sh
python scripts/benchmark.py --url https://your-domain.com --concurrency 5 --requests 50 --output docs/verification/performance-server.json
```

该命令会在目标保存带 `Benchmark` 名称的预测记录，应在管理员知悉的验收窗口运行。目标为预热后单分子和五组分混合物均 P95 ≤ 3 秒、无失败。远程测试不会自动得到服务器内存数据；在服务器上记录 `docker stats --no-stream`，或结合系统监控采集测试期间峰值。

分别从国内校园网、手机网络和境外网络验证首页、文件上传、预测和下载。将设备、网络、日期和结果记入 `docs/verification/ACCEPTANCE.md` 的待验证项。这些项目目前没有执行。

## 6. 更新模型

只替换本包 `vendor/flavorgnn/` 下的副本。先核对模型结构、标签顺序、RDKit 特征、完整 state dict 和数据来源，再在离线开发环境运行 `python scripts/prepare_model.py` 生成新参考结果与清单，运行测试后部署。不要为了通过启动检查而在服务器上自动重建清单；校验失败必须调查。

CPU PyTorch 安装使用 [官方 CPU wheel 源](https://pytorch.org/get-started/locally/)。构建锁定 Web 依赖及核心 ML 版本；基础镜像使用 Python 3.13、Node 24、Caddy 2 的维护标签，上线时可进一步记录镜像 digest。替换训练数据或特征后应更新方法页说明，并重新验证感官结论的适用范围。
