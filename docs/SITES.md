# Sites 展示版

展示版是现有模型的预计算结果展示，完整的 Python 实时推理仍由本地版或后续独立服务器提供。Sites 无需访问本机，也不连接 SQLite 用户数据库。

## 本地开发与验证

所有命令从 `mixtureflavor-web` 运行。首次安装前端依赖按主 README 操作。

```powershell
npm test --prefix frontend
npm run build:showcase --prefix frontend -- --outDir dist-showcase
npm run preview --prefix frontend -- --outDir dist-showcase --port 5174 --strictPort
```

浏览器访问 `http://127.0.0.1:5174/`。默认 `npm run build --prefix frontend` 仍生成实时版到 `frontend/dist/`，两种构建不能混用。`MODE=showcase` 选择独立页面入口，只加载同域静态 JSON，没有预测 API 请求。

`scripts/prepare_showcase.py` 使用固定模型权重和示例输入预计算结构、全部分数、基线与注意力；逐项核对原预测器输出后写静态数据。它不创建 Store，不读写提交数据库。仅模型或示例变化时重新生成，日常页面调整直接复用已生成数据。

## 发布隔离

`scripts/prepare_sites.py` 仅复制前端源码、公开样例、测试、依赖清单和页面资源到 `sites-preview/`，并将发布目录的默认 `build` 固定为 `build:showcase`。不会复制 Python 后端、权重、运行目录或密码。主 Git 仓库忽略此目录，Sites 发布仓库在该目录独立管理；脚本保留其 `.git` 与 `.openai/hosting.json`。

首次站点已经注册，后续必须复用：

- Project ID：`appgprj_6abca6cc640081919de77f2213625945`
- 权限：私有；本次没有申请公开访问或额外付费资源。
- 本地身份：`sites-preview/.openai/hosting.json`
- 当前状态：已于 2026-09-30 发布，Sites 原生部署状态为 `succeeded`。
- 正式地址：https://mixtureflavor-showcase.anna-chuang-5992.chatgpt.site
- 发布源码：`22c3475ddc996d23a4ad19f3cbdc4719b8f8dd23`
- 部署 ID：`appgdep_6abcf9b81b588191869f4d8f3b63a1e0`

如以后重新从主仓库克隆项目，先恢复上述同一站点的托管源码，再更新内容；不要创建重复 Site。

## 发布流程

使用当前安装版本的 Sites 官方工作流。执行前检查插件实际路径，不依赖旧缓存版本路径。

1. 对既有站点，先用 Sites 的打开/同步流程恢复或同步它的源码，再复制本轮前端修改。首次新建站点可直接使用已准备的发布目录。
2. 保留原始 `project_id` 和 `static.directory: "dist"`，取得同站点的短期源码凭据，只经标准输入传给工作流，不写文件。
3. 在发布目录安装与锁文件一致的依赖，使用 `npm run build:showcase` 生成 `dist/`。
4. Sites 工作流校验、提交与推送当前前端源码，打包构建产物；以返回的完整提交 SHA 和归档路径保存并私有部署。
5. 等待原生部署状态为 `succeeded` 且返回 URL 后更新本文件。尚未成功时保留错误与现有站点，不重复注册。

最近发布验证：7 项前端测试全部通过，TypeScript 检查与 Vite 展示版构建成功；官方工作流已推送源码并校验归档，Sites 已确认部署成功。此前的插件缺失问题已解决。

2026-09-30 再次通过原生接口核对：部署仍为 `succeeded`，当前访问名单只有所有者，线上版本为第 1 版。验证范围与限制见 [验收记录](verification/SITES_ACCEPTANCE.md)。

## 数据与分享

结果使用 `example_id` 和 `result_kind: precomputed_example`，没有伪造的 `record_id`、提交时间或实时耗时。CSV/JSON 保留预计算来源、模型版本与合成数据说明；PNG 带预计算及研究边界说明。此 Site 不接收或保存用户分子提交；托管平台自身的访问日志不属于本项目预测数据库。

Sites 的私有访问要求使用有权限的账号。展示版通过验收后可另行调整分享范围；当前不改变访问权限。未来接入实时推理时，需要独立模型 API，并重新设计跨域访问、凭据和记录保存方式，不能直接把本机 `127.0.0.1` 地址提供给外部访客。
