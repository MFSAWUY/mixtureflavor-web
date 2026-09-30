# API 使用约定

所有例子以 `http://127.0.0.1:8000/api` 为基础。公开调用不需要注册。预测提交会保存；说明可先从 `GET /meta` 获取。预览只解析结构，不保存记录。

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | 模型已加载、预热通过及数据库可访问 |
| `GET /meta` | 模型版本、24 类标签、真实测试指标、限制、研究与保存说明 |
| `GET /examples` | 3 组真实模型示例输入和原预测器参考结果 |
| `POST /molecules/preview` | JSON `{"text":"CCO\tEthanol\nCC(=O)O\tAcid"}` |
| `POST /molecules/upload` | multipart 字段 `file`，CSV 或 SDF |
| `POST /predictions` | 下文统一预测格式 |
| `GET /admin/records?limit=25&offset=0` | 分页记录，需管理员认证 |
| `GET /admin/records/{id}` | 完整输入、结果和错误，需认证 |
| `GET /admin/export?format=json` | 全部记录，支持 json / csv，需认证 |
| `DELETE /admin/records/{id}` | 删除当前数据库中的一条记录，需认证 |
| `POST /admin/backup` | 手动创建 SQLite 快照，需认证 |

统一请求：

```json
{
  "mode": "mixture",
  "molecules": [
    {"smiles": "CCO", "name": "Ethanol"},
    {"smiles": "CC(=O)O", "name": "Acetic acid"}
  ],
  "acknowledge_storage": true
}
```

`mode` 为 `single`（恰好 1 个）、`batch`（1–100 个独立预测）或 `mixture`（2–5 种不同结构）。响应包含 `record_id`、`created_at`、`model_version`、`dataset_kind`、`notice`、`storage_notice`、`inference_ms` 和逐个分子的全部 24 项 `scores`。

混合物额外提供 `mixture.scores`、`baseline_scores`、`deltas`、`attention_matrix`、`context_scores` 与字段含义。`deltas` 等于最终分数减基线分数。`raw_interaction_logits` 保留用于研究对照，不能当作分数变化。

错误响应为 `{"error":{"code":"...","message":{"zh":"...","en":"..."},"issues":[...]},"record_id":"..."}`。解析预览错误没有记录编号；预测错误尽可能持久化并返回编号。超过传输大小的请求在中间件拒绝，不保存巨大原始正文。服务启动失败时没有可用预测接口。

兼容接口 `/v1/predict`、`/v2/predict-mixture` 及其 `/api` 前缀形式保留旧输入结构。旧接口为兼容性不增加强制确认字段，但所有响应及元数据都有保存说明，客户端应在提交前展示。`/v1` 支持旧 `include_attention`；InChI 按原项目占位能力返回明确的 501。接口文档在本地 `http://127.0.0.1:8000/docs`，Caddy 公网配置不转发 Swagger。

管理员使用 HTTP Basic 的 `Authorization` 请求头。浏览器界面将凭据仅存当前页面内存，退出或刷新后重新登录。服务器使用 HTTPS；本机回环地址仅用于开发。认证失败进行每服务实例的限速，未认证者不能读取、导出、备份或删除记录。
