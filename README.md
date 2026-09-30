# MixtureFlavor · 分子风味研究平台

独立的科研展示网站：React + TypeScript + Vite + ECharts 前端，FastAPI + PyTorch + RDKit CPU 后端。原项目代码与权重已复制到 `vendor/flavorgnn/`，网站不依赖原目录运行，不向原目录写入文件。

**这是合成数据基准演示。模型分数不是实测风味强度，注意力不能证明竞争、抑制或受体机制。** 页面、数据导出及图片均保留这一说明。

## 现在就打开

本机启动后访问 **http://127.0.0.1:8000**。

在当前电脑的 PowerShell 中运行：

```powershell
cd D:\VSCode_data\Python\Mixture-fla\mixtureflavor-web
.\START_LOCAL.ps1
```

也可以直接运行，不需要更改 PowerShell 执行策略：

```powershell
& 'D:\anaconda3\envs\flavor\python.exe' scripts/run_local.py
```

默认只监听本机。保持启动终端运行，按 Ctrl+C 停止。端口已被占用时，先使用已启动的网站，或运行 `scripts/run_local.py --port 8001`。本次交付已包含前端构建产物和本地补充依赖，当前电脑可直接使用。

管理员入口在网页底部，用户名与随机密码位于 **`runtime/local.env`**。密码不会写在 README、源代码或部署包里。首次运行会自动生成，后续启动保留原值。管理员登录后可查看原始输入和结果、导出记录、逐条删除、手动备份。

## 展示流程

1. 打开「在线预测」，选择「单分子」「两组分混合」「批量比较」示例，或输入自己的 SMILES / 上传文件。
2. 检查二维结构及选择的组分。文本每行一个 SMILES，可在制表符后添加名称；CSV 要有 `smiles` 列，可选 `name` 列；SDF 每条记录一个分子。
3. 选择分别预测或混合预测，阅读并确认数据保存说明，点击「开始预测」。
4. 查看 Top 5 风味，展开全部 24 类；混合物还显示内部基线比较、正负分数变化和可折叠注意力图。
5. 下载 CSV / JSON；每个图表右上角可下载带研究说明、版本及记录编号的 PNG。

默认中文，右上角切换英文；支持桌面与手机布局。没有浓度或配比控件。

| 输入限制 | 数值 |
| --- | --- |
| CSV / SDF 文件 | ≤ 5 MiB（界面简写为 5 MB），CSV 使用 UTF-8 |
| 单分子 / 批量 | 1–100 个分子 |
| 混合物 | 2–5 种不同的规范结构 |
| 每个分子 | ≤ 128 个原子、SMILES ≤ 2000 字符 |
| 每次请求 | ≤ 4096 个原子 |

混合物拒绝 `CCO` 与 `OCC` 这类规范化后相同的重复项；批量独立预测保留重复项并提示。错误会定位到输入行或文件记录编号，不跳过非法分子，不改用宽松特征解析器。

## 结果含义与模型一致性

单分子使用独立的 FlavorGNN 权重；混合物使用完整的 MixtureFlavorNet 权重。模型在 CPU 上运行，默认 2 个计算线程，同时处理最多 2 个推理任务，其余请求短暂排队。

混合物变化量的定义：

```text
baseline_score = sigmoid(logsumexp(component_logits))
mixture_score  = sigmoid(baseline_logit + learned_scale × interaction_logit)
score_delta    = mixture_score − baseline_score
```

内部基线来自混合模型，不是独立单分子分数的平均值。旧接口的 `interaction_delta` 保留原含义（未缩放的 logit 修正项），网页使用新增的 `score_deltas` / `deltas`。注意力是最后一层多头平均权重，可以不对称，不表示正负作用强度。

`vendor/flavorgnn/model-manifest.json` 固定源代码、权重、配置、特征维数、RDKit 版本及参考结果。启动时执行 SHA-256 校验、严格权重加载、单分子与混合物预热对照；缺少或不兼容的权重会使服务启动失败。

当前参考输出由**复制的原始预测器**生成，保存在 `reference-predictions.json`，不来自网页适配器。页面方法页读取随包保存的真实测试指标并注明合成测试集，AUROC 不称为准确率。真实感官数据或竞争机制验证仍需后续研究。

## 文件组织

```text
mixtureflavor-web/
  frontend/              React 界面、图表、CSV/SDF 示例与已构建的 dist/
  backend/               新的 API、严格模型适配、上传校验、SQLite 与备份
  vendor/flavorgnn/       复制的源代码、配置、两套权重和测试指标
  examples/              原预测器实际生成的三组演示数据
  scripts/               启动、测试、基准、打包、恢复与原文件校验工具
  tests/                 模型一致性及 API 验收
  runtime/               本机密码、数据库、备份和测试临时文件；不打包
  deploy/                Dockerfile、Caddy 和详细部署手册
  docs/verification/     本地验收报告及性能数据
  compose.yaml           同机部署入口
```

## 环境安装与开发

在这台电脑上，原 `flavor` Conda 环境只用于运行已有 PyTorch / RDKit；新增 Web 依赖已安装到本目录 `.python-deps/`，没有升级原环境。需要重建补充依赖时：

```powershell
& 'D:\anaconda3\envs\flavor\python.exe' -m pip install --target .python-deps -r backend/requirements-local.txt -c backend/requirements-web.lock.txt
```

在全新电脑上，建议先创建本目录的虚拟环境。示例为 Linux / macOS；Windows 使用 `.venv\Scripts\python.exe` 代替 `.venv/bin/python`：

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r backend/requirements.txt -c backend/requirements-web.lock.txt
.venv/bin/python scripts/run_local.py
```

依赖已安装且使用随包 `frontend/dist/` 时，运行网站不需要联网。部署包不包含 Python、Conda、Node 或安装依赖，离线答辩前请先在目标电脑完成安装与演练。

修改前端后重新构建（Node 24）：

```sh
npm ci --prefix frontend
npm test --prefix frontend
npm run build --prefix frontend
```

开发时在两个终端分别运行 `python scripts/run_local.py` 和 `npm run dev --prefix frontend`，访问 http://127.0.0.1:5173。Vite 将 `/api` 转发到本地 8000 端口。

## 验证与重新打包

```powershell
& 'D:\anaconda3\envs\flavor\python.exe' scripts/test.py
& 'D:\anaconda3\envs\flavor\python.exe' scripts/benchmark.py
& 'D:\anaconda3\envs\flavor\python.exe' scripts/verify_source.py
& 'D:\anaconda3\envs\flavor\python.exe' scripts/package_release.py
```

新环境测试前安装 `backend/requirements-dev.txt`。`benchmark.py` 默认自行启动临时服务，用独立数据库发起 5 个并发、两组各 50 次 HTTP 请求，记录 API P95 和采样峰值 RSS；不会污染正式提交记录。`verify_source.py` 对照实施前快照检查相邻原目录中的 106 个文件，仅适用于保留原目录的本机。

部署压缩包位于 `release/`，包括真实权重、源码、前端构建和文档；排除密码、提交数据库、备份、缓存和本地依赖。压缩包旁提供 SHA-256 校验文件。重新打包前先重新构建前端。

## 服务器与其他托管方式

参见 [部署、备份与恢复手册](deploy/README.md) 和 [接口说明](docs/API.md)。首版配置面向 Linux x86-64 CPU、Docker Compose、同机同域名。当前没有购买服务器或域名，没有进行公网发布。

可将同一个前端构建放到静态托管平台，但动态预测和提交数据库仍需后端。当前浏览器请求使用同源 `/api`，所以 Vercel 等平台需要将该路径反向代理到模型服务；GitHub Pages 无原生同域代理，需要额外适配 API 地址与明确的 CORS 来源后才能在线预测。直接上传前端不能获得 Python 推理能力。学校服务器和本地演示均可使用本包。
