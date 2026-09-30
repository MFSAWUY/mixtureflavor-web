import { useRef, useState } from 'react'
import { ArrowRight, Beaker, Check, ChevronDown, FileUp, FlaskConical, Info, LoaderCircle, Plus, ScanLine, Sparkles, Trash2, TriangleAlert, Upload, X } from 'lucide-react'
import { ApiError, api, errorText } from './api'
import { Results } from './Results'
import type { Example, Lang, Meta, Prediction, Preview } from './types'

export function Workspace({ lang, meta, examples, serviceError, initialExample, onRetry }: { lang: Lang; meta: Meta | null; examples: Example[]; serviceError: boolean; initialExample?: Example; onRetry: () => void }) {
  const t = (zh: string, en: string) => lang === 'zh' ? zh : en
  const [inputTab, setInputTab] = useState<'text' | 'file'>('text')
  const [text, setText] = useState(initialExample?.molecules.map(x => `${x.smiles}\t${x.name}`).join('\n') ?? '')
  const [mode, setMode] = useState<'individual' | 'mixture'>(initialExample?.mode === 'mixture' ? 'mixture' : 'individual')
  const [preview, setPreview] = useState<Preview | null>(null)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [acknowledged, setAcknowledged] = useState(false)
  const [result, setResult] = useState<Prediction | null>(null)
  const [busy, setBusy] = useState<'preview' | 'predict' | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [fileName, setFileName] = useState('')
  const [dragging, setDragging] = useState(false)
  const [visible, setVisible] = useState(6)
  const fileInput = useRef<HTMLInputElement>(null)
  const resultAnchor = useRef<HTMLDivElement>(null)
  const count = selected.size
  const modeValid = mode === 'mixture' ? count >= 2 && count <= 5 : count >= 1
  function invalidate() { setPreview(null); setSelected(new Set()); setResult(null); setError(null); setVisible(6) }
  function applyPreview(value: Preview) { setPreview(value); setSelected(new Set(value.molecules.map(m => m.id))); setVisible(6); setResult(null) }
  async function checkText(value = text) {
    if (!value.trim()) return
    setBusy('preview'); setError(null); setResult(null)
    try { applyPreview(await api<Preview>('/molecules/preview', { method: 'POST', body: JSON.stringify({ text: value }) })) }
    catch (err) { setError(err); setPreview(null); setSelected(new Set()) }
    finally { setBusy(null) }
  }
  function loadExample(example: Example) {
    if (busy) return
    const value = example.molecules.map(x => `${x.smiles}\t${x.name}`).join('\n')
    invalidate(); setText(value); setInputTab('text'); setFileName(''); setMode(example.mode === 'mixture' ? 'mixture' : 'individual')
    void checkText(value)
  }
  async function upload(file?: File) {
    if (!file || busy) return
    invalidate(); setFileName(file.name)
    if (file.size > 5 * 1024 * 1024) return setError(new ApiError({ error: { message: { zh: '文件不得超过 5 MB。', en: 'Files must not exceed 5 MB.' } } }, 413))
    if (!/\.(csv|sdf)$/i.test(file.name)) return setError(new ApiError({ error: { message: { zh: '仅支持 CSV 或 SDF 文件。', en: 'Only CSV and SDF files are supported.' } } }, 422))
    setBusy('preview')
    try {
      const data = new FormData(); data.append('file', file)
      applyPreview(await api<Preview>('/molecules/upload', { method: 'POST', body: data }))
    } catch (err) { setError(err) }
    finally { setBusy(null); if (fileInput.current) fileInput.current.value = '' }
  }
  async function predict() {
    if (!preview || !modeValid || !acknowledged || busy) return
    setBusy('predict'); setError(null); setResult(null)
    try {
      const molecules = preview.molecules.filter(x => selected.has(x.id)).map(x => ({ name: x.name, smiles: x.smiles }))
      const value = await api<Prediction>('/predictions', { method: 'POST', body: JSON.stringify({ mode: mode === 'mixture' ? 'mixture' : (count === 1 ? 'single' : 'batch'), molecules, acknowledge_storage: true }) })
      setResult(value)
      window.setTimeout(() => resultAnchor.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80)
    } catch (err) { setError(err) }
    finally { setBusy(null) }
  }
  const modeText = mode === 'mixture' ? t('混合物预测', 'Mixture prediction') : t('逐个分子预测', 'Individual predictions')
  return <main className="workspace page-container">
    <div className="page-intro"><div><span className="eyebrow">MOLECULES IN. INSIGHTS OUT.</span><h1>{t('探索你的风味分子', 'Explore your flavor molecules')}</h1><p>{t('从一个结构出发，理解多个分子相遇后的预测变化。', 'Start with a structure. Explore the predicted changes when molecules meet.')}</p></div><span className="outline-badge"><FlaskConical size={15} />{t('科研演示 · v1.0', 'RESEARCH DEMO · v1.0')}</span></div>
    {serviceError && <div className="alert warning" role="alert"><TriangleAlert size={18} /><div>{t('预测服务暂时未连接。确认后端已启动后重试。', 'The prediction service is offline. Start the backend, then retry.')}</div><button className="text-button" onClick={onRetry}>{t('重新连接', 'Reconnect')}</button></div>}
    <div className="workspace-grid">
      <aside className="input-column">
        <section className="surface input-card">
          <div className="step-heading"><span className="step-number">01</span><h2>{t('添加分子', 'Add molecules')}</h2><span className="muted small-text">SMILES / CSV / SDF</span></div>
          <div className="segmented"><button disabled={!!busy} className={inputTab === 'text' ? 'active' : ''} onClick={() => setInputTab('text')}><ScanLine size={15} />{t('输入 SMILES', 'Enter SMILES')}</button><button disabled={!!busy} className={inputTab === 'file' ? 'active' : ''} onClick={() => setInputTab('file')}><Upload size={15} />{t('上传文件', 'Upload file')}</button></div>
          {inputTab === 'text' ? <><label htmlFor="smiles-input" className="field-label">{t('每行一个分子', 'One molecule per line')}<span>{t('可在制表符后添加名称', 'Optional name after a tab')}</span></label><textarea id="smiles-input" disabled={!!busy} value={text} onChange={e => { setText(e.target.value); invalidate() }} placeholder={'CCO\tEthanol\nCC(=O)O\tAcetic acid'} spellCheck={false} rows={5} maxLength={220000} /><div className="input-help"><span>{t('最多 100 个分子 · 不支持浓度或配比输入', 'Up to 100 molecules · no concentration inputs')}</span>{text && <button disabled={!!busy} className="icon-button" title={t('清空输入', 'Clear input')} aria-label={t('清空输入', 'Clear input')} onClick={() => { setText(''); invalidate() }}><Trash2 size={14} /></button>}</div><button className="button secondary full" disabled={!!busy || !text.trim() || !meta} onClick={() => void checkText()}>{busy === 'preview' ? <LoaderCircle className="spin" size={16} /> : <ScanLine size={16} />}{t('检查并预览结构', 'Validate & preview structures')}</button></> : <><div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void upload(e.dataTransfer.files[0]) }}><FileUp size={30} /><strong>{t('将文件拖放到这里', 'Drop your file here')}</strong><span>{t('或选择本地 CSV / SDF 文件', 'or choose a local CSV / SDF file')}</span><button className="button secondary small" disabled={!!busy || !meta} onClick={() => fileInput.current?.click()}>{busy === 'preview' ? <LoaderCircle size={15} className="spin" /> : <Plus size={15} />}{t('选择文件', 'Choose file')}</button><small>{t('最大 5 MB · 最多 100 个分子', '5 MB maximum · up to 100 molecules')}</small><input ref={fileInput} type="file" accept=".csv,.sdf" hidden onChange={e => void upload(e.target.files?.[0])} /></div>{fileName && <div className="file-name"><FileUp size={14} /><span>{fileName}</span>{preview && <Check size={14} />}</div>}<div className="template-links"><a href="/template.csv" download>{t('下载 CSV 模板', 'CSV template')} ↗</a><a href="/example.sdf" download>{t('下载 SDF 示例', 'SDF example')} ↗</a></div><p className="input-help">{t('CSV 需要 smiles 列，可选 name 列。文件应使用 UTF-8 编码。', 'CSV requires a smiles column; name is optional. Use UTF-8 encoding.')}</p></>}
          <div className="example-section"><span className="field-label"><Sparkles size={13} />{t('先试一个示例', 'TRY AN EXAMPLE')}</span><div className="example-buttons">{examples.map(example => <button disabled={!!busy || !meta} key={example.id} onClick={() => loadExample(example)} title={example.description[lang]}>{example.id === 'single' ? t('单分子', 'Single') : example.id === 'mixture' ? t('两组分混合', 'Mixture') : t('批量比较', 'Batch')}<ArrowRight size={13} /></button>)}</div></div>
        </section>
        {preview && <section className="surface preview-card"><div className="section-row"><h3>{t('结构预览', 'Structure preview')} <span className="count-badge">{preview.count}</span></h3><button disabled={!!busy} className="text-button" onClick={() => { setSelected(selected.size === preview.count ? new Set() : new Set(preview.molecules.map(x => x.id))); setResult(null) }}>{selected.size === preview.count ? t('取消全选', 'Deselect all') : t('选择全部', 'Select all')}</button></div><div className="molecule-grid">{preview.molecules.slice(0, visible).map(m => <label className={`molecule-card ${selected.has(m.id) ? 'selected' : ''}`} key={m.id}><div className="molecule-card-top"><span>{m.id}</span><input aria-label={t('选择 ', 'Select ') + m.name} type="checkbox" checked={selected.has(m.id)} disabled={!!busy} onChange={() => { setSelected(old => { const s = new Set(old); s.has(m.id) ? s.delete(m.id) : s.add(m.id); return s }); setResult(null) }} /></div><img src={'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(m.svg)} alt={t('分子结构：', 'Molecular structure: ') + m.name} /><strong title={m.name}>{m.name}</strong><span className="molecule-formula">{m.formula} <span>· {m.molecular_weight}</span></span><code title={m.canonical_smiles}>{m.canonical_smiles}</code></label>)}</div>{preview.count > visible && <button className="text-button more-button" onClick={() => setVisible(x => x + 12)}>{t('显示更多分子', 'Show more molecules')}<ChevronDown size={14} /></button>}{preview.duplicates.length > 0 && <p className="inline-warning"><Info size={14} />{t('发现重复结构。逐个预测可保留；混合物预测需取消重复项。', 'Duplicate structures found. Deselect duplicates for mixture prediction.')}</p>}</section>}
        <section className="surface input-card"><div className="step-heading"><span className="step-number">02</span><h2>{t('选择预测方式', 'Choose a prediction mode')}</h2></div><div className="mode-options"><button disabled={!!busy} className={`mode-option ${mode === 'individual' ? 'chosen' : ''}`} onClick={() => { setMode('individual'); setResult(null); setError(null) }}><span className="radio-dot" /><span><strong>{t('分别预测每个分子', 'Predict each molecule')}</strong><small>{t('查看单个或多个分子的独立风味谱', 'Independent profiles for one or more molecules')}</small></span><Beaker size={19} /></button><button disabled={!!busy} className={`mode-option ${mode === 'mixture' ? 'chosen' : ''}`} onClick={() => { setMode('mixture'); setResult(null); setError(null) }}><span className="radio-dot" /><span><strong>{t('作为一个混合物预测', 'Predict as one mixture')}</strong><small>{t('选择 2–5 种分子，观察整体与变化', 'Select 2–5 distinct molecules and explore changes')}</small></span><FlaskConical size={19} /></button></div>{preview && !modeValid && <p className="inline-warning">{mode === 'mixture' ? t('请在结构预览中选择 2–5 种不同分子。', 'Select 2–5 distinct molecules in the preview.') : t('请至少选择一个分子。', 'Select at least one molecule.')}</p>}
          <div className="storage-disclosure"><label><input type="checkbox" checked={acknowledged} onChange={e => setAcknowledged(e.target.checked)} disabled={!!busy} /><span>{t('我已知悉本次提交将被保存', 'I understand this submission will be stored')}</span></label><p>{meta?.storage_notice[lang] ?? t('输入、结果、时间和模型版本将用于课题分析与复现。', 'Inputs, results, time and model version are stored for research analysis and reproducibility.')}{t(' 删除后的备份记录最多额外保留 7 个每日快照。', ' Deleted records may remain in the latest seven daily backups.')}</p></div>
          <button className="button primary full predict-button" disabled={!!busy || !preview || !modeValid || !acknowledged || !meta} onClick={() => void predict()}>{busy === 'predict' ? <LoaderCircle className="spin" size={18} /> : <Sparkles size={18} />}{busy === 'predict' ? t('正在预测…', 'Predicting…') : t('开始预测', 'Run prediction')}<ArrowRight size={18} /></button><div className="prediction-footnote">{preview ? `${count} ${t('个已选分子', 'selected')} · ${modeText}` : t('先添加分子并检查结构', 'Add molecules and validate their structures first')}</div>
        </section>
        {!!error && <div className="alert error" role="alert"><TriangleAlert size={18} /><div><strong>{errorText(error, lang)}</strong>{error instanceof ApiError && error.issues.map((issue, index) => <p key={index}>{issue.record ? `${t('记录', 'Record')} ${issue.record}: ` : ''}{typeof issue.message === 'string' ? issue.message : issue.message?.[lang]}{issue.same_as ? `${t('与记录相同', 'same as record')} ${issue.same_as}` : ''}{issue.location ? ' ' + issue.location.join(' → ') : ''}</p>)}{error instanceof ApiError && error.recordId && <small>ID: {error.recordId}</small>}</div><button className="icon-button" onClick={() => setError(null)} aria-label={t('关闭错误提示', 'Dismiss error')}><X size={14} /></button></div>}
      </aside>
      <div className="results-column" ref={resultAnchor}>
        {busy === 'predict' ? <div className="surface empty-state"><div className="empty-orbit loading-orbit"><FlaskConical size={37} /></div><h2>{t('正在计算分子风味', 'Predicting molecular flavors')}</h2><p>{t('正在构建分子图并运行模型。完成后，结果将显示在这里。', 'Building molecular graphs and running the model. Your results will appear here.')}</p><div className="loading-line" /></div> : result && meta ? <Results key={result.record_id} result={result} lang={lang} meta={meta} /> : <div className="surface empty-state"><div className="empty-orbit"><FlaskConical size={38} /><span className="orbit-dot one" /><span className="orbit-dot two" /><span className="orbit-dot three" /></div><span className="eyebrow">A LITTLE MOLECULE. A BIGGER PICTURE.</span><h2>{t('风味探索，从这里开始', 'Your flavor story starts here')}</h2><p>{t('添加一个或多个分子，查看预测风味，\n探索它们混合后的变化。', 'Add one or more molecules to discover predicted\nflavors and explore what changes in a mixture.')}</p><div className="empty-steps"><span><b>1</b>{t('添加分子', 'Add')}</span><i /><span><b>2</b>{t('检查结构', 'Validate')}</span><i /><span><b>3</b>{t('探索结果', 'Explore')}</span></div><div className="empty-divider" /><div className="empty-features"><span><Check size={14} />{t('24 类风味标签', '24 flavor labels')}</span><span><Check size={14} />{t('可下载预测结果', 'Downloadable results')}</span><span><Check size={14} />{t('混合物变化可视化', 'Visual mixture comparisons')}</span></div><div className="empty-note"><Info size={16} /><span>{t('这是一个合成数据研究演示，预测结果不等同于感官实测。', 'A synthetic-benchmark research demo. Predictions are not sensory measurements.')}</span></div></div>}
      </div>
    </div>
  </main>
}
