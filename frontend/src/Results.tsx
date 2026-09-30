import { useEffect, useState } from 'react'
import { ArrowDownRight, ArrowUpRight, Check, ChevronDown, Download, Info, Layers, ChartNoAxesCombined } from 'lucide-react'
import { Chart, grid, scoreAxis, tooltip } from './Charts'
import { predictionBlob } from './api'
import type { Lang, Meta, DisplayPrediction } from './types'

export function Results({ result, lang, meta }: { result: DisplayPrediction; lang: Lang; meta: Meta }) {
  const [exports, setExports] = useState({ csv: '', json: '' })
  useEffect(() => {
    const csv = URL.createObjectURL(predictionBlob(result, 'csv')), json = URL.createObjectURL(predictionBlob(result, 'json'))
    setExports({ csv, json })
    return () => { URL.revokeObjectURL(csv); URL.revokeObjectURL(json) }
  }, [result])
  const t = (zh: string, en: string) => lang === 'zh' ? zh : en
  const label = (id: string) => meta.labels.find(x => x.id === id)?.[lang] ?? id
  const [selected, setSelected] = useState(result.mixture ? 'mixture' : result.molecules[0].id)
  const [all, setAll] = useState(false)
  const [radar, setRadar] = useState(false)
  const active = selected === 'mixture' ? result.mixture!.scores : result.molecules.find(x => x.id === selected)!.scores
  const ranked = [...active].sort((a, b) => b.score - a.score)
  const visible = ranked.slice(0, all ? 24 : 5)
  const mix = result.mixture
  const changes = mix ? [...mix.deltas].sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, all ? 24 : 8) : []
  const maxUp = mix && [...mix.deltas].sort((a, b) => b.score - a.score)[0]
  const maxDown = mix && [...mix.deltas].sort((a, b) => a.score - b.score)[0]
  const precomputed = 'example_id' in result
  const resultId = precomputed ? `example-${result.example_id}` : result.record_id.slice(0, 8)
  const filename = `${result.model_version}-${resultId}`
  const title = selected === 'mixture' ? t('混合物风味谱', 'Mixture flavor profile') : (result.molecules.find(x => x.id === selected)?.name ?? '')
  const barOption = {
    tooltip, grid, xAxis: scoreAxis,
    yAxis: { type: 'category', inverse: true, data: visible.map(x => label(x.label)), axisLine: { show: false }, axisTick: { show: false } },
    series: [{ type: 'bar', data: visible.map(x => x.score), barWidth: 16, showBackground: true, backgroundStyle: { color: '#f2f5f2', borderRadius: 4 }, itemStyle: { color: '#237864', borderRadius: [0, 4, 4, 0] }, label: { show: true, position: 'right', formatter: (p: any) => p.value.toFixed(3), color: '#1b443a' } }],
  }
  const radarLabels = meta.labels.slice(0, 8)
  const radarOption = { tooltip: { renderMode: 'richText' }, radar: { indicator: radarLabels.map(x => ({ name: x[lang], max: 1 })), radius: '65%', axisName: { color: '#526361' } }, series: [{ type: 'radar', data: [{ value: radarLabels.map(l => active.find(x => x.label === l.id)?.score ?? 0), name: title, areaStyle: { color: 'rgba(35,120,100,.16)' }, lineStyle: { color: '#237864' }, itemStyle: { color: '#237864' } }] }] }
  return <section className="results" aria-live="polite">
    <div className="result-heading"><div><span className="eyebrow"><span className="success-dot"><Check size={11} /></span>{precomputed ? t('预计算示例 · 非实时预测', 'PRECOMPUTED EXAMPLE · NOT LIVE') : t('预测已完成', 'PREDICTION COMPLETE')}</span><h2>{t('读懂分子，也读懂混合。', 'A clearer picture of flavor.')}</h2></div><div className="result-export"><a className="button small secondary" href={exports.csv || undefined} download={`mixtureflavor-${resultId}.csv`}><Download size={14} /> CSV</a><a className="button small secondary" href={exports.json || undefined} download={`mixtureflavor-${resultId}.json`}>JSON</a></div></div>
    <div className="result-meta"><span>{result.molecules.length} {t('个分子', 'molecules')}</span>{!precomputed && <span>{t('推理', 'Inference')} {result.inference_ms.toFixed(1)} ms</span>}<span>{precomputed ? t('示例编号', 'Example ID') : 'ID'} {resultId}</span><span className="demo-tag">{t('合成数据演示', 'Synthetic demo')}</span></div>
    <div className="surface result-card">
      <div className="section-row"><div><h3>{t('主要风味预测', 'Predicted flavors')}</h3><p>{t('各标签独立评分，不要求总和为 1。', 'Each label is scored independently; scores do not sum to one.')}</p></div><select aria-label={t('选择查看的分子或混合物', 'Select molecule or mixture')} value={selected} onChange={e => setSelected(e.target.value)}>{mix && <option value="mixture">{t('混合物整体', 'Complete mixture')}</option>}{result.molecules.map(m => <option key={m.id} value={m.id}>{m.name} · {m.id}</option>)}</select></div>
      <div className="flavor-chips">{ranked.slice(0, 5).map((x, i) => <div className={`flavor-chip tone-${i}`} key={x.label}><span>{label(x.label)}</span><b>{x.score.toFixed(3)}</b></div>)}</div>
      <div className="chart-controls"><div className="segmented compact"><button className={!radar ? 'active' : ''} onClick={() => setRadar(false)}>{t('条形图', 'Bars')}</button><button className={radar ? 'active' : ''} onClick={() => setRadar(true)}>{t('雷达图', 'Radar')}</button></div><button className="text-button" onClick={() => setAll(!all)}>{all ? t('收起为 Top 5', 'Show top 5') : t('查看全部 24 类', 'Show all 24')}<ChevronDown size={14} className={all ? 'rotated' : ''} /></button></div>
      <Chart precomputed={precomputed} title={radar ? t('固定 8 类风味对照 · 模型分数', 'Eight fixed labels · model score') : title + t(' · 模型分数', ' · model score')} option={radar ? radarOption : barOption} height={radar ? 330 : (all ? 660 : 255)} lang={lang} filename={filename + '-profile'} />
      {radar && <p className="muted small-text">{t('雷达图固定显示甜、苦、酸、咸、鲜、果香、花香与辛香，完整结果见条形图。', 'Radar uses a fixed set of eight labels. Use the bars for the complete profile.')}</p>}
    </div>
    {mix && <div className="surface result-card">
      <div className="section-row"><div><span className="eyebrow">{t('混合物视角', 'THE MIXTURE PERSPECTIVE')}</span><h3>{t('混合后，哪些分数发生了变化？', 'What changes in the mixture?')}</h3></div><Layers size={22} className="muted" /></div>
      <p className="explainer">{t('Δ 分数 = 混合物预测 − 模型内部无交互项基线。正值为增强趋势，负值为减弱趋势；这不是感官实测或机制结论。', 'Δ score = mixture prediction − the model’s internal baseline without its interaction correction. Signs indicate predicted trends, not sensory measurements or a mechanistic finding.')}</p>
      <div className="change-highlights">{maxUp && maxUp.score > 0 && <div className="change-up"><ArrowUpRight size={19} /><span>{label(maxUp.label)}<small>{t('最大增强趋势', 'Largest positive change')}</small></span><b>+{maxUp.score.toFixed(3)}</b></div>}{maxDown && maxDown.score < 0 && <div className="change-down"><ArrowDownRight size={19} /><span>{label(maxDown.label)}<small>{t('最大减弱趋势', 'Largest negative change')}</small></span><b>{maxDown.score.toFixed(3)}</b></div>}</div>
      <Chart precomputed={precomputed} lang={lang} filename={filename + '-changes'} title={t('风味分数变化 · 按绝对变化排序', 'Score changes · ordered by magnitude')} height={all ? 660 : 300} option={{ tooltip, grid, xAxis: { type: 'value', min: -1, max: 1, splitLine: { lineStyle: { color: '#edf0ed' } } }, yAxis: { type: 'category', inverse: true, data: changes.map(x => label(x.label)), axisLine: { show: false }, axisTick: { show: false } }, series: [{ type: 'bar', barWidth: 14, data: changes.map(x => ({ value: x.score, itemStyle: { color: x.score >= 0 ? '#237864' : '#b98249' } })), label: { show: true, position: 'right', color: '#425752', formatter: (p: any) => (p.value > 0 ? '+' : '') + p.value.toFixed(3) }, markLine: { silent: true, symbol: 'none', data: [{ xAxis: 0 }], lineStyle: { color: '#a6b4ad', type: 'solid' }, label: { show: false } } }] }} />
      <details className="details"><summary><ChartNoAxesCombined size={16} />{t('并排比较内部基线与混合预测', 'Compare internal baseline and mixture prediction')}<ChevronDown size={15} /></summary><Chart precomputed={precomputed} lang={lang} filename={filename + '-baseline'} title={t('内部基线与混合预测 · 模型分数', 'Internal baseline and mixture prediction · scores')} height={all ? 1060 : 370} option={{ tooltip, grid: { ...grid, top: 42 }, legend: { top: 0, data: [t('内部基线', 'Internal baseline'), t('混合预测', 'Mixture')] }, xAxis: scoreAxis, yAxis: { type: 'category', inverse: true, data: changes.map(x => label(x.label)), axisLine: { show: false }, axisTick: { show: false } }, series: [{ type: 'bar', name: t('内部基线', 'Internal baseline'), barWidth: 9, itemStyle: { color: '#bbccc2' }, data: changes.map(d => mix.baseline_scores.find(x => x.label === d.label)!.score) }, { type: 'bar', name: t('混合预测', 'Mixture'), barWidth: 9, itemStyle: { color: '#237864' }, data: changes.map(d => mix.scores.find(x => x.label === d.label)!.score) }] }} /><p className="muted small-text">{t('内部基线来自混合模型自身，不是单分子预测分数的平均值。', 'The internal baseline is part of the mixture model, not the average of standalone predictions.')}</p></details>
    </div>}
    {mix && <details className="surface attention-details"><summary><Layers size={18} /><span>{t('模型关注了哪些组分关系？', 'Which component relationships received attention?')}</span><span className="subtle-tag">{t('辅助解释', 'EXPLORATORY')}</span><ChevronDown size={16} /></summary><div className="attention-content"><p className="explainer"><Info size={16} />{t('此图为最后一层平均注意力，可能不对称。颜色深浅不表示抑制、协同或受体竞争强度。', 'Last-layer mean attention can be asymmetric. Color does not encode inhibition, synergy, or receptor competition.')}</p><Chart precomputed={precomputed} lang={lang} filename={filename + '-attention'} title={t('组分注意力 · 行关注列', 'Component attention · row attends to column')} height={350} option={{ tooltip: { position: 'top', renderMode: 'richText' }, grid: { left: 60, right: 60, top: 24, bottom: 85 }, xAxis: { type: 'category', data: result.molecules.map(x => x.id), splitArea: { show: true } }, yAxis: { type: 'category', inverse: true, data: result.molecules.map(x => x.id), splitArea: { show: true } }, visualMap: { min: 0, max: 1, calculable: false, orient: 'horizontal', left: 'center', bottom: 10, inRange: { color: ['#f0f5ed', '#9bcaac', '#1b6955'] } }, series: [{ type: 'heatmap', data: mix.attention_matrix.flatMap((row, i) => row.map((v, j) => [j, i, v])), label: { show: true, formatter: (p: any) => p.value[2].toFixed(2) } }] }} /><div className="matrix-legend">{result.molecules.map(x => <span key={x.id}><b>{x.id}</b> {x.name}</span>)}</div></div></details>}
    <div className="science-note"><Info size={17} /><p>{precomputed && <>{result.provenance[lang]}<br /></>}{result.notice[lang]}<br /><small>{t('模型版本', 'Model version')}: {result.model_version}</small></p></div>
  </section>
}
