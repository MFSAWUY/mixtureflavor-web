import { useCallback, useEffect, useState } from 'react'
import { ArrowRight, BookOpen, ChevronRight, Globe2, Info, Leaf, Menu, Microscope, X } from 'lucide-react'
import { Methods, MoleculeIllustration } from './App'
import { Results } from './Results'
import type { Lang, ShowcaseData } from './types'
import './showcase.css'

function locationState() {
  const [page, example] = location.hash.slice(1).split('/')
  return { page: page === 'predict' || page === 'methods' ? page : 'home', example: example || 'mixture' }
}

export default function ShowcaseApp() {
  const [lang, setLang] = useState<Lang>(() => localStorage.getItem('mixtureflavor-lang') === 'en' ? 'en' : 'zh')
  const [route, setRoute] = useState(locationState)
  const [data, setData] = useState<ShowcaseData | null>(null)
  const [failed, setFailed] = useState(false)
  const [menu, setMenu] = useState(false)
  const t = (zh: string, en: string) => lang === 'zh' ? zh : en
  const load = useCallback(async () => {
    setFailed(false)
    try {
      const response = await fetch('/showcase-data.json')
      if (!response.ok) throw new Error('Example data unavailable')
      const payload = await response.json() as ShowcaseData
      if (payload.schema_version !== 1 || payload.examples.length !== 3) throw new Error('Unsupported example data')
      setData(payload)
    } catch { setFailed(true) }
  }, [])
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    const update = () => { setRoute(locationState()); setMenu(false); window.scrollTo(0, 0) }
    addEventListener('hashchange', update)
    return () => removeEventListener('hashchange', update)
  }, [])
  useEffect(() => {
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en'
    document.title = t('MixtureFlavor · 风味预测示例展示', 'MixtureFlavor · Precomputed flavor showcase')
    localStorage.setItem('mixtureflavor-lang', lang)
  }, [lang])
  const example = data?.examples.find(x => x.id === route.example) ?? data?.examples[0]
  return <div className="app-shell showcase-app">
    <a className="skip-link" href="#main-content" onClick={e => { e.preventDefault(); document.getElementById('main-content')?.focus() }}>{t('跳到正文', 'Skip to content')}</a>
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="#home" aria-label="MixtureFlavor home"><img src="/favicon.svg" alt="" /><span>Mixture<span>Flavor</span><small>{t('分子风味研究展示', 'MOLECULAR FLAVOR SHOWCASE')}</small></span></a>
      <nav className={menu ? 'nav open' : 'nav'} aria-label={t('主要导航', 'Main navigation')}>{(['home', 'predict', 'methods'] as const).map(page => <a key={page} href={'#' + page} className={route.page === page ? 'active' : ''} aria-current={route.page === page ? 'page' : undefined}>{page === 'home' ? t('项目介绍', 'Overview') : page === 'predict' ? t('示例探索', 'Explore examples') : t('方法与验证', 'Methods')}</a>)}</nav>
      <div className="header-actions"><span className="showcase-status"><Microscope size={15} />{t('预计算展示', 'PRECOMPUTED')}</span><button className="language-button" onClick={() => setLang(lang === 'zh' ? 'en' : 'zh')} aria-label={t('切换为英文', 'Switch to Chinese')}><Globe2 size={16} />{lang === 'zh' ? 'EN' : '中文'}</button><button className="icon-button menu-toggle" aria-label={t('展开导航', 'Toggle navigation')} aria-expanded={menu} onClick={() => setMenu(!menu)}>{menu ? <X size={22} /> : <Menu size={22} />}</button></div>
    </div></header>
    <div className="showcase-banner"><div className="page-container"><Info size={16} /><span>{t('预计算示例 · 非实时预测。本页不接收分子上传，也不保存用户预测记录。', 'Precomputed examples, not live predictions. This site accepts no molecular uploads and stores no prediction submissions.')}</span></div></div>
    <div id="main-content" tabIndex={-1}>
      {route.page === 'home' && <main className="home-page"><section className="hero page-container"><div className="hero-copy"><span className="hero-kicker"><span />FROM MOLECULES TO MIXTURES</span><h1>{t('每一个分子，', 'Every molecule.')}<br />{t('都有它的', 'A world of ')}<span className="hero-accent">{t('风味故事。', 'flavor.')}</span></h1><p className="hero-description">{t('连接分子结构与风味感知。通过真实模型预计算的示例，探索单分子的风味，以及它们混合后的预测变化。', 'Connect molecular structure with flavor through precomputed model examples. Explore molecular profiles and predicted changes in mixtures.')}</p><div className="hero-buttons"><a className="button primary large" href="#predict/mixture">{t('探索混合物示例', 'Explore a mixture')}<ArrowRight size={18} /></a><a className="button ghost large" href="#methods"><BookOpen size={17} />{t('了解研究方法', 'Explore the science')}</a></div><p className="muted small-text">{t('合成数据基准 · 三组真实模型示例 · 结果与图表可下载', 'Synthetic benchmark · Three model examples · Data and chart exports')}</p></div><MoleculeIllustration lang={lang} /></section>
      <section className="home-facts page-container"><div><b>24</b><span>{t('风味标签', 'Flavor labels')}<small>{t('味觉、香气与口腔感受', 'Taste, aroma & mouthfeel')}</small></span></div><div><b>03</b><span>{t('预计算示例', 'Precomputed examples')}<small>{t('单分子、批量与混合物', 'Single, batch & mixture')}</small></span></div><div><b>GNN</b><span>{t('结构驱动预测', 'Structure-based prediction')}<small>{t('从分子图到风味谱', 'From graphs to flavor profiles')}</small></span></div><div className="facts-note"><Microscope size={24} /><span>{t('合成数据研究演示', 'Synthetic-benchmark demo')}<small>{t('用于方法展示与交互探索', 'For methods & exploration')}</small></span></div></section>
      <section className="page-container showcase-home-examples"><div className="section-intro left"><span className="eyebrow">THREE WAYS TO EXPLORE</span><h2>{t('从单个结构，到整体风味', 'From a single structure to the whole mixture')}</h2><p>{t('选择一个示例，查看结构、风味分数和模型解释。', 'Choose an example to inspect structures, scores and model interpretation.')}</p></div><div className="showcase-example-grid">{data?.examples.map((item, i) => <a className="surface showcase-example-link" href={'#predict/' + item.id} key={item.id}><span className="eyebrow">0{i + 1}</span><h3>{item.name[lang]}</h3><p>{item.description[lang]}</p><span>{t('查看示例', 'View example')}<ArrowRight size={17} /></span></a>)}</div></section></main>}
      {(!data || failed) && <div className="page-container showcase-loading" role={failed ? 'alert' : 'status'}><p>{failed ? t('示例数据加载失败，请重试。', 'Could not load example data. Please retry.') : t('正在加载预计算示例…', 'Loading precomputed examples…')}</p>{failed && <button className="button secondary" onClick={() => void load()}>{t('重新加载', 'Try again')}</button>}</div>}
      {route.page === 'predict' && data && example && <main className="page-container showcase-workspace"><div className="page-intro"><div><span className="eyebrow">EXPLORE THE MODEL OUTPUT</span><h1>{t('从结构出发，探索风味。', 'Start with structure. Explore flavor.')}</h1><p>{t('选择已计算的示例，对比结果与混合变化。', 'Explore saved model examples and compare the predicted changes.')}</p></div><span className="outline-badge"><Microscope size={15} />{t('预计算示例', 'PRECOMPUTED EXAMPLES')}</span></div><div className="showcase-layout"><aside className="showcase-input"><section className="surface showcase-picker"><span className="eyebrow">01 / {t('选择示例', 'CHOOSE AN EXAMPLE')}</span><h2>{t('三个视角，一起探索', 'Three perspectives')}</h2><div className="showcase-options">{data.examples.map(item => <a key={item.id} href={'#predict/' + item.id} aria-current={item.id === example.id ? 'true' : undefined} className={item.id === example.id ? 'active' : ''}><span><b>{item.name[lang]}</b><small>{item.description[lang]}</small></span><ChevronRight size={17} /></a>)}</div><p className="muted small-text">{t('任意 SMILES、CSV / SDF 的上传与实时预测，请使用本地完整版。', 'For arbitrary SMILES, CSV / SDF uploads and live predictions, use the full local application.')}</p></section><section className="surface showcase-structures"><span className="eyebrow">02 / {t('分子结构', 'MOLECULAR STRUCTURES')}</span><h2>{example.description[lang]}</h2>{example.structures.map(molecule => <article className="showcase-molecule" key={molecule.id}><div className="section-row"><h3>{molecule.name}</h3><span className="subtle-tag">{molecule.id}</span></div><div className="showcase-svg" role="img" aria-label={molecule.name + t('的二维结构', ' 2D structure')} dangerouslySetInnerHTML={{ __html: molecule.svg }} /><div className="showcase-formula"><b>{molecule.formula}</b><span>{molecule.molecular_weight.toFixed(2)} g/mol</span></div><code>{molecule.smiles}</code></article>)}</section></aside><Results key={example.id} result={example.result} lang={lang} meta={data.meta} /></div></main>}
      {route.page === 'methods' && data && <><div className="page-container showcase-method-note"><p>{t('以下介绍完整模型的方法与适用范围。本展示版只提供三组预计算示例。', 'The following describes the full model and its scope. This showcase provides three precomputed examples only.')}</p></div><Methods lang={lang} meta={data.meta} onRetry={load} showcase /></>}
    </div>
    <footer className="site-footer"><div className="page-container footer-inner"><div><span className="footer-brand"><Leaf size={17} />MixtureFlavor</span><p>{t('从分子到混合物，探索风味的可能。', 'From molecules to mixtures. Explore the possibilities of flavor.')}</p></div><div className="footer-links"><span>{t('合成数据 · 预计算示例展示', 'Synthetic data · precomputed examples')}</span><a href="#methods">{t('方法与数据说明', 'Methods & data')}<ChevronRight size={12} /></a></div></div></footer>
  </div>
}
