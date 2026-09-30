import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, RadarChart } from 'echarts/charts'
import { GridComponent, LegendComponent, MarkLineComponent, RadarComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Download } from 'lucide-react'
import { download } from './api'
import type { Lang } from './types'

echarts.use([BarChart, HeatmapChart, RadarChart, GridComponent, LegendComponent, MarkLineComponent, RadarComponent, TooltipComponent, VisualMapComponent, CanvasRenderer])

export function Chart({ option, height = 280, title, lang, filename, precomputed = false }: { option: echarts.EChartsCoreOption; height?: number; title: string; lang: Lang; filename: string; precomputed?: boolean }) {
  const element = useRef<HTMLDivElement>(null)
  const instance = useRef<echarts.EChartsType | null>(null)
  const latest = useRef(option)
  latest.current = option
  const caption = (precomputed ? (lang === 'zh' ? '预计算示例 · ' : 'Precomputed example · ') : '') + (lang === 'zh' ? '合成数据演示 · 分数不等同于实测风味强度' : 'Synthetic demo · scores are not measured intensities')
  useEffect(() => {
    if (!element.current) return
    function resize() {
      if (!element.current || element.current.clientWidth === 0) return
      if (!instance.current) {
        instance.current = echarts.init(element.current, undefined, { renderer: 'canvas' })
        instance.current.setOption({ animation: false, textStyle: { fontFamily: 'Segoe UI, Microsoft YaHei, sans-serif', fontSize: 12, color: '#56666a' }, ...latest.current }, true)
      }
      instance.current.resize()
    }
    const observer = new ResizeObserver(resize)
    observer.observe(element.current)
    resize()
    return () => { observer.disconnect(); instance.current?.dispose(); instance.current = null }
  }, [])
  useEffect(() => {
    instance.current?.setOption({ animation: false, textStyle: { fontFamily: 'Inter, Segoe UI, Microsoft YaHei, sans-serif', fontSize: 12, color: '#56666a' }, ...option }, true)
    instance.current?.resize()
  }, [option, height])
  async function save() {
    const data = instance.current?.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#ffffff' })
    if (!data) return
    const img = new Image()
    img.src = data
    await img.decode()
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(img.width, 1000)
    canvas.height = img.height + 210
    const context = canvas.getContext('2d')!
    context.fillStyle = '#fff'; context.fillRect(0, 0, canvas.width, canvas.height)
    context.fillStyle = '#183e35'; context.font = 'bold 26px sans-serif'
    context.fillText('MixtureFlavor · ' + title, 28, 42, canvas.width - 56)
    context.drawImage(img, (canvas.width - img.width) / 2, 64)
    context.font = '22px sans-serif'; context.fillStyle = '#56666a'
    context.fillText(caption, 28, canvas.height - 90, canvas.width - 56)
    context.font = '20px sans-serif'
    context.fillText(lang === 'zh' ? '变化 = 混合分数 − 内部基线；注意力不代表因果作用或竞争强度。' : 'Change = mixture score − internal baseline; attention is not causal or competition strength.', 28, canvas.height - 60, canvas.width - 56)
    context.font = '16px monospace'
    context.fillText(filename, 28, canvas.height - 28, canvas.width - 56)
    canvas.toBlob(blob => { if (blob) download(blob, filename + '.png') }, 'image/png')
  }
  return <div className="chart-frame"><div className="chart-title"><span>{title}</span><button className="icon-button" type="button" onClick={save} title={lang === 'zh' ? '下载此图（PNG）' : 'Download chart (PNG)'} aria-label={lang === 'zh' ? '下载此图（PNG）' : 'Download chart (PNG)'}><Download size={15} /></button></div><div ref={element} style={{ height }} role="img" aria-label={title} /><div className="chart-caption">{caption}</div></div>
}

export const tooltip = { trigger: 'axis', renderMode: 'richText', confine: true } as const
export const grid = { left: 8, right: 44, top: 25, bottom: 32, containLabel: true }
export const scoreAxis = { type: 'value', min: 0, max: 1, splitLine: { lineStyle: { color: '#edf0ed' } }, axisLabel: { formatter: (x: number) => x.toFixed(1) } }
