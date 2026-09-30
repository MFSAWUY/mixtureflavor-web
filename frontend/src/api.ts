import type { Lang, Localized, Prediction } from './types'

export class ApiError extends Error {
  messages: Localized
  issues: { record?: number; same_as?: number; message?: Localized | string; location?: (string | number)[] }[]
  recordId?: string
  constructor(payload: any, status: number) {
    const messages = payload?.error?.message ?? { zh: status === 401 ? '用户名或密码不正确。' : `请求失败（${status}），请稍后重试。`, en: status === 401 ? 'Incorrect username or password.' : `Request failed (${status}). Please retry.` }
    super(messages.en)
    this.messages = messages; this.issues = payload?.error?.issues ?? []; this.recordId = payload?.record_id
  }
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), 60000)
  try {
    const response = await fetch(`/api${path}`, { ...options, signal: controller.signal, headers: { ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options.headers } })
    const payload = await response.json().catch(() => null)
    if (!response.ok) throw new ApiError(payload, response.status)
    if (payload === null) throw new ApiError(null, 502)
    return payload as T
  } finally { clearTimeout(timeout) }
}

export function errorText(error: unknown, lang: Lang) {
  if (error instanceof ApiError) return error.messages[lang]
  return lang === 'zh' ? '无法连接预测服务，或请求已超时。请稍后重试。' : 'The prediction service is unavailable or the request timed out. Please retry.'
}

export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = filename; a.hidden = true
  document.body.append(a); a.click()
  window.setTimeout(() => { a.remove(); URL.revokeObjectURL(url) }, 1000)
}

function csvCell(value: unknown): string {
  let text = String(value ?? '')
  if (typeof value === 'string' && /^[=+\-@\t\r]/.test(text)) text = "'" + text
  return '"' + text.replaceAll('"', '""') + '"'
}

export function predictionBlob(result: Prediction, format: 'json' | 'csv'): Blob {
  if (format === 'json') return new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' })
  const rows: unknown[][] = [['record_id', 'created_at', 'model_version', 'dataset_kind', 'type', 'name', 'smiles', 'label', 'score', 'baseline_score', 'score_delta', 'notice']]
  const prefix = [result.record_id, result.created_at, result.model_version, result.dataset_kind]
  result.molecules.forEach(m => m.scores.forEach(s => rows.push([...prefix, 'single_molecule', m.name, m.smiles, s.label, s.score, '', '', result.notice.en])))
  if (result.mixture) {
    const mix = result.mixture
    mix.scores.forEach(s => rows.push([...prefix, 'mixture', 'Mixture', result.molecules.map(x => x.smiles).join(' | '), s.label, s.score, mix.baseline_scores.find(x => x.label === s.label)?.score, mix.deltas.find(x => x.label === s.label)?.score, result.notice.en]))
  }
  return new Blob(['\ufeff' + rows.map(row => row.map(csvCell).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8' })
}
