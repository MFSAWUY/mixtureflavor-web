import { useState } from 'react'
import { ArrowLeft, ArrowRight, Database, Download, LoaderCircle, LockKeyhole, LogOut, RefreshCw, ShieldCheck, Trash2, X } from 'lucide-react'
import { api, download, errorText } from './api'
import type { Lang, RecordItem } from './types'

export function Admin({ lang }: { lang: Lang }) {
  const t = (zh: string, en: string) => lang === 'zh' ? zh : en
  const [user, setUser] = useState('admin'), [password, setPassword] = useState('')
  const [auth, setAuth] = useState(''), [records, setRecords] = useState<RecordItem[]>([])
  const [total, setTotal] = useState(0), [offset, setOffset] = useState(0), [busy, setBusy] = useState(false)
  const [error, setError] = useState(''), [message, setMessage] = useState('')
  const [detail, setDetail] = useState<RecordItem | null>(null), [deleteId, setDeleteId] = useState<string | null>(null)
  const header = (token = auth) => ({ Authorization: token })
  async function load(token = auth, page = offset) {
    setBusy(true); setError('')
    try {
      const data = await api<{ records: RecordItem[]; total: number }>(`/admin/records?limit=25&offset=${page}`, { headers: header(token) })
      setRecords(data.records); setTotal(data.total); setOffset(page); setAuth(token); setPassword('')
    } catch (err) { setError(errorText(err, lang)) }
    finally { setBusy(false) }
  }
  function login(e: React.FormEvent) {
    e.preventDefault()
    const bytes = new TextEncoder().encode(`${user}:${password}`)
    void load('Basic ' + btoa(String.fromCharCode(...bytes)), 0)
  }
  async function exportAll(format: 'json' | 'csv') {
    setBusy(true); setError('')
    try {
      const r = await fetch(`/api/admin/export?format=${format}`, { headers: header() })
      if (!r.ok) throw Error('Export failed')
      download(await r.blob(), `mixtureflavor-records.${format}`)
    } catch { setError(t('导出失败，请重新登录后重试。', 'Export failed. Sign in again and retry.')) }
    finally { setBusy(false) }
  }
  async function show(id: string) {
    try { setDetail(await api<RecordItem>(`/admin/records/${id}`, { headers: header() })) }
    catch (err) { setError(errorText(err, lang)) }
  }
  async function remove() {
    if (!deleteId) return
    setBusy(true)
    try { await api(`/admin/records/${deleteId}`, { method: 'DELETE', headers: header() }); setDeleteId(null); setDetail(null); await load(auth, 0) }
    catch (err) { setError(errorText(err, lang)) }
    finally { setBusy(false) }
  }
  async function backup() {
    setBusy(true); setMessage(''); setError('')
    try { const value = await api<{ backup: string }>('/admin/backup', { method: 'POST', headers: header() }); setMessage(t('备份完成：', 'Backup created: ') + value.backup) }
    catch (err) { setError(errorText(err, lang)) }
    finally { setBusy(false) }
  }
  if (!auth) return <main className="page-container admin-page"><div className="surface login-card"><div className="login-icon"><LockKeyhole size={25} /></div><span className="eyebrow">PRIVATE RESEARCH RECORDS</span><h1>{t('管理员登录', 'Administrator sign-in')}</h1><p>{t('查看、导出和管理课题预测记录。', 'View, export, and manage research prediction records.')}</p><form onSubmit={login}><label htmlFor="admin-user">{t('用户名', 'Username')}</label><input id="admin-user" autoComplete="username" value={user} onChange={e => setUser(e.target.value)} required /><label htmlFor="admin-password">{t('密码', 'Password')}</label><input id="admin-password" autoComplete="current-password" type="password" value={password} onChange={e => setPassword(e.target.value)} required /><button className="button primary full" disabled={busy}>{busy && <LoaderCircle size={16} className="spin" />}{t('登录', 'Sign in')}<ArrowRight size={16} /></button></form>{error && <div className="alert error" role="alert">{error}</div>}<small>{t('登录凭据仅保留在当前页面内存中。', 'Credentials remain only in this page’s memory.')}</small></div></main>
  return <main className="page-container admin-page"><div className="page-intro"><div><span className="eyebrow">RESEARCH ADMINISTRATION</span><h1>{t('预测记录', 'Prediction records')}</h1><p>{t('已保存的课题提交，默认仅管理员可见。', 'Stored research submissions, visible only to administrators.')}</p></div><button className="button secondary" onClick={() => { setAuth(''); setRecords([]); setDetail(null); setDeleteId(null); setError(''); setMessage('') }}><LogOut size={16} />{t('退出登录', 'Sign out')}</button></div><div className="admin-toolbar"><span><Database size={18} /><strong>{total}</strong>{t('条记录', 'records')}</span><div><button className="button small secondary" disabled={busy} onClick={() => void load()}><RefreshCw size={14} />{t('刷新', 'Refresh')}</button><button className="button small secondary" disabled={busy} onClick={() => void backup()}><ShieldCheck size={15} />{t('立即备份', 'Backup now')}</button><button className="button small secondary" disabled={busy} onClick={() => void exportAll('csv')}><Download size={14} />CSV</button><button className="button small secondary" disabled={busy} onClick={() => void exportAll('json')}>JSON</button></div></div>{error && <div className="alert error" role="alert">{error}</div>}{message && <div className="alert success" role="status">{message}</div>}<div className="surface table-wrap"><table><thead><tr><th>{t('记录编号', 'Record ID')}</th><th>{t('提交时间', 'Submitted')}</th><th>{t('模式', 'Mode')}</th><th>{t('状态', 'Status')}</th><th>{t('操作', 'Actions')}</th></tr></thead><tbody>{records.length === 0 && <tr><td colSpan={5} className="table-empty">{t('尚无提交记录。', 'No submissions yet.')}</td></tr>}{records.map(r => <tr key={r.id}><td><button className="record-link" onClick={() => void show(r.id)}>{r.id.slice(0, 8)} ↗</button></td><td>{new Date(r.created_at).toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-GB')}</td><td>{r.mode === 'mixture' ? t('混合物', 'Mixture') : r.mode === 'single' ? t('单分子', 'Single') : r.mode === 'batch' ? t('批量', 'Batch') : t('格式无效', 'Invalid')}</td><td><span className={`status-tag ${r.status}`}>{r.status === 'success' ? t('成功', 'Success') : r.status === 'failed' ? t('失败', 'Failed') : r.status === 'interrupted' ? t('已中断', 'Interrupted') : t('处理中', 'Running')}</span></td><td><button className="icon-button danger" onClick={() => setDeleteId(r.id)} title={t('删除记录', 'Delete record')} aria-label={t('删除记录 ', 'Delete record ') + r.id.slice(0, 8)}><Trash2 size={15} /></button></td></tr>)}</tbody></table></div><div className="pagination"><button className="button small secondary" disabled={busy || offset === 0} onClick={() => void load(auth, Math.max(0, offset - 25))}><ArrowLeft size={14} />{t('上一页', 'Previous')}</button><span>{total ? `${offset + 1}–${Math.min(offset + 25, total)} / ${total}` : '0 / 0'}</span><button className="button small secondary" disabled={busy || offset + 25 >= total} onClick={() => void load(auth, offset + 25)}>{t('下一页', 'Next')}<ArrowRight size={14} /></button></div><p className="muted small-text">{t('每天自动备份，保留最近 7 个快照。删除的记录可能仍存在于轮换到期前的备份中。恢复操作请按部署文档在停止服务后执行。', 'Backups run daily and keep the latest seven snapshots. Deleted records may remain in older snapshots until rotation. Stop the service before restoring a backup as described in the deployment guide.')}</p>
    {detail && <div className="modal-backdrop" onClick={() => setDetail(null)}><section className="surface modal large-modal" role="dialog" aria-modal="true" aria-label={t('记录详情', 'Record details')} onClick={e => e.stopPropagation()}><div className="section-row"><h2>{t('记录详情', 'Record details')}</h2><button className="icon-button" autoFocus aria-label={t('关闭', 'Close')} onClick={() => setDetail(null)}><X size={20} /></button></div><pre>{JSON.stringify(detail, null, 2)}</pre></section></div>}
    {deleteId && <div className="modal-backdrop"><section className="surface modal" role="dialog" aria-modal="true" aria-label={t('确认删除', 'Confirm deletion')}><div className="login-icon"><Trash2 size={24} /></div><h2>{t('删除这条预测记录？', 'Delete this prediction record?')}</h2><p>{t('这会从当前数据库永久删除记录。旧备份将在轮换到期后移除。', 'This permanently deletes the record from the live database. Older backup copies expire with rotation.')}</p><code>{deleteId}</code><div className="modal-actions"><button className="button secondary" autoFocus disabled={busy} onClick={() => setDeleteId(null)}>{t('取消', 'Cancel')}</button><button className="button danger-button" disabled={busy} onClick={() => void remove()}>{t('确认删除', 'Delete record')}</button></div></section></div>}
  </main>
}
