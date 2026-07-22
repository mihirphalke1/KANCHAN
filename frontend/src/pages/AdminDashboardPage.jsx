import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BarChart3, Clock, Building2, AlertTriangle, ShieldAlert, Copy,
  Users, Download, RefreshCw, ChevronRight, CheckCircle2, XCircle,
} from 'lucide-react'
import MobileTabBar from '@/components/MobileTabBar'
import { apiFetch, getEvaluator } from '@/lib/auth'
import styles from './AdminDashboardPage.module.css'

export default function AdminDashboardPage() {
  const navigate = useNavigate()
  const evaluator = getEvaluator()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [branchFilter, setBranchFilter] = useState('')
  const [regionFilter, setRegionFilter] = useState('')

  const isFullAccess = evaluator?.role === 'admin'

  const load = async (branch_id, region_id) => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      if (branch_id) params.set('branch_id', branch_id)
      if (region_id) params.set('region_id', region_id)
      const res = await apiFetch(`/api/admin/overview?${params.toString()}`)
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Could not load the admin dashboard')
      }
      setData(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const regions = useMemo(() => {
    if (!data) return []
    const seen = new Map()
    Object.values(data.available_branches || {}).forEach(b => {
      if (b.region_id && !seen.has(b.region_id)) seen.set(b.region_id, b.region_name)
    })
    return Array.from(seen.entries())
  }, [data])

  const handleRegionChange = (val) => {
    setRegionFilter(val)
    setBranchFilter('')
    load(null, val || null)
  }
  const handleBranchChange = (val) => {
    setBranchFilter(val)
    load(val || null, null)
  }

  const exportCsv = async (view) => {
    const params = new URLSearchParams({ view })
    if (branchFilter) params.set('branch_id', branchFilter)
    if (regionFilter) params.set('region_id', regionFilter)
    const res = await apiFetch(`/api/admin/export?${params.toString()}`)
    if (!res.ok) return
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `kanchan_admin_${view}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const goToHistory = (params) => {
    const qs = new URLSearchParams(params).toString()
    navigate(`/history?${qs}`)
  }

  return (
    <div className={styles.page}>
      <header className={styles.topbar}>
        <button className={styles.brand} onClick={() => navigate('/')} title="Go to home page">
          <div className={styles.brandMark}><img src="/logo.png" alt="KANCHAN-AI logo" /></div>
          <div>
            <div className={styles.brandName}>KANCHAN<span className={styles.brandAi}>-AI</span></div>
            <div className={styles.brandSub}>Regional Admin Dashboard</div>
          </div>
        </button>
        <nav className={styles.topNav}>
          <button className={styles.navItem} onClick={() => navigate('/dashboard')}>
            <BarChart3 size={16} strokeWidth={1.75} /><span>Analyse</span>
          </button>
          <button className={styles.navItem} onClick={() => navigate('/history')}>
            <Clock size={16} strokeWidth={1.75} /><span>Case History</span>
          </button>
          <button className={`${styles.navItem} ${styles.navActive}`}>
            <ShieldAlert size={16} strokeWidth={1.75} /><span>Admin</span>
          </button>
        </nav>
      </header>

      <main className={styles.main}>
        <div className={styles.scopeBar}>
          <div className={styles.scopeInfo}>
            <Building2 size={14} />
            {data?.scope?.restricted_to_own_branch
              ? <span>Scoped to your branch — <strong>{data.scope.branch_ids[0]}</strong></span>
              : <span>Regional admin — all branches unless filtered</span>}
          </div>
          {isFullAccess && (
            <div className={styles.filters}>
              <select className={styles.filterSelect} value={regionFilter} onChange={e => handleRegionChange(e.target.value)}>
                <option value="">All regions</option>
                {regions.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
              </select>
              <select className={styles.filterSelect} value={branchFilter} onChange={e => handleBranchChange(e.target.value)}>
                <option value="">All branches</option>
                {Object.entries(data?.available_branches || {}).map(([id, b]) => (
                  <option key={id} value={id}>{b.name || id}</option>
                ))}
              </select>
            </div>
          )}
          <button className={styles.refreshBtn} onClick={() => load(branchFilter || null, regionFilter || null)} disabled={loading}>
            <RefreshCw size={13} className={loading ? styles.spin : ''} /> Refresh
          </button>
        </div>

        {loading && <div className={styles.loadingBlock}><div className={styles.spinner} /> Loading dashboard…</div>}
        {error && <div className={styles.errorBlock}><AlertTriangle size={16} /> {error}</div>}

        {!loading && !error && data && (
          <>
            <KpiRow data={data} />
            <VerdictSection data={data} onDrill={goToHistory} onExport={() => exportCsv('verdicts')} />
            <BenfordSection data={data} onExport={() => exportCsv('benford')} />
            <ContradictionSection data={data} onDrill={goToHistory} onExport={() => exportCsv('contradiction')} />
            <DuplicateSection data={data} onDrill={goToHistory} onExport={() => exportCsv('duplicates')} />
            <OfficerSection data={data} onDrill={goToHistory} onExport={() => exportCsv('officers')} />
          </>
        )}
      </main>
      <MobileTabBar />
    </div>
  )
}

function SectionHeader({ icon: Icon, title, sub, onExport }) {
  return (
    <div className={styles.sectionHeader}>
      <div className={styles.sectionTitle}><Icon size={15} /> {title}</div>
      <div className={styles.sectionRight}>
        {sub && <span className={styles.sectionSub}>{sub}</span>}
        {onExport && (
          <button className={styles.exportBtn} onClick={onExport} title="Export this view as CSV">
            <Download size={12} /> CSV
          </button>
        )}
      </div>
    </div>
  )
}

function KpiRow({ data }) {
  const totals = data.verdict_distribution.reduce((acc, r) => ({
    genuine: acc.genuine + r.genuine, borderline: acc.borderline + r.borderline, reject: acc.reject + r.reject,
  }), { genuine: 0, borderline: 0, reject: 0 })
  const alertCount = data.benford_alerts.filter(a => a.alert).length
  const crossBranchDupes = data.duplicate_feed.filter(d => d.cross_branch).length

  return (
    <div className={styles.kpiRow}>
      <KpiTile label="Total cases" value={data.total_cases_in_scope} cls="total" />
      <KpiTile label="Genuine" value={totals.genuine} cls="genuine" />
      <KpiTile label="Borderline" value={totals.borderline} cls="borderline" />
      <KpiTile label="Reject" value={totals.reject} cls="reject" />
      <KpiTile label="Benford alerts" value={alertCount} cls={alertCount ? 'reject' : 'total'} />
      <KpiTile label="Cross-branch dupes" value={crossBranchDupes} cls={crossBranchDupes ? 'borderline' : 'total'} />
    </div>
  )
}

function KpiTile({ label, value, cls }) {
  return (
    <div className={`${styles.kpiTile} ${styles[`kpi_${cls}`]}`}>
      <div className={styles.kpiValue}>{value}</div>
      <div className={styles.kpiLabel}>{label}</div>
    </div>
  )
}

function VerdictSection({ data, onDrill, onExport }) {
  const rows = data.verdict_distribution
  const max = Math.max(1, ...rows.map(r => r.total))
  return (
    <section className={styles.card}>
      <SectionHeader icon={BarChart3} title="Verdict distribution by branch" onExport={onExport} />
      {rows.length === 0 && <p className={styles.empty}>No cases in scope.</p>}
      <div className={styles.barList}>
        {rows.map(r => (
          <div key={r.branch_id} className={styles.barRow}>
            <button className={styles.barRowLabel} onClick={() => onDrill({ branch: r.branch_id })} title="View cases">
              {r.branch_name || r.branch_id} <span className={styles.barRowTotal}>{r.total}</span>
            </button>
            <div className={styles.barTrack}>
            <div className={styles.stackedBar} style={{ width: `${(r.total / max) * 100}%` }}>
              {r.genuine > 0 && (
                <button className={`${styles.seg} ${styles.segGenuine}`} style={{ flex: r.genuine }}
                        onClick={() => onDrill({ branch: r.branch_id, verdict: 'GENUINE' })} title={`${r.genuine} genuine`} />
              )}
              {r.borderline > 0 && (
                <button className={`${styles.seg} ${styles.segBorderline}`} style={{ flex: r.borderline }}
                        onClick={() => onDrill({ branch: r.branch_id, verdict: 'BORDERLINE' })} title={`${r.borderline} borderline`} />
              )}
              {r.reject > 0 && (
                <button className={`${styles.seg} ${styles.segReject}`} style={{ flex: r.reject }}
                        onClick={() => onDrill({ branch: r.branch_id, verdict: 'REJECT' })} title={`${r.reject} reject`} />
              )}
            </div>
            </div>
          </div>
        ))}
      </div>
      <div className={styles.legend}>
        <LegendItem cls="segGenuine" label="Genuine" />
        <LegendItem cls="segBorderline" label="Borderline" />
        <LegendItem cls="segReject" label="Reject" />
      </div>
    </section>
  )
}

function LegendItem({ cls, label }) {
  return (
    <div className={styles.legendItem}>
      <span className={`${styles.legendDot} ${styles[cls]}`} />
      {label}
    </div>
  )
}

function BenfordSection({ data, onExport }) {
  const alerts = data.benford_alerts
  return (
    <section className={styles.card}>
      <SectionHeader icon={AlertTriangle} title="Benford's Law alerts"
                     sub={`${alerts.filter(a => a.alert).length} active`} onExport={onExport} />
      {alerts.length === 0 && <p className={styles.empty}>No branch or evaluator has enough samples yet.</p>}
      <div className={styles.alertList}>
        {alerts.map((a, i) => (
          <div key={i} className={`${styles.alertRow} ${a.alert ? styles.alertActive : styles.alertOk}`}>
            <div className={styles.alertIcon}>
              {a.alert ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
            </div>
            <div className={styles.alertBody}>
              <div className={styles.alertTitle}>
                {a.scope_type === 'branch' ? `Branch ${a.branch_id}` : `Evaluator ${a.evaluator_id}`}
              </div>
              <div className={styles.alertMsg}>{a.message}</div>
            </div>
            <div className={styles.alertStat}>n={a.n_samples}<br />p={a.p_value}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function ContradictionSection({ data, onDrill, onExport }) {
  const rows = data.contradiction_rate
  return (
    <section className={styles.card}>
      <SectionHeader icon={ShieldAlert} title="Contradiction-flag rate by branch" onExport={onExport} />
      {rows.length === 0 && <p className={styles.empty}>No cases in scope.</p>}
      <div className={styles.barList}>
        {rows.map(r => (
          <div key={r.branch_id} className={styles.barRow}>
            <button className={styles.barRowLabel} onClick={() => onDrill({ branch: r.branch_id })}>
              {r.branch_name || r.branch_id} <span className={styles.barRowTotal}>{Math.round(r.rate * 100)}%</span>
            </button>
            <div className={styles.rateTrack}>
              <div className={styles.rateFill} style={{ width: `${r.rate * 100}%` }} />
            </div>
            <span className={styles.rateSub}>{r.flagged}/{r.total}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function DuplicateSection({ data, onDrill, onExport }) {
  const rows = data.duplicate_feed
  return (
    <section className={styles.card}>
      <SectionHeader icon={Copy} title="Cross-branch photo duplicate feed"
                     sub={`${rows.filter(r => r.cross_branch).length} cross-branch`} onExport={onExport} />
      {rows.length === 0 && <p className={styles.empty}>No duplicate or reused photos detected.</p>}
      <div className={styles.dupeList}>
        {rows.slice(0, 20).map((r, i) => (
          <div key={i} className={`${styles.dupeRow} ${r.cross_branch ? styles.dupeCross : ''}`}>
            <div className={styles.dupeBranches}>
              {r.branches.map(b => <span key={b} className={styles.dupeBranchChip}>{b}</span>)}
            </div>
            <div className={styles.dupeCases}>
              {r.case_ids.slice(0, 6).map(cid => (
                <button key={cid} className={styles.dupeCaseChip}
                        onClick={() => onDrill({ case: cid })} title="Open case">
                  #{cid}
                </button>
              ))}
              {r.case_ids.length > 6 && <span className={styles.dupeMore}>+{r.case_ids.length - 6} more</span>}
            </div>
            <div className={styles.dupeMeta}>{r.occurrences}× · {r.image_roles.join(', ')}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function OfficerSection({ data, onDrill, onExport }) {
  const rows = data.officer_activity
  return (
    <section className={styles.card}>
      <SectionHeader icon={Users} title="Officer activity" onExport={onExport} />
      {rows.length === 0 && <p className={styles.empty}>No cases in scope.</p>}
      <div className={styles.officerTable}>
        <div className={styles.officerHeaderRow}>
          <span>Officer</span><span>Cases</span><span>Genuine</span><span>Border.</span>
          <span>Reject</span><span>Pending 2nd sign</span><span>Override rate</span>
        </div>
        {rows.map(r => (
          <button key={r.evaluator_id} className={styles.officerRow}
                  onClick={() => r.branch_id && onDrill({ branch: r.branch_id })}>
            <span className={styles.officerName}>{r.name}<span className={styles.officerId}>{r.evaluator_id}</span></span>
            <span>{r.total}</span>
            <span className={styles.chipGenuine}>{r.genuine}</span>
            <span className={styles.chipBorderline}>{r.borderline}</span>
            <span className={styles.chipReject}>{r.reject}</span>
            <span>{r.pending_checker > 0 ? <><Clock size={11} /> {r.pending_checker}</> : '—'}</span>
            <span className={r.override_rate > 0.2 ? styles.chipReject : ''}>
              {r.signed_off > 0 ? `${Math.round(r.override_rate * 100)}%` : '—'}
            </span>
            <ChevronRight size={13} className={styles.officerChevron} />
          </button>
        ))}
      </div>
    </section>
  )
}
