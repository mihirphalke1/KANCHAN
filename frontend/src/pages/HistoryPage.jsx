import { useEffect, useState, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Search, CheckCircle2, AlertTriangle, XCircle,
  Clock, ChevronRight, BarChart3, X, Filter,
  Calendar, Tag, Building2, SlidersHorizontal
} from 'lucide-react'
import MobileTabBar from '@/components/MobileTabBar'
import VerdictCard from '@/components/VerdictCard'
import CustomerInfoCard from '@/components/CustomerInfoCard'
import ProcessTrace from '@/components/ProcessTrace'
import MediaEvidence from '@/components/MediaEvidence'
import GoldVsGemsCard from '@/components/GoldVsGemsCard'
import XRayView from '@/components/XRayView'
import SignalBars from '@/components/SignalBars'
import ContradictionAlert from '@/components/ContradictionAlert'
import DensityDetails from '@/components/DensityDetails'
import CompositionCard from '@/components/CompositionCard'
import BenfordStatus from '@/components/BenfordStatus'
import SHAPBreakdown from '@/components/SHAPBreakdown'
import styles from './HistoryPage.module.css'

const VERDICT_META = {
  GENUINE:    { icon: CheckCircle2, cls: 'genuine',    label: 'Genuine'    },
  BORDERLINE: { icon: AlertTriangle, cls: 'borderline', label: 'Borderline' },
  REJECT:     { icon: XCircle,      cls: 'reject',     label: 'Reject'     },
}
const KARAT_OPTIONS   = ['All', '14', '18', '22', '24']
const VERDICT_OPTIONS = ['All', 'GENUINE', 'BORDERLINE', 'REJECT']

export default function HistoryPage() {
  const navigate = useNavigate()
  const [cases, setCases]       = useState([])
  const [total, setTotal]       = useState(0)
  const [loading, setLoading]   = useState(true)
  const [selected, setSelected] = useState(null)

  const [search, setSearch]   = useState('')
  const [verdict, setVerdict] = useState('All')
  const [karat, setKarat]     = useState('All')
  const [branch, setBranch]   = useState('All')
  const [sortDir, setSortDir] = useState('desc')

  useEffect(() => {
    fetch('/api/history?limit=500')
      .then(r => r.json())
      .then(d => {
        const all = d.cases || []
        setCases(all)
        setTotal(d.total || 0)
        if (all.length > 0) setSelected(all[0])
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const branches = useMemo(() => {
    const s = new Set(cases.map(c => c.branch_id).filter(Boolean))
    return ['All', ...Array.from(s).sort()]
  }, [cases])

  const counts = useMemo(() => ({
    genuine:    cases.filter(c => c.verdict?.risk_level === 'GENUINE').length,
    borderline: cases.filter(c => c.verdict?.risk_level === 'BORDERLINE').length,
    reject:     cases.filter(c => c.verdict?.risk_level === 'REJECT').length,
  }), [cases])

  const filtered = useMemo(() => {
    let list = [...cases]
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(c =>
        c.item_description?.toLowerCase().includes(q) ||
        c.case_id?.toLowerCase().includes(q) ||
        c.branch_id?.toLowerCase().includes(q)
      )
    }
    if (verdict !== 'All') list = list.filter(c => c.verdict?.risk_level === verdict)
    if (karat   !== 'All') list = list.filter(c => String(c.declared_karat) === karat)
    if (branch  !== 'All') list = list.filter(c => c.branch_id === branch)
    list.sort((a, b) => {
      const ta = a.timestamp || '', tb = b.timestamp || ''
      return sortDir === 'desc' ? tb.localeCompare(ta) : ta.localeCompare(tb)
    })
    return list
  }, [cases, search, verdict, karat, branch, sortDir])

  const clearFilters = () => { setSearch(''); setVerdict('All'); setKarat('All'); setBranch('All') }
  const hasFilters   = search || verdict !== 'All' || karat !== 'All' || branch !== 'All'

  return (
    <div className={styles.page}>

      {/* ── Main ── */}
      <div className={styles.main}>

        {/* ── Top navbar ── */}
        <header className={styles.topbar}>
          <button className={styles.topbarLeft} onClick={() => navigate('/')} title="Go to home page">
            <div className={styles.brandMark}><img src="/logo.png" alt="KANCHAN-AI logo" /></div>
            <div>
              <div className={styles.brandName}>KANCHAN<span className={styles.brandAi}>-AI</span></div>
              <div className={styles.brandSub}>Gold Loan Division</div>
            </div>
          </button>

          <div className={styles.topbarRight}>
            <span className={styles.recordCount}>{total} total records</span>
            <div className={styles.statusPill}>
              <span className={styles.statusDot} />
              <span>Canara Bank · Branch system</span>
            </div>
            <nav className={styles.topNav}>
              <button className={styles.navItem} onClick={() => navigate('/dashboard')}>
                <BarChart3 size={16} strokeWidth={1.75} />
                <span>Dashboard</span>
              </button>
              <button className={`${styles.navItem} ${styles.navActive}`}>
                <Clock size={16} strokeWidth={1.75} />
                <span>Case History</span>
              </button>
            </nav>
          </div>
        </header>

        {/* ── Stat chips (SAMHITA style) ── */}
        <div className={styles.statsRow}>
          <StatChip
            icon={BarChart3} label="Total" value={total}
            sub="all records" cls="total"
          />
          <StatChip
            icon={CheckCircle2} label="Genuine" value={counts.genuine}
            sub="attestations" cls="genuine"
            active={verdict === 'GENUINE'} onClick={() => setVerdict(v => v === 'GENUINE' ? 'All' : 'GENUINE')}
          />
          <StatChip
            icon={AlertTriangle} label="Borderline" value={counts.borderline}
            sub="attestations" cls="borderline"
            active={verdict === 'BORDERLINE'} onClick={() => setVerdict(v => v === 'BORDERLINE' ? 'All' : 'BORDERLINE')}
          />
          <StatChip
            icon={XCircle} label="Rejected" value={counts.reject}
            sub="attestations" cls="reject"
            active={verdict === 'REJECT'} onClick={() => setVerdict(v => v === 'REJECT' ? 'All' : 'REJECT')}
          />
        </div>

        {/* ── Body ── */}
        <div className={styles.body}>

          {/* List panel */}
          <div className={styles.listPanel}>
            {/* Filter bar */}
            <div className={styles.filterBar}>
              <div className={styles.searchWrap}>
                <Search size={13} className={styles.searchIcon} />
                <input
                  className={styles.searchInput}
                  placeholder="Search description, case ID, branch…"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                />
                {search && (
                  <button className={styles.clearSearch} onClick={() => setSearch('')}>
                    <X size={12} />
                  </button>
                )}
              </div>
              <div className={styles.filterRow}>
                <FilterPill
                  icon={Tag} value={verdict} options={VERDICT_OPTIONS}
                  onChange={setVerdict} label="Verdict"
                />
                <FilterPill
                  icon={Building2} value={branch} options={branches}
                  onChange={setBranch} label="Branch"
                />
                <FilterPill
                  icon={Calendar} value={karat !== 'All' ? `${karat}K` : 'Karat'}
                  options={KARAT_OPTIONS.map(o => o === 'All' ? 'All' : `${o}K`)}
                  onChange={v => setKarat(v === 'All' ? 'All' : v.replace('K', ''))} label="Karat"
                />
                <button
                  className={`${styles.sortPill} ${sortDir === 'asc' ? styles.active : ''}`}
                  onClick={() => setSortDir(d => d === 'desc' ? 'asc' : 'desc')}
                >
                  <Clock size={11} />
                  {sortDir === 'desc' ? 'Newest' : 'Oldest'}
                </button>
                {hasFilters && (
                  <button className={styles.clearBtn} onClick={clearFilters}>
                    <X size={11} /> Clear
                  </button>
                )}
              </div>
              <div className={styles.filterResult}>
                <SlidersHorizontal size={11} />
                {filtered.length} of {total} cases
                {hasFilters && ' (filtered)'}
              </div>
            </div>

            {/* Case list */}
            <div className={styles.caseList}>
              {loading && (
                <div className={styles.listEmpty}>
                  <div className={styles.spinner} />
                  Loading cases…
                </div>
              )}
              {!loading && filtered.length === 0 && (
                <div className={styles.listEmpty}>
                  <Filter size={28} className={styles.emptyIcon} />
                  <p>No cases match the current filters.</p>
                  {hasFilters && (
                    <button className={styles.clearLink} onClick={clearFilters}>Clear filters</button>
                  )}
                </div>
              )}
              {filtered.map(c => (
                <CaseRow
                  key={c.case_id}
                  c={c}
                  active={selected?.case_id === c.case_id}
                  onClick={() => setSelected(c)}
                />
              ))}
            </div>
          </div>

          {/* Detail panel */}
          <div className={styles.detailPanel}>
            {selected
              ? <CaseDetail c={selected} onCaseUpdated={updated => {
                  setSelected(updated)
                  setCases(cs => cs.map(c => c.case_id === updated.case_id ? updated : c))
                }} />
              : (
                <div className={styles.noDetail}>
                  <BarChart3 size={36} className={styles.noDetailIcon} />
                  <p className={styles.noDetailText}>Select a case to view full analysis</p>
                </div>
              )
            }
          </div>
        </div>
      </div>
      <MobileTabBar />
    </div>
  )
}

/* ── Sub-components ── */

function StatChip({ icon: Icon, label, value, sub, cls, active, onClick }) {
  return (
    <button
      className={`${styles.statChip} ${styles[`chip_${cls}`]} ${active ? styles.chipActive : ''} ${onClick ? styles.chipClickable : ''}`}
      onClick={onClick}
    >
      <div className={styles.chipIcon}><Icon size={16} strokeWidth={2} /></div>
      <div>
        <div className={styles.chipValue}>{value}</div>
        <div className={styles.chipLabel}>{label}</div>
      </div>
      <div className={styles.chipSub}>{sub}</div>
    </button>
  )
}

function FilterPill({ icon: Icon, value, options, onChange, label }) {
  return (
    <div className={styles.filterPillWrap}>
      <Icon size={11} className={styles.filterPillIcon} />
      <select
        className={styles.filterPill}
        value={value}
        onChange={e => onChange(e.target.value)}
      >
        {options.map(o => (
          <option key={o} value={o}>{o === 'All' ? `All ${label}s` : o}</option>
        ))}
      </select>
    </div>
  )
}

function CaseRow({ c, active, onClick }) {
  const rl  = c.verdict?.risk_level || 'BORDERLINE'
  const cfg = VERDICT_META[rl] || VERDICT_META.BORDERLINE
  const Icon = cfg.icon
  const dt   = new Date(c.timestamp)
  const date = isNaN(dt) ? '-' : dt.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })
  const time = isNaN(dt) ? '-' : dt.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })

  return (
    <button
      className={`${styles.caseRow} ${active ? styles.caseRowActive : ''}`}
      onClick={onClick}
    >
      <div className={`${styles.rowBar} ${styles[`bar_${cfg.cls}`]}`} />
      <div className={`${styles.rowIconWrap} ${styles[`icon_${cfg.cls}`]}`}>
        <Icon size={14} strokeWidth={2.5} />
      </div>
      <div className={styles.rowBody}>
        <div className={styles.rowTop}>
          <span className={styles.rowDesc}>{c.item_description || 'Gold item'}</span>
          <span className={`${styles.rowVerdict} ${styles[`vrd_${cfg.cls}`]}`}>{rl}</span>
        </div>
        <div className={styles.rowMeta}>
          <span className={styles.rowKarat}>{c.declared_karat}K</span>
          <span className={styles.rowDot}>·</span>
          <span className={styles.rowCase}>#{c.case_id}</span>
          {c.branch_id && c.branch_id !== 'default' && (
            <><span className={styles.rowDot}>·</span><span>{c.branch_id}</span></>
          )}
        </div>
      </div>
      <div className={styles.rowTime}>
        <div>{date}</div>
        <div className={styles.rowTimeTime}>{time}</div>
      </div>
      <ChevronRight size={14} className={styles.rowChevron} />
    </button>
  )
}

function CaseDetail({ c, onCaseUpdated }) {
  const { modality_scores, contradiction, fusion, benford, verdict, case_id } = c
  const rl  = verdict?.risk_level || 'BORDERLINE'
  const cfg = VERDICT_META[rl] || VERDICT_META.BORDERLINE
  const Icon = cfg.icon
  const dt   = new Date(c.timestamp)
  const timeStr = isNaN(dt) ? '-' : dt.toLocaleString('en-IN', {
    weekday: 'short', day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className={styles.detail}>
      <div className={styles.detailHeader}>
        <div className={styles.detailHeaderLeft}>
          <div className={`${styles.detailVerdictIcon} ${styles[`icon_${cfg.cls}`]}`}>
            <Icon size={20} strokeWidth={2} />
          </div>
          <div>
            <h2 className={styles.detailTitle}>{c.item_description || 'Gold item'}</h2>
            <p className={styles.detailMeta}>
              {c.declared_karat}K gold
              {c.branch_id && c.branch_id !== 'default' && ` · ${c.branch_id}`}
              {' · '}{timeStr}
            </p>
          </div>
        </div>
        <span className={`${styles.detailVerdictBadge} ${styles[`badge_${cfg.cls}`]}`}>{rl}</span>
      </div>

      <div className={styles.detailBody}>
        <VerdictCard verdict={verdict} caseId={case_id} />
        <CustomerInfoCard caseId={case_id} customer={c.customer} evaluator={c.evaluator} onUpdated={onCaseUpdated} />
        <ProcessTrace trace={c.verification_trace} caseData={c} />
        <MediaEvidence caseData={c} />
        <GoldVsGemsCard caseData={c} />
        <XRayView caseData={c} />
        <SignalBars scores={modality_scores} />
        {contradiction?.flags?.length > 0 && (
          <ContradictionAlert contradiction={contradiction} />
        )}
        <DensityDetails density={modality_scores?.density} />
        <CompositionCard
          composition={c.composition}
          weightDry={modality_scores?.density?.weight_dry}
          goldGemSplit={c.media?.xray?.gold_gem_split}
        />
        {fusion?.shap_values && <SHAPBreakdown shap={fusion.shap_values} />}
        {benford && <BenfordStatus benford={benford} />}
      </div>
    </div>
  )
}
