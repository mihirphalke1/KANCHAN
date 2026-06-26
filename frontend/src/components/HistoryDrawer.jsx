import { useEffect, useState } from 'react'
import { X, CheckCircle2, AlertTriangle, XCircle, ChevronRight } from 'lucide-react'
import styles from './HistoryDrawer.module.css'

const ICONS = {
  GENUINE:    { icon: CheckCircle2, cls: 'genuine' },
  BORDERLINE: { icon: AlertTriangle, cls: 'borderline' },
  REJECT:     { icon: XCircle, cls: 'reject' },
}

export default function HistoryDrawer({ onClose }) {
  const [cases, setCases]   = useState([])
  const [total, setTotal]   = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/history?limit=30')
      .then(r => r.json())
      .then(d => {
        setCases(d.cases || [])
        setTotal(d.total || 0)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  return (
    <>
      <div className={styles.overlay} onClick={onClose} />
      <aside className={styles.drawer}>
        <div className={styles.header}>
          <h2 className={styles.title}>Case History</h2>
          <div className={styles.headerRight}>
            <span className={styles.count}>{total} cases</span>
            <button className={styles.closeBtn} onClick={onClose} aria-label="Close">
              <X size={18} />
            </button>
          </div>
        </div>

        <div className={styles.body}>
          {loading && <p className={styles.loading}>Loading cases…</p>}
          {!loading && cases.length === 0 && (
            <div className={styles.empty}>
              <p>No cases analysed yet.</p>
              <p>Run your first analysis to see results here.</p>
            </div>
          )}
          {cases.map(c => <CaseRow key={c.case_id} c={c} />)}
        </div>
      </aside>
    </>
  )
}

function CaseRow({ c }) {
  const cfg  = ICONS[c.verdict?.risk_level] || ICONS.BORDERLINE
  const Icon = cfg.icon
  const dt   = new Date(c.timestamp)
  const time = isNaN(dt) ? '—' : dt.toLocaleString('en-IN', {
    day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
  })

  return (
    <div className={styles.row}>
      <div className={`${styles.rowIcon} ${styles[cfg.cls]}`}>
        <Icon size={16} strokeWidth={2.5} />
      </div>
      <div className={styles.rowBody}>
        <div className={styles.rowTop}>
          <span className={styles.rowDesc}>{c.item_description || 'Gold item'}</span>
          <span className={`${styles.rowVerdict} ${styles[cfg.cls]}`}>
            {c.verdict?.risk_level}
          </span>
        </div>
        <div className={styles.rowMeta}>
          <span>{c.declared_karat}K</span>
          <span className={styles.dot}>·</span>
          <span>#{c.case_id}</span>
          <span className={styles.dot}>·</span>
          <span>{time}</span>
        </div>
      </div>
    </div>
  )
}
