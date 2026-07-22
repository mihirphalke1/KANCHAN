import { useState } from 'react'
import { ListChecks, ChevronDown, CircleCheck, CircleAlert, CircleMinus } from 'lucide-react'
import styles from './ProcessTrace.module.css'

function mediaUrl(path) {
  if (!path) return null
  if (path.startsWith('data:')) return path          // embedded image
  return '/' + path.replace(/\\/g, '/').replace(/^data\//, '')
}

const STATUS_META = {
  done:    { icon: CircleCheck, cls: 'ok' },
  flag:    { icon: CircleAlert, cls: 'flag' },
  skipped: { icon: CircleMinus, cls: 'skip' },
}

function Step({ n, step, stages }) {
  const [open, setOpen] = useState(false)
  const meta = STATUS_META[step.status] || STATUS_META.done
  const Icon = meta.icon
  const stageKey = step.details?.stage_image
  const stageImg = stageKey && stages?.[stageKey] ? mediaUrl(stages[stageKey]) : null
  const detailEntries = Object.entries(step.details || {}).filter(([k]) => k !== 'stage_image')

  return (
    <li className={`${styles.step} ${styles[meta.cls]}`}>
      <button type="button" className={styles.stepHeader} onClick={() => setOpen(o => !o)}>
        <span className={styles.stepNum}>{n}</span>
        <Icon size={15} className={styles.stepIcon} />
        <span className={styles.stepTitle}>{step.step}</span>
        <span className={styles.stepSummary}>{step.summary}</span>
        <ChevronDown size={14} className={`${styles.chevron} ${open ? styles.chevronOpen : ''}`} />
      </button>

      {open && (
        <div className={styles.stepBody}>
          {step.formula && (
            <div className={styles.formula}>{step.formula}</div>
          )}
          {detailEntries.length > 0 && (
            <div className={styles.detailTable}>
              {detailEntries.map(([k, v]) => (
                <div key={k} className={styles.detailRow}>
                  <span>{k}</span>
                  <span className={styles.detailValue}>
                    {Array.isArray(v) ? v.map((x, i) => <div key={i}>{String(x)}</div>) : String(v)}
                  </span>
                </div>
              ))}
            </div>
          )}
          {stageImg && (
            <img src={stageImg} alt={step.step} className={styles.stageImg} loading="lazy" />
          )}
          {step.source && (
            <div className={styles.source}>Source: {step.source}</div>
          )}
        </div>
      )}
    </li>
  )
}

export default function ProcessTrace({ trace, caseData }) {
  const [expanded, setExpanded] = useState(true)
  if (!trace?.length) return null

  const stages = caseData?.media?.xray?.stages
  const flags = trace.filter(s => s.status === 'flag').length

  return (
    <div className={styles.card}>
      <button type="button" className={styles.titleRow} onClick={() => setExpanded(e => !e)}>
        <h3 className={styles.title}>
          <ListChecks size={14} />
          How This Result Was Reached — Step by Step
          <span className={styles.count}>{trace.length} steps{flags > 0 ? ` · ${flags} flagged` : ''}</span>
        </h3>
        <ChevronDown size={15} className={`${styles.chevron} ${expanded ? styles.chevronOpen : ''}`} />
      </button>
      {expanded && (
        <ol className={styles.steps}>
          {trace.map((s, i) => (
            <Step key={i} n={i + 1} step={s} stages={stages} />
          ))}
        </ol>
      )}
      {!expanded && (
        <p className={styles.hint}>
          Every step of the detection — inputs, formula, outputs, and data source — for officer verification.
        </p>
      )}
    </div>
  )
}
