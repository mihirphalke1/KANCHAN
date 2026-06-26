import { useEffect, useRef, useState } from 'react'
import { Eye, Weight, Mic, Brush } from 'lucide-react'
import styles from './SignalBars.module.css'

const SIGNALS = [
  { key: 'image',    label: 'Visual',   icon: Eye,    description: 'Color & surface analysis' },
  { key: 'density',  label: 'Density',  icon: Weight, description: 'Archimedes test result' },
  { key: 'acoustic', label: 'Acoustic', icon: Mic,    description: 'MFCC-ΔΔ tap test' },
  { key: 'streak',   label: 'Streak',   icon: Brush,  description: 'Touchstone test' },
]

function getRiskColor(risk) {
  if (risk < 0.3) return 'green'
  if (risk < 0.6) return 'amber'
  return 'red'
}

function Bar({ signal, score }) {
  const [width, setWidth] = useState(0)
  const pct   = Math.round((score?.risk_score ?? 0.5) * 100)
  const color = getRiskColor(score?.risk_score ?? 0.5)
  const Icon  = signal.icon

  useEffect(() => {
    const t = requestAnimationFrame(() => {
      setTimeout(() => setWidth(pct), 50)
    })
    return () => cancelAnimationFrame(t)
  }, [pct])

  return (
    <div className={styles.barRow}>
      <div className={styles.barMeta}>
        <div className={styles.barLabelGroup}>
          <span className={styles.barIcon}>
            <Icon size={14} />
          </span>
          <div>
            <span className={styles.barLabel}>{signal.label}</span>
            <span className={styles.barDesc}>{signal.description}</span>
          </div>
        </div>
        <div className={styles.barRight}>
          <span className={`${styles.statusDot} ${styles[`dot_${color}`]}`} aria-hidden="true" />
          <span className={`${styles.barPct} ${styles[color]}`}>{pct}%</span>
          {score?.mode && <span className={styles.barMode}>{score.mode}</span>}
        </div>
      </div>
      <div className={styles.track}>
        <div
          className={`${styles.fill} ${styles[color]}`}
          style={{ '--bar-width': `${width}%`, width: `${width}%` }}
        />
      </div>
    </div>
  )
}

export default function SignalBars({ scores }) {
  if (!scores) return null

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Signal Breakdown</h3>
      <div className={styles.bars}>
        {SIGNALS.map(sig => (
          <Bar key={sig.key} signal={sig} score={scores[sig.key]} />
        ))}
      </div>
      <p className={styles.footer}>Risk score per modality — lower is better</p>
    </div>
  )
}
