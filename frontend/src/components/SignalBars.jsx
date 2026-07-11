import { useEffect, useRef, useState } from 'react'
import { Eye, Weight, Mic, Brush, Scan } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './SignalBars.module.css'

const SIGNALS = [
  { key: 'image',    label: 'Photo',       icon: Eye,    description: 'Colour & surface from the photos' },
  { key: 'density',  label: 'Weight test', icon: Weight, description: 'Weight in air vs in water' },
  { key: 'acoustic', label: 'Sound test',  icon: Mic,    description: 'Ring of the item when tapped' },
  { key: 'streak',   label: 'Streak',      icon: Brush,  description: 'Touchstone streak colour' },
  { key: 'xray',     label: 'Material scan', icon: Scan, description: 'Metal vs stones from the photo' },
]

function getRiskColor(risk) {
  if (risk < 0.3) return 'green'
  if (risk < 0.6) return 'amber'
  return 'red'
}

function isNotPerformed(score) {
  return !score || (score.mode || '').startsWith('no_')
}

function Bar({ signal, score }) {
  const [width, setWidth] = useState(0)
  const notPerformed = isNotPerformed(score)
  const pct   = Math.round((score?.risk_score ?? 0.5) * 100)
  const color = getRiskColor(score?.risk_score ?? 0.5)
  const Icon  = signal.icon

  useEffect(() => {
    if (notPerformed) return
    const t = requestAnimationFrame(() => {
      setTimeout(() => setWidth(pct), 50)
    })
    return () => cancelAnimationFrame(t)
  }, [pct, notPerformed])

  return (
    <div className={`${styles.barRow} ${notPerformed ? styles.notPerformed : ''}`}>
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
          {notPerformed ? (
            <span className={styles.naTag}>Not performed</span>
          ) : (
            <>
              <span className={`${styles.statusDot} ${styles[`dot_${color}`]}`} aria-hidden="true" />
              <span className={`${styles.barPct} ${styles[color]}`}>{pct}%</span>
            </>
          )}
        </div>
      </div>
      <div className={styles.track}>
        {!notPerformed && (
          <div
            className={`${styles.fill} ${styles[color]}`}
            style={{ '--bar-width': `${width}%`, width: `${width}%` }}
          />
        )}
      </div>
    </div>
  )
}

export default function SignalBars({ scores }) {
  if (!scores) return null

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>
        Test Results
        <InfoTip text="Each test's individual risk before combination. 'Not performed' tests carry no evidence either way — run more tests for a stronger verdict." side="right" />
      </h3>
      <div className={styles.bars}>
        {SIGNALS
          .filter(sig => sig.key !== 'xray' || (scores.xray && scores.xray.mode !== 'no_xray'))
          .map(sig => (
            <Bar key={sig.key} signal={sig} score={scores[sig.key]} />
          ))}
      </div>
      <p className={styles.footer}>Risk per test — lower is better</p>
    </div>
  )
}
