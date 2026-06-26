import { useEffect, useState } from 'react'
import styles from './LoadingState.module.css'

const STEPS = [
  { label: 'Computing density via Archimedes…', delay: 0 },
  { label: 'Extracting MFCC-ΔΔ acoustic features…', delay: 1200 },
  { label: 'Running EfficientNet-B3 visual analysis…', delay: 2400 },
  { label: 'Analysing touchstone streak…', delay: 3200 },
  { label: 'Computing cross-modal contradictions…', delay: 4000 },
  { label: 'Fusing signals with XGBoost…', delay: 4800 },
  { label: 'Generating LLM verdict…', delay: 5400 },
]

export default function LoadingState() {
  const [step, setStep] = useState(0)

  useEffect(() => {
    const timers = STEPS.map((s, i) =>
      setTimeout(() => setStep(i), s.delay)
    )
    return () => timers.forEach(clearTimeout)
  }, [])

  return (
    <div className={styles.wrap}>
      <div className={styles.spinner}>
        <div className={styles.ring} />
        <div className={styles.core} />
      </div>
      <p className={styles.step}>{STEPS[step].label}</p>
      <div className={styles.progress}>
        {STEPS.map((s, i) => (
          <div
            key={i}
            className={`${styles.dot} ${i <= step ? styles.done : ''}`}
          />
        ))}
      </div>
      <p className={styles.hint}>Analysing all four modalities simultaneously</p>
    </div>
  )
}
