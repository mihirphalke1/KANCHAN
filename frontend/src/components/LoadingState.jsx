import { useEffect, useState } from 'react'
import styles from './LoadingState.module.css'

// Narrates the actual pipeline stages in plain language, in the order they
// really run (see app/routers/analyze.py) — reads like someone working
// through the case by hand, not a bare "loading…" spinner for an API call.
const STEPS = [
  { label: 'Validating the density reading…', delay: 0 },
  { label: 'Mapping the stones in your photos…', delay: 900 },
  { label: 'Building the material scan image…', delay: 1900 },
  { label: 'Listening to how the item rings…', delay: 2900 },
  { label: 'Checking the touchstone streak…', delay: 3900 },
  { label: 'Cross-checking every signal against the others…', delay: 4700 },
  { label: 'Working out the loan valuation…', delay: 5500 },
  { label: 'Writing up the verdict…', delay: 6300 },
]

// The verdict step above used to be where this screen would freeze if the
// AI explanation call ran long — there was nothing after it. The backend
// now bounds that wait on its own (VERDICT_TIMEOUT_S), but this screen
// should never look stuck regardless, so it keeps talking once the main
// sequence runs out.
const STILL_WORKING = [
  'Still writing up the verdict…',
  'Double-checking the numbers…',
  'Finishing the report…',
]
const TAIL_CYCLE_MS = 3500
const LONG_WAIT_MS  = 15000

export default function LoadingState() {
  const [step, setStep]         = useState(0)
  const [tailIndex, setTailIndex] = useState(-1)   // -1 = still in the main STEPS
  const [longWait, setLongWait] = useState(false)

  useEffect(() => {
    const timers = STEPS.map((s, i) => setTimeout(() => setStep(i), s.delay))
    const lastDelay = STEPS[STEPS.length - 1].delay

    let tailInterval
    const tailStart = setTimeout(() => {
      setTailIndex(0)
      tailInterval = setInterval(() => {
        setTailIndex(i => (i + 1) % STILL_WORKING.length)
      }, TAIL_CYCLE_MS)
    }, lastDelay + 1500)

    const longWaitTimer = setTimeout(() => setLongWait(true), LONG_WAIT_MS)

    return () => {
      timers.forEach(clearTimeout)
      clearTimeout(tailStart)
      clearTimeout(longWaitTimer)
      if (tailInterval) clearInterval(tailInterval)
    }
  }, [])

  const label = tailIndex >= 0 ? STILL_WORKING[tailIndex] : STEPS[step].label

  return (
    <div className={styles.wrap}>
      <div className={styles.spinner}>
        <div className={styles.ring} />
        <div className={styles.core} />
      </div>
      <p className={styles.step}>{label}</p>
      <div className={styles.progress}>
        {STEPS.map((s, i) => (
          <div
            key={i}
            className={`${styles.dot} ${i <= step ? styles.done : ''}`}
          />
        ))}
      </div>
      <p className={styles.hint}>
        {longWait
          ? 'A thorough check can take up to a minute — thanks for waiting.'
          : 'Checking the weight, stones, sound, and streak test all at once'}
      </p>
    </div>
  )
}
