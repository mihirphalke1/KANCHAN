import { AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import styles from './ContradictionAlert.module.css'

const PLAIN_NAMES = {
  density: 'weight test', image: 'photo', acoustic: 'sound test',
  streak: 'streak', xray: 'material scan', photo: 'photo',
}

function plain(text) {
  // Replace standalone modality tokens only — never inside hyphenated words
  // like "low-density core" (which must stay literal English).
  return String(text).replace(
    /(?<![-\w])(density|image|acoustic|streak|xray)(?![-\w])/g,
    m => PLAIN_NAMES[m] || m
  )
}

export default function ContradictionAlert({ contradiction, scores }) {
  const [expanded, setExpanded] = useState(true)
  if (!contradiction || !contradiction.flags?.length) return null

  const absent = new Set(
    Object.entries(scores || {})
      .filter(([, s]) => (s?.mode || '').startsWith('no_'))
      .map(([k]) => k)
  )
  const pairVisible = ([pair]) => !pair.split('↔').some(m => absent.has(m))

  const score = contradiction.contradiction_score
  const level = score > 0.5 ? 'high' : score > 0.3 ? 'medium' : 'low'

  return (
    <div className={`${styles.card} ${styles[level]}`}>
      <button className={styles.header} onClick={() => setExpanded(e => !e)}>
        <div className={styles.titleGroup}>
          <AlertTriangle size={16} strokeWidth={2.5} />
          <span className={styles.title}>
            Tests Disagree — Needs Attention
          </span>
        </div>
        <div className={styles.headerRight}>
          <span className={styles.score}>{Math.round(score * 100)}% disagreement</span>
          {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </div>
      </button>

      {expanded && (
        <div className={styles.body}>
          {contradiction.flags.map((flag, i) => (
            <div key={i} className={styles.flag}>
              <div className={styles.flagBullet} />
              <p>{plain(flag)}</p>
            </div>
          ))}

          {contradiction.cross_pairs && (
            <div className={styles.pairs}>
              {Object.entries(contradiction.cross_pairs)
                .filter(pairVisible)
                .sort(([, a], [, b]) => b - a)
                .map(([pair, val]) => (
                  <div key={pair} className={styles.pair}>
                    <span className={styles.pairName}>{plain(pair)}</span>
                    <div className={styles.pairTrack}>
                      <div
                        className={`${styles.pairFill} ${val > 0.35 ? styles.hot : styles.cool}`}
                        style={{ width: `${val * 100}%` }}
                      />
                    </div>
                    <span className={styles.pairVal}>{(val * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
