import { AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react'
import { useState } from 'react'
import styles from './ContradictionAlert.module.css'

export default function ContradictionAlert({ contradiction }) {
  const [expanded, setExpanded] = useState(true)
  if (!contradiction || !contradiction.flags?.length) return null

  const score = contradiction.contradiction_score
  const level = score > 0.5 ? 'high' : score > 0.3 ? 'medium' : 'low'

  return (
    <div className={`${styles.card} ${styles[level]}`}>
      <button className={styles.header} onClick={() => setExpanded(e => !e)}>
        <div className={styles.titleGroup}>
          <AlertTriangle size={16} strokeWidth={2.5} />
          <span className={styles.title}>
            Cross-Modal Contradiction Detected
            <span className={styles.badge}>Novelty 3</span>
          </span>
        </div>
        <div className={styles.headerRight}>
          <span className={styles.score}>{Math.round(score * 100)}% conflict</span>
          {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
        </div>
      </button>

      {expanded && (
        <div className={styles.body}>
          {contradiction.flags.map((flag, i) => (
            <div key={i} className={styles.flag}>
              <div className={styles.flagBullet} />
              <p>{flag}</p>
            </div>
          ))}

          {contradiction.cross_pairs && (
            <div className={styles.pairs}>
              {Object.entries(contradiction.cross_pairs)
                .sort(([, a], [, b]) => b - a)
                .map(([pair, val]) => (
                  <div key={pair} className={styles.pair}>
                    <span className={styles.pairName}>{pair}</span>
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
