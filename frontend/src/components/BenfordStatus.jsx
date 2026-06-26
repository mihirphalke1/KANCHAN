import { BarChart3, ShieldCheck, ShieldAlert } from 'lucide-react'
import styles from './BenfordStatus.module.css'

export default function BenfordStatus({ benford }) {
  if (!benford) return null

  const { status, alert, message, n_samples, p_value, digit_observed, digit_expected } = benford
  const hasData = status !== 'insufficient_data' && digit_observed?.length === 9

  return (
    <div className={`${styles.card} ${alert ? styles.alert : styles.ok}`}>
      <div className={styles.header}>
        <div className={styles.iconTitle}>
          {alert
            ? <ShieldAlert size={18} strokeWidth={2.5} />
            : <ShieldCheck size={18} strokeWidth={2.5} />}
          <div>
            <h3 className={styles.title}>
              Benford's Law Monitor
              <span className={styles.badge}>Novelty 2</span>
            </h3>
            <p className={styles.msg}>{message}</p>
          </div>
        </div>
        {n_samples > 0 && (
          <div className={styles.stats}>
            <Stat label="Samples" value={n_samples} />
            {p_value != null && <Stat label="p-value" value={p_value.toFixed(4)} mono />}
          </div>
        )}
      </div>

      {hasData && (
        <div className={styles.chartSection}>
          <h4 className={styles.chartTitle}>
            <BarChart3 size={13} />
            First-Digit Distribution
          </h4>
          <div className={styles.chart}>
            {digit_observed.map((obs, i) => {
              const exp  = digit_expected?.[i] ?? 0
              const digit = i + 1
              const maxH  = 60
              const obsPx = Math.round(obs  * maxH / 0.32)
              const expPx = Math.round(exp  * maxH / 0.32)
              return (
                <div key={digit} className={styles.col}>
                  <div className={styles.colBars} style={{ height: maxH }}>
                    <div
                      className={`${styles.bar} ${styles.obsBar} ${alert ? styles.alertBar : styles.okBar}`}
                      style={{ height: obsPx }}
                      title={`Digit ${digit}: observed ${(obs*100).toFixed(1)}%`}
                    />
                    <div
                      className={`${styles.bar} ${styles.expBar}`}
                      style={{ height: expPx }}
                      title={`Digit ${digit}: expected ${(exp*100).toFixed(1)}%`}
                    />
                  </div>
                  <span className={styles.colLabel}>{digit}</span>
                </div>
              )
            })}
          </div>
          <div className={styles.legend}>
            <span className={`${styles.dot} ${alert ? styles.alertDot : styles.okDot}`} />
            <span>Observed</span>
            <span className={`${styles.dot} ${styles.expDot}`} />
            <span>Benford expected</span>
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value, mono }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statLabel}>{label}</span>
      <span className={`${styles.statValue} ${mono ? styles.mono : ''}`}>{value}</span>
    </div>
  )
}
