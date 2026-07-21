import { BarChart3, ShieldCheck, ShieldAlert } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './BenfordStatus.module.css'

export default function BenfordStatus({ benford, evaluator }) {
  if (!benford) return null

  const { status, alert, message, n_samples, p_value, digit_observed, digit_expected } = benford
  const hasData = status !== 'insufficient_data' && digit_observed?.length === 9
  const evAlert = evaluator?.alert

  return (
    <div className={`${styles.card} ${alert ? styles.alert : styles.ok}`}>
      {evAlert && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12,
          padding: '10px 12px', borderRadius: 8, background: '#fef2f2',
          border: '1px solid #fca5a5', color: '#991b1b', fontSize: 13, fontWeight: 600,
        }}>
          <ShieldAlert size={16} strokeWidth={2.5} />
          This officer’s own readings show an unusual pattern
          {evaluator?.n_samples ? ` (${evaluator.n_samples} records)` : ''} — localised to
          them, not the whole branch. Recommend audit of their appraisals.
        </div>
      )}
      <div className={styles.header}>
        <div className={styles.iconTitle}>
          {alert
            ? <ShieldAlert size={18} strokeWidth={2.5} />
            : <ShieldCheck size={18} strokeWidth={2.5} />}
          <div>
            <h3 className={styles.title}>
              Branch Records Check
              <InfoTip text="Genuinely measured numbers start with 1 far more often than with 9 — a known natural pattern. When someone types made-up readings instead of measuring, the pattern breaks. This watches the whole branch's records, not this item." side="right" />
            </h3>
            <p className={styles.msg}>
              {alert
                ? 'The pattern of this branch’s recorded weight readings looks unusual — this can happen when readings are being entered by hand instead of measured. Worth a supervisor review.'
                : 'The pattern of this branch’s recorded weight readings looks normal — consistent with genuinely measured values.'}
            </p>
          </div>
        </div>
        {n_samples > 0 && (
          <div className={styles.stats}>
            <Stat label="Records checked" value={n_samples} />
          </div>
        )}
      </div>

      {hasData && (
        <div className={styles.chartSection}>
          <h4 className={styles.chartTitle}>
            <BarChart3 size={13} />
            How the readings start (first digit of each measurement)
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
            <span>This branch</span>
            <span className={`${styles.dot} ${styles.expDot}`} />
            <span>Natural pattern</span>
            {p_value != null && (
              <span className={styles.techNote}>Benford first-digit test, p = {p_value.toFixed(4)}</span>
            )}
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
