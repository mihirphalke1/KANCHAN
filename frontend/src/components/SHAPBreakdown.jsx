import styles from './SHAPBreakdown.module.css'

const FEATURE_LABELS = {
  image_risk:               'Visual risk',
  density_risk:             'Density risk',
  acoustic_risk:            'Acoustic risk',
  streak_risk:              'Streak risk',
  contra_density_acoustic:  'Conflict: Density ↔ Acoustic',
  contra_density_image:     'Conflict: Density ↔ Visual',
  contra_image_streak:      'Conflict: Visual ↔ Streak',
  contra_acoustic_image:    'Conflict: Acoustic ↔ Visual',
  contra_density_streak:    'Conflict: Density ↔ Streak',
  contra_acoustic_streak:   'Conflict: Acoustic ↔ Streak',
}

export default function SHAPBreakdown({ shap }) {
  if (!shap) return null

  const entries = Object.entries(shap)
    .map(([k, v]) => ({ key: k, label: FEATURE_LABELS[k] || k, value: v }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 6)

  const maxAbs = Math.max(...entries.map(e => Math.abs(e.value)), 0.001)

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>
        Feature Importance
        <span className={styles.subtitle}>SHAP values — XGBoost explainability</span>
      </h3>
      <div className={styles.rows}>
        {entries.map(({ key, label, value }) => {
          const pct = (Math.abs(value) / maxAbs) * 100
          const positive = value > 0
          return (
            <div key={key} className={styles.row}>
              <span className={styles.featureLabel}>{label}</span>
              <div className={styles.barGroup}>
                <div className={styles.track}>
                  <div
                    className={`${styles.fill} ${positive ? styles.pos : styles.neg}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className={`${styles.value} ${positive ? styles.posText : styles.negText}`}>
                  {positive ? '+' : ''}{value.toFixed(3)}
                </span>
              </div>
            </div>
          )
        })}
      </div>
      <p className={styles.legend}>
        <span className={styles.posLegend}>Positive</span> = pushes toward spurious ·
        <span className={styles.negLegend}> Negative</span> = pushes toward genuine
      </p>
    </div>
  )
}
