import InfoTip from './ui/InfoTip'
import styles from './SHAPBreakdown.module.css'

const FEATURE_LABELS = {
  image_risk:               'Photo checks (material scan)',
  density_risk:             'Weight-in-water test',
  acoustic_risk:            'Sound test',
  streak_risk:              'Touchstone streak',
  contra_density_acoustic:  'Weight & sound tests disagree',
  contra_density_image:     'Weight & photo tests disagree',
  contra_image_streak:      'Photo & streak tests disagree',
  contra_acoustic_image:    'Sound & photo tests disagree',
  contra_density_streak:    'Weight & streak tests disagree',
  contra_acoustic_streak:   'Sound & streak tests disagree',
}

// Which modality each fusion feature involves — used to hide rows for
// tests that were never performed (they enter the model only as neutral
// defaults and must not be displayed as results).
const FEATURE_MODALITIES = {
  image_risk:               ['image'],
  density_risk:             ['density'],
  acoustic_risk:            ['acoustic'],
  streak_risk:              ['streak'],
  contra_density_acoustic:  ['density', 'acoustic'],
  contra_density_image:     ['density', 'image'],
  contra_image_streak:      ['image', 'streak'],
  contra_acoustic_image:    ['acoustic', 'image'],
  contra_density_streak:    ['density', 'streak'],
  contra_acoustic_streak:   ['acoustic', 'streak'],
}

const MODALITY_NAMES = {
  image: 'photo', acoustic: 'sound', streak: 'streak', density: 'weight',
}

export default function SHAPBreakdown({ shap, scores, fusionMode }) {
  if (!shap) return null
  const isLogodds = fusionMode === 'logodds'

  const absent = new Set(
    Object.entries(scores || {})
      .filter(([, s]) => {
        const m = s?.mode || ''
        return m.startsWith('no_') || m === 'dsip_unusable'
      })
      .map(([k]) => k)
  )
  // The visual slot carries the material scan when the CNN probe is off —
  // it counts as performed whenever the DSIP scan ran usably.
  if (scores?.xray?.mode === 'dsip_xray') absent.delete('image')
  else absent.add('image')

  // Plain-language strength of a log-odds contribution
  const strength = (v) => {
    const a = Math.abs(v)
    const word = a < 0.35 ? 'slight' : a < 1.1 ? 'moderate' : a < 2.5 ? 'strong' : 'decisive'
    return `${word} push toward ${v > 0 ? 'fake' : 'genuine'}`
  }

  const entries = Object.entries(shap)
    .map(([k, v]) => ({ key: k, label: FEATURE_LABELS[k] || k, value: v }))
    .filter(({ key }) => !(FEATURE_MODALITIES[key] || []).some(m => absent.has(m)))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 6)

  const absentNames = [...absent].map(m => MODALITY_NAMES[m] || m)

  const maxAbs = Math.max(...entries.map(e => Math.abs(e.value)), 0.001)

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>
        What Influenced This Decision
        {isLogodds && (
          <InfoTip text="These numbers ADD UP to the decision: red pushes toward fake, green toward genuine, and the total sets the overall risk. You can verify the verdict by summing them." side="right" />
        )}
        <span className={styles.subtitle}>
          {isLogodds ? 'each test’s exact evidence contribution' : 'model attribution (SHAP)'}
        </span>
      </h3>
      <div className={styles.rows}>
        {entries.map(({ key, label, value }) => {
          const pct = (Math.abs(value) / maxAbs) * 100
          const positive = value > 0
          return (
            <div key={key} className={styles.row}>
              <span className={styles.featureLabel}>
                {label}
                {isLogodds && (
                  <span className={`${styles.strength} ${positive ? styles.posText : styles.negText}`}>
                    {strength(value)}
                  </span>
                )}
              </span>
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
        <span className={styles.posLegend}>Red</span> = pushed the decision toward fake ·
        <span className={styles.negLegend}> Green</span> = pushed toward genuine
      </p>
      {absentNames.length > 0 && (
        <p className={styles.legend}>
          Tests not performed ({absentNames.join(', ')}) are excluded — they enter the
          model only as neutral placeholders and carry no evidence.
        </p>
      )}
    </div>
  )
}
