import { AlertTriangle } from 'lucide-react'
import styles from './DensityDetails.module.css'

export default function DensityDetails({ density }) {
  if (!density) return null

  const {
    measured_density, expected_nominal, expected_low, expected_high,
    deviation_pct, karat_verdict, closest_fake, tungsten_warning
  } = density

  const inRange = karat_verdict === 'IN_RANGE'
  const tungsten = karat_verdict === 'TUNGSTEN_BLIND_SPOT' || tungsten_warning

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Density Analysis</h3>

      {tungsten && (
        <div className={styles.tungstenWarn}>
          <div className={styles.tungstenTitle}>
            <AlertTriangle size={14} />
            <strong>Tungsten Blind-Spot Warning</strong>
          </div>
          <p>Measured density is near tungsten (19.25 g/cm³), which is indistinguishable from 24K gold by density alone. Rely on acoustic and visual signals.</p>
        </div>
      )}

      <div className={styles.gauge}>
        <div className={styles.gaugeBar}>
          <div
            className={`${styles.gaugeFill} ${inRange ? styles.good : styles.bad}`}
            style={{ width: `${Math.min(100, (measured_density / 22) * 100)}%` }}
          />
          <div
            className={styles.rangeIndicator}
            style={{
              left:  `${(expected_low  / 22) * 100}%`,
              width: `${((expected_high - expected_low) / 22) * 100}%`,
            }}
          />
        </div>
        <div className={styles.gaugeLabels}>
          <span>0</span>
          <span>11 g/cm³</span>
          <span>22 g/cm³</span>
        </div>
      </div>

      <div className={styles.table}>
        <Row label="Measured Density"
          value={`${measured_density} g/cm³`}
          mono highlight={!inRange}
        />
        <Row label="Expected Range"
          value={`${expected_low} – ${expected_high} g/cm³`}
          mono
        />
        <Row label="Nominal"
          value={`${expected_nominal} g/cm³`}
          mono
        />
        <Row label="Deviation"
          value={`${deviation_pct > 0 ? '+' : ''}${deviation_pct}%`}
          mono highlight={Math.abs(deviation_pct) > 5}
        />
        <Row label="Verdict"
          value={VERDICT_TEXT[karat_verdict] || karat_verdict}
          status={inRange ? 'ok' : 'warn'}
        />
        {closest_fake && (
          <Row label="Closest Fake Metal"
            value={closest_fake.charAt(0).toUpperCase() + closest_fake.slice(1)}
            status="warn"
          />
        )}
      </div>
    </div>
  )
}

function Row({ label, value, mono, highlight, status }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}</span>
      <span className={`${styles.rowValue} ${mono ? styles.mono : ''} ${highlight ? styles.highlight : ''} ${status === 'ok' ? styles.ok : ''} ${status === 'warn' ? styles.warn : ''}`}>
        {value}
      </span>
    </div>
  )
}

const VERDICT_TEXT = {
  IN_RANGE:            'Within expected range',
  LOW_DENSITY:         'Below expected - density too low',
  HIGH_DENSITY:        'Above expected - density too high',
  TUNGSTEN_BLIND_SPOT: 'Matches tungsten signature',
}
