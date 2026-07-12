import { AlertTriangle } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './DensityDetails.module.css'

export default function DensityDetails({ density }) {
  if (!density) return null

  const {
    measured_density, expected_nominal, expected_low, expected_high,
    deviation_pct, karat_verdict, closest_fake, tungsten_warning,
    sigma, conformity_probability, measurement_adequate, water_temp_c,
    best_match, misdeclared_purity
  } = density

  const inRange = karat_verdict === 'IN_RANGE' || karat_verdict === 'IDENTIFIED_FROM_PHYSICS'
  const tungsten = karat_verdict === 'TUNGSTEN_BLIND_SPOT' || tungsten_warning

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Weight-in-Water Test (Density)</h3>

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
          value={sigma != null ? `${measured_density} ± ${sigma} g/cm³` : `${measured_density} g/cm³`}
          mono highlight={!inRange}
          tip="The ± is the measurement's honest uncertainty, from the balance's precision. Heavier items measure more precisely than light ones."
        />
        {expected_low != null && (
          <Row label={karat_verdict === 'IDENTIFIED_FROM_PHYSICS' ? 'Matched Range (physics)' : 'Expected Range'}
            value={`${expected_low} – ${expected_high} g/cm³`}
            mono
          />
        )}
        {expected_nominal != null && (
          <Row label="Nominal"
            value={`${expected_nominal} g/cm³`}
            mono
          />
        )}
        {conformity_probability != null && (
          <Row label="Chance it matches declared karat"
            value={`${(conformity_probability * 100).toFixed(1)}%`}
            mono
            status={conformity_probability > 0.8 ? 'ok' : 'warn'}
            tip="The probability that the item's true density falls inside the declared karat's range, given the measurement and its uncertainty. Low values mean the metal is unlikely to be what was declared."
          />
        )}
        {water_temp_c != null && (
          <Row label="Water Temperature"
            value={`${water_temp_c} °C (buoyancy-corrected)`}
            mono
          />
        )}
        {deviation_pct != null && (
          <Row label="Deviation"
            value={`${deviation_pct > 0 ? '+' : ''}${deviation_pct}%`}
            mono highlight={Math.abs(deviation_pct) > 5}
          />
        )}
        <Row label="Verdict"
          value={VERDICT_TEXT[karat_verdict] || karat_verdict}
          status={inRange ? 'ok' : 'warn'}
        />
        {best_match?.name && (
          <Row label="What the physics says it is"
            value={
              best_match.kind === 'karat'
                ? `${best_match.name} (${Math.round((best_match.probability || 0) * 100)}% match)`
                : best_match.name.charAt(0).toUpperCase() + best_match.name.slice(1)
            }
            status={best_match.kind === 'karat' && !misdeclared_purity ? 'ok' : 'warn'}
            tip="The declared karat is never trusted — it is the claim under test. This row inverts the question: given the measured density, which material does the physics actually point to?"
          />
        )}
        {closest_fake && best_match?.kind !== 'karat' && (
          <Row label="Closest Fake Metal"
            value={closest_fake.charAt(0).toUpperCase() + closest_fake.slice(1)}
            status="warn"
          />
        )}
      </div>

      {misdeclared_purity && (
        <p className={styles.adequacyNote}>
          The metal is consistent with genuine {best_match.name} — the declared karat
          appears to be over-stated. Consider revaluation at {best_match.karat}K and
          re-run the analysis with the corrected declaration.
        </p>
      )}

      {measurement_adequate === false && (
        <p className={styles.adequacyNote}>
          The balance's precision (±{sigma} g/cm³) cannot fully resolve the declared
          karat band for an item this light — re-weigh with a finer balance or rely
          on the other modalities.
        </p>
      )}
    </div>
  )
}

function Row({ label, value, mono, highlight, status, tip }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}{tip && <InfoTip text={tip} />}</span>
      <span className={`${styles.rowValue} ${mono ? styles.mono : ''} ${highlight ? styles.highlight : ''} ${status === 'ok' ? styles.ok : ''} ${status === 'warn' ? styles.warn : ''}`}>
        {value}
      </span>
    </div>
  )
}

const VERDICT_TEXT = {
  IN_RANGE:                'Within expected range',
  LOW_DENSITY:             'Below expected - density too low',
  HIGH_DENSITY:            'Above expected - density too high',
  TUNGSTEN_BLIND_SPOT:     'Matches tungsten signature',
  IDENTIFIED_FROM_PHYSICS: 'Identified from physics — no declaration needed',
  NOT_GOLD:                'Matches no gold alloy',
}
