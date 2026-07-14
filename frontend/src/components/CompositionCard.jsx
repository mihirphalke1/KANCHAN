import { Gem, AlertTriangle } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './CompositionCard.module.css'

const HUE_COLOURS = {
  red:        '#DC2626',
  green:      '#15803D',
  blue:       '#2563EB',
  other:      '#6B7280',
  colourless: '#D4D4D8',
}
const HUE_LABELS = {
  red:        'Ruby-like',
  green:      'Emerald-like',
  blue:       'Sapphire-like',
  other:      'Other stone',
  colourless: 'Colourless stone — confirm it’s a stone',
}

export default function CompositionCard({ composition, weightDry }) {
  if (!composition) return null

  const {
    gold_mass_g, stone_mass_g, gold_mass_fraction, model_valid,
    stone_frac_photo, stone_frac_implied,
    rho_predicted, consistency_z, adjusted_density_risk,
    hidden_volume_flag, gems = [], note,
  } = composition

  const goldPct = Math.round((gold_mass_fraction ?? 0) * 100)
  const showMass = model_valid !== false && gold_mass_g != null

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>
        <Gem size={14} />
        Composition — Gold vs Stones
      </h3>

      {(hidden_volume_flag || model_valid === false) && (
        <div className={styles.hiddenWarn}>
          <AlertTriangle size={14} />
          <span>{note}</span>
        </div>
      )}

      {/* Mass split bar — only when the mixture model actually applies */}
      {showMass && (
        <div className={styles.massSection}>
          <div className={styles.massBar}>
            <div className={styles.massGold} style={{ width: `${goldPct}%` }} />
            <div className={styles.massStone} style={{ width: `${100 - goldPct}%` }} />
          </div>
          <div className={styles.massLegend}>
            <span><i className={styles.dotGold} /> Gold ≈ {gold_mass_g} g ({goldPct}%)</span>
            <span><i className={styles.dotStone} /> Stones/other ≈ {stone_mass_g} g</span>
            {weightDry != null && <span className={styles.total}>of {weightDry} g total</span>}
          </div>
        </div>
      )}

      {/* Detected gems — only meaningful when the item is actually gold;
          when the mixture model doesn't apply, a "gold vs stones" breakdown
          is moot, so we don't list stones. */}
      {model_valid !== false && gems.length > 0 && (
        <div className={styles.gemList}>
          {gems.map((g, i) => {
            const named = g.stone_name && g.stone_name !== 'unidentified'
            const label = named
              ? g.stone_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
              : (HUE_LABELS[g.hue_class] || g.hue_class)
            return (
              <span key={i} className={styles.gemChip} title={named ? `Nearest colour match: ${Math.round((g.match_confidence || 0) * 100)}%` : undefined}>
                <i style={{ background: HUE_COLOURS[g.hue_class] || HUE_COLOURS.other }} />
                #{i + 1} {label} · {g.area_pct}%
              </span>
            )
          })}
        </div>
      )}

      {model_valid !== false && (
        <div className={styles.table}>
          <Row label="Stone volume (photo)" value={`${(stone_frac_photo * 100).toFixed(0)}%`}
            tip="How much of the item's visible area the camera detected as stones. This is a lower bound — stones on the reverse side or hidden by the setting aren't counted." />
          <Row label="Stone volume (physics)" value={`${(stone_frac_implied * 100).toFixed(0)}%`}
            warn={hidden_volume_flag}
            tip="How much non-gold volume the measured density implies. If this is much larger than what the camera sees, something non-gold is hidden inside the metal." />
          <Row label="Predicted bulk density" value={`${rho_predicted} g/cm³`}
            tip="What the density SHOULD read for the declared karat plus the stones the camera found." />
          <Row label="Consistency (z)" value={consistency_z} warn={consistency_z > 2}
            tip="Distance between the measured and predicted density, in units of total uncertainty. Below 2 = consistent; above 3 = something is wrong." />
          <Row label="Stone-corrected density risk" value={`${Math.round(adjusted_density_risk * 100)}%`}
            warn={adjusted_density_risk > 0.5}
            tip="The density risk after accounting for the detected stones — this is what feeds the final decision for stone-set items, so genuine jewellery isn't rejected for its stones." />
        </div>
      )}

      {!hidden_volume_flag && model_valid !== false && <p className={styles.note}>{note}</p>}
    </div>
  )
}

function Row({ label, value, warn, tip }) {
  return (
    <div className={styles.row}>
      <span className={styles.rowLabel}>{label}{tip && <InfoTip text={tip} />}</span>
      <span className={`${styles.rowValue} ${warn ? styles.warn : ''}`}>{value}</span>
    </div>
  )
}
