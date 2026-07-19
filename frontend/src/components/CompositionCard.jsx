import { Gem, AlertTriangle, Camera, Scale } from 'lucide-react'
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

const GOLD_GEM_LEGEND = [
  { key: 'gold',  label: 'Gold metal', colour: '#E6AA28', pctKey: 'gold_pct' },
  { key: 'gem',   label: 'Gems',       colour: '#DC285A', pctKey: 'gem_pct' },
  { key: 'other', label: 'Other',      colour: '#5A646E', pctKey: 'other_pct' },
]

/**
 * Composition — Gold vs Stones
 *
 * Two independent stories on one card:
 *   1. Photo colour split (gold_gem_split) — always shown when DSIP ran
 *   2. Physics mixture model — gold mass in grams from density; only when
 *      measured density is still in the range stones could explain
 */
export default function CompositionCard({ composition, weightDry, goldGemSplit }) {
  if (!composition && !goldGemSplit) return null

  const {
    gold_mass_g, stone_mass_g, gold_mass_fraction, model_valid,
    stone_frac_photo, stone_frac_implied,
    rho_predicted, consistency_z, adjusted_density_risk,
    hidden_volume_flag, gems = [], note,
  } = composition || {}

  const goldPct = Math.round((gold_mass_fraction ?? 0) * 100)
  const showMass = composition && model_valid !== false && gold_mass_g != null
  const showPhysicsTable = composition && model_valid !== false
  const showPhotoSplit = goldGemSplit
    && (goldGemSplit.gold_pct != null || goldGemSplit.gem_pct != null)
  const showGems = Array.isArray(gems) && gems.length > 0

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>
        <Gem size={14} />
        Composition — Gold vs Stones
        <InfoTip
          text="Two independent checks: (1) the photo colour split paints gold metal vs gems by colour — this always runs when a photo is present; (2) the physics mixture model estimates gold mass in grams from density, but only when the measured density is still high enough that stones could explain it. A fake/test density does not erase the photo split."
          side="right"
        />
      </h3>

      {/* ── Photo colour split (always when available) ── */}
      {showPhotoSplit && (
        <div className={styles.photoSection}>
          <div className={styles.sectionHead}>
            <Camera size={12} />
            <span>From the photo (colour split)</span>
            <span className={styles.sectionHint}>
              visual only · {goldGemSplit.stones_used ?? 0} stone region
              {(goldGemSplit.stones_used ?? 0) === 1 ? '' : 's'}
            </span>
          </div>
          <div className={styles.massBar}>
            {GOLD_GEM_LEGEND.map(m => {
              const pct = goldGemSplit[m.pctKey] ?? 0
              return pct > 0 ? (
                <div
                  key={m.key}
                  className={styles.massSeg}
                  style={{ width: `${pct}%`, background: m.colour }}
                  title={`${m.label}: ${pct}%`}
                />
              ) : null
            })}
          </div>
          <div className={styles.massLegend}>
            {GOLD_GEM_LEGEND.map(m => (
              <span key={m.key}>
                <i style={{ background: m.colour }} />
                {m.label} {goldGemSplit[m.pctKey] ?? 0}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Detected gems — from the photo, independent of density validity */}
      {showGems && (
        <div className={styles.gemList}>
          {gems.map((g, i) => {
            const named = g.stone_name && g.stone_name !== 'unidentified'
            const label = named
              ? g.stone_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
              : (HUE_LABELS[g.hue_class] || g.hue_class)
            return (
              <span
                key={i}
                className={styles.gemChip}
                title={named ? `Nearest colour match: ${Math.round((g.match_confidence || 0) * 100)}%` : undefined}
              >
                <i style={{ background: HUE_COLOURS[g.hue_class] || HUE_COLOURS.other }} />
                #{i + 1} {label} · {g.area_pct}%
              </span>
            )
          })}
        </div>
      )}

      {/* ── Physics mixture model ── */}
      {composition && (
        <div className={styles.physicsSection}>
          <div className={styles.sectionHead}>
            <Scale size={12} />
            <span>From density (physics mixture)</span>
          </div>

          {(hidden_volume_flag || model_valid === false) && (
            <div className={styles.hiddenWarn}>
              <AlertTriangle size={14} />
              <span>{note}</span>
            </div>
          )}

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

          {showPhysicsTable && (
            <div className={styles.table}>
              <Row
                label="Stone volume (photo)"
                value={`${(stone_frac_photo * 100).toFixed(0)}%`}
                tip="How much of the item's visible area the camera detected as stones. This is a lower bound — stones on the reverse side or hidden by the setting aren't counted."
              />
              <Row
                label="Stone volume (physics)"
                value={`${(stone_frac_implied * 100).toFixed(0)}%`}
                warn={hidden_volume_flag}
                tip="How much non-gold volume the measured density implies. If this is much larger than what the camera sees, something non-gold is hidden inside the metal."
              />
              <Row
                label="Predicted bulk density"
                value={`${rho_predicted} g/cm³`}
                tip="What the density SHOULD read for the declared karat plus the stones the camera found."
              />
              <Row
                label="Consistency (z)"
                value={consistency_z}
                warn={consistency_z > 2}
                tip="Distance between the measured and predicted density, in units of total uncertainty. Below 2 = consistent; above 3 = something is wrong."
              />
              <Row
                label="Stone-corrected density risk"
                value={`${Math.round(adjusted_density_risk * 100)}%`}
                warn={adjusted_density_risk > 0.5}
                tip="The density risk after accounting for the detected stones — this is what feeds the final decision for stone-set items, so genuine jewellery isn't rejected for its stones."
              />
            </div>
          )}

          {!hidden_volume_flag && model_valid !== false && note && (
            <p className={styles.note}>{note}</p>
          )}
        </div>
      )}
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
