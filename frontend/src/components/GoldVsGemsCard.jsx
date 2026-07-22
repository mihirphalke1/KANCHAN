import { Gem, Sparkles } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './GoldVsGemsCard.module.css'

// Same URL rule as XRayView: data URIs pass through, on-disk paths get the
// leading "data/" stripped so they resolve under the static mount.
function mediaUrl(path) {
  if (!path) return null
  if (path.startsWith('data:')) return path
  // Saved case paths use the OS separator (backslashes on Windows); normalise
  // to forward slashes before stripping the "data/" prefix so the /cases mount
  // resolves them (a browser turns "/data\cases\.." into "/data/cases/.." which
  // 404s against the /cases mount).
  return '/' + path.replace(/\\/g, '/').replace(/^data\//, '')
}

const LEGEND = [
  { key: 'gold',  label: 'Gold metal', colour: '#E6AA28', pctKey: 'gold_pct' },
  { key: 'gem',   label: 'Gems',       colour: '#DC285A', pctKey: 'gem_pct' },
  { key: 'other', label: 'Other',      colour: '#5A646E', pctKey: 'other_pct' },
]

const HUE_COLOURS = {
  red: '#DC2626', green: '#15803D', blue: '#2563EB',
  other: '#6B7280', colourless: '#D4D4D8',
}

const MODE_LABEL = {
  ml_ai:     { text: 'ML detection', colour: '#0F766E' },
  ml_sam:    { text: 'ML detection', colour: '#6D28D9' },
  classical: { text: 'Classical',    colour: '#6B7280' },
}

export default function GoldVsGemsCard({ caseData }) {
  const xray = caseData?.media?.xray
  if (!xray?.stages || xray.background_removed !== true) return null

  const split = xray.gold_gem_split
  if (!split) return null

  const goldMassG = caseData?.composition?.gold_mass_g
  const gemWeight = caseData?.gem_weight
  const overlay = xray.stages.gold_gem
  const goldPct = split.gold_pct ?? 0
  const gemPct = split.gem_pct ?? 0

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <h3 className={styles.title}>
          <Sparkles size={16} />
          Gold vs Gems
          <InfoTip
            side="right"
            text="The ornament separated into gold metal and gemstones from the photo's own colours — gold is matched to the item's own metal colour across shadow and glare (chromaticity manifold), stones are outlined by an ML boundary pass. Gold weight comes from the density physics; gem carats are estimated from the photo scale when a calibration card is present. Visual/advisory — it does not change the loan decision."
          />
        </h3>
        <div className={styles.headline}>
          <span className={styles.headlineGold}>{goldPct}% gold</span>
          <span className={styles.headlineSep}>·</span>
          <span className={styles.headlineGem}>{gemPct}% gems</span>
          {(() => {
            const m = MODE_LABEL[xray.stone_detection_mode]
            return m ? (
              <span className={styles.modeBadge}
                    style={{ background: `${m.colour}14`, color: m.colour }}
                    title="How the stones were detected: the ML vision system finds and identifies each stone, with MobileSAM refining the boundaries.">
                {m.text}
              </span>
            ) : null
          })()}
        </div>
      </div>

      <div className={styles.body}>
        {overlay && (
          <div className={styles.hero}>
            <img src={mediaUrl(overlay)} alt="Gold vs gems overlay"
                 onError={e => { e.target.style.visibility = 'hidden' }} />
          </div>
        )}

        <div className={styles.right}>
          <div className={styles.bar}>
            {LEGEND.map(m => {
              const pct = split[m.pctKey] ?? 0
              return pct > 0 ? (
                <div key={m.key} className={styles.seg}
                     style={{ width: `${pct}%`, background: m.colour }}
                     title={`${m.label}: ${pct}%`} />
              ) : null
            })}
          </div>
          <div className={styles.legend}>
            {LEGEND.map(m => (
              <span key={m.key} className={styles.legendItem}>
                <i style={{ background: m.colour }} />
                {m.label} {split[m.pctKey] ?? 0}%
              </span>
            ))}
          </div>

          <div className={styles.weights}>
            {goldMassG != null && (
              <div className={styles.weightRow}>
                <span className={styles.weightLabel}>Gold weight (physics)</span>
                <span className={styles.weightValue}>≈ {goldMassG} g</span>
              </div>
            )}
            {gemWeight?.total_carat != null ? (
              <div className={styles.weightRow}>
                <span className={styles.weightLabel}>Gem weight (from size)</span>
                <span className={styles.weightValue}>
                  ≈ {gemWeight.total_carat} ct
                  <span className={styles.range}>
                    {' '}({gemWeight.total_carat_low}–{gemWeight.total_carat_high})
                  </span>
                </span>
              </div>
            ) : gemWeight ? (
              <div className={styles.weightRow}>
                <span className={styles.weightLabel}>Gem weight</span>
                <span className={styles.weightMuted}>needs calibration card</span>
              </div>
            ) : null}
          </div>

          {Array.isArray(gemWeight?.stones) && gemWeight.stones.length > 0 && (
            <div className={styles.gemChips}>
              {gemWeight.stones.map((g, i) => {
                const named = g.stone_name && g.stone_name !== 'unidentified'
                const label = named
                  ? g.stone_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
                  : (g.hue_class || 'stone')
                return (
                  <span key={i} className={styles.gemChip}>
                    <i style={{ background: HUE_COLOURS[g.hue_class] || HUE_COLOURS.other }} />
                    <Gem size={11} />
                    #{i + 1} {label}
                    {g.diameter_mm != null && <> · {g.diameter_mm} mm</>}
                    {g.est_carat != null && <> · ≈{g.est_carat} ct</>}
                    {g.diameter_mm == null && g.area_pct != null && <> · {g.area_pct}%</>}
                  </span>
                )
              })}
            </div>
          )}

          {gemWeight?.note && <p className={styles.note}>{gemWeight.note}</p>}
        </div>
      </div>
    </div>
  )
}
