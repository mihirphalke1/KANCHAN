import { useState } from 'react'
import { Scan, ZoomIn } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import Lightbox from './ui/Lightbox'
import styles from './XRayView.module.css'

// Build a URL that works in both dev (proxied) and prod (same origin)
function mediaUrl(path) {
  if (!path) return null
  if (path.startsWith('data:')) return path          // embedded image
  // legacy path = "data/cases/{id}/xray/material.png" → "/cases/{id}/..."
  return '/' + path.replace(/\\/g, '/').replace(/^data\//, '')
}

const STAGES = [
  { key: 'original',  label: 'Original',
    desc: 'Your photo, exactly as uploaded. Everything below is worked out from this one image.' },
  { key: 'grey',      label: 'Greyscale',
    desc: 'The photo in shades of grey — brightness only, no colour. Polished gold looks bright, stones and shadows look darker. This is what the rest of the analysis measures.' },
  { key: 'invert',    label: 'Inverted (X-ray look)',
    desc: 'Brightness flipped: dense metal turns dark and lighter parts turn bright — the way a real X-ray film looks. This is where the “X-ray” name comes from; it is still your ordinary photo, just re-shaded.' },
  { key: 'threshold', label: 'Sorted into materials',
    desc: 'Every pixel is sorted into one of four brightness bands — the step that tells metal, joints, stones and bright facets apart.' },
  { key: 'histogram', label: 'Brightness chart',
    desc: 'A tally of how bright the item’s pixels are, from dark (left) to bright (right), coloured by material band. The dashed lines mark where the four bands are split. Shown for technical verification — an officer doesn’t need to read it.' },
  { key: 'sobel',     label: 'Edges',
    desc: 'Only the outlines — the lines where one material meets another (a stone against its gold setting, the rim of a prong). Flat, even surfaces stay dark.' },
  { key: 'hsv',       label: 'Colour (HSV)',
    desc: 'The photo converted to HSV and redrawn at full strength — every pixel keeps its own hue but is shown at maximum saturation and brightness. This is what "a stone looks different by colour" means made visible: the gold reads as one steady colour, and a stone stands out clearly even if it looked dull or shadowed in the original photo.' },
  { key: 'material',  label: 'Material map',
    desc: 'The four material bands shown in colour, with the item lifted off its background. This is the core visual result — metal, stones and joints separated, the same idea as a lab element map but from a normal photo.' },
  { key: 'gold_gem',  label: 'Gold vs Gems',
    desc: 'Colour-based split of the ornament: warm gold for metal that matches the item’s own gold colour, jewel tones for each detected gem, and slate for anything else (solder, rhodium plating, unexplained patches). Built from Lab colour distance + HSV gold band + the stone mask — visual only, does not change the loan decision.' },
  { key: 'gems',      label: 'Stones found',
    desc: 'Stones the camera picked out on its own, numbered — it never reads the description. Clear or white stones get a “?”: the officer confirms whether each one is really a stone.' },
  { key: 'heatmap',   label: 'Heat view',
    desc: 'Brightness shown on a blue-to-red colour scale. It makes faint differences on the surface easier to spot than plain grey.' },
]

// Must stay in sync with CLASS_COLOURS_RGB in app/utils/xray.py
const MATERIALS = [
  { key: 'gemstone', label: 'Gemstone', colour: '#DC2626' },
  { key: 'joint',    label: 'Joints',   colour: '#0F766E' },
  { key: 'metal',    label: 'Metal',    colour: '#1E3A5F' },
  { key: 'facet',    label: 'Facets',   colour: '#D97706' },
]

// Must stay in sync with GOLD_GEM_COLOURS_BGR in app/utils/xray.py
const GOLD_GEM_LEGEND = [
  { key: 'gold',  label: 'Gold metal', colour: '#E6AA28', pctKey: 'gold_pct' },
  { key: 'gem',   label: 'Gems',       colour: '#DC285A', pctKey: 'gem_pct' },
  { key: 'other', label: 'Other',      colour: '#5A646E', pctKey: 'other_pct' },
]

function riskColour(risk) {
  if (risk < 0.3) return '#15803D'
  if (risk < 0.6) return '#B45309'
  return '#B91C1C'
}

// Stone hue swatch colours (kept in sync with GoldVsGemsCard).
const HUE_COLOURS = {
  red: '#DC2626', green: '#15803D', blue: '#2563EB',
  other: '#6B7280', colourless: '#D4D4D8',
}

// How each detection mode is labelled in the badge row.
const MODE_META = {
  ml_ai:     { label: 'AI-led + ML verified', style: { background: '#0F766E18', color: '#0F766E' },
               title: 'An AI vision model is the primary judge of how many stones there are and what each one is; the ML/MobileSAM detector supplies precise boundaries and a safety net. Both agree = highest confidence; a stone only the ML saw is kept as secondary evidence.' },
  ml_sam:    { label: 'ML-refined (MobileSAM)', style: { background: '#6D28D918', color: '#6D28D9' },
               title: 'Stone boundaries refined by a pretrained MobileSAM segmentation pass' },
  classical: { label: 'Classical detection', style: undefined,
               title: 'MobileSAM unavailable this run — classical colour-threshold detection used' },
}

// Per-stone agreement chip (who saw this stone). The "verify" suffix is added
// from the stone's own status, since under AI-primary an AI-led find can be
// confirmed outright.
const AGREEMENT_META = {
  both:    { label: 'AI + ML agree', dot: '#15803D', title: 'Identified by the AI (primary judge) and independently outlined by the ML model — highest confidence.' },
  ai_only: { label: 'AI (primary)',  dot: '#0F766E', title: 'Identified by the AI vision model, the primary judge of stone type and count. The ML detector did not separately outline it.' },
  ml_only: { label: 'ML only',       dot: '#6B7280', title: 'Found only by the secondary ML detector; the AI (primary judge) did not confirm it — treat as supporting evidence.' },
}

function prettyStoneName(s) {
  const n = s?.stone_name
  if (n && n !== 'unidentified' && n !== 'stone') {
    return n.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }
  return (s?.hue_class || 'stone').replace(/\b\w/g, c => c.toUpperCase())
}

export default function XRayView({ caseData }) {
  const xray = caseData?.media?.xray
  const xrayScore = caseData?.modality_scores?.xray
  // When the item can't be separated from its background, every derived
  // number (gems, inclusions, composition) is measured on the scene, not
  // the item — so we show NO findings, and default to the plain photo.
  const usable = xray?.background_removed === true
  // The AI stone judge reads the ornament directly, so it can find stones even
  // when the backdrop could not be separated (a busy/leafy background). In that
  // case we still surface the stones + the gem overlay, while the area-based
  // numbers (composition, gold-vs-gems) stay hidden since they need the item mask.
  const hasStones = Array.isArray(xray?.stones) && xray.stones.length > 0
  const aiFoundDespiteBg = !usable && hasStones
  // Prefer the new gold-vs-gems map when available; fall back to material map;
  // when unusable-but-AI-found-stones, show the gem overlay; else the plain photo.
  const defaultStage = usable
    ? (xray?.stages?.gold_gem ? 'gold_gem' : 'material')
    : (aiFoundDespiteBg && xray?.stages?.gems ? 'gems' : 'original')
  const [active, setActive] = useState(defaultStage)
  const [zoom, setZoom] = useState(null)

  if (!xray?.stages) return null

  const { stages, composition = {}, thresholds = {}, gem_regions,
          item_area_pct, inclusions_unexplained, stone_detection_mode,
          stone_agreement, stones, gold_gem_split } = xray
  const activeStage = STAGES.find(s => s.key === active) || STAGES[0]
  const hasScore = xrayScore?.mode === 'dsip_xray'
  const showSignals = hasScore || xrayScore?.mode === 'dsip_unusable'
  const showGoldGem = usable && gold_gem_split && stages.gold_gem

  return (
    <div className={styles.card}>
      <div className={styles.titleRow}>
        <h3 className={styles.title}>
          <Scan size={14} />
          Photo Material Scan
          <InfoTip text="Classical image processing separates the item from the backdrop and splits the surface into material classes. The Gold vs Gems stage paints metal that matches the item’s own gold colour in warm gold, and each detected stone in a jewel tone — colour difference only, never the typed description. Stone boundaries are refined by MobileSAM when available. Click the small images to see every processing stage." side="right" />
        </h3>
        <div className={styles.badgeRow}>
          {hasScore && (
            <span
              className={styles.badge}
              style={{ background: `${riskColour(xrayScore.risk_score)}18`, color: riskColour(xrayScore.risk_score) }}
            >
              risk {Math.round(xrayScore.risk_score * 100)}%
            </span>
          )}
          {(usable || hasStones) && Number.isFinite(gem_regions) && (
            <span className={styles.badge}>
              {gem_regions} gem{gem_regions === 1 ? '' : 's'} detected
            </span>
          )}
          {(usable || hasStones) && stone_detection_mode && (() => {
            const m = MODE_META[stone_detection_mode] || MODE_META.classical
            return <span className={styles.badge} title={m.title} style={m.style}>{m.label}</span>
          })()}
          {usable && stone_agreement && stone_agreement.ai_used === false && (
            <span
              className={styles.badge}
              style={{ background: '#B4530912', color: '#B45309' }}
              title="The AI vision judge is off, so stones are shown from the ML detector only. Set FIREWORKS_API_KEY (or GOOGLE_API_KEY) and USE_AI_STONE_CONFIRM=1 to let the AI lead stone identification and count."
            >
              AI check off
            </span>
          )}
          {usable && Number.isFinite(inclusions_unexplained) && inclusions_unexplained > 0 && (
            <span className={styles.badge} style={{ background: '#B91C1C18', color: '#B91C1C' }}>
              {inclusions_unexplained} unexplained inclusion{inclusions_unexplained === 1 ? '' : 's'}
            </span>
          )}
          {xray.background_removed != null && (
            <span className={styles.badge}
              style={usable ? undefined : { background: '#B4530918', color: '#B45309' }}
              title={usable ? 'Stats computed on item pixels only'
                            : 'The item could not be told apart from the background'}>
              {usable ? `item ${item_area_pct}% of frame` : 'Busy background — retake on a plain surface'}
            </span>
          )}
        </div>
      </div>

      <div className={styles.viewer}>
        <button
          type="button"
          className={styles.zoomBtn}
          onClick={() => setZoom(mediaUrl(stages[active]))}
          title="Click to zoom"
        >
          <img
            src={mediaUrl(stages[active])}
            alt={activeStage.label}
            className={styles.mainImg}
            onError={e => { e.target.style.visibility = 'hidden' }}
          />
          <span className={styles.zoomHint}><ZoomIn size={13} /> Click to zoom</span>
        </button>
        <div className={styles.stageCaption}>
          <strong>{activeStage.label}</strong>
          <span>{activeStage.desc}</span>
        </div>
      </div>

      <div className={styles.thumbStrip}>
        {STAGES.filter(s => stages[s.key]).map(s => (
          <button
            key={s.key}
            type="button"
            className={`${styles.thumb} ${active === s.key ? styles.thumbActive : ''}`}
            onClick={() => setActive(s.key)}
            title={s.desc}
          >
            <img src={mediaUrl(stages[s.key])} alt={s.label} loading="lazy" />
            <span>{s.label}</span>
          </button>
        ))}
      </div>

      {showGoldGem && (
        <div className={styles.compSection}>
          <div className={styles.sectionLabel}>
            Gold vs gems (colour split)
            <span className={styles.thresholdNote}>
              {gold_gem_split.stones_used ?? 0} stone region{gold_gem_split.stones_used === 1 ? '' : 's'} · visual only
            </span>
          </div>
          <div className={styles.compBar}>
            {GOLD_GEM_LEGEND.map(m => {
              const pct = gold_gem_split[m.pctKey] ?? 0
              return pct > 0 ? (
                <div
                  key={m.key}
                  className={styles.compSeg}
                  style={{ width: `${pct}%`, background: m.colour }}
                  title={`${m.label}: ${pct}%`}
                />
              ) : null
            })}
          </div>
          <div className={styles.legend}>
            {GOLD_GEM_LEGEND.map(m => (
              <span key={m.key} className={styles.legendItem}>
                <i style={{ background: m.colour }} />
                {m.label} {gold_gem_split[m.pctKey] ?? 0}%
              </span>
            ))}
          </div>
        </div>
      )}

      {hasStones && (
        <div className={styles.compSection}>
          <div className={styles.sectionLabel}>
            Detected stones
            {stone_agreement?.ai_used && (
              <span className={styles.thresholdNote}>
                {aiFoundDespiteBg
                  ? 'found by the AI directly in the photo (background could not be separated)'
                  : `${stone_agreement.n_both || 0} confirmed by both · ${stone_agreement.n_ai_only || 0} AI-only · ${stone_agreement.n_ml_only || 0} ML-only`}
              </span>
            )}
          </div>
          <div className={styles.stoneList}>
            {stones.map((s, i) => {
              const a = AGREEMENT_META[s.agreement] || AGREEMENT_META.ml_only
              const needsVerify = s.status && s.status !== 'confirmed'
              const dot = needsVerify ? '#B45309' : a.dot
              const label = needsVerify ? `${a.label} — verify` : a.label
              return (
                <div key={i} className={styles.stoneRow}>
                  <span className={styles.stoneIdx}>#{i + 1}</span>
                  <i className={styles.stoneSwatch}
                     style={{ background: HUE_COLOURS[s.hue_class] || HUE_COLOURS.other }} />
                  <span className={styles.stoneName}>{prettyStoneName(s)}</span>
                  {s.colour && <span className={styles.stoneColour}>{s.colour}</span>}
                  <span className={styles.stoneAgree} title={a.title} style={{ color: dot }}>
                    <i style={{ background: dot }} />{label}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {usable && (
        <div className={styles.compSection}>
          <div className={styles.sectionLabel}>
            Material composition
            {thresholds.t1 != null && (
              <span className={styles.thresholdNote}>
                T1 {thresholds.t1} · T2 {thresholds.t2} · T3 {thresholds.t3}
              </span>
            )}
          </div>
          <div className={styles.compBar}>
            {MATERIALS.map(m => {
              const pct = composition[m.key] ?? 0
              return pct > 0 ? (
                <div
                  key={m.key}
                  className={styles.compSeg}
                  style={{ width: `${pct}%`, background: m.colour }}
                  title={`${m.label}: ${pct}%`}
                />
              ) : null
            })}
          </div>
          <div className={styles.legend}>
            {MATERIALS.map(m => (
              <span key={m.key} className={styles.legendItem}>
                <i style={{ background: m.colour }} />
                {m.label} {composition[m.key] ?? 0}%
              </span>
            ))}
          </div>
        </div>
      )}

      {showSignals && xrayScore.signals?.length > 0 && (
        <div className={styles.compSection}>
          <div className={styles.sectionLabel}>
            Findings
            {xrayScore.fusion_contribution && (
              <span className={styles.thresholdNote}>
                feeds fusion: {Math.round(xrayScore.fusion_contribution.xray_weight * 100)}% of visual channel
              </span>
            )}
          </div>
          <ul className={styles.findings}>
            {xrayScore.signals.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      <Lightbox src={zoom} alt={activeStage.label} onClose={() => setZoom(null)} />
    </div>
  )
}
