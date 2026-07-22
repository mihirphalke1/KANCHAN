import { useMemo, useState } from 'react'
import { Box, Scan, ZoomIn } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import Lightbox from './ui/Lightbox'
import Mesh3dViewer from './Mesh3dViewer'
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
  { key: 'mesh3d',    label: '3D Model',
    desc: 'Interactive 3D reconstruction from the uploaded photo (Microsoft TRELLIS on Hugging Face). Generated in the background after analysis — illustrative only, not used for the loan decision.' },
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

// How each detection mode is labelled in the badge row. The vision model and
// the segmentation detector are presented as one system ("ML") to the officer.
const MODE_META = {
  ml_ai:     { label: 'ML detection', style: { background: '#0F766E18', color: '#0F766E' },
               title: 'Stones are found, counted and identified by the ML vision system and its segmentation pass, then reconciled. Stones both stages agree on are highest confidence; ones only one stage saw are flagged for review.' },
  ml_sam:    { label: 'ML detection', style: { background: '#6D28D918', color: '#6D28D9' },
               title: 'Stone boundaries refined by a pretrained MobileSAM segmentation pass' },
  classical: { label: 'Classical detection', style: undefined,
               title: 'MobileSAM unavailable this run — classical colour-threshold detection used' },
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

  // Group identical stones (same name + colour) so a pavé of a dozen
  // "Diamond · White" reads as ONE compact row with a ×count, instead of a
  // dozen near-identical lines running down the screen. `flagged` counts how
  // many in the group still need an officer to confirm.
  const groupedStones = useMemo(() => {
    const list = caseData?.media?.xray?.stones || []
    const map = new Map()
    list.forEach((s) => {
      const name = prettyStoneName(s)
      const key = `${name}|${s.colour || ''}|${s.hue_class || ''}`
      if (!map.has(key)) {
        map.set(key, { name, colour: s.colour, hue_class: s.hue_class, count: 0, flagged: 0 })
      }
      const g = map.get(key)
      g.count += 1
      if (s.needs_review || (s.status && s.status !== 'confirmed')) g.flagged += 1
    })
    return [...map.values()].sort((a, b) => b.count - a.count)
  }, [caseData])

  if (!xray?.stages) return null

  const { stages, composition = {}, thresholds = {}, gem_regions,
          item_area_pct, inclusions_unexplained, stone_detection_mode,
          stone_agreement, stones, gold_gem_split, mesh3d } = xray
  const mesh3dState = mesh3d || caseData?.media?.mesh3d
  const caseId = caseData?.case_id
  const activeStage = STAGES.find(s => s.key === active) || STAGES[0]
  const hasScore = xrayScore?.mode === 'dsip_xray'
  const showSignals = hasScore || xrayScore?.mode === 'dsip_unusable'
  const showGoldGem = usable && gold_gem_split && stages.gold_gem
  const showMesh3dTab = Boolean(caseId && (mesh3dState || stages))

  return (
    <div className={styles.card}>
      {/* Background poller — keeps the TRELLIS job warm while other stages are open */}
      {showMesh3dTab && active !== 'mesh3d' && (
        <Mesh3dViewer caseId={caseId} initial={mesh3dState} visible={false} />
      )}

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
              title="The ML vision model is off, so stones are shown from the segmentation detector only. Set FIREWORKS_API_KEY (or GOOGLE_API_KEY) and USE_AI_STONE_CONFIRM=1 to let the full ML pipeline lead stone identification and count."
            >
              ML vision off
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
        {active === 'mesh3d' ? (
          <>
            <Mesh3dViewer caseId={caseId} initial={mesh3dState} visible />
            <div className={styles.stageCaption}>
              <strong>{activeStage.label}</strong>
              <span>{activeStage.desc}</span>
            </div>
          </>
        ) : (
          <>
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
          </>
        )}
      </div>

      <div className={styles.thumbStrip}>
        {STAGES.filter(s => s.key === 'mesh3d' ? showMesh3dTab : stages[s.key]).map(s => (
          <button
            key={s.key}
            type="button"
            className={`${styles.thumb} ${active === s.key ? styles.thumbActive : ''}`}
            onClick={() => setActive(s.key)}
            title={s.desc}
          >
            {s.key === 'mesh3d' ? (
              <span className={styles.thumb3d}>
                <Box size={18} />
              </span>
            ) : (
              <img src={mediaUrl(stages[s.key])} alt={s.label} loading="lazy" />
            )}
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
            <span className={styles.thresholdNote}>
              {aiFoundDespiteBg
                ? 'found by ML directly in the photo (background could not be separated)'
                : `${stones.length} detected`
                  + (stone_agreement?.n_needs_review ? ` · ${stone_agreement.n_needs_review} flagged for review` : '')}
            </span>
          </div>
          <div className={styles.stoneGrid}>
            {groupedStones.map((g, i) => (
              <div key={i} className={styles.stoneChip}>
                <i className={styles.stoneSwatch}
                   style={{ background: HUE_COLOURS[g.hue_class] || HUE_COLOURS.other }} />
                <span className={styles.stoneName}>{g.name}</span>
                {g.count > 1 && <span className={styles.stoneCount}>×{g.count}</span>}
                {g.colour && <span className={styles.stoneColour}>{g.colour}</span>}
                {g.flagged > 0 && (
                  <span className={styles.stoneFlag}
                        title="These need an officer to confirm each one is really a stone (clear/white stones can be hard to tell apart from bright metal).">
                    {g.flagged} to review
                  </span>
                )}
              </div>
            ))}
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
