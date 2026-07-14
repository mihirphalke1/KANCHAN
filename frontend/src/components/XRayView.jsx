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
  return '/' + path.replace(/^data\//, '')
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

function riskColour(risk) {
  if (risk < 0.3) return '#15803D'
  if (risk < 0.6) return '#B45309'
  return '#B91C1C'
}

export default function XRayView({ caseData }) {
  const xray = caseData?.media?.xray
  const xrayScore = caseData?.modality_scores?.xray
  // When the item can't be separated from its background, every derived
  // number (gems, inclusions, composition) is measured on the scene, not
  // the item — so we show NO findings, and default to the plain photo.
  const usable = xray?.background_removed === true
  const [active, setActive] = useState(usable ? 'material' : 'original')
  const [zoom, setZoom] = useState(null)

  if (!xray?.stages) return null

  const { stages, composition = {}, thresholds = {}, gem_regions,
          item_area_pct, inclusions_unexplained, stone_detection_mode } = xray
  const activeStage = STAGES.find(s => s.key === active) || STAGES[0]
  const hasScore = xrayScore?.mode === 'dsip_xray'
  const showSignals = hasScore || xrayScore?.mode === 'dsip_unusable'

  return (
    <div className={styles.card}>
      <div className={styles.titleRow}>
        <h3 className={styles.title}>
          <Scan size={14} />
          Photo Material Scan
          <InfoTip text="Classical image processing separates the item from the backdrop and splits the surface into material classes. Stone boundaries are refined by a pretrained MobileSAM segmentation pass when available (falls back to classical colour-threshold detection otherwise) — stone TYPE is still decided by the calibrated, auditable colour/saturation rule, never a black box. Click the small images to see every processing stage." side="right" />
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
          {usable && Number.isFinite(gem_regions) && (
            <span className={styles.badge}>
              {gem_regions} gem{gem_regions === 1 ? '' : 's'} detected
            </span>
          )}
          {usable && stone_detection_mode && (
            <span
              className={styles.badge}
              title={stone_detection_mode === 'ml_sam'
                ? 'Stone boundaries refined by a pretrained MobileSAM segmentation pass'
                : 'MobileSAM unavailable this run — classical colour-threshold detection used'}
              style={stone_detection_mode === 'ml_sam' ? { background: '#6D28D918', color: '#6D28D9' } : undefined}
            >
              {stone_detection_mode === 'ml_sam' ? 'ML-refined (MobileSAM)' : 'Classical detection'}
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
