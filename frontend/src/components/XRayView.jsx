import { useState } from 'react'
import { Scan } from 'lucide-react'
import InfoTip from './ui/InfoTip'
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
    desc: 'The uploaded photograph — the source signal for every stage below.' },
  { key: 'grey',      label: 'Greyscale',
    desc: 'Brightness only (L = 0.21R + 0.72G + 0.07B, weighted like the human eye). Gold reflects strongly → bright; stones absorb their colour band → darker.' },
  { key: 'invert',    label: 'Inverted',
    desc: 'P′ = 255 − P, the radiographic negative: dense metal turns dark, lighter materials turn bright — the way an X-ray film reads.' },
  { key: 'threshold', label: 'Thresholded',
    desc: 'Brightness is cut at three points (T1/T2/T3, set automatically from this item) into four material classes — the core segmentation step.' },
  { key: 'histogram', label: 'Histogram',
    desc: 'How the item’s pixels distribute by brightness, coloured by material class with the cut points marked. Valleys between peaks are natural material boundaries — check the cuts sit in valleys.' },
  { key: 'sobel',     label: 'Sobel edges',
    desc: 'Rate of brightness change. Bright lines = boundaries between materials (metal↔stone, prong edges); dark = uniform surface.' },
  { key: 'material',  label: 'Material map',
    desc: 'The four classes in false colour with boundaries overlaid in white — the same presentation as an X-ray fluorescence element map. Backdrop is greyed out.' },
  { key: 'gems',      label: 'Gem detection',
    desc: 'Stones found independently of the description: coloured stones outlined and numbered, colourless candidates marked with a “?” for confirmation.' },
  { key: 'heatmap',   label: 'Heatmap',
    desc: 'Brightness on the blue→red scientific scale (as in thermal/NDT imaging) — reveals subtle variations invisible in greyscale.' },
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
  const [active, setActive] = useState('material')

  if (!xray?.stages) return null

  const { stages, composition = {}, thresholds = {}, gem_regions,
          background_removed, item_area_pct, inclusions_unexplained } = xray
  const activeStage = STAGES.find(s => s.key === active) || STAGES[0]
  const hasScore = xrayScore?.mode === 'dsip_xray'

  return (
    <div className={styles.card}>
      <div className={styles.titleRow}>
        <h3 className={styles.title}>
          <Scan size={14} />
          Photo Material Scan
          <InfoTip text="Classical image processing on the photo — no AI guesswork. It separates the item from the backdrop, splits the surface into material classes, and finds each stone. Click the small images to see every processing stage." side="right" />
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
          {Number.isFinite(gem_regions) && (
            <span className={styles.badge}>
              {gem_regions} gem{gem_regions === 1 ? '' : 's'} detected
            </span>
          )}
          {Number.isFinite(inclusions_unexplained) && inclusions_unexplained > 0 && (
            <span className={styles.badge} style={{ background: '#B91C1C18', color: '#B91C1C' }}>
              {inclusions_unexplained} unexplained inclusion{inclusions_unexplained === 1 ? '' : 's'}
            </span>
          )}
          {background_removed != null && (
            <span className={styles.badge} title="Stats computed on item pixels only">
              {background_removed ? `item ${item_area_pct}% of frame` : 'backdrop not separable'}
            </span>
          )}
        </div>
      </div>

      <div className={styles.viewer}>
        <img
          src={mediaUrl(stages[active])}
          alt={activeStage.label}
          className={styles.mainImg}
          onError={e => { e.target.style.visibility = 'hidden' }}
        />
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

      {hasScore && xrayScore.signals?.length > 0 && (
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
    </div>
  )
}
