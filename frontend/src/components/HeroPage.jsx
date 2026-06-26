import { Shield, ArrowRight, Check, Minus, ChevronRight } from 'lucide-react'
import styles from './HeroPage.module.css'

const STATS = [
  { num: '99.9%', label: 'Acoustic AUC' },
  { num: '4',     label: 'AI modalities' },
  { num: '3',     label: 'Novel signals' },
  { num: '<10s',  label: 'Time to verdict' },
]

const STEPS = [
  { n: 1, title: 'Density test', desc: 'Archimedes principle: dry vs. submerged weight gives density. 24K gold = 19.32 g/cm³.', badge: 'always required' },
  { n: 2, title: 'Acoustic ring', desc: 'Tap-test recording. MFCC-ΔΔ captures the unique ring decay of pure gold vs. fakes.', badge: 'Novelty 1' },
  { n: 3, title: 'Visual analysis', desc: 'EfficientNet-B3 embeddings detect surface plating, discoloration, and forgery marks.', badge: '4–6 photos' },
  { n: 4, title: 'AI verdict', desc: 'XGBoost fuses all signals. SHAP values explain every decision. LLM writes the action.', badge: 'SHAP explainable' },
]

const NOVELTIES = [
  {
    n: 'Novelty 1',
    title: 'MFCC-ΔΔ acoustic fingerprinting',
    desc: 'Delta-delta MFCC captures the temporal ring decay of pure gold. Fake items — plated brass, tungsten core — produce a fundamentally different acoustic envelope, undetectable by the human ear.',
    pill: 'SVM · RBF kernel · AUC 0.999',
  },
  {
    n: 'Novelty 2',
    title: "Benford's Law density monitor",
    desc: "At branch population level, first digits of submerged weight measurements follow Benford's distribution for genuine items. Organised fraud rings show detectable deviation — catching collusion.",
    pill: 'Population-level fraud ring detection',
  },
  {
    n: 'Novelty 3',
    title: 'Cross-modal contradiction detection',
    desc: 'Density says genuine, acoustic says fake? That contradiction is itself a signal. Six modality-pair scores are injected into XGBoost — critical for tungsten-core fraud where density alone passes.',
    pill: '0.40× contradiction boost in fusion',
  },
]

const FRAUD_ROWS = [
  { type: 'Gold-plated base metal', sub: 'Brass/copper core with gold layer', trad: 'Acid test (destructive) or XRF', kanchan: 'Density + visual + acoustic', full: true },
  { type: 'Tungsten-core bar', sub: 'Density ≈ 24K gold (19.25 vs 19.32)', trad: 'XRF or drill test — invasive', kanchan: 'Cross-modal contradiction detection', full: true },
  { type: 'Under-karat item', sub: '22K declared, 18K actual', trad: 'Touchstone + acid reagent', kanchan: 'Density deviation model', full: true },
  { type: 'Coordinated branch fraud', sub: 'Multiple staff colluding on appraisals', trad: 'Manual audit — weeks later', kanchan: "Benford's Law realtime monitor", full: true },
  { type: 'Surface-only plating', sub: 'Thin gold layer on silver/alloy', trad: 'Visual inspection — unreliable', kanchan: 'EfficientNet-B3 visual probe', full: true },
  { type: 'Recycled/mixed gold', sub: 'Melted mix below declared karat', trad: 'Fire assay — lab only', kanchan: 'Density range check (partial)', full: false },
]

export default function HeroPage({ onLaunch }) {
  return (
    <div className={styles.page}>
      <nav className={styles.nav}>
        <div className={styles.navBrand}>
          <div className={styles.navMark}><Shield size={16} strokeWidth={2.5} /></div>
          <span className={styles.navName}>KANCHAN<span>-AI</span></span>
        </div>
        <div className={styles.navRight}>
          <span className={styles.navBadge}>SuRaksha Cyber Hackathon 2.0</span>
          <button className={styles.navCta} onClick={onLaunch}>Open dashboard <ArrowRight size={14} /></button>
        </div>
      </nav>

      <section className={styles.hero}>
        <div className={styles.heroEyebrow}>Canara Bank · IISc Bangalore</div>
        <h1 className={styles.heroH1}>
          Detect <span className={styles.heroAccent}>Spurious Gold</span><br />
          in Under 10 Seconds
        </h1>
        <p className={styles.heroSub}>
          Non-destructive, branch-deployable fraud detection for gold loan appraisal.
          Four AI modalities fused into a single verdict — with full SHAP explainability for every decision.
        </p>
        <div className={styles.heroActions}>
          <button className={styles.heroCta} onClick={onLaunch}>
            <ArrowRight size={16} /> Launch dashboard
          </button>
          <button className={styles.heroSecondary}>Watch 60-second demo</button>
        </div>
      </section>

      <div className={styles.statsBar}>
        {STATS.map(s => (
          <div key={s.label} className={styles.stat}>
            <span className={styles.statNum}>{s.num}</span>
            <span className={styles.statLabel}>{s.label}</span>
          </div>
        ))}
      </div>

      <div className={styles.contentWrap}>
        <section className={styles.section}>
          <p className={styles.sectionTag}>How it works</p>
          <h2 className={styles.sectionH2}>Four independent signals. One fused verdict.</h2>
          <p className={styles.sectionSub}>Each modality runs independently. The XGBoost fusion engine detects when signals contradict each other — a key indicator of sophisticated fraud.</p>
          <div className={styles.steps}>
            {STEPS.map(s => (
              <div key={s.n} className={styles.step}>
                <div className={styles.stepNum}>{s.n}</div>
                <div className={styles.stepTitle}>{s.title}</div>
                <div className={styles.stepDesc}>{s.desc}</div>
                <span className={styles.stepBadge}>{s.badge}</span>
              </div>
            ))}
          </div>
        </section>

        <section className={styles.section}>
          <p className={styles.sectionTag}>What makes us different</p>
          <h2 className={styles.sectionH2}>Three novel contributions</h2>
          <p className={styles.sectionSub}>Built specifically for the Indian gold loan context — addressing fraud vectors that traditional XRF and visual inspection miss entirely.</p>
          <div className={styles.novelties}>
            {NOVELTIES.map(n => (
              <div key={n.n} className={styles.novelty}>
                <div className={styles.noveltyN}>{n.n}</div>
                <div className={styles.noveltyTitle}>{n.title}</div>
                <p className={styles.noveltyDesc}>{n.desc}</p>
                <span className={styles.noveltyPill}>{n.pill}</span>
              </div>
            ))}
          </div>
        </section>

        <section className={styles.section}>
          <p className={styles.sectionTag}>Detection coverage</p>
          <h2 className={styles.sectionH2}>Fraud types we detect</h2>
          <div className={styles.fraudTable}>
            <div className={`${styles.fraudHeader} ${styles.fraudCol1}`}>Fraud type</div>
            <div className={`${styles.fraudHeader} ${styles.fraudCol2}`}>Traditional method</div>
            <div className={`${styles.fraudHeader} ${styles.fraudCol3}`}>KANCHAN-AI</div>
            {FRAUD_ROWS.map((row, i) => (
              <div key={i} className={styles.fraudRowGroup}>
                <div className={`${styles.fraudCell} ${styles.fraudType}`}>
                  <span className={styles.fraudTypeName}>{row.type}</span>
                  <span className={styles.fraudTypeSub}>{row.sub}</span>
                </div>
                <div className={`${styles.fraudCell} ${styles.fraudTrad}`}>{row.trad}</div>
                <div className={`${styles.fraudCell} ${styles.fraudKanchan} ${row.full ? styles.full : styles.partial}`}>
                  {row.full ? <Check size={13} strokeWidth={2.5} /> : <Minus size={13} strokeWidth={2} />}
                  {row.kanchan}
                </div>
              </div>
            ))}
          </div>
        </section>

        <section className={styles.section}>
          <p className={styles.sectionTag}>Live verdict preview</p>
          <h2 className={styles.sectionH2}>What a bank officer sees</h2>
          <div className={styles.previewWrap}>
            <div className={styles.previewCard}>
              <div className={styles.previewCaseId}>KANCHAN-AI · Case #a4f7c821</div>
              <div className={styles.previewTopRow}>
                <span className={styles.previewVerdict}>Genuine</span>
                <span className={styles.previewConf}>HIGH confidence · Approve loan</span>
              </div>
              <div className={styles.previewMeter}>
                <div className={styles.previewMeterLabel}>Fusion risk score — 18%</div>
                <div className={styles.previewTrack}><div className={styles.previewFill} style={{ width: '18%' }} /></div>
              </div>
              {[
                { label: 'Density',  pct: 12, color: '#4ade80' },
                { label: 'Acoustic', pct: 8,  color: '#4ade80' },
                { label: 'Visual',   pct: 21, color: '#4ade80' },
                { label: 'Streak',   pct: 34, color: '#fbbf24' },
              ].map(b => (
                <div key={b.label} className={styles.previewBarRow}>
                  <span className={styles.previewBarLabel}>{b.label}</span>
                  <div className={styles.previewBarTrack}>
                    <div className={styles.previewBarFill} style={{ width: `${b.pct}%`, background: b.color }} />
                  </div>
                  <span className={styles.previewBarPct}>{b.pct}%</span>
                </div>
              ))}
              <div className={styles.previewTags}>
                <span className={styles.previewTag}>22K declared · 17.82 g/cm³</span>
                <span className={styles.previewTag}>Ring decay nominal</span>
                <span className={styles.previewTag}>No surface anomaly</span>
                <span className={styles.previewTag}>via Groq Llama-3</span>
              </div>
            </div>
            <div className={styles.previewCta}>
              <h3 className={styles.previewCtaTitle}>Ready to try it?</h3>
              <p className={styles.previewCtaDesc}>Upload an item's photos, audio, and weight measurements to get a real verdict in under 10 seconds.</p>
              <button className={styles.previewCtaBtn} onClick={onLaunch}>
                Open analysis dashboard <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </section>
      </div>

      <footer className={styles.footer}>
        <p>KANCHAN-AI · SuRaksha Cyber Hackathon 2.0 · Canara Bank · IISc Bangalore</p>
      </footer>
    </div>
  )
}
