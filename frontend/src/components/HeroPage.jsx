import { useNavigate } from 'react-router-dom'
import { Shield, ArrowRight, Check, Minus, Zap, Eye, Waves, GitMerge } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import styles from './HeroPage.module.css'

const STATS = [
  { num: '99.9%', label: 'Acoustic AUC' },
  { num: '4',     label: 'AI modalities' },
  { num: '3',     label: 'Novel signals' },
  { num: '<10s',  label: 'Time to verdict' },
]

const STEPS = [
  {
    n: 1,
    title: 'Density test',
    desc:  'Archimedes principle — dry vs. submerged weight. 24K gold = 19.32 g/cm³. Non-destructive, no chemicals.',
    badge: 'Always required',
    Icon:  Zap,
  },
  {
    n: 2,
    title: 'Acoustic ring',
    desc:  'MFCC-ΔΔ features capture ring decay of pure gold. Plated or tungsten-core items sound fundamentally different.',
    badge: 'Novelty 1',
    Icon:  Waves,
  },
  {
    n: 3,
    title: 'Visual analysis',
    desc:  'EfficientNet-B3 embeddings detect surface plating, discoloration, and forgery marks across multiple photo angles.',
    badge: '4–6 photos',
    Icon:  Eye,
  },
  {
    n: 4,
    title: 'AI fusion',
    desc:  'XGBoost fuses all four signals. SHAP explains every decision. Groq LLM writes a branch-ready action statement.',
    badge: 'SHAP explainable',
    Icon:  GitMerge,
  },
]

const NOVELTIES = [
  {
    n:     '01',
    tag:   'Novelty 1',
    title: 'MFCC-ΔΔ Acoustic Fingerprinting',
    desc:  'Delta-delta MFCC captures temporal ring decay unique to pure gold. Plated brass or tungsten-core items produce a fundamentally different acoustic envelope — undetectable by the human ear.',
    pill:  'SVM · RBF kernel · AUC 0.999',
  },
  {
    n:     '02',
    tag:   'Novelty 2',
    title: "Benford's Law Density Monitor",
    desc:  "First digits of submerged weights follow Benford's distribution for genuine items. Fraud rings show detectable deviation — catching branch-level collusion in real time, not weeks later.",
    pill:  'Population-level fraud ring detection',
  },
  {
    n:     '03',
    tag:   'Novelty 3',
    title: 'Cross-Modal Contradiction Detection',
    desc:  'Density passes but acoustic fails? That contradiction is itself a high-value signal. Six modality-pair scores are fed into XGBoost — critical for tungsten-core fraud where density alone passes.',
    pill:  '0.40× contradiction boost in fusion',
  },
]

const FRAUD_ROWS = [
  { type: 'Gold-plated base metal',   sub: 'Brass or copper core with gold layer',    trad: 'Acid test (destructive) or XRF',    kanchan: 'Density + visual + acoustic',         full: true  },
  { type: 'Tungsten-core bar',        sub: 'Density ≈ 24K gold (19.25 vs 19.32)',      trad: 'XRF or drill test — invasive',       kanchan: 'Cross-modal contradiction detection', full: true  },
  { type: 'Under-karat item',         sub: '22K declared, 18K actual',                 trad: 'Touchstone + acid reagent',          kanchan: 'Density deviation model',             full: true  },
  { type: 'Coordinated fraud ring',   sub: 'Multiple staff colluding on appraisals',   trad: 'Manual audit — weeks later',         kanchan: "Benford's Law realtime monitor",      full: true  },
  { type: 'Surface-only plating',     sub: 'Thin gold layer on silver or alloy',       trad: 'Visual inspection — unreliable',     kanchan: 'EfficientNet-B3 visual probe',        full: true  },
  { type: 'Recycled or mixed gold',   sub: 'Melted mix below declared karat',          trad: 'Fire assay — lab only',              kanchan: 'Density range check (partial)',        full: false },
]

export default function HeroPage() {
  const navigate = useNavigate()

  return (
    <div className={styles.page}>
      <div className={styles.dotGrid} aria-hidden="true" />
      <div className={styles.spotlight} aria-hidden="true" />
      <div className={styles.spotlightYellow} aria-hidden="true" />

      {/* ── Nav ── */}
      <nav className={styles.nav}>
        <div className={styles.navBrand}>
          <div className={styles.navMark}><Shield size={16} strokeWidth={2.5} /></div>
          <span className={styles.navName}>KANCHAN<span className={styles.navAI}>-AI</span></span>
        </div>
        <div className={styles.navRight}>
          <span className={styles.navBadge}>
            <span className={styles.navDot} />
            SuRaksha Cyber Hackathon 2.0
          </span>
          <Button size="sm" onClick={() => navigate('/dashboard')}>
            Dashboard <ArrowRight size={13} />
          </Button>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className={styles.hero}>
        <div className={styles.heroEyebrow}>
          <span className={styles.eyebrowDot} />
          Canara Bank · IISc Bangalore
        </div>
        <h1 className={styles.heroH1}>
          Detect{' '}
          <span className={styles.heroAccent}>Spurious Gold</span>
          <br />
          in{' '}
          <span className={styles.heroAccentYellow}>Under 10 Seconds</span>
        </h1>
        <p className={styles.heroSub}>
          Non-destructive, branch-deployable fraud detection for gold loan appraisal.
          Four AI modalities fused into one verdict — with full SHAP explainability for every decision.
        </p>
        <div className={styles.heroActions}>
          <Button size="lg" onClick={() => navigate('/dashboard')}>
            Launch dashboard <ArrowRight size={16} />
          </Button>
          <Button size="lg" variant="outline">
            View architecture
          </Button>
        </div>
      </section>

      {/* ── Stats ── */}
      <div style={{ maxWidth: 900, margin: '0 auto', padding: '0 24px 96px' }}>
        <div className={styles.statsBar}>
          {STATS.map(s => (
            <div key={s.label} className={styles.stat}>
              <span className={styles.statNum}>{s.num}</span>
              <span className={styles.statLabel}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Content ── */}
      <div className={styles.contentWrap}>

        {/* Steps */}
        <section className={styles.section}>
          <p className={styles.sectionLabel}>How it works</p>
          <h2 className={styles.sectionH2}>Four independent signals.<br />One fused verdict.</h2>
          <p className={styles.sectionSub}>Each modality runs independently. XGBoost detects when signals contradict each other — a critical indicator of sophisticated fraud.</p>
          <div className={styles.steps}>
            {STEPS.map(s => (
              <div key={s.n} className={styles.step}>
                <div className={styles.stepTop}>
                  <span className={styles.stepNum}>{s.n}</span>
                  <Badge variant="secondary">{s.badge}</Badge>
                </div>
                <div className={styles.stepIcon}><s.Icon size={20} strokeWidth={1.75} /></div>
                <div className={styles.stepTitle}>{s.title}</div>
                <div className={styles.stepDesc}>{s.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Novelties */}
        <section className={styles.section}>
          <p className={styles.sectionLabel}>What makes us different</p>
          <h2 className={styles.sectionH2}>Three novel contributions</h2>
          <p className={styles.sectionSub}>Built for the Indian gold loan context — addressing fraud vectors that traditional XRF and visual inspection miss entirely.</p>
          <div className={styles.novelties}>
            {NOVELTIES.map(n => (
              <div key={n.n} className={styles.novelty}>
                <div className={styles.noveltyHeader}>
                  <span className={styles.noveltyN}>{n.n}</span>
                  <Badge variant="secondary">{n.tag}</Badge>
                </div>
                <div className={styles.noveltyTitle}>{n.title}</div>
                <p className={styles.noveltyDesc}>{n.desc}</p>
                <span className={styles.noveltyPill}>{n.pill}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Fraud table */}
        <section className={styles.section}>
          <p className={styles.sectionLabel}>Detection coverage</p>
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
                  {row.full
                    ? <Check size={13} strokeWidth={2.5} />
                    : <Minus size={13} strokeWidth={2} />}
                  {row.kanchan}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Preview */}
        <section className={styles.section}>
          <p className={styles.sectionLabel}>Live verdict preview</p>
          <h2 className={styles.sectionH2}>What a bank officer sees</h2>
          <div className={styles.previewWrap}>
            <div className={styles.previewCard}>
              <div className={styles.previewCaseId}>KANCHAN-AI · Case #a4f7c821</div>
              <div className={styles.previewTopRow}>
                <span className={styles.previewVerdict}>Genuine</span>
                <span className={styles.previewConf}>HIGH confidence · Approve loan</span>
              </div>
              <div className={styles.previewMeter}>
                <div className={styles.previewMeterLabel}>
                  <span>Fusion risk score</span>
                  <strong>18%</strong>
                </div>
                <div className={styles.previewTrack}>
                  <div className={styles.previewFill} style={{ width: '18%' }} />
                </div>
              </div>
              {[
                { label: 'Density',  pct: 12, color: 'var(--ok)' },
                { label: 'Acoustic', pct: 8,  color: 'var(--ok)' },
                { label: 'Visual',   pct: 21, color: 'var(--ok)' },
                { label: 'Streak',   pct: 34, color: 'var(--warn)' },
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
              <h3 className={styles.previewCtaTitle}>
                Ready to try it?
              </h3>
              <p className={styles.previewCtaDesc}>
                Upload photos, audio, and weight measurements. Get a real SHAP-explainable verdict in under 10 seconds.
              </p>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <Button size="lg" onClick={() => navigate('/dashboard')}>
                  Open analysis dashboard <ArrowRight size={16} />
                </Button>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {[
                  'No destructive testing required',
                  'Branch-deployable — no lab needed',
                  'SHAP explanation for every verdict',
                ].map(f => (
                  <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 14, color: 'var(--text-mid)' }}>
                    <div style={{ width: 20, height: 20, borderRadius: '50%', background: 'var(--brand-muted)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                      <Check size={11} strokeWidth={2.5} color="var(--brand)" />
                    </div>
                    {f}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div className={styles.footerLogo}><Shield size={12} /></div>
          <p>KANCHAN-AI · SuRaksha Cyber Hackathon 2.0 · Canara Bank · IISc Bangalore</p>
        </div>
      </footer>
    </div>
  )
}
