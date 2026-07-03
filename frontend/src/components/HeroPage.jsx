import { useNavigate } from 'react-router-dom'
import {
  Shield, ArrowRight, Check, Minus, Zap, Eye, Waves, GitMerge,
  ChevronRight, BarChart3, Brain, FlaskConical, Clock, AlertTriangle,
  CheckCircle2, XCircle, Lock, Cpu, TrendingUp, Database
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import styles from './HeroPage.module.css'

const STATS = [
  { num: '0.999', label: 'Acoustic AUC',      sub: 'SVM on MFCC-ΔΔ' },
  { num: '4',     label: 'AI modalities',      sub: 'Independent signals' },
  { num: '3',     label: 'Novel contributions', sub: 'Research-grade' },
  { num: '<10s',  label: 'Time to verdict',    sub: 'Branch-deployable' },
]

const PIPELINE = [
  { id: 'density',  label: 'Density',  sub: 'Archimedes',    color: 'blue',   Icon: FlaskConical },
  { id: 'acoustic', label: 'Acoustic', sub: 'MFCC-ΔΔ SVM',  color: 'purple', Icon: Waves },
  { id: 'visual',   label: 'Visual',   sub: 'EfficientNet',  color: 'teal',   Icon: Eye },
  { id: 'streak',   label: 'Streak',   sub: 'HSV analysis',  color: 'amber',  Icon: FlaskConical },
]

const STEPS = [
  {
    n: '01', title: 'Density test', badge: 'Physics — always required', Icon: FlaskConical,
    desc: 'Archimedes principle: dry ÷ (dry − submerged) = density. 24K gold = 19.32 g/cm³. Impossible to fake without perfectly matching weight ratio.',
  },
  {
    n: '02', title: 'Acoustic ring', badge: 'Novelty 1', Icon: Waves,
    desc: 'MFCC-ΔΔ features capture the ring-decay envelope of genuine gold. Plated brass or tungsten-core items produce a fundamentally different temporal signature — imperceptible to the human ear.',
  },
  {
    n: '03', title: 'Visual analysis', badge: '2–4 photos recommended', Icon: Eye,
    desc: 'EfficientNet-B3 embeddings detect surface plating, discoloration, and forgery marks. LogReg probe trained on genuine jewellery (DS-3) vs. surface defects (DS-2).',
  },
  {
    n: '04', title: 'AI fusion + LLM', badge: 'SHAP explainable', Icon: GitMerge,
    desc: 'XGBoost fuses 10 features (4 modality + 6 contradiction pairs). SHAP explains every decision. Groq Llama-3 70B writes a branch-ready plain-English action statement.',
  },
]

const NOVELTIES = [
  {
    n: '01', tag: 'Novelty 1', Icon: Waves,
    title: 'MFCC-ΔΔ Acoustic Fingerprinting',
    desc: 'Delta-delta MFCC captures temporal ring decay unique to pure gold. Plated brass or tungsten-core items produce a significantly different acoustic envelope — undetectable to the human ear but trivial for our SVM.',
    pill: 'SVM · RBF kernel · AUC 0.999',
    color: 'blue',
  },
  {
    n: '02', tag: 'Novelty 2', Icon: BarChart3,
    title: "Benford's Law Density Monitor",
    desc: "First significant digits of submerged weights follow Benford's distribution for genuine items. Systematic deviations expose fraud rings — catching branch-level collusion in real time, weeks ahead of manual audit.",
    pill: 'Chi-squared test · p < 0.05 alert',
    color: 'gold',
  },
  {
    n: '03', tag: 'Novelty 3', Icon: GitMerge,
    title: 'Cross-Modal Contradiction Detection',
    desc: 'Density passes but acoustic fails? That contradiction is itself a high-value signal. Six modality-pair contradiction scores are fed into XGBoost — critical for tungsten-core fraud where density alone passes at ≥19.25 g/cm³.',
    pill: '0.40× contradiction boost in fusion',
    color: 'purple',
  },
]

const FRAUD_ROWS = [
  { type: 'Gold-plated base metal',  sub: 'Brass / copper core',        trad: 'Acid test (destructive)', kanchan: 'Density + visual + acoustic',        full: true  },
  { type: 'Tungsten-core bar',       sub: 'Density ≈ 24K (19.25 vs 19.32)', trad: 'XRF or drill — invasive',   kanchan: 'Cross-modal contradiction',          full: true  },
  { type: 'Under-karat item',        sub: '22K declared, 18K actual',   trad: 'Touchstone + acid',       kanchan: 'Density deviation model',            full: true  },
  { type: 'Coordinated fraud ring',  sub: 'Staff collusion on appraisals', trad: 'Manual audit — weeks later', kanchan: "Benford's Law realtime monitor",     full: true  },
  { type: 'Surface-only plating',    sub: 'Thin layer on silver/alloy', trad: 'Visual — unreliable',     kanchan: 'EfficientNet-B3 visual probe',        full: true  },
  { type: 'Recycled / mixed alloy',  sub: 'Melted below declared karat', trad: 'Fire assay — lab only',  kanchan: 'Density range check',                full: false },
]

export default function HeroPage() {
  const navigate = useNavigate()

  return (
    <div className={styles.page}>
      <div className={styles.dotGrid} aria-hidden />
      <div className={styles.spotlight} aria-hidden />
      <div className={styles.spotlightGold} aria-hidden />

      {/* ── Nav ── */}
      <nav className={styles.nav}>
        <div className={styles.navBrand}>
          <div className={styles.navMark}><Shield size={15} strokeWidth={2.5} /></div>
          <span className={styles.navName}>KANCHAN<span className={styles.aiSuffix}>-AI</span></span>
        </div>
        <div className={styles.navLinks}>
          <a href="#pipeline" className={styles.navLink}>Pipeline</a>
          <a href="#novelties" className={styles.navLink}>Novelties</a>
          <a href="#coverage" className={styles.navLink}>Coverage</a>
        </div>
        <div className={styles.navRight}>
          <span className={styles.navPill}>
            <span className={styles.liveDot} />
            Live Demo
          </span>
          <Button size="default" onClick={() => navigate('/dashboard')}>
            Open dashboard <ArrowRight size={14} />
          </Button>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className={styles.hero}>
        <div className={styles.heroEyebrow}>
          <Lock size={11} strokeWidth={2.5} />
          Canara Bank · Gold Loan Division · AI Fraud Detection
        </div>

        <h1 className={styles.heroH1}>
          <span className={styles.h1Line1}>Detect Spurious Gold</span>
          <span className={styles.h1Line2}>
            in <em className={styles.heroAccent}>Under 10 Seconds</em>
          </span>
        </h1>

        <p className={styles.heroSub}>
          Non-destructive, branch-deployable fraud detection for gold loan appraisal.
          Four independent AI modalities fused into one SHAP-explainable verdict —
          no lab, no acid, no damage.
        </p>

        <div className={styles.heroActions}>
          <Button size="lg" className={styles.heroPrimary} onClick={() => navigate('/dashboard')}>
            Launch dashboard
            <ArrowRight size={16} />
          </Button>
          <Button size="lg" variant="outline" className={styles.heroSecondary} onClick={() => navigate('/history')}>
            View case history
          </Button>
        </div>

        <div className={styles.heroBadges}>
          {['Non-destructive', 'SHAP-explainable', 'Branch-deployable', 'LLM verdict'].map(b => (
            <span key={b} className={styles.heroBadge}>
              <Check size={10} strokeWidth={3} /> {b}
            </span>
          ))}
        </div>
      </section>

      {/* ── Stats ── */}
      <div className={styles.statsWrap}>
        <div className={styles.statsGrid}>
          {STATS.map(s => (
            <div key={s.label} className={styles.stat}>
              <span className={styles.statNum}>{s.num}</span>
              <span className={styles.statLabel}>{s.label}</span>
              <span className={styles.statSub}>{s.sub}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Pipeline Architecture ── */}
      <div id="pipeline" className={styles.pipelineSection}>
        <div className={styles.sectionEyebrow}>System architecture</div>
        <h2 className={styles.sectionH2}>Four modalities. One fused verdict.</h2>
        <p className={styles.sectionSub}>Each signal runs independently. Contradiction between modalities is itself a fraud signal — critical for tungsten-core attacks where density alone passes.</p>

        <div className={styles.pipelineFlow}>
          {/* Inputs */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Inputs</p>
            {['Weight (dry + submerged)', 'Audio (tap test)', 'Item photos (2–4)', 'Streak photo'].map(i => (
              <div key={i} className={styles.pipelineInput}>{i}</div>
            ))}
          </div>

          <div className={styles.pipelineArrow}><ChevronRight size={20} /></div>

          {/* Modalities */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Modalities</p>
            {PIPELINE.map(m => (
              <div key={m.id} className={`${styles.pipelineModal} ${styles[`modal_${m.color}`]}`}>
                <m.Icon size={14} strokeWidth={2} />
                <div>
                  <div className={styles.pipelineModalName}>{m.label}</div>
                  <div className={styles.pipelineModalSub}>{m.sub}</div>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.pipelineArrow}><ChevronRight size={20} /></div>

          {/* Fusion */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Fusion</p>
            <div className={styles.pipelineFusion}>
              <Cpu size={18} strokeWidth={1.75} />
              <div>
                <div className={styles.pipelineFusionTitle}>XGBoost</div>
                <div className={styles.pipelineFusionSub}>10 features + SHAP</div>
              </div>
            </div>
            <div className={`${styles.pipelineFusion} ${styles.fusionContra}`} style={{ marginTop: 8 }}>
              <GitMerge size={14} strokeWidth={2} />
              <div>
                <div className={styles.pipelineFusionTitle}>Contradiction</div>
                <div className={styles.pipelineFusionSub}>6 cross-modal pairs</div>
              </div>
            </div>
          </div>

          <div className={styles.pipelineArrow}><ChevronRight size={20} /></div>

          {/* Verdict */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Verdict</p>
            <div className={styles.pipelineVerdict}>
              <div className={styles.verdictGenuine}><CheckCircle2 size={16} /> Genuine</div>
              <div className={styles.verdictBorderline}><AlertTriangle size={16} /> Borderline</div>
              <div className={styles.verdictReject}><XCircle size={16} /> Reject</div>
            </div>
            <div className={styles.pipelineLLM}>
              <Brain size={14} />
              <span>Groq Llama-3 70B</span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.contentWrap}>

        {/* ── How it works ── */}
        <section className={styles.section}>
          <div className={styles.sectionEyebrow}>How it works</div>
          <h2 className={styles.sectionH2}>Each signal, explained.</h2>
          <p className={styles.sectionSub}>Run sequentially or in parallel — the system works with whatever inputs are available, degrading gracefully when modalities are missing.</p>
          <div className={styles.steps}>
            {STEPS.map(s => (
              <div key={s.n} className={styles.step}>
                <div className={styles.stepHeader}>
                  <span className={styles.stepN}>{s.n}</span>
                  <span className={styles.stepBadge}>{s.badge}</span>
                </div>
                <div className={styles.stepIconWrap}>
                  <s.Icon size={22} strokeWidth={1.75} />
                </div>
                <h3 className={styles.stepTitle}>{s.title}</h3>
                <p className={styles.stepDesc}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Novelties ── */}
        <section id="novelties" className={styles.section}>
          <div className={styles.sectionEyebrow}>What makes us different</div>
          <h2 className={styles.sectionH2}>Three novel contributions</h2>
          <p className={styles.sectionSub}>Addressing fraud vectors that traditional XRF and visual inspection miss entirely — built for the Indian gold loan context.</p>
          <div className={styles.novelties}>
            {NOVELTIES.map(n => (
              <div key={n.n} className={`${styles.novelty} ${styles[`novelty_${n.color}`]}`}>
                <div className={styles.noveltyTop}>
                  <span className={styles.noveltyTag}>{n.tag}</span>
                  <div className={styles.noveltyIconWrap}>
                    <n.Icon size={16} strokeWidth={2} />
                  </div>
                </div>
                <div className={styles.noveltyNum}>{n.n}</div>
                <h3 className={styles.noveltyTitle}>{n.title}</h3>
                <p className={styles.noveltyDesc}>{n.desc}</p>
                <div className={styles.noveltyPill}>{n.pill}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Fraud coverage ── */}
        <section id="coverage" className={styles.section}>
          <div className={styles.sectionEyebrow}>Detection coverage</div>
          <h2 className={styles.sectionH2}>Fraud types we detect</h2>
          <div className={styles.fraudTable}>
            <div className={styles.fraudHead}>
              <div className={styles.fraudTh} style={{ gridColumn: '1' }}>Fraud type</div>
              <div className={styles.fraudTh} style={{ gridColumn: '2' }}>Traditional method</div>
              <div className={styles.fraudTh} style={{ gridColumn: '3' }}>KANCHAN-AI approach</div>
            </div>
            {FRAUD_ROWS.map((row, i) => (
              <div key={i} className={`${styles.fraudRow} ${row.full ? styles.fraudRowFull : styles.fraudRowPartial}`}>
                <div className={styles.fraudCell}>
                  <div className={styles.fraudLeftBar} />
                  <div>
                    <div className={styles.fraudName}>{row.type}</div>
                    <div className={styles.fraudSub}>{row.sub}</div>
                  </div>
                </div>
                <div className={styles.fraudCell}>
                  <span className={styles.fraudTrad}>{row.trad}</span>
                </div>
                <div className={styles.fraudCell}>
                  {row.full
                    ? <span className={styles.fraudCover}><Check size={12} strokeWidth={2.5} />{row.kanchan}</span>
                    : <span className={styles.fraudPartial}><Minus size={12} strokeWidth={2} />{row.kanchan}</span>
                  }
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Live preview ── */}
        <section className={styles.section}>
          <div className={styles.sectionEyebrow}>Live verdict preview</div>
          <h2 className={styles.sectionH2}>What a bank officer sees</h2>
          <div className={styles.previewGrid}>
            <div className={styles.previewCard}>
              <div className={styles.previewHeader}>
                <div className={styles.previewMono}>KANCHAN-AI · Case #a4f7c821</div>
                <span className={styles.previewStatusOk}>GENUINE · HIGH CONFIDENCE</span>
              </div>
              <div className={styles.previewVerdict}>Genuine</div>
              <div className={styles.previewSub}>22K gold · 17.82 g/cm³ · BLR-001</div>

              <div className={styles.previewMeter}>
                <div className={styles.previewMeterRow}>
                  <span>Fusion risk score</span><strong>18%</strong>
                </div>
                <div className={styles.previewTrack}>
                  <div className={styles.previewFill} style={{ width: '18%' }} />
                </div>
              </div>

              <div className={styles.previewBars}>
                {[
                  { label: 'Density',  pct: 12, ok: true  },
                  { label: 'Acoustic', pct: 8,  ok: true  },
                  { label: 'Visual',   pct: 21, ok: true  },
                  { label: 'Streak',   pct: 34, ok: false },
                ].map(b => (
                  <div key={b.label} className={styles.previewBarRow}>
                    <span className={styles.previewBarLabel}>{b.label}</span>
                    <div className={styles.previewBarTrack}>
                      <div
                        className={styles.previewBarFill}
                        style={{ width: `${b.pct}%`, background: b.ok ? 'var(--ok)' : 'var(--warn)' }}
                      />
                    </div>
                    <span className={styles.previewBarPct}>{b.pct}%</span>
                  </div>
                ))}
              </div>

              <div className={styles.previewExplanation}>
                All four analysis signals are consistent with genuine 22K gold at the declared karat. No suspicious patterns were detected. Proceed with loan approval.
              </div>

              <div className={styles.previewTags}>
                {['Ring decay nominal', 'No surface anomaly', 'Density within 22K range', 'via Groq Llama-3'].map(t => (
                  <span key={t} className={styles.previewTag}>{t}</span>
                ))}
              </div>
            </div>

            <div className={styles.previewCta}>
              <div className={styles.ctaIcon}><Shield size={28} strokeWidth={1.75} /></div>
              <h3 className={styles.ctaTitle}>Ready to analyse a gold item?</h3>
              <p className={styles.ctaDesc}>
                Upload item photos, record a tap test, enter the weight measurements.
                Get a SHAP-explainable verdict with a branch-ready action statement in under 10 seconds.
              </p>
              <Button size="lg" className={styles.ctaBtn} onClick={() => navigate('/dashboard')}>
                Open analysis dashboard <ArrowRight size={16} />
              </Button>
              <div className={styles.ctaChecks}>
                {['No destructive testing required', 'Works with any standard scale', 'Groq LLM plain-English verdict', 'Full case history with audit trail'].map(f => (
                  <div key={f} className={styles.ctaCheck}>
                    <div className={styles.ctaCheckIcon}><Check size={10} strokeWidth={3} /></div>
                    {f}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* ── Tech stack ── */}
        <section className={styles.techSection}>
          <div className={styles.techGrid}>
            {[
              { icon: Database, label: 'Backend', items: ['FastAPI', 'scikit-learn', 'XGBoost', 'librosa'] },
              { icon: Brain,    label: 'AI/ML',   items: ['EfficientNet-B3', 'MFCC-ΔΔ SVM', 'Fusion XGBoost', 'SHAP'] },
              { icon: Cpu,      label: 'LLM',     items: ['Groq Llama-3 70B', 'Gemini 1.5 Flash', 'Rule heuristic'] },
              { icon: TrendingUp, label: 'Frontend', items: ['React + Vite', 'ShadCN UI', 'Radix primitives'] },
            ].map(t => (
              <div key={t.label} className={styles.techCard}>
                <div className={styles.techCardHeader}>
                  <t.icon size={16} strokeWidth={2} />
                  <span>{t.label}</span>
                </div>
                <div className={styles.techItems}>
                  {t.items.map(i => <span key={i} className={styles.techItem}>{i}</span>)}
                </div>
              </div>
            ))}
          </div>
        </section>

      </div>

      {/* ── Footer ── */}
      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div className={styles.footerBrand}>
            <div className={styles.footerMark}><Shield size={13} /></div>
            <span className={styles.footerName}>KANCHAN<span className={styles.aiSuffix}>-AI</span></span>
          </div>
          <p className={styles.footerText}>
            KANCHAN-AI · Canara Bank · Gold Loan Division · 2026
          </p>
          <div className={styles.footerLinks}>
            <button className={styles.footerLink} onClick={() => navigate('/dashboard')}>Dashboard</button>
            <button className={styles.footerLink} onClick={() => navigate('/history')}>History</button>
          </div>
        </div>
      </footer>
    </div>
  )
}
