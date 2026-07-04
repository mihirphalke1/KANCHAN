import { useNavigate } from 'react-router-dom'
import {
  Shield, ArrowRight, Check, Eye, Waves, GitMerge,
  ChevronRight, BarChart3, Brain, FlaskConical, AlertTriangle,
  CheckCircle2, XCircle, Lock, Cpu, Scale, Mic, Camera, Layers
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

const NOVELTIES = [
  {
    n: '01', tag: 'Novelty 1', Icon: Waves,
    title: 'MFCC-ΔΔ Acoustic Fingerprinting',
    desc: 'Genuine gold rings differently. Our SVM hears what the human ear cannot - plated brass and tungsten cores fail the tap test.',
    pill: 'SVM · RBF kernel · AUC 0.999',
    color: 'blue',
  },
  {
    n: '02', tag: 'Novelty 2', Icon: BarChart3,
    title: "Benford's Law Density Monitor",
    desc: 'Weight digits follow Benford’s distribution - systematic deviations expose branch-level fraud rings in real time.',
    pill: 'Chi-squared test · p < 0.05 alert',
    color: 'gold',
  },
  {
    n: '03', tag: 'Novelty 3', Icon: GitMerge,
    title: 'Cross-Modal Contradiction Detection',
    desc: 'Density passes but acoustic fails? The contradiction itself is a signal - how we catch tungsten-core fakes.',
    pill: '0.40× contradiction boost in fusion',
    color: 'purple',
  },
]

export default function HeroPage() {
  const navigate = useNavigate()

  return (
    <div className={styles.page}>
      <div className={styles.heroFrame}>

      {/* ── Nav ── */}
      <nav className={styles.nav}>
        <div className={styles.navBrand}>
          <div className={styles.navMark}><img src="/logo.png" alt="KANCHAN-AI logo" /></div>
          <span className={styles.navName}>KANCHAN<span className={styles.aiSuffix}>-AI</span></span>
        </div>
        <div className={styles.navLinks}>
          <a href="#pipeline" className={styles.navLink}>Pipeline</a>
          <a href="#novelties" className={styles.navLink}>Novelties</a>
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
        <img src="/gold.png" alt="" aria-hidden className={styles.heroWreath} />
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
          Four AI signals fused into one explainable verdict -
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

      </div>{/* /heroFrame */}

      {/* ── Pipeline Architecture ── */}
      <div id="pipeline" className={styles.pipelineSection}>
        <div className={styles.sectionEyebrow}>System architecture</div>
        <h2 className={styles.sectionH2}>Four modalities. One fused verdict.</h2>
        <p className={styles.sectionSub}>Each signal runs independently - contradiction between them is itself a fraud signal.</p>

        <div className={styles.pipelineFlow}>
          {/* Inputs */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Inputs</p>
            {[
              { label: 'Weight (dry + submerged)', Icon: Scale },
              { label: 'Audio (tap test)',          Icon: Mic },
              { label: 'Item photos (2–4)',         Icon: Camera },
              { label: 'Streak photo',              Icon: Layers },
            ].map(({ label, Icon }) => (
              <div key={label} className={styles.pCard}>
                <span className={`${styles.pIcon} ${styles.pIcon_gray}`}><Icon size={15} strokeWidth={2} /></span>
                <div className={styles.pText}>
                  <div className={styles.pTitle}>{label}</div>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.pipelineArrow}><ChevronRight size={20} /></div>

          {/* Modalities */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Modalities</p>
            {PIPELINE.map(m => (
              <div key={m.id} className={styles.pCard}>
                <span className={`${styles.pIcon} ${styles[`pIcon_${m.color}`]}`}><m.Icon size={15} strokeWidth={2} /></span>
                <div className={styles.pText}>
                  <div className={styles.pTitle}>{m.label}</div>
                  <div className={styles.pSub}>{m.sub}</div>
                </div>
              </div>
            ))}
          </div>

          <div className={styles.pipelineArrow}><ChevronRight size={20} /></div>

          {/* Fusion */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Fusion</p>
            <div className={styles.pCard}>
              <span className={`${styles.pIcon} ${styles.pIcon_blue}`}><Cpu size={15} strokeWidth={2} /></span>
              <div className={styles.pText}>
                <div className={styles.pTitle}>XGBoost</div>
                <div className={styles.pSub}>10 features + SHAP</div>
              </div>
            </div>
            <div className={styles.pCard}>
              <span className={`${styles.pIcon} ${styles.pIcon_purple}`}><GitMerge size={15} strokeWidth={2} /></span>
              <div className={styles.pText}>
                <div className={styles.pTitle}>Contradiction</div>
                <div className={styles.pSub}>6 cross-modal pairs</div>
              </div>
            </div>
          </div>

          <div className={styles.pipelineArrow}><ChevronRight size={20} /></div>

          {/* Verdict */}
          <div className={styles.pipelineCol}>
            <p className={styles.pipelineColLabel}>Verdict</p>
            <div className={styles.pCard}>
              <span className={`${styles.pIcon} ${styles.pIcon_green}`}><CheckCircle2 size={15} strokeWidth={2} /></span>
              <div className={styles.pText}>
                <div className={styles.pTitle}>Genuine</div>
                <div className={styles.pSub}>Approve loan</div>
              </div>
            </div>
            <div className={styles.pCard}>
              <span className={`${styles.pIcon} ${styles.pIcon_amber}`}><AlertTriangle size={15} strokeWidth={2} /></span>
              <div className={styles.pText}>
                <div className={styles.pTitle}>Borderline</div>
                <div className={styles.pSub}>Escalate for review</div>
              </div>
            </div>
            <div className={styles.pCard}>
              <span className={`${styles.pIcon} ${styles.pIcon_red}`}><XCircle size={15} strokeWidth={2} /></span>
              <div className={styles.pText}>
                <div className={styles.pTitle}>Reject</div>
                <div className={styles.pSub}>Decline loan</div>
              </div>
            </div>
            <div className={`${styles.pCard} ${styles.pCardDashed}`}>
              <span className={`${styles.pIcon} ${styles.pIcon_gray}`}><Brain size={15} strokeWidth={2} /></span>
              <div className={styles.pText}>
                <div className={styles.pTitle}>Groq Llama-3 70B</div>
                <div className={styles.pSub}>Plain-English verdict</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.contentWrap}>

        {/* ── Novelties ── */}
        <section id="novelties" className={styles.section}>
          <div className={styles.sectionEyebrow}>What makes us different</div>
          <h2 className={styles.sectionH2}>Three novel contributions</h2>
          <p className={styles.sectionSub}>Fraud vectors that XRF and visual inspection miss entirely.</p>
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
                Photos, a tap-test recording, and two weights - get an explainable
                verdict in under 10 seconds.
              </p>
              <Button size="lg" className={styles.ctaBtn} onClick={() => navigate('/dashboard')}>
                Open analysis dashboard <ArrowRight size={16} />
              </Button>
              <div className={styles.ctaChecks}>
                {['No destructive testing', 'Works with any standard scale', 'Full case history & audit trail'].map(f => (
                  <div key={f} className={styles.ctaCheck}>
                    <div className={styles.ctaCheckIcon}><Check size={10} strokeWidth={3} /></div>
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
          <div className={styles.footerBrand}>
            <div className={styles.footerMark}><img src="/logo.png" alt="" /></div>
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
