import { CheckCircle2, AlertTriangle, XCircle, Copy, Check, Download } from 'lucide-react'
import { useState } from 'react'
import styles from './VerdictCard.module.css'

const VERDICT_CONFIG = {
  GENUINE: {
    color:       'green',
    icon:        CheckCircle2,
    iconImg:     '/tick.svg',
    label:       'Genuine',
    loanColor:   'green',
  },
  BORDERLINE: {
    color:       'amber',
    icon:        AlertTriangle,
    label:       'Borderline',
    loanColor:   'amber',
  },
  REJECT: {
    color:       'red',
    icon:        XCircle,
    label:       'Reject',
    loanColor:   'red',
  },
}

const LOAN_LABELS = {
  APPROVE:  { text: 'Approve Loan',   cls: 'loanApprove' },
  HOLD:     { text: 'Hold for Review', cls: 'loanHold' },
  DECLINE:  { text: 'Decline Loan',   cls: 'loanDecline' },
}

export default function VerdictCard({ verdict, caseId }) {
  const [copied, setCopied] = useState(false)

  if (!verdict) return null

  const cfg    = VERDICT_CONFIG[verdict.risk_level] || VERDICT_CONFIG.BORDERLINE
  const Icon   = cfg.icon
  const loan   = LOAN_LABELS[verdict.loan_action] || LOAN_LABELS.HOLD

  const copyCase = () => {
    navigator.clipboard.writeText(caseId)
    setCopied(true)
    setTimeout(() => setCopied(false), 1800)
  }

  return (
    <div className={`${styles.card} ${styles[cfg.color]}`}>
      <div className={styles.topRow}>
        <div className={styles.verdictBadge}>
          {cfg.iconImg
            ? <img src={cfg.iconImg} alt="" className={styles.verdictIconImg} />
            : <Icon size={22} strokeWidth={2} />}
          <span className={styles.verdictLabel}>{cfg.label}</span>
        </div>
        <div className={styles.metaRight}>
          <span className={`${styles.confidence} ${styles[`conf${verdict.confidence}`]}`}>
            {verdict.confidence} confidence
          </span>
          <span className={`${styles.loanTag} ${styles[loan.cls]}`}>
            {loan.text}
          </span>
        </div>
      </div>

      {verdict.fusion_risk !== undefined && (
        <div className={styles.riskMeter}>
          <div className={styles.riskMeterHeader}>
            <span className={styles.riskMeterLabel}>Fusion risk</span>
            <span className={`${styles.riskMeterValue} ${styles[cfg.color]}`}>
              {Math.round(verdict.fusion_risk * 100)}%
            </span>
          </div>
          <div className={styles.riskTrack}>
            <div
              className={`${styles.riskFill} ${styles[cfg.color]}`}
              style={{ width: `${verdict.fusion_risk * 100}%` }}
            />
          </div>
        </div>
      )}

      {verdict.plain_english && (
        <p className={styles.explanation}>{verdict.plain_english}</p>
      )}

      {verdict.action && (
        <div className={styles.actionBox}>
          <span className={styles.actionLabel}>Recommended action</span>
          <p className={styles.actionText}>{verdict.action}</p>
        </div>
      )}

      <div className={styles.footer}>
        <button className={styles.caseId} onClick={copyCase} title="Copy case ID">
          Case #{caseId}
          {copied
            ? <Check size={12} className={styles.copyIcon} />
            : <Copy size={12} className={styles.copyIcon} />}
        </button>
        <div className={styles.footerRight}>
          {verdict.llm_provider && verdict.llm_provider !== 'heuristic' && (
            <span className={styles.llmTag}>via {verdict.llm_provider}</span>
          )}
          {caseId && (
            <a
              href={`/api/history/${caseId}/report`}
              target="_blank"
              rel="noreferrer"
              className={styles.downloadBtn}
              title="Download PDF report"
            >
              <Download size={12} />
              Download Report
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
