import { IndianRupee, ShieldAlert, ShieldCheck, Gem, ScanLine } from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './EvidencePanel.module.css'

function formatInr(n) {
  if (n === null || n === undefined) return '—'
  return '₹' + Number(n).toLocaleString('en-IN', { maximumFractionDigits: 0 })
}

export default function EvidencePanel({ caseData }) {
  if (!caseData) return null
  const { ltv, integrity, gem_grid: gemGrid, hallmark, kyc } = caseData
  const dupFlags  = integrity?.duplicate_photo_flags || []
  const fiducial  = integrity?.fiducial

  if (!ltv && !dupFlags.length && !fiducial && !gemGrid && !hallmark && !kyc) return null

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Loan &amp; Integrity Evidence</h3>

      {ltv && (
        <div className={styles.block}>
          <div className={styles.blockHead}>
            <IndianRupee size={14} />
            <span>Assessed Value &amp; RBI-Tiered LTV</span>
            <InfoTip text="Net gold weight × configured gold rate = assessed value. Max loan uses the RBI's tiered LTV structure (85% up to ₹2.5L, 80% up to ₹5L, 75% above) — not the old flat 75% cap." side="right" />
          </div>
          {ltv.net_gold_breakdown &&
            (ltv.net_gold_breakdown.method === 'stone_weight_deduction' ||
             ltv.net_gold_breakdown.method === 'trade_deduction') && (
            <div className={styles.deduction}>
              <span>Total weight <b>{ltv.net_gold_breakdown.gross_weight_g} g</b></span>
              <span className={styles.minus}>−</span>
              <span>stones ≈ <b>{ltv.net_gold_breakdown.stone_weight_g} g</b>
                {ltv.net_gold_breakdown.n_stones
                  ? ` (${ltv.net_gold_breakdown.n_stones}${ltv.net_gold_breakdown.stone_carat_total ? `, ~${ltv.net_gold_breakdown.stone_carat_total} ct` : ''})`
                  : ''}</span>
              <span className={styles.minus}>=</span>
              <span>net gold <b>{ltv.net_gold_weight_g} g</b></span>
            </div>
          )}
          {ltv.net_gold_breakdown?.calibration_flag && (
            <p className={styles.calibrationWarning}>
              <ShieldAlert size={12} /> {ltv.net_gold_breakdown.calibration_flag}
            </p>
          )}
          <div className={styles.ltvGrid}>
            <div>
              <span className={styles.ltvLabel}>Net gold</span>
              <span className={styles.ltvValue}>{ltv.net_gold_weight_g} g</span>
              {ltv.net_gold_breakdown && (
                <InfoTip text={ltv.net_gold_breakdown.explanation} side="right" />
              )}
            </div>
            <div>
              <span className={styles.ltvLabel}>Rate applied</span>
              <span className={styles.ltvValue}>
                {formatInr(ltv.rate_per_gram_inr)}/g
                {ltv.valuation_karat ? ` · ${ltv.valuation_karat}K` : ''}
              </span>
              <InfoTip text="Derived from the daily 999-fine (24K) gold rate × BIS fineness for the declared karat — not a flat rate applied regardless of purity." side="right" />
            </div>
            <div><span className={styles.ltvLabel}>Assessed value</span><span className={styles.ltvValue}>{formatInr(ltv.assessed_value_inr)}</span></div>
            <div><span className={styles.ltvLabel}>LTV tier</span><span className={styles.ltvValue}>{Math.round(ltv.ltv_pct * 100)}% · {ltv.tier}</span></div>
            <div><span className={styles.ltvLabel}>Max eligible loan</span><span className={`${styles.ltvValue} ${styles.ltvHighlight}`}>{formatInr(ltv.max_loan_inr)}</span></div>
          </div>

          {ltv.matched_karat && ltv.matched_karat !== ltv.valuation_karat && ltv.rate_per_gram_calculated_karat && (
            <p className={styles.note}>
              For comparison only: the density physics matched <b>{ltv.matched_karat}K</b> (
              {formatInr(ltv.rate_per_gram_calculated_karat)}/g) rather than the declared {ltv.valuation_karat}K —
              this valuation still uses the declared karat's rate above. A genuine mismatch should be resolved
              by re-running the analysis with the corrected declared karat, not by silently reweighting here.
            </p>
          )}

          {ltv.status !== 'final' && (
            <p className={styles.indicativeWarning}>
              <ShieldAlert size={12} /> INDICATIVE VALUATION ONLY — borrower-present attestation not recorded.
              This assessed value, LTV, and maximum loan are provisional until the borrower is attested present.
            </p>
          )}
        </div>
      )}

      {kyc && (
        <div className={styles.block}>
          <div className={styles.blockHead}>
            <ShieldCheck size={14} />
            <span>Core-Banking KYC Lookup</span>
            <span className={styles.simBadge}>SIMULATED</span>
          </div>
          <p className={styles.note}>
            {kyc.record?.name} · KYC {kyc.record?.kyc_status} · {kyc.record?.existing_gold_loans} existing gold loan(s) on file
          </p>
        </div>
      )}

      {hallmark && (
        <div className={styles.block}>
          <div className={styles.blockHead}>
            <ScanLine size={14} />
            <span>Hallmark / HUID</span>
            <span className={styles.simBadge}>SIMULATED</span>
          </div>
          <p className={styles.note}>
            {hallmark.registry_found
              ? `Registry: ${hallmark.registry_record.purity_karat}K, ${hallmark.registry_record.jeweller_name}, status ${hallmark.registry_record.status}`
              : 'No match in the simulated BIS registry'}
            {' · '}Engraving risk {Math.round((hallmark.engraving?.risk_score ?? 0.5) * 100)}%
            {hallmark.cross_check && (hallmark.cross_check.match === false) && (
              <strong className={styles.warnText}> — registry purity does not match declared karat</strong>
            )}
          </p>
        </div>
      )}

      {gemGrid && gemGrid.total_stones > 0 && (
        <div className={styles.block}>
          <div className={styles.blockHead}>
            <Gem size={14} />
            <span>Gemstone Grid</span>
            <InfoTip text="Item divided into a 4×4 grid; stones bucketed per cell. Diameter/weight-deduction estimates only appear when today's calibration card was detected in the photo, giving a real-world scale reference." side="right" />
          </div>
          <p className={styles.note}>
            {gemGrid.total_stones} stone(s) detected across the grid
            {gemGrid.total_deduction_g != null
              ? ` — estimated deduction ${gemGrid.total_deduction_g} g (scale from today's calibration card)`
              : ' — no scale reference detected, sizes not estimated; deduction to be confirmed by the officer'}
          </p>
        </div>
      )}

      {(dupFlags.length > 0 || fiducial) && (
        <div className={`${styles.block} ${dupFlags.length ? styles.blockAlert : ''}`}>
          <div className={styles.blockHead}>
            <ShieldAlert size={14} />
            <span>Evaluator Integrity Checks</span>
          </div>
          {dupFlags.length > 0 ? (
            dupFlags.map((f, i) => (
              <p key={i} className={styles.alertNote}>
                {f.image_role} matches evidence already on file for case #{f.matches_case_id}
                {f.matched_at ? ` (submitted ${new Date(f.matched_at).toLocaleDateString()})` : ''} — possible photo reuse.
              </p>
            ))
          ) : (
            <p className={styles.note}>No duplicate/reused photos detected against the system-wide index.</p>
          )}
          {fiducial && (
            <p className={styles.note}>
              Calibration card: {fiducial.detected ? (
                fiducial.checksum_valid === false
                  ? 'detected, but today\'s checksum did not match — possible re-used photo'
                  : fiducial.checksum_valid
                    ? 'detected and verified for today'
                    : 'detected (checksum unreadable, likely off-angle)'
              ) : 'not detected — no tamper-evidence or scale reference for this photo'}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
