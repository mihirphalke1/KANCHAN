import { useState } from 'react'
import { UserCheck, Clock, CheckCircle2, XCircle } from 'lucide-react'
import { apiFetch, getEvaluator } from '@/lib/auth'

/**
 * Maker-checker dual sign-off (P3-12). A BORDERLINE / HELD case cannot close
 * until a SECOND, different authorised officer approves or rejects it.
 *
 * Props:
 *   approval  the case's approval block
 *   caseId    case id (needed to POST a sign-off)
 *   allowAction  show Approve/Reject controls (History view only, not the live
 *                just-created result where maker == current user)
 *   onSignedOff  callback(updatedApproval) after a successful sign-off
 */
export default function MakerCheckerCard({ approval, caseId, allowAction = false, onSignedOff }) {
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState('')
  const [err, setErr] = useState(null)

  if (!approval?.maker_checker_required) return null

  const me = getEvaluator()
  const isMaker = me && me.evaluator_id === approval.maker_id
  const status = approval.status
  const pending = status === 'pending_checker'

  const cfg = pending
    ? { bg: '#fffbeb', border: '#fcd34d', ink: '#92400e', Icon: Clock,
        label: 'Pending second sign-off' }
    : status === 'approved'
    ? { bg: '#ecfdf5', border: '#6ee7b7', ink: '#065f46', Icon: CheckCircle2,
        label: 'Dual sign-off complete — approved' }
    : { bg: '#fef2f2', border: '#fca5a5', ink: '#991b1b', Icon: XCircle,
        label: 'Dual sign-off — rejected' }

  async function signoff(decision) {
    setErr(null); setBusy(true)
    try {
      const res = await apiFetch(`/api/history/${caseId}/signoff`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, note: note || null }),
      })
      const data = await res.json()
      if (!res.ok) { setErr(data.detail || 'Sign-off failed'); return }
      onSignedOff?.(data.approval)
    } catch (e) {
      setErr('Network error during sign-off')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{ ...S.card, background: cfg.bg, borderColor: cfg.border }}>
      <div style={{ ...S.header, color: cfg.ink }}>
        <cfg.Icon size={18} strokeWidth={2} />
        <span>Maker-checker · {cfg.label}</span>
      </div>
      <div style={S.meta}>
        <span><b>Maker:</b> {approval.maker_name || approval.maker_id || '—'}</span>
        {approval.checker_id && (
          <span><b>Checker:</b> {approval.checker_name || approval.checker_id}
            {approval.signed_at && ` · ${approval.signed_at.slice(0, 19).replace('T', ' ')}`}
          </span>
        )}
        {approval.note && <span><b>Note:</b> {approval.note}</span>}
      </div>

      {pending && allowAction && !isMaker && (
        <div style={S.actionWrap}>
          <input
            style={S.input}
            placeholder="Optional note for the audit trail…"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
          <div style={S.btnRow}>
            <button style={{ ...S.btn, ...S.approve }} disabled={busy}
                    onClick={() => signoff('approve')}>
              <UserCheck size={14} /> Approve & close
            </button>
            <button style={{ ...S.btn, ...S.reject }} disabled={busy}
                    onClick={() => signoff('reject')}>
              <XCircle size={14} /> Reject
            </button>
          </div>
        </div>
      )}

      {pending && allowAction && isMaker && (
        <p style={S.blocked}>
          Segregation of duties: you assessed this case, so a different officer
          must sign it off.
        </p>
      )}

      {pending && !allowAction && (
        <p style={S.blocked}>
          A second authorised officer must sign this off from the History page
          before the case can close.
        </p>
      )}

      {err && <p style={S.err}>{err}</p>}
    </div>
  )
}

const S = {
  card: { border: '1px solid', borderRadius: 12, padding: '16px 18px', marginBottom: 16 },
  header: { display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: 14 },
  meta: { display: 'flex', flexDirection: 'column', gap: 3, marginTop: 8, fontSize: 12.5, color: '#57534e' },
  actionWrap: { marginTop: 14 },
  input: {
    width: '100%', padding: '8px 10px', borderRadius: 8, border: '1px solid #e7e5e4',
    fontSize: 13, marginBottom: 10, boxSizing: 'border-box',
  },
  btnRow: { display: 'flex', gap: 10 },
  btn: {
    display: 'inline-flex', alignItems: 'center', gap: 6, border: 'none',
    borderRadius: 8, padding: '9px 16px', fontWeight: 600, fontSize: 13, cursor: 'pointer',
  },
  approve: { background: '#059669', color: '#fff' },
  reject: { background: '#dc2626', color: '#fff' },
  blocked: { marginTop: 10, fontSize: 12.5, color: '#78716c', fontStyle: 'italic' },
  err: { marginTop: 8, fontSize: 12.5, color: '#b91c1c' },
}
