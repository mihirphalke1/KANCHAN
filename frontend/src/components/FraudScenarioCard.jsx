import { ShieldAlert } from 'lucide-react'

/**
 * Bank fraud-scenario classification (P3-13). Renders the internal verdict +
 * contradiction pattern translated into the bank's own Sl.1–Sl.8 spurious-gold
 * vocabulary. Read-only; shows only when at least one scenario matched.
 */
export default function FraudScenarioCard({ scenarios }) {
  const matched = scenarios?.matched || []
  if (!matched.length) return null

  return (
    <div style={S.card}>
      <div style={S.header}>
        <ShieldAlert size={18} strokeWidth={2} />
        <span>Bank fraud-scenario classification</span>
      </div>
      <p style={S.sub}>
        Verdict mapped to the bank’s standard spurious-gold scenarios.
      </p>
      <div style={S.list}>
        {matched.map((m) => (
          <div key={m.code} style={S.row}>
            <span style={S.sl}>{m.sl}</span>
            <div>
              <div style={S.title}>{m.title}</div>
              {m.evidence && <div style={S.evidence}>{m.evidence}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

const S = {
  card: {
    background: '#fff7ed', border: '1px solid #fdba74', borderRadius: 12,
    padding: '16px 18px', marginBottom: 16,
  },
  header: {
    display: 'flex', alignItems: 'center', gap: 8, color: '#9a3412',
    fontWeight: 700, fontSize: 14,
  },
  sub: { color: '#b45309', fontSize: 12, margin: '4px 0 12px' },
  list: { display: 'flex', flexDirection: 'column', gap: 10 },
  row: { display: 'flex', gap: 12, alignItems: 'flex-start' },
  sl: {
    flex: '0 0 auto', background: '#ea580c', color: '#fff', fontWeight: 700,
    fontSize: 11, borderRadius: 6, padding: '3px 8px', minWidth: 34,
    textAlign: 'center',
  },
  title: { color: '#7c2d12', fontWeight: 600, fontSize: 13 },
  evidence: { color: '#9a3412', fontSize: 12, marginTop: 2, lineHeight: 1.4 },
}
