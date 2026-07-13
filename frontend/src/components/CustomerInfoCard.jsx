import { useState } from 'react'
import { Pencil, X } from 'lucide-react'
import styles from './CustomerInfoCard.module.css'

const FIELDS = [
  { key: 'name',         label: 'Customer Name' },
  { key: 'account_no',   label: 'Account Number' },
  { key: 'loan_app_no',  label: 'Loan Application No.' },
  { key: 'officer_name', label: 'Assessment Officer' },
  { key: 'phone',        label: 'Phone Number' },
  { key: 'email',        label: 'Email' },
  { key: 'address',      label: 'Address', full: true },
  { key: 'notes',        label: 'Notes', full: true, textarea: true },
]

// Restores the customer-details section removed from the live form, plus
// lets an officer fill in anything not captured at intake (phone, email,
// address, notes) once the case is already in History.
export default function CustomerInfoCard({ caseId, customer, onUpdated }) {
  const [editing, setEditing] = useState(false)
  const [saving, setSaving]   = useState(false)
  const [error, setError]     = useState(null)
  const [form, setForm]       = useState(() => Object.fromEntries(
    FIELDS.map(f => [f.key, customer?.[f.key] || ''])
  ))

  const filled = FIELDS.filter(f => customer?.[f.key])

  const startEdit = () => {
    setForm(Object.fromEntries(FIELDS.map(f => [f.key, customer?.[f.key] || ''])))
    setError(null)
    setEditing(true)
  }

  const save = async () => {
    setSaving(true)
    setError(null)
    try {
      const res = await fetch(`/api/history/${caseId}`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(form),
      })
      if (!res.ok) throw new Error('Save failed')
      const updated = await res.json()
      onUpdated?.(updated)
      setEditing(false)
    } catch (e) {
      setError('Could not save — try again.')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.title}>Customer Information</span>
        {!editing && (
          <button type="button" className={styles.editBtn} onClick={startEdit}>
            <Pencil size={11} /> {filled.length ? 'Edit' : 'Add details'}
          </button>
        )}
      </div>

      {!editing && (
        filled.length > 0 ? (
          <div className={styles.table}>
            {filled.map(f => (
              <div key={f.key} className={styles.row}>
                <span className={styles.rowLabel}>{f.label}</span>
                <span className={styles.rowValue}>{customer[f.key]}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className={styles.empty}>No customer details recorded for this case yet.</p>
        )
      )}

      {editing && (
        <div className={styles.form}>
          {FIELDS.map(f => (
            <div key={f.key} className={`${styles.field} ${f.full ? styles.full : ''}`}>
              <label className={styles.label} htmlFor={`ci_${f.key}`}>{f.label}</label>
              {f.textarea ? (
                <textarea
                  id={`ci_${f.key}`}
                  className={styles.input}
                  value={form[f.key]}
                  onChange={e => setForm(s => ({ ...s, [f.key]: e.target.value }))}
                />
              ) : (
                <input
                  id={`ci_${f.key}`}
                  className={styles.input}
                  type="text"
                  value={form[f.key]}
                  onChange={e => setForm(s => ({ ...s, [f.key]: e.target.value }))}
                />
              )}
            </div>
          ))}
          {error && <p className={styles.error}>{error}</p>}
          <div className={styles.actions}>
            <button type="button" className={styles.cancelBtn} onClick={() => setEditing(false)} disabled={saving}>
              <X size={12} /> Cancel
            </button>
            <button type="button" className={styles.saveBtn} onClick={save} disabled={saving}>
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
