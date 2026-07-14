import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ShieldCheck, LogIn, Loader2 } from 'lucide-react'
import CameraCapture from '@/components/CameraCapture'
import { setSession, setSelfieCaptured, getEvaluator, getToken, authHeaders } from '@/lib/auth'
import styles from './LoginPage.module.css'

// Evaluator Integrity Layer entry point: ID + PIN, then a mandatory session
// selfie before any case can be analysed. This replaces the old free-text
// "officer_name" field — every case downstream is traceable to a person who
// was physically present and authenticated at login, not just a typed name.
export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const startAtSelfie = location.state?.step === 'selfie' && getToken()

  const [step, setStep]           = useState(startAtSelfie ? 'selfie' : 'login')
  const [evaluatorId, setEvaluatorId] = useState('')
  const [pin, setPin]             = useState('')
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [selfieFile, setSelfieFile] = useState(null)

  const evaluator = getEvaluator()

  const handleLogin = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('evaluator_id', evaluatorId)
      fd.append('pin', pin)
      const res = await fetch('/api/auth/login', { method: 'POST', body: fd })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Invalid evaluator ID or PIN')
      }
      const data = await res.json()
      setSession(data.token, { ...data, selfie_captured: false })
      setStep('selfie')
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const submitSelfie = async () => {
    if (!selfieFile) return
    setLoading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('selfie', selfieFile)
      const res = await fetch('/api/auth/selfie', { method: 'POST', body: fd, headers: authHeaders() })
      if (!res.ok) throw new Error('Could not save session selfie — try again')
      setSelfieCaptured(true)
      const dest = location.state?.from?.pathname || '/dashboard'
      navigate(dest, { replace: true })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.card}>
        <div className={styles.brand}>
          <ShieldCheck size={22} />
          <span>KANCHAN<span className={styles.brandAi}>-AI</span></span>
        </div>
        <p className={styles.sub}>Evaluator sign-in — Canara Bank Gold Loan Division</p>

        {step === 'login' && (
          <form className={styles.form} onSubmit={handleLogin}>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="evaluator_id">Evaluator ID</label>
              <input
                id="evaluator_id" className={styles.input} type="text" autoFocus
                placeholder="e.g. EMP-1001" value={evaluatorId}
                onChange={e => setEvaluatorId(e.target.value)} required
              />
            </div>
            <div className={styles.field}>
              <label className={styles.label} htmlFor="pin">PIN</label>
              <input
                id="pin" className={styles.input} type="password" inputMode="numeric"
                placeholder="4-digit PIN" value={pin}
                onChange={e => setPin(e.target.value)} required
              />
            </div>
            {error && <p className={styles.error}>{error}</p>}
            <button type="submit" className={styles.submitBtn} disabled={loading}>
              {loading ? <Loader2 size={16} className={styles.spin} /> : <LogIn size={16} />}
              Sign in
            </button>
            <p className={styles.demoHint}>Pilot demo credentials: EMP-1001 / 1234</p>
          </form>
        )}

        {step === 'selfie' && (
          <div className={styles.form}>
            <p className={styles.selfieIntro}>
              Signed in as <strong>{evaluator?.name}</strong> ({evaluator?.evaluator_id}). Capture a session
              selfie before analysing any items — this is what makes every case traceable to you specifically.
            </p>
            <CameraCapture facingMode="user" label="session selfie" onCapture={setSelfieFile} />
            {error && <p className={styles.error}>{error}</p>}
            <button
              type="button" className={styles.submitBtn}
              disabled={!selfieFile || loading} onClick={submitSelfie}
            >
              {loading ? <Loader2 size={16} className={styles.spin} /> : <ShieldCheck size={16} />}
              Confirm & start session
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
