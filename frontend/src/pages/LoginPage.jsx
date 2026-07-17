import { useCallback, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ShieldCheck, LogIn, Loader2, LogOut } from 'lucide-react'
import LivenessCapture from '@/components/LivenessCapture'
import { setSession, setSelfieCaptured, getEvaluator, getToken, authHeaders, clearSession } from '@/lib/auth'
import styles from './LoginPage.module.css'

// Evaluator Integrity Layer entry point: ID + PIN, then a mandatory session
// selfie (face scan) before any case can be analysed.
export default function LoginPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const startAtSelfie = location.state?.step === 'selfie' && getToken()

  const [step, setStep] = useState(startAtSelfie ? 'selfie' : 'login')
  const [evaluatorId, setEvaluatorId] = useState('')
  const [pin, setPin] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadingSelfie, setUploadingSelfie] = useState(false)
  const submittingRef = useRef(false)

  const evaluator = getEvaluator()

  const handleLogout = () => {
    clearSession()
    setStep('login')
    setError(null)
    setUploadingSelfie(false)
    submittingRef.current = false
  }

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

  // Called by LivenessCapture when the 5 s face scan finishes and a photo
  // is taken — uploads the selfie and continues straight to the dashboard.
  const handleSelfieCapture = useCallback(async (file) => {
    if (!file) {
      submittingRef.current = false
      setUploadingSelfie(false)
      return
    }
    if (submittingRef.current) return
    submittingRef.current = true
    setUploadingSelfie(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('selfie', file)
      const res = await fetch('/api/auth/selfie', {
        method: 'POST',
        body: fd,
        headers: authHeaders(),
      })
      if (res.status === 401) {
        clearSession()
        setStep('login')
        setError('Session expired — please sign in again.')
        submittingRef.current = false
        setUploadingSelfie(false)
        return
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Could not save session selfie — try again')
      }
      setSelfieCaptured(true)
      const dest = location.state?.from?.pathname || '/dashboard'
      navigate(dest, { replace: true })
    } catch (e) {
      setError(e.message)
      submittingRef.current = false
      setUploadingSelfie(false)
    }
  }, [location.state, navigate])

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
            <div className={styles.selfieHeader}>
              <p className={styles.selfieIntro}>
                Signed in as <strong>{evaluator?.name}</strong> ({evaluator?.evaluator_id}).
                Centre your face in the oval — the moment it locks, a photo is
                taken and you continue to the dashboard.
              </p>
              <button
                type="button"
                className={styles.logoutBtn}
                onClick={handleLogout}
                title="Log out and return to sign-in"
                disabled={uploadingSelfie}
              >
                <LogOut size={14} /> Log out
              </button>
            </div>
            <LivenessCapture onCapture={handleSelfieCapture} />
            {error && (
              <p className={styles.error}>
                {error}
                {' '}Use Retake to try again.
              </p>
            )}
            {uploadingSelfie && !error && (
              <p className={styles.uploadingHint}>
                <Loader2 size={14} className={styles.spin} />
                Saving session photo &amp; opening dashboard…
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
