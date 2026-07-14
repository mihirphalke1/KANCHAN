import { useRef, useState } from 'react'
import { Mic, RotateCcw, Check, AlertTriangle, Volume2, Upload } from 'lucide-react'
import styles from './AudioCapture.module.css'

// Ambient-noise gate: measures ~0.7s of room noise via the Web Audio API
// before allowing a tap recording. A noisy branch floor produces
// inconsistent ring-frequency readings across branches — this SOP gate
// (quiet check -> record) is the software half of that fix; the other half
// is a cheap foam-lined tap box at each branch.
const NOISE_RMS_THRESHOLD = 0.02

// Live mic recording is the default (and the noise gate only means anything
// for a live recording). An explicit "upload instead" escape hatch stays
// available for demos/dev without a real item to tap — uploaded audio skips
// the noise gate entirely and is labelled distinctly, never shown as recorded.
export default function AudioCapture({ label, onCapture }) {
  const [stage, setStage]         = useState('idle')   // idle | checking_noise | noisy | ready | recording | done
  const [error, setError]         = useState(null)
  const [fileName, setFileName]   = useState(null)
  const [source, setSource]       = useState(null)   // 'mic' | 'upload'
  const streamRef = useRef(null)
  const fileRef   = useRef(null)

  const measureAmbientNoise = async (stream) => {
    const AudioCtx = window.AudioContext || window.webkitAudioContext
    const ctx = new AudioCtx()
    const source = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 2048
    source.connect(analyser)
    const data = new Float32Array(analyser.fftSize)
    await new Promise(r => setTimeout(r, 700))
    analyser.getFloatTimeDomainData(data)
    let sum = 0
    for (let i = 0; i < data.length; i++) sum += data[i] * data[i]
    ctx.close()
    return Math.sqrt(sum / data.length)
  }

  const beginCheck = async () => {
    setError(null)
    setStage('checking_noise')
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Microphone capture is not supported in this browser.')
      setStage('idle')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const rms = await measureAmbientNoise(stream)
      if (rms > NOISE_RMS_THRESHOLD) {
        setStage('noisy')
        stream.getTracks().forEach(t => t.stop())
        streamRef.current = null
      } else {
        setStage('ready')
      }
    } catch {
      setError('Microphone permission denied — allow microphone access to record the tap test.')
      setStage('idle')
    }
  }

  const startRecording = async () => {
    let stream = streamRef.current
    if (!stream) {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
    }
    const chunks = []
    const recorder = new MediaRecorder(stream)
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data) }
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
      const file = new File([blob], `tap-${Date.now()}.webm`, { type: blob.type })
      setFileName(file.name)
      setSource('mic')
      onCapture(file)
      streamRef.current?.getTracks().forEach(t => t.stop())
      streamRef.current = null
      setStage('done')
    }
    recorder.start()
    setStage('recording')
    setTimeout(() => { if (recorder.state === 'recording') recorder.stop() }, 3000)
  }

  const handleUpload = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setFileName(file.name)
    setSource('upload')
    onCapture(file)
    setStage('done')
  }

  const reset = () => {
    setStage('idle')
    setFileName(null)
    setSource(null)
    onCapture(null)
  }

  return (
    <div className={styles.wrap}>
      {stage === 'idle' && (
        <>
          <button type="button" className={styles.checkBtn} onClick={beginCheck}>
            <Volume2 size={15} /> Check room noise — {label}
          </button>
          <button type="button" className={styles.uploadLink} onClick={() => fileRef.current?.click()}>
            <Upload size={11} /> Upload a recording instead (demo)
          </button>
          <input
            ref={fileRef} type="file" accept="audio/*" className={styles.hiddenFileInput}
            onChange={handleUpload}
          />
        </>
      )}
      {stage === 'checking_noise' && <p className={styles.hint}>Listening for background noise…</p>}
      {stage === 'noisy' && (
        <div className={styles.warn}>
          <AlertTriangle size={13} />
          <span>Too noisy to record a reliable tap — move to a quieter spot.</span>
          <button type="button" className={styles.retryBtn} onClick={beginCheck}>Re-check</button>
        </div>
      )}
      {stage === 'ready' && (
        <button type="button" className={styles.recordBtn} onClick={startRecording}>
          <Mic size={15} /> Tap the item now — recording 3s
        </button>
      )}
      {stage === 'recording' && (
        <div className={styles.recording}>
          <span className={styles.recDot} /> Recording… tap the item once
        </div>
      )}
      {stage === 'done' && (
        <div className={`${styles.done} ${source === 'upload' ? styles.uploadDone : ''}`}>
          <Check size={13} /> <span>{label} {source === 'upload' ? 'uploaded (demo)' : 'recorded'}</span>
          <button type="button" className={styles.retakeBtn} onClick={reset}>
            <RotateCcw size={12} /> Redo
          </button>
        </div>
      )}
      {error && <p className={styles.error}>{error}</p>}
    </div>
  )
}
