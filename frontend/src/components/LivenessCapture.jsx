/**
 * LivenessCapture — session selfie via face detection only.
 *
 * Flow:
 *   idle → starting → detecting_face → capturing → confirmed → preview
 *                                              ↘ failed (capture error)
 *
 * The moment a face is centred in the oval, a high-res photo is taken
 * and handed to the parent (which continues to the dashboard).
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { AlertTriangle, Camera, Check, Loader2, RefreshCw, ScanFace, Upload } from 'lucide-react'
import { authHeaders, clearSession } from '@/lib/auth'
import styles from './LivenessCapture.module.css'

// ── Constants ────────────────────────────────────────────────────────────────

const FACE_POLL_MS = 250    // snappy poll so lock-in feels instant
const CAPTURE_HOLD_MS = 700 // brief verified beat before handing off

// Centre-oval parameters — MUST stay in sync with OVAL_*_RATIO in liveness.py
const OVAL_CX_RATIO = 0.50
const OVAL_CY_RATIO = 0.50
const OVAL_RX_RATIO = 0.25
const OVAL_RY_RATIO = 0.33
const DETECT_W = 320
const DETECT_H = 240

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Returns true when the nose landmark (in 320×240 coords) sits inside the oval.
 * Mirrors the Python _in_center_oval() so the visual guide and the acceptance
 * zone are always the exact same region — no frontend/backend mismatch.
 */
function isNoseInOval(nose) {
  if (!nose) return false
  const dx = (nose[0] - DETECT_W * OVAL_CX_RATIO) / (DETECT_W * OVAL_RX_RATIO)
  const dy = (nose[1] - DETECT_H * OVAL_CY_RATIO) / (DETECT_H * OVAL_RY_RATIO)
  return dx * dx + dy * dy <= 1.0
}

// ── Component ────────────────────────────────────────────────────────────────

export default function LivenessCapture({ onCapture }) {
  // idle | starting | detecting_face | capturing | confirmed | failed | preview
  const [phase, setPhase] = useState('idle')
  const [faceFound, setFaceFound] = useState(false)
  const [faceCentered, setFaceCentered] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [statusMsg, setStatusMsg] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [flash, setFlash] = useState(false)

  const videoRef = useRef(null)
  const captureCanvas = useRef(null)
  const overlayCanvas = useRef(null)
  const streamRef = useRef(null)
  const pollTimer = useRef(null)
  const pollBusy = useRef(false)
  const phaseRef = useRef('idle')
  const fileRef = useRef(null)
  const lastOverlayMode = useRef('idle')
  const lastBboxRef = useRef(null)
  const capturingRef = useRef(false) // guard against double capture

  useEffect(() => { phaseRef.current = phase }, [phase])

  useEffect(() => () => {
    clearInterval(pollTimer.current)
    stopStream()
  }, [])

  const stopStream = () => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
  }

  const captureBlob = useCallback((w = 320, h = 240, q = 0.72) =>
    new Promise(resolve => {
      const video = videoRef.current
      const canvas = captureCanvas.current
      if (!video || !canvas || !video.videoWidth) { resolve(null); return }
      canvas.width = w
      canvas.height = h
      canvas.getContext('2d').drawImage(video, 0, 0, w, h)
      canvas.toBlob(b => resolve(b || null), 'image/jpeg', q)
    }), [])

  const captureFullBlob = useCallback(() =>
    new Promise(resolve => {
      const video = videoRef.current
      const canvas = captureCanvas.current
      if (!video || !canvas || !video.videoWidth) { resolve(null); return }
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      canvas.getContext('2d').drawImage(video, 0, 0)
      canvas.toBlob(b => resolve(b || null), 'image/jpeg', 0.92)
    }), [])

  // mode: 'idle' | 'off-center' | 'locked' | 'capturing' | 'confirmed'
  const drawOverlay = useCallback((mode, bbox = null) => {
    const ov = overlayCanvas.current
    const video = videoRef.current
    if (!ov || !video || !video.offsetWidth) return

    ov.width = video.offsetWidth
    ov.height = video.offsetHeight
    const ctx = ov.getContext('2d')
    ctx.clearRect(0, 0, ov.width, ov.height)

    const cx = ov.width * OVAL_CX_RATIO
    const cy = ov.height * OVAL_CY_RATIO
    const rx = ov.width * OVAL_RX_RATIO
    const ry = ov.height * OVAL_RY_RATIO

    const color =
      mode === 'locked' || mode === 'capturing' || mode === 'confirmed' ? '#22c55e' :
      mode === 'off-center' ? '#f59e0b' :
      'rgba(255,255,255,0.50)'

    ctx.beginPath()
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2)
    ctx.strokeStyle = color
    ctx.lineWidth = 3.5
    ctx.setLineDash(mode === 'idle' || mode === 'off-center' ? [10, 6] : [])
    ctx.shadowColor = color
    ctx.shadowBlur = mode === 'idle' ? 6 : 22
    ctx.stroke()
    ctx.setLineDash([])
    ctx.shadowBlur = 0

    if (bbox) {
      const sx = ov.width / DETECT_W
      const sy = ov.height / DETECT_H
      ctx.beginPath()
      ctx.ellipse(
        bbox[0] * sx + bbox[2] * sx / 2,
        bbox[1] * sy + bbox[3] * sy / 2,
        bbox[2] * sx / 2 + 6,
        bbox[3] * sy / 2 + 8,
        0, 0, Math.PI * 2)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.shadowColor = color
      ctx.shadowBlur = 8
      ctx.stroke()
      ctx.shadowBlur = 0
    }
  }, [])

  useEffect(() => {
    if (phase === 'detecting_face') {
      lastOverlayMode.current = 'idle'
      lastBboxRef.current = null
      requestAnimationFrame(() => drawOverlay('idle'))
    }
  }, [phase, drawOverlay])

  // ── Instant capture the moment the face locks into the oval ────────────────

  const captureAndFinish = useCallback(async (bbox) => {
    if (capturingRef.current) return
    capturingRef.current = true

    clearInterval(pollTimer.current)
    lastBboxRef.current = bbox
    lastOverlayMode.current = 'locked'
    setFaceCentered(true)
    setPhase('capturing')
    phaseRef.current = 'capturing'
    drawOverlay('capturing', bbox)
    setFlash(true)

    // Let the shutter flash paint before freezing the frame
    await new Promise(r => setTimeout(r, 90))

    const blob = await captureFullBlob()
    setFlash(false)

    if (!blob) {
      capturingRef.current = false
      setPhase('failed')
      phaseRef.current = 'failed'
      setStatusMsg('Could not capture photo — please try again.')
      return
    }

    const file = new File([blob], `selfie-${Date.now()}.jpg`, { type: 'image/jpeg' })
    const url = URL.createObjectURL(blob)
    setPreviewUrl(url)
    setPhase('confirmed')
    phaseRef.current = 'confirmed'
    drawOverlay('confirmed', bbox)

    await new Promise(r => setTimeout(r, CAPTURE_HOLD_MS))

    stopStream()
    setPhase('preview')
    phaseRef.current = 'preview'
    onCapture(file)
  }, [captureFullBlob, drawOverlay, onCapture])

  // ── Face polling — lock & shoot on first centred frame ─────────────────────

  const startFacePolling = useCallback(() => {
    pollTimer.current = setInterval(async () => {
      if (pollBusy.current || capturingRef.current) return
      if (phaseRef.current !== 'detecting_face') {
        clearInterval(pollTimer.current)
        return
      }

      drawOverlay(lastOverlayMode.current, lastBboxRef.current)

      pollBusy.current = true
      try {
        const blob = await captureBlob(320, 240, 0.80)
        if (!blob) { pollBusy.current = false; return }

        const fd = new FormData()
        fd.append('frame', blob, 'frame.jpg')
        const res = await fetch('/api/auth/check-frame', {
          method: 'POST', headers: authHeaders(), body: fd,
        })
        if (!res.ok) {
          if (res.status === 401) { clearSession(); window.location.href = '/login'; return }
          pollBusy.current = false
          return
        }
        const data = await res.json()

        const detected = data.face_detected === true
        const bbox = data.bbox || null
        const nose = data.nose || (bbox ? [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2] : null)
        const centred = detected && isNoseInOval(nose)

        setFaceFound(detected)
        setFaceCentered(centred)

        if (centred) {
          // Instant: face in oval → capture immediately
          clearInterval(pollTimer.current)
          pollBusy.current = false
          captureAndFinish(bbox)
          return
        }

        const mode = detected ? 'off-center' : 'idle'
        lastOverlayMode.current = mode
        lastBboxRef.current = bbox
        drawOverlay(mode, bbox)
      } catch {
        // non-fatal — keep polling
      }
      pollBusy.current = false
    }, FACE_POLL_MS)
  }, [captureBlob, captureAndFinish, drawOverlay])

  // ── Camera start ───────────────────────────────────────────────────────────

  const startCamera = async () => {
    setErrorMsg(null)
    setStatusMsg(null)
    setPhase('starting')
    capturingRef.current = false
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMsg('Camera capture is not supported in this browser.')
      setPhase('idle')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      })
      streamRef.current = stream
      const video = videoRef.current
      if (video) { video.srcObject = stream; await video.play().catch(() => {}) }
      setFaceFound(false)
      setFaceCentered(false)
      setPhase('detecting_face')
      startFacePolling()
    } catch (e) {
      setErrorMsg(
        e.name === 'NotAllowedError'
          ? 'Camera permission denied — allow camera access and try again.'
          : 'Could not access the camera on this device.',
      )
      setPhase('idle')
    }
  }

  const retry = () => {
    clearInterval(pollTimer.current)
    lastOverlayMode.current = 'idle'
    lastBboxRef.current = null
    pollBusy.current = false
    capturingRef.current = false
    setFaceFound(false)
    setFaceCentered(false)
    setStatusMsg(null)
    phaseRef.current = 'detecting_face'
    setPhase('detecting_face')
    startFacePolling()
  }

  const retake = async () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(null)
    setStatusMsg(null)
    onCapture(null)
    await startCamera()
  }

  const handleUpload = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setPreviewUrl(URL.createObjectURL(file))
    setPhase('preview')
    stopStream()
    onCapture(file)
  }

  const cameraActive = !['idle', 'preview', 'starting', 'failed'].includes(phase)

  if (phase === 'preview' && previewUrl) {
    return (
      <div className={styles.wrap}>
        <div className={styles.previewFrame}>
          <img src={previewUrl} alt="Session selfie" className={styles.preview} />
          <span className={`${styles.badge} ${styles.badgeOk}`}>
            <Check size={11} /> Identity verified
          </span>
          <div className={styles.previewFooter}>
            <Check size={13} />
            Session photo captured
          </div>
        </div>
        <button type="button" className={styles.retakeBtn} onClick={retake}>
          <RefreshCw size={13} /> Retake
        </button>
      </div>
    )
  }

  return (
    <div className={styles.wrap}>
      <canvas ref={captureCanvas} style={{ display: 'none' }} />
      <input
        ref={fileRef} type="file" accept="image/*"
        style={{ display: 'none' }} onChange={handleUpload}
      />

      <div className={[
        styles.videoWrap,
        cameraActive ? styles.videoVisible : styles.videoHidden,
        phase === 'confirmed' || phase === 'capturing' ? styles.ringConfirmed : '',
        faceCentered && phase === 'detecting_face' ? styles.ringOk : '',
      ].join(' ')}>

        <video ref={videoRef} className={styles.video} muted playsInline />
        <canvas ref={overlayCanvas} className={styles.overlayCanvas} />

        {flash && <div className={styles.shutterFlash} aria-hidden />}

        {phase === 'detecting_face' && !faceFound && (
          <div className={`${styles.banner} ${styles.bannerWait}`}>
            Look at the camera — centre your face in the oval
          </div>
        )}
        {phase === 'detecting_face' && faceFound && !faceCentered && (
          <div className={`${styles.banner} ${styles.bannerWarn}`}>
            Move your face into the centre of the oval
          </div>
        )}

        {phase === 'capturing' && (
          <div className={`${styles.banner} ${styles.bannerConfirmed}`}>
            <Camera size={14} /> Face locked — capturing…
          </div>
        )}
        {phase === 'confirmed' && (
          <div className={`${styles.banner} ${styles.bannerConfirmed}`}>
            <Check size={14} /> Identity verified — continuing…
          </div>
        )}
      </div>

      {phase === 'idle' && (
        <button type="button" className={styles.startBtn} onClick={startCamera}>
          <ScanFace size={15} /> Start face scan
        </button>
      )}

      {phase === 'starting' && (
        <div className={styles.starting}>
          <Loader2 size={16} className={styles.spin} /> Starting camera…
        </div>
      )}

      {phase === 'failed' && (
        <div className={styles.failedBox}>
          <AlertTriangle size={20} className={styles.failedIcon} />
          <p className={styles.failedMsg}>
            {statusMsg || 'Face scan failed — centre your face and try again.'}
          </p>
          <button type="button" className={styles.retryBtn} onClick={retry}>
            <RefreshCw size={13} /> Try again
          </button>
        </div>
      )}

      {errorMsg && (
        <div className={styles.errorBox}>
          <AlertTriangle size={13} />
          <span>{errorMsg}</span>
          <button type="button" className={styles.inlineRetry} onClick={startCamera}>Retry</button>
        </div>
      )}

      {(phase === 'idle' || errorMsg) && (
        <button
          type="button"
          className={styles.uploadLink}
          onClick={() => fileRef.current?.click()}
          title="Skip face scan — for development/demo only"
        >
          <Upload size={11} /> Upload a photo instead (demo — skips scan)
        </button>
      )}
    </div>
  )
}
