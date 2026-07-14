import { useEffect, useRef, useState } from 'react'
import { Camera, RotateCcw, Check, AlertTriangle, Upload } from 'lucide-react'
import styles from './CameraCapture.module.css'

// Live camera capture is the default and closes the single biggest hole in
// the old evidence flow: with a bare file input, anyone can re-select an old
// photo from disk. An explicit "upload instead" escape hatch stays available
// for demos/dev where a real item isn't on hand — uploaded evidence is
// labelled distinctly (never shown as "Captured live") so the distinction
// isn't lost downstream.
export default function CameraCapture({ facingMode = 'environment', onCapture, label }) {
  const videoRef  = useRef(null)
  const streamRef = useRef(null)
  const fileRef   = useRef(null)
  const [error, setError]         = useState(null)
  const [ready, setReady]         = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [source, setSource]       = useState(null)   // 'camera' | 'upload'

  const stop = () => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    setReady(false)
  }

  const start = async () => {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera capture is not supported in this browser.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode }, audio: false })
      streamRef.current = stream
      // setReady(true) mounts the <video> element for the first time — the
      // ref isn't attached to a DOM node yet at this point in the same tick,
      // so wiring srcObject happens in the effect below once it is.
      setReady(true)
    } catch (e) {
      setError(
        e.name === 'NotAllowedError'
          ? 'Camera permission denied — allow camera access to capture live evidence.'
          : 'Could not access a camera on this device.'
      )
      setReady(false)
    }
  }

  useEffect(() => {
    if (ready && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current
      videoRef.current.play().catch(() => {})
    }
  }, [ready])

  useEffect(() => () => stop(), [])

  const capture = () => {
    const video = videoRef.current
    if (!video || !video.videoWidth) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob(blob => {
      if (!blob) return
      const file = new File([blob], `capture-${Date.now()}.jpg`, { type: 'image/jpeg' })
      setPreviewUrl(URL.createObjectURL(blob))
      setSource('camera')
      onCapture(file)
      stop()
    }, 'image/jpeg', 0.92)
  }

  const handleUpload = (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''   // allow re-selecting the same file after a retake
    if (!file) return
    setPreviewUrl(URL.createObjectURL(file))
    setSource('upload')
    onCapture(file)
  }

  const retake = () => {
    setPreviewUrl(null)
    setSource(null)
    onCapture(null)
  }

  if (previewUrl) {
    return (
      <div className={styles.wrap}>
        <div className={styles.previewFrame}>
          <img src={previewUrl} alt={label} className={styles.preview} />
          <span className={`${styles.capturedBadge} ${source === 'upload' ? styles.uploadBadge : ''}`}>
            <Check size={11} /> {source === 'upload' ? 'Uploaded (demo)' : 'Captured live'}
          </span>
        </div>
        <button type="button" className={styles.retakeBtn} onClick={retake}>
          <RotateCcw size={13} /> Retake
        </button>
      </div>
    )
  }

  return (
    <div className={styles.wrap}>
      {ready ? (
        <div className={styles.videoFrame}>
          <video ref={videoRef} className={styles.video} muted playsInline />
          <button type="button" className={styles.shutterBtn} onClick={capture} aria-label={`Capture ${label}`}>
            <Camera size={18} />
          </button>
        </div>
      ) : (
        <button type="button" className={styles.startBtn} onClick={start}>
          <Camera size={16} /> Start camera — {label}
        </button>
      )}
      {error && (
        <div className={styles.error}>
          <AlertTriangle size={13} />
          <span>{error}</span>
          <button type="button" className={styles.retryBtn} onClick={start}>Try again</button>
        </div>
      )}
      {!ready && (
        <>
          <button type="button" className={styles.uploadLink} onClick={() => fileRef.current?.click()}>
            <Upload size={11} /> Upload a photo instead (demo)
          </button>
          <input
            ref={fileRef} type="file" accept="image/*" className={styles.hiddenFileInput}
            onChange={handleUpload}
          />
        </>
      )}
    </div>
  )
}
