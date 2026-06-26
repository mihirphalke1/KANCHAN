import { useState, useRef } from 'react'
import {
  Upload, Mic, Camera, Weight, ChevronDown,
  Loader2, Sparkles, X, Image, FileAudio
} from 'lucide-react'
import styles from './AnalysisForm.module.css'

const KARATS = [
  { value: 14, label: '14K — 58.3% gold' },
  { value: 18, label: '18K — 75.0% gold' },
  { value: 22, label: '22K — 91.7% gold' },
  { value: 24, label: '24K — 99.9% gold' },
]

const LLM_PROVIDERS = [
  { value: 'groq',   label: 'Groq (Llama 3 70B) — Fast' },
  { value: 'gemini', label: 'Google Gemini 1.5 Flash' },
]

export default function AnalysisForm({ onSubmit, loading }) {
  const [form, setForm] = useState({
    item_description:  '',
    declared_karat:    22,
    weight_dry:        '',
    weight_submerged:  '',
    branch_id:         'main',
    llm_provider:      'groq',
  })
  const [images,      setImages]      = useState([])
  const [audio,       setAudio]       = useState(null)
  const [streak,      setStreak]      = useState(null)

  const imageRef  = useRef()
  const audioRef  = useRef()
  const streakRef = useRef()

  const isValid = (
    form.item_description.trim() &&
    form.weight_dry &&
    form.weight_submerged &&
    parseFloat(form.weight_dry) > parseFloat(form.weight_submerged)
  )

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!isValid || loading) return
    const fd = new FormData()
    Object.entries(form).forEach(([k, v]) => fd.append(k, v))
    images.forEach(img => fd.append('images', img))
    if (audio)  fd.append('audio', audio)
    if (streak) fd.append('streak_image', streak)
    onSubmit(fd)
  }

  const removeImage = (i) => setImages(imgs => imgs.filter((_, idx) => idx !== i))

  return (
    <form className={styles.form} onSubmit={handleSubmit} noValidate>
      <div className={styles.formHeader}>
        <h2 className={styles.formTitle}>Item Analysis</h2>
        <p className={styles.formSub}>Enter all available details for best accuracy</p>
      </div>

      {/* ── Item Details ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>Item Details</legend>

        <div className={styles.field}>
          <label htmlFor="item_description" className={styles.label}>Description</label>
          <input
            id="item_description"
            className={styles.input}
            type="text"
            placeholder="e.g. 22K gold necklace, set with rubies"
            value={form.item_description}
            onChange={e => set('item_description', e.target.value)}
            required
          />
        </div>

        <div className={styles.row}>
          <div className={styles.field}>
            <label htmlFor="declared_karat" className={styles.label}>Declared Karat</label>
            <div className={styles.selectWrap}>
              <select
                id="declared_karat"
                className={styles.select}
                value={form.declared_karat}
                onChange={e => set('declared_karat', parseInt(e.target.value))}
              >
                {KARATS.map(k => (
                  <option key={k.value} value={k.value}>{k.label}</option>
                ))}
              </select>
              <ChevronDown className={styles.selectIcon} size={15} />
            </div>
          </div>

          <div className={styles.field}>
            <label htmlFor="branch_id" className={styles.label}>Branch ID</label>
            <input
              id="branch_id"
              className={styles.input}
              type="text"
              placeholder="e.g. BLR-001"
              value={form.branch_id}
              onChange={e => set('branch_id', e.target.value)}
            />
          </div>
        </div>
      </fieldset>

      {/* ── Density Inputs ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <Weight size={13} />
          Density Test (Archimedes)
        </legend>

        <div className={styles.row}>
          <div className={styles.field}>
            <label htmlFor="weight_dry" className={styles.label}>
              Dry Weight <span className={styles.unit}>g</span>
            </label>
            <input
              id="weight_dry"
              className={styles.input}
              type="number"
              step="0.01"
              min="0.5"
              placeholder="0.00"
              value={form.weight_dry}
              onChange={e => set('weight_dry', e.target.value)}
              required
            />
          </div>
          <div className={styles.field}>
            <label htmlFor="weight_submerged" className={styles.label}>
              Submerged <span className={styles.unit}>g</span>
            </label>
            <input
              id="weight_submerged"
              className={styles.input}
              type="number"
              step="0.01"
              min="0.1"
              placeholder="0.00"
              value={form.weight_submerged}
              onChange={e => set('weight_submerged', e.target.value)}
              required
            />
          </div>
        </div>

        {form.weight_dry && form.weight_submerged && (
          (() => {
            const d = parseFloat(form.weight_dry)
            const s = parseFloat(form.weight_submerged)
            if (d > s && s > 0) {
              const density = (d / (d - s)).toFixed(2)
              return (
                <div className={styles.densityPreview}>
                  <span>Calculated density:</span>
                  <strong>{density} g/cm³</strong>
                </div>
              )
            }
            if (d <= s) return (
              <p className={styles.fieldError}>Submerged weight must be less than dry weight</p>
            )
          })()
        )}
      </fieldset>

      {/* ── Media Uploads ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <Camera size={13} />
          Visual Evidence
        </legend>

        {/* Multi-angle photos */}
        <div className={styles.field}>
          <label className={styles.label}>
            Item Photos <span className={styles.optional}>(multiple angles)</span>
          </label>
          <div
            className={styles.dropzone}
            onClick={() => imageRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => {
              e.preventDefault()
              const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
              setImages(imgs => [...imgs, ...files])
            }}
          >
            <Image size={22} className={styles.dropzoneIcon} />
            <span className={styles.dropzoneText}>
              {images.length > 0
                ? `${images.length} photo${images.length > 1 ? 's' : ''} selected`
                : 'Tap to upload or drag photos here'}
            </span>
            <span className={styles.dropzoneSub}>JPG, PNG, WebP — multiple angles recommended</span>
          </div>
          <input
            ref={imageRef}
            type="file"
            accept="image/*"
            multiple
            className={styles.hiddenInput}
            onChange={e => setImages(imgs => [...imgs, ...Array.from(e.target.files)])}
          />
          {images.length > 0 && (
            <div className={styles.thumbnails}>
              {images.map((img, i) => (
                <div key={i} className={styles.thumb}>
                  <img src={URL.createObjectURL(img)} alt={`Angle ${i+1}`} />
                  <button
                    type="button"
                    className={styles.thumbRemove}
                    onClick={() => removeImage(i)}
                    aria-label={`Remove photo ${i+1}`}
                  >
                    <X size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Streak photo */}
        <div className={styles.field}>
          <label className={styles.label}>
            Touchstone Streak Photo <span className={styles.optional}>(optional)</span>
          </label>
          <div
            className={`${styles.dropzone} ${styles.dropzoneSmall}`}
            onClick={() => streakRef.current?.click()}
          >
            <Camera size={18} className={styles.dropzoneIcon} />
            <span className={styles.dropzoneText}>
              {streak ? streak.name : 'Upload streak image'}
            </span>
          </div>
          <input
            ref={streakRef}
            type="file"
            accept="image/*"
            className={styles.hiddenInput}
            onChange={e => setStreak(e.target.files[0] || null)}
          />
          {streak && (
            <div className={styles.fileTag}>
              <Camera size={12} />
              <span>{streak.name}</span>
              <button type="button" onClick={() => setStreak(null)}><X size={10} /></button>
            </div>
          )}
        </div>
      </fieldset>

      {/* ── Audio ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <Mic size={13} />
          Acoustic Test <span className={styles.noveltyBadge}>Novelty 1</span>
        </legend>
        <p className={styles.helpText}>
          Tap the item with a metal stylus. Record the ring sound on a smartphone.
        </p>
        <div
          className={`${styles.dropzone} ${styles.dropzoneSmall}`}
          onClick={() => audioRef.current?.click()}
        >
          <FileAudio size={18} className={styles.dropzoneIcon} />
          <span className={styles.dropzoneText}>
            {audio ? audio.name : 'Upload WAV or M4A recording'}
          </span>
        </div>
        <input
          ref={audioRef}
          type="file"
          accept="audio/*"
          className={styles.hiddenInput}
          onChange={e => setAudio(e.target.files[0] || null)}
        />
        {audio && (
          <div className={styles.fileTag}>
            <FileAudio size={12} />
            <span>{audio.name}</span>
            <button type="button" onClick={() => setAudio(null)}><X size={10} /></button>
          </div>
        )}
      </fieldset>

      {/* ── LLM Provider ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>AI Verdict Model</legend>
        <div className={styles.field}>
          <div className={styles.selectWrap}>
            <select
              className={styles.select}
              value={form.llm_provider}
              onChange={e => set('llm_provider', e.target.value)}
            >
              {LLM_PROVIDERS.map(p => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
            <ChevronDown className={styles.selectIcon} size={15} />
          </div>
        </div>
      </fieldset>

      {/* ── Submit ── */}
      <button
        type="submit"
        className={styles.submitBtn}
        disabled={!isValid || loading}
        aria-busy={loading}
      >
        {loading ? (
          <>
            <Loader2 size={18} className={styles.spinner} />
            Analysing item…
          </>
        ) : (
          <>
            <Sparkles size={18} />
            Analyse Gold Item
          </>
        )}
      </button>

      {!isValid && form.item_description === '' && (
        <p className={styles.formHint}>Fill in item description and weight measurements to enable analysis</p>
      )}
    </form>
  )
}
