import { useState, useRef } from 'react'
import {
  Upload, Mic, Camera, Weight, ChevronDown,
  Loader2, Sparkles, X, Image, FileAudio, User
} from 'lucide-react'
import InfoTip from './ui/InfoTip'
import styles from './AnalysisForm.module.css'

const KARATS = [
  { value: 14, label: '14K - 58.3% gold' },
  { value: 18, label: '18K - 75.0% gold' },
  { value: 22, label: '22K - 91.7% gold' },
  { value: 24, label: '24K - 99.9% gold' },
]

// CRC Handbook water density (g/cm³), linear interpolation — mirrors app/utils/density.py
const WATER_DENSITY_TABLE = [
  [10, 0.9997026], [15, 0.9991026], [20, 0.9982071], [25, 0.9970479],
  [30, 0.9956502], [35, 0.9940349], [40, 0.9922152],
]
function waterDensity(tempC) {
  const pts = WATER_DENSITY_TABLE
  if (tempC <= pts[0][0]) return pts[0][1]
  if (tempC >= pts[pts.length - 1][0]) return pts[pts.length - 1][1]
  for (let i = 0; i < pts.length - 1; i++) {
    const [t0, r0] = pts[i], [t1, r1] = pts[i + 1]
    if (tempC >= t0 && tempC <= t1) return r0 + (r1 - r0) * (tempC - t0) / (t1 - t0)
  }
  return pts[pts.length - 1][1]
}

const BLANK_FORM = {
  item_description:  '',
  declared_karat:    22,
  weight_dry:        '',
  weight_submerged:  '',
  water_temp_c:      '25',
  branch_id:         'BLR-001',
  customer_name:     '',
  officer_name:      '',
  llm_provider:      'groq',
}

export default function AnalysisForm({ onSubmit, loading }) {
  const [form, setForm]     = useState(BLANK_FORM)
  const [images, setImages] = useState([])
  const [audio,  setAudio]  = useState(null)
  const [streak, setStreak] = useState(null)

  const imageRef  = useRef()
  const audioRef  = useRef()
  const streakRef = useRef()

  // Live validation — errors show as you type, and block the submit button.
  const dryW  = parseFloat(form.weight_dry)
  const subW  = parseFloat(form.weight_submerged)
  const tempC = parseFloat(form.water_temp_c)

  const tempError =
    form.water_temp_c !== '' && (isNaN(tempC) || tempC <= 0 || tempC >= 45)
      ? 'Water temperature must be between 0 and 45 °C'
      : null
  const weightError =
    form.weight_dry && dryW <= 0
      ? 'Dry weight must be greater than zero'
      : form.weight_dry && form.weight_submerged && subW <= 0
        ? 'Submerged weight must be greater than zero'
        : form.weight_dry && form.weight_submerged && dryW <= subW
          ? 'Submerged weight must be less than dry weight — check for air bubbles or a touching container'
          : null

  const isValid = (
    form.item_description.trim() &&
    form.weight_dry &&
    form.weight_submerged &&
    !weightError &&
    !tempError
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
    // the browser's own copies are what the results screen displays —
    // the server never stores the uploads
    onSubmit(fd, { images, audio, streak })
  }

  const removeImage = (i) => setImages(imgs => imgs.filter((_, idx) => idx !== i))

  return (
    <form className={styles.form} onSubmit={handleSubmit} noValidate>
      <div className={styles.formHeader}>
        <h2 className={styles.formTitle}>Item Analysis</h2>
        <p className={styles.formSub}>Enter all available details for best accuracy</p>
      </div>

      {/* ── Customer Information ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <User size={13} />
          Customer Information
        </legend>

        <div className={styles.row}>
          <div className={styles.field}>
            <label htmlFor="customer_name" className={styles.label}>Customer Name</label>
            <input
              id="customer_name"
              className={styles.input}
              type="text"
              placeholder="Full name"
              value={form.customer_name}
              onChange={e => set('customer_name', e.target.value)}
            />
          </div>
          <div className={styles.field}>
            <label htmlFor="officer_name" className={styles.label}>Assessment Officer</label>
            <input
              id="officer_name"
              className={styles.input}
              type="text"
              placeholder="Name (EMP-XXXX)"
              value={form.officer_name}
              onChange={e => set('officer_name', e.target.value)}
            />
          </div>
        </div>
      </fieldset>

      {/* ── Item Details ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>Item Details</legend>

        <div className={styles.field}>
          <label htmlFor="item_description" className={styles.label}>
            Description
            <InfoTip text="Describe the item as the customer declares it, including stones (e.g. '22K necklace with rubies'). The photo analysis cross-checks this — declared stones the camera can't find raise a flag." />
          </label>
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
            <label htmlFor="declared_karat" className={styles.label}>
              Declared Karat
              <InfoTip text="The purity the customer claims. The weight-in-water test checks the measured density against this karat's expected range." />
            </label>
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

        <div className={`${styles.row} ${styles.row3}`}>
          <div className={styles.field}>
            <label htmlFor="weight_dry" className={styles.label}>
              Dry Weight <span className={styles.unit}>g</span>
              <InfoTip text="Weigh the item in air on the branch balance. Note the reading to 0.01 g." />
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
              <InfoTip text="Weigh again with the item fully underwater, hanging from a thin thread. It must not touch the sides or bottom of the container, and no air bubbles should cling to it." />
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
          <div className={styles.field}>
            <label htmlFor="water_temp_c" className={styles.label}>
              Water Temp <span className={styles.unit}>°C</span>
              <InfoTip text="Water density changes with temperature, and the density result is corrected for it. If you don't have a thermometer, leave 25 °C — the error is small but noted." />
            </label>
            <input
              id="water_temp_c"
              className={styles.input}
              type="number"
              step="0.5"
              min="1"
              max="44"
              placeholder="25"
              value={form.water_temp_c}
              onChange={e => set('water_temp_c', e.target.value)}
            />
          </div>
        </div>

        {tempError && <p className={styles.fieldError}>{tempError}</p>}
        {weightError && <p className={styles.fieldError}>{weightError}</p>}
        {!tempError && !weightError && form.weight_dry && form.weight_submerged &&
          dryW > subW && subW > 0 && (
          (() => {
            const t = isNaN(tempC) ? 25 : tempC
            // Buoyancy-corrected Archimedes: rho = W_dry * rho_water(T) / (W_dry - W_sub)
            const rhoWater = waterDensity(t)
            const density  = (dryW * rhoWater / (dryW - subW)).toFixed(2)
            // sigma_rho ~ sqrt(2) * scale_sigma / displaced * rho  (0.005 g balance)
            const sigmaVal = Math.SQRT2 * 0.005 / (dryW - subW) * dryW * rhoWater / (dryW - subW)
            const sigma    = sigmaVal >= 0.01 ? sigmaVal.toFixed(2) : sigmaVal.toPrecision(1)
            return (
              <div className={styles.densityPreview}>
                <span>Calculated density:</span>
                <strong>{density} ± {sigma} g/cm³</strong>
              </div>
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
            <InfoTip text="Place the item on a plain, single-colour backdrop, centred in the frame — no fingers, props or other objects. The system separates the item from the backdrop and finds the stones automatically." />
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
            <span className={styles.dropzoneSub}>JPG, PNG, WebP - multiple angles recommended</span>
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
          Sound Test
        </legend>
        <p className={styles.helpText}>
          Tap the item once with a metal stylus and record the ring on a smartphone,
          in a quiet room. This test catches filled-core fakes that pass the weight
          test — record it whenever possible.
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
