import { useEffect, useRef, useState } from 'react'
import {
  Mic, Camera, Weight, ChevronDown, ChevronRight,
  Loader2, Sparkles, X, User, RefreshCw, Plus, MapPin,
  UsbIcon, Search, ShieldCheck, Flashlight, ScanLine,
} from 'lucide-react'
import InfoTip from './ui/InfoTip'
import CameraCapture from './CameraCapture'
import AudioCapture from './AudioCapture'
import { apiFetch } from '@/lib/auth'
import { connectScale, isWebHIDSupported } from '@/lib/usbScale'
import styles from './AnalysisForm.module.css'

const KARATS = [
  { value: 24, label: '24K - 99.9% gold' },
  { value: 23, label: '23K - 95.8% gold' },
  { value: 22, label: '22K - 91.7% gold' },
  { value: 18, label: '18K - 75.0% gold' },
  { value: 14, label: '14K - 58.3% gold' },
  { value: 0,  label: 'Not declared — identify from physics' },
]

const TAP_POSITIONS = ["Point B (3 o’clock)", "Point C (6 o’clock)", "Point D (9 o’clock)"]

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
  customer_account:  '',
  loan_app_no:       '',
  huid:              '',
  hollow_item:       false,
  // Defaults to checked — the common case is the borrower standing right
  // there at the counter. An officer running a case with no borrower present
  // (e.g. a re-check) explicitly unchecks it, which is what marks the LTV as
  // indicative-only rather than final (see EvidencePanel's LTV card).
  borrower_present:  true,
}

// Image-first flow: photos + weights get a verdict; text details are an
// OPTIONAL refinement that unlocks after the first evaluation. Every photo
// and tap recording is LIVE-CAPTURED (camera/mic APIs), never a file
// picker — see CameraCapture / AudioCapture for why.
export default function AnalysisForm({ onSubmit, loading, hasResult }) {
  const [form, setForm]       = useState(BLANK_FORM)
  const [photos, setPhotos]   = useState([null])          // up to 4 live-captured angles
  const [photoSources, setPhotoSources] = useState([null]) // 'camera' | 'upload' per slot
  const [primaryTap, setPrimaryTap] = useState(null)
  const [extraTaps, setExtraTaps]   = useState([])         // spatial acoustic grid points
  const [streak, setStreak]   = useState(null)
  const [streakSource, setStreakSource] = useState(null)
  const [uv, setUv]           = useState(null)
  const [uvSource, setUvSource] = useState(null)
  const [hallmarkPhoto, setHallmarkPhoto] = useState(null)
  const [moreOpen, setMoreOpen] = useState(false)

  const [weightSource, setWeightSource] = useState('manual')
  const [scaleBusy, setScaleBusy]       = useState(null)   // 'dry' | 'submerged' | null
  const [scaleError, setScaleError]     = useState(null)

  // Water-displacement reading (g) — a cup of water sits on the scale,
  // tared to zero; the item hangs from a hand-held thread into the water
  // (nothing touching the scale but the cup), and the scale shows the mass
  // of water displaced directly. `form.weight_submerged` (what the backend
  // formula actually uses — see app/utils/density.py) is then just
  // dry_weight - displaced, derived below so no backend change is needed:
  // this is the same Archimedes physics, just measured without needing the
  // item itself suspended from the scale.
  const [waterDisplaced, setWaterDisplaced] = useState('')

  useEffect(() => {
    const displaced = parseFloat(waterDisplaced)
    const dry = parseFloat(form.weight_dry)
    if (waterDisplaced !== '' && !isNaN(displaced) && !isNaN(dry)) {
      set('weight_submerged', (dry - displaced).toFixed(2))
    } else if (waterDisplaced === '') {
      set('weight_submerged', '')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [waterDisplaced, form.weight_dry])

  const [geo, setGeo] = useState(null)
  useEffect(() => {
    if (!navigator.geolocation) return
    navigator.geolocation.getCurrentPosition(
      pos => setGeo({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
      () => setGeo(null),
      { timeout: 6000 },
    )
  }, [])

  const [cardLoading, setCardLoading] = useState(false)
  const printCalibrationCard = async () => {
    setCardLoading(true)
    try {
      const res = await apiFetch(`/api/fiducial/today?branch_id=${encodeURIComponent(form.branch_id || 'BLR-001')}`)
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      window.open(url, '_blank', 'noopener')
    } catch {
      // best-effort — printing the card is a convenience, never a blocker
    } finally {
      setCardLoading(false)
    }
  }

  const [kycRecord, setKycRecord]   = useState(null)
  const [kycLoading, setKycLoading] = useState(false)
  const [kycError, setKycError]     = useState(null)

  const [hallmarkResult, setHallmarkResult] = useState(null)
  const [hallmarkLoading, setHallmarkLoading] = useState(false)
  const [hallmarkError, setHallmarkError] = useState(null)

  // Live validation — errors show as you type, and block the submit button.
  const dryW  = parseFloat(form.weight_dry)
  const subW  = parseFloat(form.weight_submerged)
  const tempC = parseFloat(form.water_temp_c)

  const tempError =
    form.water_temp_c !== '' && (isNaN(tempC) || tempC <= 0 || tempC >= 45)
      ? 'Water temperature must be between 0 and 45 °C'
      : null
  const displacedW = parseFloat(waterDisplaced)
  const weightError =
    form.weight_dry && dryW <= 0
      ? 'Dry weight must be greater than zero'
      : waterDisplaced && displacedW <= 0
        ? 'Water displaced must be greater than zero'
        : form.weight_dry && form.weight_submerged && subW <= 0
          ? 'Water displaced looks too high for this item’s weight — check the reading and re-tare the scale'
          : form.weight_dry && form.weight_submerged && dryW <= subW
            ? 'Water displaced looks too high for this item’s weight — check the reading and re-tare the scale'
            : null

  const capturedPhotos = photos.filter(Boolean)

  // The three core tests are mandatory: weight + photo + sound. Known frauds
  // passed because a single factor was trusted alone — approval requires all.
  const isValid = (
    form.weight_dry &&
    form.weight_submerged &&
    !weightError &&
    !tempError &&
    capturedPhotos.length > 0 &&
    primaryTap
  )

  const missingCore = [
    !form.weight_dry || !form.weight_submerged ? 'weight readings' : null,
    capturedPhotos.length === 0 ? 'item photo' : null,
    !primaryTap ? 'tap recording' : null,
  ].filter(Boolean)

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }))

  const useScale = async (which) => {
    setScaleError(null)
    setScaleBusy(which)
    try {
      const { grams } = await connectScale()
      if (grams === null) {
        setScaleError('No weight reading received — check the scale is on and stable, or enter manually.')
      } else if (which === 'dry') {
        set('weight_dry', grams.toFixed(2))
        setWeightSource('usb_device')
      } else {
        // Scale is tared to the water cup at this point — this reading IS
        // the displaced-water mass directly, not the item's own weight.
        setWaterDisplaced(grams.toFixed(2))
        setWeightSource('usb_device')
      }
    } catch (e) {
      setScaleError(e.message)
    } finally {
      setScaleBusy(null)
    }
  }

  const runKycLookup = async () => {
    if (!form.customer_account) return
    setKycLoading(true)
    setKycError(null)
    setKycRecord(null)
    try {
      const res = await apiFetch(`/api/kyc/lookup?account_no=${encodeURIComponent(form.customer_account)}`)
      const data = await res.json()
      if (!res.ok || !data.found) {
        setKycError(data.note || 'No matching record found')
      } else {
        setKycRecord(data)
        if (data.record?.name) set('customer_name', data.record.name)
      }
    } catch {
      setKycError('Lookup failed — check the connection and try again')
    } finally {
      setKycLoading(false)
    }
  }

  const runHallmarkVerify = async () => {
    if (!hallmarkPhoto) return
    setHallmarkLoading(true)
    setHallmarkError(null)
    try {
      const fd = new FormData()
      fd.append('huid', form.huid)
      fd.append('declared_karat', form.declared_karat)
      fd.append('hallmark_image', hallmarkPhoto)
      const res = await apiFetch('/api/hallmark/verify', { method: 'POST', body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error('Hallmark verification failed')
      setHallmarkResult(data)
    } catch (e) {
      setHallmarkError(e.message)
    } finally {
      setHallmarkLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!isValid || loading) return
    const fd = new FormData()
    Object.entries(form).forEach(([k, v]) => fd.append(k, v))
    fd.append('weight_capture_source', weightSource)
    if (geo) { fd.append('geo_lat', geo.lat); fd.append('geo_lon', geo.lon) }
    if (hallmarkResult) fd.append('hallmark_result', JSON.stringify(hallmarkResult))
    if (kycRecord)      fd.append('kyc_record', JSON.stringify(kycRecord))

    // Per-image capture provenance, aligned to the `images` order, so the
    // backend can enforce live-capture-only evidence in production
    // (ALLOW_UPLOAD_EVIDENCE=0) and record the source either way.
    photos.forEach((img, idx) => {
      if (!img) return
      fd.append('images', img)
      fd.append('image_sources', photoSources[idx] || 'camera')
    })
    fd.append('audio', primaryTap)
    extraTaps.forEach(t => { if (t.file) { fd.append('tap_audio', t.file); fd.append('tap_positions', t.position) } })
    if (streak) { fd.append('streak_image', streak); fd.append('streak_source', streakSource || 'camera') }
    if (uv)     { fd.append('uv_image', uv); fd.append('uv_source', uvSource || 'camera') }

    onSubmit(fd, { images: capturedPhotos, audio: primaryTap, streak })
  }

  const addPhotoSlot = () => photos.length < 4 && (setPhotos(p => [...p, null]), setPhotoSources(s => [...s, null]))
  const setPhotoAt = (i, file, source = 'camera') => {
    setPhotos(p => p.map((x, idx) => idx === i ? file : x))
    setPhotoSources(s => s.map((x, idx) => idx === i ? (file ? source : null) : x))
  }
  const removePhotoSlot = (i) => {
    setPhotos(p => p.length > 1 ? p.filter((_, idx) => idx !== i) : [null])
    setPhotoSources(s => s.length > 1 ? s.filter((_, idx) => idx !== i) : [null])
  }

  const addTapPoint = () => extraTaps.length < 3 &&
    setExtraTaps(t => [...t, { position: TAP_POSITIONS[t.length] || `Point ${t.length + 2}`, file: null }])
  const setTapFile = (i, file) => setExtraTaps(t => t.map((x, idx) => idx === i ? { ...x, file } : x))
  const removeTapPoint = (i) => setExtraTaps(t => t.filter((_, idx) => idx !== i))

  return (
    <form className={styles.form} onSubmit={handleSubmit} noValidate>
      <div className={styles.formHeader}>
        <h2 className={styles.formTitle}>Item Analysis</h2>
        <p className={styles.formSub}>Three core tests — live photos, weights, tap sound. Text details can be added after the first result</p>
      </div>

      {/* ── 1. Customer details ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <User size={13} />
          Customer Information
        </legend>
        <div className={styles.row}>
          <div className={styles.field}>
            <label htmlFor="customer_name_top" className={styles.label}>Customer Name</label>
            <input
              id="customer_name_top" className={styles.input} type="text"
              placeholder="Full name" value={form.customer_name}
              onChange={e => set('customer_name', e.target.value)}
            />
          </div>
          <div className={styles.field}>
            <label htmlFor="customer_account" className={styles.label}>
              Account Number
              <InfoTip text="Looks up an already-KYC'd customer record from Canara's core banking system. SIMULATED — no live core-banking integration is wired; this reproduces the lookup contract against a local demo dataset (try 04581234567)." />
            </label>
            <div className={styles.inlineRow}>
              <input
                id="customer_account" className={styles.input} type="text"
                placeholder="e.g. 04581234567" value={form.customer_account}
                onChange={e => set('customer_account', e.target.value)}
              />
              <button type="button" className={styles.smallActionBtn} onClick={runKycLookup} disabled={!form.customer_account || kycLoading}>
                {kycLoading ? <Loader2 size={13} className={styles.spinner} /> : <Search size={13} />}
              </button>
            </div>
          </div>
        </div>
        {kycRecord && (
          <div className={styles.kycResult}>
            <span className={styles.simBadge}>SIMULATED</span>
            <span>{kycRecord.record.name} · KYC {kycRecord.record.kyc_status} · {kycRecord.record.existing_gold_loans} existing gold loan(s)</span>
          </div>
        )}
        {kycError && <p className={styles.fieldError}>{kycError}</p>}
        <label className={styles.checkRow}>
          <input
            type="checkbox" checked={form.borrower_present}
            onChange={e => set('borrower_present', e.target.checked)}
          />
          <span>
            Borrower is physically present at valuation
            <InfoTip text="RBI requires the borrower present when the pledge is valued. Until this is attested, the loan valuation (LTV) is treated as provisional and cannot be finalised for disbursal." />
          </span>
        </label>
      </fieldset>

      {/* ── 2. Photos — live capture only ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <Camera size={13} />
          Item Photos — live capture
        </legend>
        <p className={styles.helpText}>
          Place the item on a plain backdrop with today's calibration card visible in-frame
          (<InfoTip text="Print today's card from Dashboard → Calibration Card. It gives a real-world scale reference for stone sizing and proves the photo was taken today, not reused." />).
          Live camera capture is the default — an upload fallback stays available underneath for demos.
        </p>
        <button type="button" className={styles.addBtn} onClick={printCalibrationCard} disabled={cardLoading} style={{ marginTop: 0, marginBottom: 12 }}>
          {cardLoading ? <Loader2 size={13} className={styles.spinner} /> : <ScanLine size={13} />}
          Print today's calibration card
        </button>
        <div className={styles.photoGrid}>
          {photos.map((file, i) => (
            <div key={i} className={styles.photoSlot}>
              <CameraCapture facingMode="environment" label={`angle ${i + 1}`} onCapture={f => setPhotoAt(i, f)} />
              {photos.length > 1 && (
                <button type="button" className={styles.slotRemove} onClick={() => removePhotoSlot(i)}>
                  <X size={11} /> Remove slot
                </button>
              )}
            </div>
          ))}
        </div>
        {photos.length < 4 && (
          <button type="button" className={styles.addBtn} onClick={addPhotoSlot}>
            <Plus size={13} /> Add another angle
          </button>
        )}
      </fieldset>

      {/* ── 3. Weights + declared karat ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <Weight size={13} />
          Weight Test (in air & in water)
        </legend>

        {isWebHIDSupported() && (
          <p className={styles.helpText}>
            <UsbIcon size={11} style={{ display: 'inline', verticalAlign: -2 }} /> USB scale supported on this device — connect once per weighing, or enter manually below.
          </p>
        )}

        <div className={`${styles.row} ${styles.row3}`}>
          <div className={styles.field}>
            <label htmlFor="weight_dry" className={styles.label}>
              Dry Weight <span className={styles.unit}>g</span>
              <InfoTip text="Weigh the item in air. Connect a USB/HID scale for a tamper-resistant reading, or note the reading to 0.01 g manually." />
            </label>
            <div className={styles.inlineRow}>
              <input
                id="weight_dry" className={styles.input} type="number" step="0.01" min="0.5"
                placeholder="0.00" value={form.weight_dry}
                onChange={e => { set('weight_dry', e.target.value); setWeightSource('manual') }} required
              />
              {isWebHIDSupported() && (
                <button type="button" className={styles.smallActionBtn} onClick={() => useScale('dry')} disabled={scaleBusy !== null}>
                  {scaleBusy === 'dry' ? <Loader2 size={13} className={styles.spinner} /> : <UsbIcon size={13} />}
                </button>
              )}
            </div>
          </div>
          <div className={styles.field}>
            <label htmlFor="water_displaced" className={styles.label}>
              Water Displaced <span className={styles.unit}>g</span>
              <InfoTip text="Tare the scale with a cup of water on it. Lower the item in on a hand-held thread, not touching the sides or bottom — the reading shown is the water displaced." />
            </label>
            <div className={styles.inlineRow}>
              <input
                id="water_displaced" className={styles.input} type="number" step="0.01" min="0.01"
                placeholder="0.20" value={waterDisplaced}
                onChange={e => { setWaterDisplaced(e.target.value); setWeightSource('manual') }} required
              />
              {isWebHIDSupported() && (
                <button type="button" className={styles.smallActionBtn} onClick={() => useScale('submerged')} disabled={scaleBusy !== null}>
                  {scaleBusy === 'submerged' ? <Loader2 size={13} className={styles.spinner} /> : <UsbIcon size={13} />}
                </button>
              )}
            </div>
          </div>
          <div className={styles.field}>
            <label htmlFor="water_temp_c" className={styles.label}>
              Water Temp <span className={styles.unit}>°C</span>
              <InfoTip text="Water density changes with temperature, and the density result is corrected for it. Leave 25 °C if you don't have a thermometer." />
            </label>
            <input
              id="water_temp_c" className={styles.input} type="number" step="0.5" min="1" max="44"
              placeholder="25" value={form.water_temp_c}
              onChange={e => set('water_temp_c', e.target.value)}
            />
          </div>
        </div>
        {scaleError && <p className={styles.fieldError}>{scaleError}</p>}
        {weightSource === 'usb_device' && <p className={styles.usbConfirm}><UsbIcon size={11} /> Weight captured from USB scale</p>}

        <div className={styles.field}>
          <label htmlFor="declared_karat" className={styles.label}>
            Declared Karat
            <InfoTip text="The purity the customer claims. The weight-in-water test checks the measured density against this karat's expected range." />
          </label>
          <div className={styles.selectWrap}>
            <select
              id="declared_karat" className={styles.select} value={form.declared_karat}
              onChange={e => set('declared_karat', parseInt(e.target.value))}
            >
              {KARATS.map(k => <option key={k.value} value={k.value}>{k.label}</option>)}
            </select>
            <ChevronDown className={styles.selectIcon} size={15} />
          </div>
        </div>

        <label className={styles.checkRow}>
          <input
            type="checkbox" checked={form.hollow_item}
            onChange={e => set('hollow_item', e.target.checked)}
          />
          <span>
            Hollow item (bangle, jhumka dome, etc.)
            <InfoTip text="A hollow piece should read lighter than solid gold. If it's hollow but the density reads solid-gold, that's the dense-core-fill signature — the ring-pitch sound test becomes the decisive check." />
          </span>
        </label>

        {tempError && <p className={styles.fieldError}>{tempError}</p>}
        {weightError && <p className={styles.fieldError}>{weightError}</p>}
        {!tempError && !weightError && form.weight_dry && form.weight_submerged &&
          dryW > subW && subW > 0 && (
          (() => {
            const t = isNaN(tempC) ? 25 : tempC
            const rhoWater = waterDensity(t)
            const density  = (dryW * rhoWater / (dryW - subW)).toFixed(2)
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

      {/* ── 4. Sound test — third core test + spatial acoustic mapping ── */}
      <fieldset className={styles.section}>
        <legend className={styles.sectionLabel}>
          <Mic size={13} />
          Sound Test
        </legend>
        <p className={styles.helpText}>
          Tap the item once with a metal stylus and record the ring. This is what catches
          filled-core fakes that pass the weight test — no approval is possible without it.
        </p>
        <AudioCapture label="Primary tap point" onCapture={setPrimaryTap} />

        <div className={styles.spatialBlock}>
          <div className={styles.spatialHead}>
            <span className={styles.spatialTitle}>
              Spatial acoustic mapping
              <InfoTip text="Tap 2-4 distinct points on the item. A genuine solid piece rings at a consistent pitch everywhere; a hidden partial core (tungsten/lead under a gold shell) rings differently depending on where you tap — this catches what a single tap and the average density number both miss." />
            </span>
            <span className={styles.noveltyBadge}>Novel</span>
          </div>
          {extraTaps.map((t, i) => (
            <div key={i} className={styles.tapRow}>
              <AudioCapture label={t.position} onCapture={f => setTapFile(i, f)} />
              <button type="button" className={styles.slotRemove} onClick={() => removeTapPoint(i)}>
                <X size={11} /> Remove point
              </button>
            </div>
          ))}
          {extraTaps.length < 3 && (
            <button type="button" className={styles.addBtn} onClick={addTapPoint}>
              <Plus size={13} /> Add another tap point ({extraTaps.length}/3)
            </button>
          )}
        </div>
      </fieldset>

      {/* ── 5. Optional extra tests, collapsed ── */}
      <fieldset className={styles.section}>
        <button type="button" className={styles.moreToggle} onClick={() => setMoreOpen(o => !o)}>
          {moreOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          More tests (optional) — streak, UV, hallmark
          {(streak || uv || hallmarkPhoto) && <span className={styles.moreCount}>
            {[streak, uv, hallmarkPhoto].filter(Boolean).length} added
          </span>}
        </button>

        {moreOpen && (
          <div className={styles.moreBody}>
            <div className={styles.field}>
              <label className={styles.label}>
                <Camera size={12} /> Touchstone Streak Photo
                <InfoTip text="Rub the item on the touchstone and photograph the streak mark. HSV hue analysed against per-karat reference bands — catches surface-level plating a rub cuts through." />
              </label>
              <CameraCapture facingMode="environment" label="streak photo" onCapture={(f, s) => { setStreak(f); setStreakSource(s) }} />
            </div>

            <div className={styles.field}>
              <label className={styles.label}>
                <Flashlight size={12} /> UV-Pass ("Blacklight") Photo
                <InfoTip text="Photograph the item under a UV-A torch. Solid genuine gold should not fluoresce; rhodium plating and many synthetic stones do — an assistive signal, not conclusive on its own." />
              </label>
              <CameraCapture facingMode="environment" label="UV-pass photo" onCapture={(f, s) => { setUv(f); setUvSource(s) }} />
            </div>

            <div className={styles.field}>
              <label className={styles.label}>
                <ScanLine size={12} /> Hallmark / HUID
                <InfoTip text="BIS has no public developer API — HUID lookup here is against a SIMULATED local registry. A HUID is trivial to copy, so this also checks the engraving itself (stroke width/spacing) for a laser-engraving signature, and cross-checks the registry's purity against the physics-measured karat." />
              </label>
              <input
                className={styles.input} type="text" placeholder="6-character HUID (e.g. AZ4E58C2)"
                value={form.huid} onChange={e => set('huid', e.target.value.toUpperCase())}
                style={{ marginBottom: 8 }}
              />
              <CameraCapture facingMode="environment" label="hallmark macro photo" onCapture={setHallmarkPhoto} />
              {hallmarkPhoto && (
                <button type="button" className={styles.addBtn} onClick={runHallmarkVerify} disabled={hallmarkLoading} style={{ marginTop: 8 }}>
                  {hallmarkLoading ? <Loader2 size={13} className={styles.spinner} /> : <ShieldCheck size={13} />}
                  Verify hallmark
                </button>
              )}
              {hallmarkError && <p className={styles.fieldError}>{hallmarkError}</p>}
              {hallmarkResult && (
                <div className={styles.kycResult}>
                  <span className={styles.simBadge}>SIMULATED</span>
                  <span>
                    {hallmarkResult.registry_found
                      ? `Registry: ${hallmarkResult.registry_record.purity_karat}K, ${hallmarkResult.registry_record.status}`
                      : 'No registry match'} · Engraving risk {Math.round(hallmarkResult.engraving.risk_score * 100)}%
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </fieldset>

      {/* ── Analyse ── */}
      <button type="submit" className={styles.submitBtn} disabled={!isValid || loading} aria-busy={loading}>
        {loading ? (
          <><Loader2 size={18} className={styles.spinner} /> Analysing item…</>
        ) : (
          <><Sparkles size={18} /> {hasResult ? 'Analyse Again' : 'Analyse Gold Item'}</>
        )}
      </button>

      {!isValid && missingCore.length > 0 && (
        <p className={styles.formHint}>
          All three core tests are required — still missing: {missingCore.join(', ')}
        </p>
      )}
      {geo && (
        <p className={styles.geoHint}><MapPin size={11} /> Location captured for evidence stamping</p>
      )}

      {/* ── 6. Officer inputs — unlocks after the first evaluation ── */}
      {hasResult && (
        <fieldset className={`${styles.section} ${styles.refineSection}`}>
          <legend className={styles.sectionLabel}>
            <User size={13} />
            Officer Inputs — refine this result
          </legend>
          <p className={styles.helpText}>
            Add anything the customer declared or details for the record, then re-analyse.
            Declared stones are cross-checked against the photo — a declaration can raise
            a flag but never lower the risk.
          </p>

          <div className={styles.field}>
            <label htmlFor="item_description" className={styles.label}>
              Item description
              <InfoTip text="Describe the item as the customer declares it, including stones (e.g. '22K necklace with rubies'). The photo analysis cross-checks this — declared stones the camera can't find raise a flag." />
            </label>
            <input
              id="item_description" className={styles.input} type="text"
              placeholder="e.g. 22K gold necklace, set with rubies"
              value={form.item_description} onChange={e => set('item_description', e.target.value)}
            />
          </div>

          <div className={styles.row}>
            <div className={styles.field}>
              <label htmlFor="loan_app_no" className={styles.label}>Loan Application No.</label>
              <input
                id="loan_app_no" className={styles.input} type="text"
                placeholder="e.g. LA-2026-00123" value={form.loan_app_no}
                onChange={e => set('loan_app_no', e.target.value)}
              />
            </div>
            <div className={styles.field}>
              <label htmlFor="branch_id" className={styles.label}>Branch ID</label>
              <input
                id="branch_id" className={styles.input} type="text"
                placeholder="e.g. BLR-001" value={form.branch_id}
                onChange={e => set('branch_id', e.target.value)}
              />
            </div>
          </div>

          <button type="submit" className={styles.refineBtn} disabled={!isValid || loading}>
            <RefreshCw size={14} />
            Re-analyse with details
          </button>
        </fieldset>
      )}
    </form>
  )
}
