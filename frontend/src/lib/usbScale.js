// USB/HID digital scale integration — deployment-ready, hardware-optional.
//
// Most branch jeweller scales expose a USB-HID "POS scale" report (the same
// class Dymo/CAS/A&D and similar POS-integrated scales use): a small fixed
// report with a status byte and a little-endian weight value. This module
// requests a HID device via the browser's WebHID API and parses that common
// report shape. It is intentionally permissive about vendor/product IDs
// (requestDevice with an empty filter lets the branch pick whichever scale
// is plugged in) since we don't have a specific unit to calibrate against —
// treat DEFAULT_UNIT_DIVISOR as the one constant a deployment team tunes
// per actual scale model.
//
// Falls back cleanly: if WebHID isn't supported, permission is denied, or
// the connected device's reports don't parse as a plausible weight, the
// caller keeps using manual entry — this NEVER blocks the weighing step.

const DEFAULT_UNIT_DIVISOR = 100   // many POS scales report grams * 100

export function isWebHIDSupported() {
  return typeof navigator !== 'undefined' && !!navigator.hid
}

// Parses the common POS-scale HID report shape:
//   byte0: report id / status, bytes[2..3]: little-endian weight units
// Returns grams, or null if the report doesn't look like a weight.
function parseWeightReport(dataView) {
  if (dataView.byteLength < 4) return null
  const raw = dataView.getUint16(2, true)
  if (raw === 0 || raw > 20000) return null   // implausible for jewellery
  return raw / DEFAULT_UNIT_DIVISOR
}

// Connects to a USB HID scale and resolves with { device, read } where
// read() returns a Promise<number|null> resolving the next weight reading.
export async function connectScale() {
  if (!isWebHIDSupported()) {
    throw new Error('WebHID is not supported in this browser — use Chrome/Edge on desktop, or enter weights manually.')
  }
  const devices = await navigator.hid.requestDevice({ filters: [] })
  const device = devices[0]
  if (!device) throw new Error('No device selected')
  if (!device.opened) await device.open()

  return new Promise((resolve) => {
    let settled = false
    const onReport = (event) => {
      const grams = parseWeightReport(event.data)
      if (grams !== null && !settled) {
        settled = true
        device.removeEventListener('inputreport', onReport)
        resolve({ device, grams })
      }
    }
    device.addEventListener('inputreport', onReport)
    // Timeout: if nothing plausible arrives in 4s, hand control back to
    // manual entry rather than leaving the evaluator staring at a spinner.
    setTimeout(() => {
      if (!settled) {
        settled = true
        device.removeEventListener('inputreport', onReport)
        resolve({ device, grams: null })
      }
    }, 4000)
  })
}
