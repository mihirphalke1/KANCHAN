/*
 * Minimal service worker — installability + an offline app shell only.
 *
 * Deliberately NOT a full offline case-submission queue: /api/* requests
 * (analysis, auth, history) are never intercepted here, so evidence capture
 * (multipart photos/audio/blobs) behaves exactly as it does without a service
 * worker — no risk of a queued-but-silently-stale submission. A branch
 * officer who loses signal mid-analysis sees the same "request failed" error
 * as today; they are not told it "saved for later" when it didn't.
 *
 * What this DOES do: cache-first for the hashed, immutable Vite build assets
 * (JS/CSS/fonts) and the static /public files, and a network-first shell for
 * navigations so the app still opens (to its own login-required state) with
 * no connectivity — the PWA installability + "opens offline" bar without
 * touching correctness-critical requests.
 */
const CACHE_NAME = 'kanchan-shell-v1'
const SHELL_URL = '/'

self.addEventListener('install', (event) => {
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    ).then(() => self.clients.claim())
  )
})

function isStaticAsset(url) {
  return (
    url.pathname.startsWith('/assets/') ||
    /\.(png|jpg|jpeg|svg|ico|webmanifest|woff2?)$/.test(url.pathname)
  )
}

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') return

  const url = new URL(request.url)
  if (url.origin !== self.location.origin) return
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/cases/')) return

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((res) => {
          caches.open(CACHE_NAME).then((c) => c.put(SHELL_URL, res.clone()))
          return res
        })
        .catch(() => caches.match(SHELL_URL))
    )
    return
  }

  if (isStaticAsset(url)) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached
        return fetch(request).then((res) => {
          if (res.ok) caches.open(CACHE_NAME).then((c) => c.put(request, res.clone()))
          return res
        })
      })
    )
  }
})
