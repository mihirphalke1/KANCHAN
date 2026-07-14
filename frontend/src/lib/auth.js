const TOKEN_KEY     = 'kanchan_token'
const EVALUATOR_KEY = 'kanchan_evaluator'

export function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

export function getEvaluator() {
  try {
    return JSON.parse(localStorage.getItem(EVALUATOR_KEY) || 'null')
  } catch {
    return null
  }
}

export function setSession(token, evaluator) {
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(EVALUATOR_KEY, JSON.stringify(evaluator))
}

export function setSelfieCaptured(captured = true) {
  const e = getEvaluator()
  if (e) localStorage.setItem(EVALUATOR_KEY, JSON.stringify({ ...e, selfie_captured: captured }))
}

export function clearSession() {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(EVALUATOR_KEY)
}

export function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// Wraps fetch with the evaluator's bearer token; a 401 means the session
// expired or was never valid, so it clears local state and bounces to
// /login rather than letting the app sit in a half-authenticated state.
export async function apiFetch(url, options = {}) {
  const headers = { ...(options.headers || {}), ...authHeaders() }
  const res = await fetch(url, { ...options, headers })
  if (res.status === 401) {
    clearSession()
    if (!window.location.pathname.startsWith('/login')) {
      window.location.href = '/login'
    }
  }
  return res
}
