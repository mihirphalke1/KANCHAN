import { Navigate, useLocation } from 'react-router-dom'
import { getToken, getEvaluator, clearSession } from '@/lib/auth'

// Route guard for the Evaluator Integrity Layer: no session -> /login;
// session without a captured selfie -> clear it and require a fresh login.
// Jumping into the selfie step with a stale token causes 401s if the server
// was restarted, so we always force a full re-authentication instead.
export default function RequireAuth({ children }) {
  const location  = useLocation()
  const token     = getToken()
  const evaluator = getEvaluator()

  if (!token || !evaluator) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  if (!evaluator.selfie_captured) {
    clearSession()
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  return children
}
