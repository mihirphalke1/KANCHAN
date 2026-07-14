import { Navigate, useLocation } from 'react-router-dom'
import { getToken, getEvaluator } from '@/lib/auth'

// Route guard for the Evaluator Integrity Layer: no session -> /login;
// session without a captured selfie -> /login (selfie step) — every case
// must be traceable to a person who was physically present at login, so
// the app never lets an evaluator reach the analysis screen without one.
export default function RequireAuth({ children }) {
  const location  = useLocation()
  const token     = getToken()
  const evaluator = getEvaluator()

  if (!token || !evaluator) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }
  if (!evaluator.selfie_captured) {
    return <Navigate to="/login" state={{ from: location, step: 'selfie' }} replace />
  }
  return children
}
