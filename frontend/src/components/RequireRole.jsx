import { Navigate } from 'react-router-dom'
import RequireAuth from './RequireAuth'
import { getEvaluator } from '@/lib/auth'

// Kept in sync with the backend default for ADMIN_DASHBOARD_ROLES
// (app/routers/admin.py) — this only gates which nav link/route renders;
// the server independently re-checks the role on every admin API call.
export const ADMIN_DASHBOARD_ROLES = ['branch_manager', 'admin']

// Role gate for privileged routes (the admin dashboard). Layered on top of
// RequireAuth: an unauthenticated user still bounces to /login, but an
// authenticated evaluator whose role isn't in `roles` is redirected to their
// ordinary dashboard rather than being shown a bare 403 page.
export default function RequireRole({ roles, children }) {
  return (
    <RequireAuth>
      <RoleCheck roles={roles}>{children}</RoleCheck>
    </RequireAuth>
  )
}

function RoleCheck({ roles, children }) {
  const evaluator = getEvaluator()
  if (!evaluator || !roles.includes(evaluator.role)) {
    return <Navigate to="/dashboard" replace />
  }
  return children
}
