"""
Shared rate limiter (slowapi) — protects the expensive/abusable endpoints.

/api/analyze runs an LLM vision call plus several CV/ML pipelines per request;
unbounded request volume is both a runaway-API-cost risk and a DoS surface.
/api/auth/login is a PIN brute-force target (4-digit PINs — see app/auth.py).
Both are throttled per-client-IP. Limits are generous enough not to interfere
with normal branch-officer usage (a handful of analyses/logins a minute) and
are configurable via env for a real deployment.
"""
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

ANALYZE_RATE_LIMIT = os.getenv("ANALYZE_RATE_LIMIT", "20/minute")
LOGIN_RATE_LIMIT    = os.getenv("LOGIN_RATE_LIMIT", "10/minute")
