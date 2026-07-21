"""
RBI-tiered Loan-to-Value calculation + configurable gold rate.

Per RBI's Lending Against Gold Collateral Directions (draft April 2025,
finalised November 2025, effective 1 April 2026): a tiered maximum LTV
replaces the old flat 75% cap —
    loan amount <= Rs 2.5 lakh   -> 85%
    Rs 2.5 lakh < amount <= 5L   -> 80%
    loan amount > Rs 5 lakh      -> 75%
Tiers are declared here, not buried in a form — a bank can adjust its own
risk appetite by editing this table without touching the decision pipeline,
the same externalisation pattern as data/decision_policy.json.
"""
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

GOLD_RATE_PATH = Path("data/gold_rate.json")
DEFAULT_RATE_PER_GRAM_INR = 6800.0   # last-resort default if no file and no feed

# Live rate feed (P4-17). Set GOLD_RATE_API_URL to an IBJA/equivalent endpoint
# that returns JSON; GOLD_RATE_JSON_PATH is the dot-path to the ₹/gram value in
# that JSON (e.g. "rates.gold_999_per_gram"). refresh_gold_rate() fetches it and
# caches to gold_rate.json; the static file remains the explicit fallback if the
# feed is unset or unreachable. The hot path (load_gold_rate) only reads the
# cache, so an analysis never blocks on the network.
GOLD_RATE_API_URL   = os.getenv("GOLD_RATE_API_URL", "").strip()
GOLD_RATE_JSON_PATH = os.getenv("GOLD_RATE_JSON_PATH", "rate_per_gram_inr").strip()
GOLD_RATE_MAX_AGE_DAYS = int(os.getenv("GOLD_RATE_MAX_AGE_DAYS", "2"))


def _rate_age_days(updated_at) -> int | None:
    if not updated_at:
        return None
    try:
        d = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00")).date()
        return (date.today() - d).days
    except Exception:
        return None


def _dig(obj, dotted: str):
    cur = obj
    for key in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur


def refresh_gold_rate(timeout_s: float = 8.0) -> dict:
    """Fetch the live gold rate from GOLD_RATE_API_URL and cache it to
    gold_rate.json. Returns the resulting rate dict. On any failure (feed unset,
    network error, unparseable payload) the existing static file is left intact
    and returned — the feed is an enhancement, never a hard dependency."""
    if not GOLD_RATE_API_URL:
        current = load_gold_rate()
        current["refresh"] = "no GOLD_RATE_API_URL configured — using static file"
        return current
    try:
        import httpx
        resp = httpx.get(GOLD_RATE_API_URL, timeout=timeout_s)
        resp.raise_for_status()
        rate = float(_dig(resp.json(), GOLD_RATE_JSON_PATH))
        if rate <= 0:
            raise ValueError(f"non-positive rate {rate}")
        record = {
            "rate_per_gram_inr": round(rate, 2),
            "updated_at":        date.today().isoformat(),
            "source":            f"ibja_api ({GOLD_RATE_API_URL})",
            "rate_source":       "ibja_api",
        }
        GOLD_RATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLD_RATE_PATH.write_text(json.dumps(record, indent=2))
        logger.info("Gold rate refreshed from live feed: ₹%.2f/g", rate)
        return record
    except Exception as e:
        logger.warning("Live gold-rate refresh failed (%s) — keeping static file", e)
        current = load_gold_rate()
        current["refresh"] = f"live feed unreachable ({e}) — using static fallback"
        return current

TIERS = [
    {"max_amount": 250_000, "ltv_pct": 0.85, "label": "up to ₹2,50,000"},
    {"max_amount": 500_000, "ltv_pct": 0.80, "label": "₹2,50,001 – ₹5,00,000"},
    {"max_amount": float("inf"), "ltv_pct": 0.75, "label": "above ₹5,00,000"},
]


def load_gold_rate() -> dict:
    if GOLD_RATE_PATH.exists():
        try:
            rate = json.loads(GOLD_RATE_PATH.read_text())
            age = _rate_age_days(rate.get("updated_at"))
            rate["age_days"] = age
            rate["stale"] = bool(age is not None and age > GOLD_RATE_MAX_AGE_DAYS)
            return rate
        except Exception as e:
            logger.warning("Could not read gold rate config: %s", e)
    return {"rate_per_gram_inr": DEFAULT_RATE_PER_GRAM_INR, "updated_at": None,
            "source": "default", "age_days": None, "stale": True}


def compute_ltv(assessed_value_inr: float) -> dict:
    """Greedy tier selection: try the highest-LTV tier first; a tier is
    self-consistent when the loan it implies actually falls inside that
    tier's amount band (loan = value * ltv%, and the band is defined on the
    loan amount, not the collateral value)."""
    for tier in TIERS:
        candidate_loan = assessed_value_inr * tier["ltv_pct"]
        if candidate_loan <= tier["max_amount"]:
            return {
                "assessed_value_inr": round(assessed_value_inr, 2),
                "ltv_pct":            tier["ltv_pct"],
                "max_loan_inr":       round(candidate_loan, 2),
                "tier":               tier["label"],
                "source": "RBI Lending Against Gold Collateral Directions, 2025 (tiered LTV, effective 1 Apr 2026)",
            }
    last = TIERS[-1]
    return {
        "assessed_value_inr": round(assessed_value_inr, 2),
        "ltv_pct":             last["ltv_pct"],
        "max_loan_inr":        round(assessed_value_inr * last["ltv_pct"], 2),
        "tier":                last["label"],
        "source": "RBI Lending Against Gold Collateral Directions, 2025 (tiered LTV, effective 1 Apr 2026)",
    }


G_PER_CARAT = 0.2

# Hard ceiling on how much of the gross weight ANY stone-deduction method may
# claim. A stone-set ornament is still mostly gold — no plausible pavé/melee
# layout gets remotely close to this, so it never fires on a real case. It
# exists purely as a last-line guard: if a size→carat estimate is thrown off by
# a calibration error (px_per_mm), the error compounds CUBICALLY into volume/
# weight, and without a ceiling it can consume the entire gross weight and
# silently report net_gold_weight_g = 0 — which is what happened before this
# fix (3 stones, miscalibrated scale → 77.7 ct ≈ 15.5 g claimed against a 10 g
# item). Net gold can never be pushed below this floor fraction of gross.
MAX_STONE_FRACTION = 0.5


def compute_net_gold_weight(gross_weight_g: float, gem_weight_result: dict = None,
                            composition_result: dict = None, gem_grid_result: dict = None) -> dict:
    """Net gold weight = total measured weight − estimated weight of the set
    stones, so LTV is charged on the GOLD only (a stone-set ornament isn't
    lent against at the gold rate for the parts that aren't gold).

    Method priority (each falls through to the next):
      1. trade_deduction — the standard jeweller per-stone deduction from
         gem_grid.build_grid_stats: diameter (from the calibration card) looked
         up against a small, BOUNDED per-size-band gram table (max ~0.09 g per
         stone). This is the industry-standard method and, critically, cannot
         blow up from a calibration error the way a volume estimate can — it's
         used as PRIMARY whenever a card was present.
      2. stone_weight_deduction — the DIRECT per-stone size→carat→gram volume
         estimate (gem_weight.estimate_gem_weights). More precise in principle,
         but volumetric (weight ∝ diameter³), so a scale error compounds
         cubically — only used when the bounded trade-deduction isn't
         available, and always capped at MAX_STONE_FRACTION of gross as a
         backstop against exactly that failure mode.
      3. density_composition — the density mixture model's gold mass (gold+stone
         physics) when there's no card for a size-based estimate at all.
      4. gross_weight — plain-metal item, or nothing to deduct.

    Returns a breakdown dict (always includes net_gold_weight_g) for the audit
    trail and the report. Never raises.
    """
    gross = float(gross_weight_g or 0.0)
    gw = gem_weight_result or {}
    grid = gem_grid_result or {}
    comp = composition_result or {}

    trade_g = grid.get("total_deduction_g")
    if trade_g is not None and gross > 0:
        stone_g = min(gross * MAX_STONE_FRACTION, trade_g)
        result = {
            "net_gold_weight_g": round(max(0.0, gross - stone_g), 3),
            "gross_weight_g":    round(gross, 3),
            "stone_weight_g":    round(stone_g, 3),
            "n_stones":          grid.get("total_stones", 0),
            "method":            "trade_deduction",
            "explanation": ("Net gold = total weight − standard jeweller per-stone deduction "
                            "(diameter from the calibration card → bounded gram table by size band)."),
        }
        # Cross-check against the volumetric estimate, when available, purely
        # for audit visibility — a wide disagreement usually means px_per_mm is
        # off and the officer should double-check the calibration card. Never
        # changes which number is used.
        total_ct = gw.get("total_carat")
        if total_ct is not None:
            volumetric_g = total_ct * G_PER_CARAT
            if trade_g > 0 and volumetric_g > 3 * trade_g:
                result["calibration_flag"] = (
                    f"Volumetric stone-weight estimate ({volumetric_g:.2f} g) disagrees sharply with "
                    f"the trade-deduction estimate ({trade_g:.2f} g) — check the calibration card is "
                    "flat, in-frame, and today's print; px_per_mm may be off."
                )
        return result

    total_ct = gw.get("total_carat")
    if total_ct is not None and gross > 0:
        total_ct_hi = gw.get("total_carat_high") or total_ct
        # Capped at MAX_STONE_FRACTION, not just at gross — a volumetric
        # estimate thrown off by a calibration error must never be allowed to
        # claim the whole item and zero out net gold.
        stone_g = min(gross * MAX_STONE_FRACTION, total_ct * G_PER_CARAT)
        stone_g_hi = min(gross * MAX_STONE_FRACTION, total_ct_hi * G_PER_CARAT)
        result = {
            "net_gold_weight_g": round(max(0.0, gross - stone_g), 3),
            "gross_weight_g":    round(gross, 3),
            "stone_weight_g":    round(stone_g, 3),
            "stone_weight_g_high": round(stone_g_hi, 3),
            "stone_carat_total": round(total_ct, 3),
            "n_stones":          gw.get("n_stones", 0),
            "method":            "stone_weight_deduction",
            "explanation": ("Net gold = total weight − estimated weight of the set stones "
                            "(each stone's photo size × calibration card → carat → grams)."),
        }
        if total_ct * G_PER_CARAT > gross * MAX_STONE_FRACTION:
            result["calibration_flag"] = (
                f"Raw volumetric estimate ({total_ct * G_PER_CARAT:.2f} g) exceeded "
                f"{MAX_STONE_FRACTION*100:.0f}% of the item's gross weight and was capped — "
                "check the calibration card is flat, in-frame, and today's print; px_per_mm may be off."
            )
        return result

    if comp.get("model_valid") and comp.get("gold_mass_g") is not None:
        net = float(comp["gold_mass_g"])
        return {
            "net_gold_weight_g": round(max(0.0, net), 3),
            "gross_weight_g":    round(gross, 3),
            "stone_weight_g":    round(max(0.0, gross - net), 3),
            "method":            "density_composition",
            "explanation": ("Net gold from the density mixture model (gold + stones); "
                            "no calibration card was present for a size-based stone estimate."),
        }

    return {
        "net_gold_weight_g": round(gross, 3),
        "gross_weight_g":    round(gross, 3),
        "stone_weight_g":    0.0,
        "method":            "gross_weight",
        "explanation": ("No stones detected or no way to estimate their weight — "
                        "net gold = total measured weight."),
    }


def assess_value(net_gold_weight_g: float) -> dict:
    rate = load_gold_rate()
    value = net_gold_weight_g * rate["rate_per_gram_inr"]
    ltv = compute_ltv(value)
    return {
        "net_gold_weight_g":  round(net_gold_weight_g, 3),
        "rate_per_gram_inr":  rate["rate_per_gram_inr"],
        "rate_updated_at":    rate.get("updated_at"),
        "rate_source":        rate.get("rate_source", "static"),
        "rate_stale":         rate.get("stale", False),
        **ltv,
    }
