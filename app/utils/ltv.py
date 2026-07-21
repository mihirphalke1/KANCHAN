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
from pathlib import Path

logger = logging.getLogger(__name__)

GOLD_RATE_PATH = Path("data/gold_rate.json")
DEFAULT_RATE_PER_GRAM_INR = 6800.0   # admin-configurable — no live market-rate API wired

TIERS = [
    {"max_amount": 250_000, "ltv_pct": 0.85, "label": "up to ₹2,50,000"},
    {"max_amount": 500_000, "ltv_pct": 0.80, "label": "₹2,50,001 – ₹5,00,000"},
    {"max_amount": float("inf"), "ltv_pct": 0.75, "label": "above ₹5,00,000"},
]


def load_gold_rate() -> dict:
    if GOLD_RATE_PATH.exists():
        try:
            return json.loads(GOLD_RATE_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read gold rate config: %s", e)
    return {"rate_per_gram_inr": DEFAULT_RATE_PER_GRAM_INR, "updated_at": None, "source": "default"}


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


def compute_net_gold_weight(gross_weight_g: float, gem_weight_result: dict = None,
                            composition_result: dict = None) -> dict:
    """Net gold weight = total measured weight − estimated weight of the set
    stones, so LTV is charged on the GOLD only (a stone-set ornament isn't
    lent against at the gold rate for the parts that aren't gold).

    Method priority (each falls through to the next):
      1. stone_weight_deduction — subtract the DIRECT per-stone weight estimate
         (each detected stone's photo size → carat → grams, from
         gem_weight.estimate_gem_weights). Needs a calibration card for scale.
         This is the method the officer sees itemised.
      2. density_composition — the density mixture model's gold mass (gold+stone
         physics) when there's no card for a size-based estimate.
      3. gross_weight — plain-metal item, or nothing to deduct.

    Returns a breakdown dict (always includes net_gold_weight_g) for the audit
    trail and the report. Never raises.
    """
    gross = float(gross_weight_g or 0.0)
    gw = gem_weight_result or {}
    comp = composition_result or {}

    total_ct = gw.get("total_carat")
    if total_ct is not None and gross > 0:
        total_ct_hi = gw.get("total_carat_high") or total_ct
        stone_g = min(gross, total_ct * G_PER_CARAT)     # can't exceed the item
        stone_g_hi = min(gross, total_ct_hi * G_PER_CARAT)
        return {
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
        **ltv,
    }
