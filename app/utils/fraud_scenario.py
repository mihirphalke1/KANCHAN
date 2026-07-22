"""fraud_scenario mapping (P3-13).

Translates KANCHAN's internal verdict + contradiction pattern into the bank's
own 8-scenario spurious-gold vocabulary (Sl.1-Sl.8, see data/fraud_scenarios.json),
so the API response and the PDF certificate speak the language the bank's fraud
desk already uses instead of the engine's internal labels.

Pure and deterministic: `map_scenarios(signals)` takes a flat dict of already-
computed case signals and returns, for each catalogued scenario, whether it is
matched and the concrete evidence that matched it. No detection happens here —
this is only a translation layer over decisions made upstream.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/fraud_scenarios.json")

_DEFAULT_CATALOG = {
    "scenarios": [
        {"sl": "Sl.1", "code": "GOLD_PLATED_BASE_METAL", "title": "Gold-plated base metal"},
        {"sl": "Sl.2", "code": "UNDER_KARAT", "title": "Under-karat / purity mis-declaration"},
        {"sl": "Sl.3", "code": "TUNGSTEN_CORE", "title": "Density-matched core fill"},
        {"sl": "Sl.4", "code": "HOLLOW_OR_FILLED", "title": "Hollow or filler-weighted item"},
        {"sl": "Sl.5", "code": "STONE_WEIGHT_INFLATION", "title": "Stone weight passed as gold"},
        {"sl": "Sl.6", "code": "SPURIOUS_HALLMARK", "title": "Spurious hallmark / HUID"},
        {"sl": "Sl.7", "code": "REPLEDGE_OR_CAP_BREACH", "title": "Re-pledging / cap breach"},
        {"sl": "Sl.8", "code": "NOT_GOLD", "title": "Not gold"},
    ]
}


def load_catalog() -> dict:
    if CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text())
        except Exception as e:
            logger.warning("Could not read fraud scenario catalog: %s", e)
    return _DEFAULT_CATALOG


def _flags_text(flags) -> str:
    return " ".join(str(f).lower() for f in (flags or []))


def _match_rules(signals: dict) -> dict:
    """Return {code: evidence_str} for every scenario the case's signals match.
    A single case can match several scenarios (e.g. hollow + stone inflation)."""
    density = signals.get("density") or {}
    karat_verdict = str(density.get("karat_verdict") or "").upper()
    closest_fake = density.get("closest_fake")
    misdeclared = bool(density.get("misdeclared_purity"))
    flags = _flags_text(signals.get("contradiction_flags"))
    stiff_core = bool(signals.get("stiff_core_flag"))
    hollow_anomaly = bool(signals.get("hollow_anomaly"))
    hollow_declared = bool(signals.get("hollow_item"))
    pledging_exceeds = bool(signals.get("pledging_exceeds"))
    hallmark = signals.get("hallmark") or {}
    n_stones = int(signals.get("n_stones") or 0)
    stone_fraction = float(signals.get("stone_weight_fraction") or 0.0)

    matched: dict[str, str] = {}

    # Sl.3 — density-matched dense core (tungsten): the one signature weight-in-
    # water cannot see, caught by the acoustic stiff-core contradiction.
    if stiff_core or "stiff" in flags or karat_verdict == "TUNGSTEN_BLIND_SPOT" or (closest_fake and "tungsten" in str(closest_fake).lower()):
        matched["TUNGSTEN_CORE"] = (
            "Acoustic ring pitch indicates a stiffer-than-gold core while density passes — "
            "dense-core-fill (tungsten) signature."
        )

    # Sl.1 — gold-plated base metal: too light for the declared karat, or a
    # base-metal best match, or plating flagged on the surface/streak.
    if karat_verdict == "LOW_DENSITY" or "plat" in flags or (closest_fake and any(
            m in str(closest_fake).lower() for m in ("brass", "copper", "silver", "steel", "base"))):
        if not misdeclared:
            matched["GOLD_PLATED_BASE_METAL"] = (
                f"Density below the declared-karat band"
                + (f"; closest base metal: {closest_fake}." if closest_fake else ".")
            )

    # Sl.2 — under-karat / purity mis-declaration: density low but consistent
    # with a genuine lower-fineness gold alloy (not a base metal).
    if misdeclared or karat_verdict == "MISDECLARED_PURITY":
        matched["UNDER_KARAT"] = (
            "Density matches a genuine but LOWER-fineness gold alloy than declared — "
            "likely purity mis-declaration."
        )

    # Sl.8 — not gold at all.
    if karat_verdict == "NOT_GOLD":
        matched["NOT_GOLD"] = "Physical measurement matches no gold alloy."

    # Sl.4 — hollow / filler-weighted: hollow anomaly (solid-gold density on a
    # piece that should be hollow) or the hollow-item contradiction.
    if hollow_anomaly or ("hollow" in flags and karat_verdict == "HIGH_DENSITY") or (
            hollow_declared and karat_verdict == "HIGH_DENSITY"):
        matched["HOLLOW_OR_FILLED"] = (
            "Item is hollow/openwork yet reads as dense/solid — filler-weighting "
            "(dense-core-fill) signature."
        )

    # Sl.5 — stone weight inflation: a large share of gross weight is set stones,
    # so charging LTV on gross would over-value the gold.
    if n_stones >= 1 and stone_fraction >= 0.15:
        matched["STONE_WEIGHT_INFLATION"] = (
            f"{n_stones} set stone(s) ≈ {round(stone_fraction * 100)}% of gross weight — "
            "gold weight must be netted of stones before valuation."
        )

    # Sl.6 — spurious hallmark / HUID.
    if hallmark:
        hm_status = str(hallmark.get("verification") or hallmark.get("status") or "").lower()
        if hallmark.get("mismatch") or "not found" in hm_status or "invalid" in hm_status or "fail" in hm_status:
            matched["SPURIOUS_HALLMARK"] = "BIS hallmark / HUID did not verify against the registry."

    # Sl.7 — re-pledging / aggregate cap breach.
    if pledging_exceeds or "pledging-cap" in flags:
        matched["REPLEDGE_OR_CAP_BREACH"] = (
            "Borrower's aggregate pledged weight breaches the RBI Para 16 cap across active loans."
        )

    return matched


def map_scenarios(signals: dict) -> dict:
    """Map case signals onto the bank's scenario catalog.

    Returns:
      matched   list of {sl, code, title, evidence} for scenarios that fired
      all       full catalog with a `matched` boolean on each (for a checklist view)
      summary   short human string ("Sl.3 (Density-matched core fill); Sl.5 (…)" or "none")
    """
    catalog = load_catalog().get("scenarios", [])
    hits = _match_rules(signals or {})
    matched, full = [], []
    for entry in catalog:
        code = entry.get("code")
        is_match = code in hits
        row = {"sl": entry.get("sl"), "code": code, "title": entry.get("title"),
               "matched": is_match, "evidence": hits.get(code)}
        full.append(row)
        if is_match:
            matched.append({k: row[k] for k in ("sl", "code", "title", "evidence")})
    summary = "; ".join(f"{m['sl']} ({m['title']})" for m in matched) or "No spurious-gold scenario matched."
    return {"matched": matched, "all": full, "summary": summary}
