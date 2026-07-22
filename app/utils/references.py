"""
Reference data sources for every physical constant in KANCHAN-AI.

The karat density bands are not copied from a lookup table — they are DERIVED
from the BIS hallmarking fineness grades plus CRC element densities using the
inverse mixture rule, which is the defensible answer to "what is the source":

    1/rho_alloy = sum_i ( w_i / rho_i )      (mass-fraction mixture rule)

A 22K (BIS 916) alloy is 91.6% Au by mass; the remaining 8.4% ranges from
copper-rich (dense red golds) to silver-rich (paler alloys), giving a density
BAND, not a point. Zinc-bearing solders extend the lower bound slightly.
"""

# ── Element densities, g/cm3 at 20 °C ──
# Source: CRC Handbook of Chemistry and Physics, 97th ed., Section 4,
# "Physical Constants of Inorganic Compounds / Properties of the Elements".
ELEMENT_DENSITIES = {
    "gold":     19.32,
    "silver":   10.49,
    "copper":   8.96,
    "zinc":     7.14,
    "tungsten": 19.25,
    "lead":     11.34,
}

# ── BIS hallmarking fineness grades (parts per 1000 by mass) ──
# Source: IS 1417:2016, Bureau of Indian Standards — "Gold and Gold Alloys,
# Jewellery/Artefacts — Fineness and Marking".
BIS_FINENESS = {24: 0.999, 23: 0.958, 22: 0.916, 18: 0.750, 14: 0.585}

# ── Gem specific gravities ──
# Source: R. Webster, "Gems: Their Sources, Descriptions and Identification",
# 5th ed. (Butterworth-Heinemann); ranges match GIA Gem Reference Guide.
GEM_SG = {
    "corundum (ruby/sapphire)": (3.95, 4.05),
    "beryl (emerald)":          (2.67, 2.78),
    "diamond":                  (3.50, 3.53),
    "pearl":                    (2.60, 2.85),
    "cubic zirconia":           (5.60, 6.00),
}

# ── Young's moduli (GPa) — stiffness, the property density cannot fake ──
# Source: CRC Handbook / ASM Metals Handbook. Sound speed v = sqrt(E/rho):
#   gold ~2020 m/s, copper ~3810, tungsten ~4620. For comparable geometry the
#   ring frequency scales with v, so a stiff core raises the pitch — the
#   physical basis of the tap test (validated on DS-1: genuine median 6153 Hz
#   vs plated-composite 7689 Hz, non-overlapping).
YOUNGS_MODULUS_GPA = {
    "gold":     79.0,
    "copper":   130.0,
    "tungsten": 411.0,
}

# ── Water density vs temperature ──
# Source: CRC Handbook, "Standard Density of Water"; consistent with
# G.S. Kell, J. Chem. Eng. Data 20:97 (1975). Table lives in
# app/utils/density.py (WATER_DENSITY_TABLE).

CITATIONS = {
    "elements":  "CRC Handbook of Chemistry and Physics, 97th ed., Sec. 4",
    "fineness":  "IS 1417:2016, Bureau of Indian Standards",
    "gems":      "Webster, 'Gems', 5th ed.; GIA Gem Reference Guide",
    "water":     "CRC Handbook, 'Standard Density of Water'; Kell (1975)",
    "scoring":   "JCGM 106:2012 — conformity assessment under measurement uncertainty",
    "mixture":   "Inverse mixture rule (mass-weighted specific volumes)",
}


def alloy_density(fineness: float, alloying: dict[str, float]) -> float:
    """Density of a gold alloy via the inverse mixture rule.
    `alloying` maps element name -> mass fraction of the NON-gold remainder."""
    inv = fineness / ELEMENT_DENSITIES["gold"]
    rest = 1.0 - fineness
    for element, frac in alloying.items():
        inv += rest * frac / ELEMENT_DENSITIES[element]
    return 1.0 / inv


def derive_karat_band(karat: int) -> dict:
    """
    Derive the density band for a karat grade from first principles:
    copper-rich alloy = lower bound, silver-rich = upper bound, with a
    zinc-solder allowance widening the lower edge (~1.5%).
    """
    f = BIS_FINENESS[karat]
    cu_rich = alloy_density(f, {"copper": 1.0})
    ag_rich = alloy_density(f, {"silver": 1.0})
    lo, hi = min(cu_rich, ag_rich), max(cu_rich, ag_rich)
    return {
        "karat":       karat,
        "fineness":    f,
        "derived_low":  round(lo * 0.985, 2),   # zinc-bearing solder allowance
        "derived_high": round(hi * 1.003, 2),   # measurement rounding headroom
        "cu_rich":     round(cu_rich, 2),
        "ag_rich":     round(ag_rich, 2),
        "source":      f"{CITATIONS['fineness']} + {CITATIONS['elements']} via {CITATIONS['mixture']}",
    }


if __name__ == "__main__":
    from app.utils.density import KARAT_DENSITY_TABLE
    print(f"{'K':>3} {'derived band':>15} {'table band':>15}")
    for k in (24, 22, 18, 14):
        d = derive_karat_band(k)
        t = KARAT_DENSITY_TABLE[k]
        print(f"{k:>3} {d['derived_low']:>7}-{d['derived_high']:<7} {t['low']:>7}-{t['high']:<7}")
