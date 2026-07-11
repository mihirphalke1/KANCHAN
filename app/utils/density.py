"""
Archimedes-principle density measurement for gold items, with honest metrology.

Physics:
    V_item  = (W_dry - W_sub) / rho_water(T)      displaced volume
    rho     = W_dry / V_item
            = W_dry * rho_water(T) / (W_dry - W_sub)

Uncertainty (propagation of the balance's repeatability sigma_w through the
formula; the dominant term is the *difference* of the two weighings):
    sigma_rho / rho ≈ sqrt(2) * sigma_w / (W_dry - W_sub)

Risk score follows JCGM 106 (conformity assessment with measurement
uncertainty): risk = P(true density outside the declared karat band | data),
computed with the Gaussian CDF. No arbitrary scale factors.
"""
import math
import os

# CRC Handbook of Chemistry and Physics, water density (g/cm3) vs temperature.
WATER_DENSITY_TABLE = [
    (10.0, 0.9997026),
    (15.0, 0.9991026),
    (20.0, 0.9982071),
    (25.0, 0.9970479),
    (30.0, 0.9956502),
    (35.0, 0.9940349),
    (40.0, 0.9922152),
]

# 1-sigma repeatability of the branch balance in grams. Instrument property —
# override per deployment with the SCALE_SIGMA_G environment variable.
# Default 0.005 g is typical for a 0.01 g-readability jeweller's balance.
SCALE_SIGMA_G = float(os.getenv("SCALE_SIGMA_G", "0.005"))

KARAT_DENSITY_TABLE = {
    24: {"nominal": 19.32, "low": 19.10, "high": 19.40},
    22: {"nominal": 17.80, "low": 17.40, "high": 18.10},
    18: {"nominal": 15.55, "low": 15.20, "high": 15.90},
    14: {"nominal": 13.07, "low": 12.80, "high": 13.40},
}

# Densities per CRC Handbook (brass varies by alloy: 8.4-8.7, CDA data)
KNOWN_FAKES = {
    "copper":   8.96,
    "brass":    8.55,
    "lead":     11.34,
    "zinc":     7.14,
    "tungsten": 19.25,
    "silver":   10.49,
}


def water_density(temp_c: float = 25.0) -> float:
    """Density of water (g/cm3) at temp_c, linear interpolation on CRC data."""
    pts = WATER_DENSITY_TABLE
    if temp_c <= pts[0][0]:
        return pts[0][1]
    if temp_c >= pts[-1][0]:
        return pts[-1][1]
    for (t0, r0), (t1, r1) in zip(pts, pts[1:]):
        if t0 <= temp_c <= t1:
            return r0 + (r1 - r0) * (temp_c - t0) / (t1 - t0)
    return pts[-1][1]


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def density_measurement(
    weight_dry: float,
    weight_submerged: float,
    water_temp_c: float = 25.0,
    scale_sigma_g: float = SCALE_SIGMA_G,
) -> dict:
    """
    Full Archimedes measurement: buoyancy-corrected density, displaced volume,
    and 1-sigma uncertainty from balance repeatability.
    """
    displaced = weight_dry - weight_submerged
    if displaced <= 0:
        raise ValueError(f"Displaced volume ≤ 0: dry={weight_dry}g, sub={weight_submerged}g")

    rho_w   = water_density(water_temp_c)
    volume  = displaced / rho_w                      # cm3
    density = weight_dry / volume                    # g/cm3

    # d(ln rho)/dW_dry = 1/W_dry - 1/displaced ; d(ln rho)/dW_sub = 1/displaced
    rel_var = (scale_sigma_g ** 2) * (
        (1.0 / weight_dry - 1.0 / displaced) ** 2 + (1.0 / displaced) ** 2
    )
    sigma = density * math.sqrt(rel_var)

    return {
        "density":       round(density, 4),
        "sigma":         round(sigma, 4),
        "volume_cm3":    round(volume, 4),
        "water_temp_c":  water_temp_c,
        "rho_water":     round(rho_w, 5),
        "scale_sigma_g": scale_sigma_g,
    }


def compute_density(
    weight_dry: float, weight_submerged: float, water_temp_c: float = 25.0
) -> float:
    """Buoyancy-corrected density (g/cm3). Kept for backward compatibility."""
    return density_measurement(weight_dry, weight_submerged, water_temp_c)["density"]


def density_risk_score(
    measured_density: float, declared_karat: int, sigma: float | None = None
) -> dict:
    """
    Risk = P(true density outside the declared karat band | measurement),
    per JCGM 106. `sigma` is the 1-sigma measurement uncertainty; when absent
    a nominal SCALE-derived floor of 0.05 g/cm3 is assumed.

    Returns risk_score plus band context, conformity probability, and a
    measurement-adequacy flag (False when the instrument cannot resolve the
    declared band — e.g. any light item against the narrow 24K band).
    """
    ref = KARAT_DENSITY_TABLE.get(declared_karat)
    if ref is None:
        valid = list(KARAT_DENSITY_TABLE.keys())
        raise ValueError(f"Unsupported karat {declared_karat}. Valid: {valid}")

    nominal, low, high = ref["nominal"], ref["low"], ref["high"]
    sigma = sigma if sigma and sigma > 0 else 0.05

    p_below = _phi((low - measured_density) / sigma)
    p_above = 1.0 - _phi((high - measured_density) / sigma)
    risk_score = min(1.0, p_below + p_above)
    conformity = 1.0 - risk_score

    deviation_pct = (measured_density - nominal) / nominal * 100

    if low <= measured_density <= high:
        karat_verdict = "IN_RANGE"
    elif measured_density < low:
        karat_verdict = "LOW_DENSITY"
    else:
        karat_verdict = "HIGH_DENSITY"

    # Instrument adequacy: can this measurement resolve the band at all?
    half_band = (high - low) / 2.0
    measurement_adequate = sigma <= half_band

    closest_fake = min(KNOWN_FAKES, key=lambda k: abs(KNOWN_FAKES[k] - measured_density))
    if abs(KNOWN_FAKES[closest_fake] - measured_density) > 1.0:
        closest_fake = None

    # Tungsten blind spot: the gold-tungsten gap (0.07 g/cm3) is smaller than
    # any branch-grade sigma, so a 24K declaration near tungsten's density is
    # unresolvable by this instrument — flag it regardless of risk score.
    tungsten_warning = (
        declared_karat == 24
        and abs(measured_density - KNOWN_FAKES["tungsten"]) < max(2 * sigma, 0.2)
    )
    if tungsten_warning:
        karat_verdict = "TUNGSTEN_BLIND_SPOT"

    return {
        "risk_score":            round(risk_score, 4),
        "measured_density":      round(measured_density, 4),
        "sigma":                 round(sigma, 4),
        "conformity_probability": round(conformity, 4),
        "measurement_adequate":  measurement_adequate,
        "expected_nominal":      nominal,
        "expected_low":          low,
        "expected_high":         high,
        "deviation_pct":         round(deviation_pct, 2),
        "karat_verdict":         karat_verdict,
        "closest_fake":          closest_fake,
        "tungsten_warning":      tungsten_warning,
    }
