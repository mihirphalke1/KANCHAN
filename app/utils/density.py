"""
Archimedes-principle density calculation for gold items.
Density = dry_weight / (dry_weight - submerged_weight)
"""

KARAT_DENSITY_TABLE = {
    24: {"nominal": 19.32, "low": 19.10, "high": 19.40},
    22: {"nominal": 17.80, "low": 17.40, "high": 18.10},
    18: {"nominal": 15.55, "low": 15.20, "high": 15.90},
    14: {"nominal": 13.07, "low": 12.80, "high": 13.40},
}

KNOWN_FAKES = {
    "copper":   8.96,
    "brass":    8.55,
    "lead":     11.34,
    "zinc":     6.90,
    "tungsten": 19.25,
    "silver":   10.49,
}


def compute_density(weight_dry: float, weight_submerged: float) -> float:
    """Return density (g/cm³) via Archimedes principle."""
    displaced = weight_dry - weight_submerged
    if displaced <= 0:
        raise ValueError(f"Displaced volume ≤ 0: dry={weight_dry}g, sub={weight_submerged}g")
    return round(weight_dry / displaced, 4)


def density_risk_score(measured_density: float, declared_karat: int) -> dict:
    """
    Compute a 0-1 risk score from how far measured density deviates from
    the expected range for the declared karat.

    Returns a dict with:
      risk_score         float  0=genuine, 1=highly suspicious
      measured_density   float
      expected_nominal   float
      expected_low       float
      expected_high      float
      deviation_pct      float  signed % deviation from nominal
      karat_verdict      str    "IN_RANGE" | "LOW_DENSITY" | "HIGH_DENSITY"
      closest_fake       str    name of closest known fake metal, or None
      tungsten_warning   bool
    """
    ref = KARAT_DENSITY_TABLE.get(declared_karat)
    if ref is None:
        valid = list(KARAT_DENSITY_TABLE.keys())
        raise ValueError(f"Unsupported karat {declared_karat}. Valid: {valid}")

    nominal = ref["nominal"]
    low     = ref["low"]
    high    = ref["high"]
    spread  = high - low

    deviation_pct = (measured_density - nominal) / nominal * 100

    if low <= measured_density <= high:
        risk_score   = 0.0
        karat_verdict = "IN_RANGE"
    else:
        if measured_density < low:
            delta = (low - measured_density) / spread
            karat_verdict = "LOW_DENSITY"
        else:
            delta = (measured_density - high) / spread
            karat_verdict = "HIGH_DENSITY"
        risk_score = min(1.0, delta * 0.6)

    closest_fake = min(KNOWN_FAKES, key=lambda k: abs(KNOWN_FAKES[k] - measured_density))
    fake_distance = abs(KNOWN_FAKES[closest_fake] - measured_density)
    if fake_distance > 1.0:
        closest_fake = None

    tungsten_warning = (
        declared_karat == 24
        and abs(measured_density - KNOWN_FAKES["tungsten"]) < 0.2
    )
    if tungsten_warning:
        karat_verdict = "TUNGSTEN_BLIND_SPOT"

    return {
        "risk_score":        round(risk_score, 4),
        "measured_density":  round(measured_density, 4),
        "expected_nominal":  nominal,
        "expected_low":      low,
        "expected_high":     high,
        "deviation_pct":     round(deviation_pct, 2),
        "karat_verdict":     karat_verdict,
        "closest_fake":      closest_fake,
        "tungsten_warning":  tungsten_warning,
    }
