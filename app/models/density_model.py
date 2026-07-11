"""
Density modality wrapper — always uses physics computation (no ML model needed).
"""
from app.utils.density import density_measurement, density_risk_score


def analyze_density(
    weight_dry: float,
    weight_submerged: float,
    declared_karat: int,
    water_temp_c: float = 25.0,
) -> dict:
    """
    Full density analysis pipeline: buoyancy-corrected Archimedes measurement
    with propagated uncertainty, scored as P(outside declared karat band).
    Returns modality_score dict compatible with fusion model input.
    """
    meas   = density_measurement(weight_dry, weight_submerged, water_temp_c)
    result = density_risk_score(meas["density"], declared_karat, sigma=meas["sigma"])

    return {
        "risk_score":             result["risk_score"],
        "confidence":             "high" if result["measurement_adequate"] else "medium",
        "mode":                   "computed",
        "weight_dry":             round(weight_dry, 3),
        "weight_submerged":       round(weight_submerged, 3),
        "measured_density":       result["measured_density"],
        "sigma":                  result["sigma"],
        "conformity_probability": result["conformity_probability"],
        "measurement_adequate":   result["measurement_adequate"],
        "volume_cm3":             meas["volume_cm3"],
        "water_temp_c":           meas["water_temp_c"],
        "rho_water":              meas["rho_water"],
        "expected_low":           result["expected_low"],
        "expected_high":          result["expected_high"],
        "expected_nominal":       result["expected_nominal"],
        "deviation_pct":          result["deviation_pct"],
        "karat_verdict":          result["karat_verdict"],
        "closest_fake":           result["closest_fake"],
        "tungsten_warning":       result["tungsten_warning"],
    }
