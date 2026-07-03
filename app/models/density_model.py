"""
Density modality wrapper — always uses physics computation (no ML model needed).
"""
from app.utils.density import compute_density, density_risk_score


def analyze_density(weight_dry: float, weight_submerged: float, declared_karat: int) -> dict:
    """
    Full density analysis pipeline.
    Returns modality_score dict compatible with fusion model input.
    """
    measured = compute_density(weight_dry, weight_submerged)
    result   = density_risk_score(measured, declared_karat)

    return {
        "risk_score":        result["risk_score"],
        "confidence":        "high",
        "mode":              "computed",
        "weight_dry":        round(weight_dry, 3),
        "weight_submerged":  round(weight_submerged, 3),
        "measured_density":  result["measured_density"],
        "expected_low":      result["expected_low"],
        "expected_high":     result["expected_high"],
        "expected_nominal":  result["expected_nominal"],
        "deviation_pct":     result["deviation_pct"],
        "karat_verdict":     result["karat_verdict"],
        "closest_fake":      result["closest_fake"],
        "tungsten_warning":  result["tungsten_warning"],
    }
