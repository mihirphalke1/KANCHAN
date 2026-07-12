"""
Stone-corrected composition analysis — two-component mixture model.

A stone-set item is a gold/stone mixture. With volume fraction f of stone:

    rho_bulk = (1 - f) * rho_gold + f * rho_stone

Every common gem is far less dense than gold, so stones always LOWER the
bulk density. Three outputs follow:

  1. f_implied = (rho_gold - rho_measured) / (rho_gold - rho_stone)
     — the stone volume fraction the physics implies.
  2. m_gold = rho_gold * (1 - f_implied) * V
     — grams of actual gold (what the loan should be valued on).
  3. Hidden-volume check: f_implied vs the stone fraction the camera
     independently measured (DSIP gem detection). Physics demanding far
     more non-gold volume than the photo shows = hidden core signature.

Direction safety: the correction can only ever EXPLAIN a low reading;
a measured density ABOVE the plain karat band is never softened.
Anti-laundering: f_photo comes from independent CV detection, never from
the item description.
"""
import math

from app.utils.density import KARAT_DENSITY_TABLE, _phi

# Stone densities (g/cm3) — CRC Handbook / standard gemmology references.
STONE_DENSITIES = {
    "red":        4.00,   # ruby (corundum)
    "blue":       4.00,   # sapphire (corundum)
    "green":      2.72,   # emerald (beryl)
    "other":      3.00,   # generic coloured stone / paste
    "colourless": 3.30,   # unconfirmed pearl (2.7) / diamond (3.52) / CZ (5.8)
}
STONE_DENSITY_SIGMA = 0.60   # spread across plausible stone identities

# 2D photo area fraction is a rough proxy for 3D volume fraction, and the
# detection itself is a LOWER bound (occlusion, reverse side, pale stones).
# The proxy sigma is therefore generous.
PHOTO_FRAC_REL_SIGMA = 0.75
PHOTO_FRAC_ABS_SIGMA = 0.04
MAX_STONE_FRAC = 0.60        # clamp: more than this is not jewellery

# Consistency-test decision curve: risk stays low while the measurement is
# within ~2 sigma of the composition prediction (standard acceptance) and
# saturates by ~3-4 sigma (standard rejection). Logistic centre/width below.
Z_CENTER = 2.5
Z_WIDTH  = 0.75


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def analyze_composition(
    weight_dry: float,
    measured_density: float,
    sigma_rho: float,
    volume_cm3: float,
    declared_karat: int,
    gems: list[dict],
    gem_area_pct: float,
) -> dict:
    """
    Full stone-corrected analysis. `gems` and `gem_area_pct` come from the
    DSIP pipeline's independent gem detection (item-relative percentages).
    """
    ref = KARAT_DENSITY_TABLE[declared_karat]
    rho_gold = ref["nominal"]

    # Area-weighted stone density from the detected gems' colour classes.
    # With no detected stones, fall back to a mid-range gem density rather
    # than 0 — a zero here would inflate the uncertainty term and make any
    # low density look falsely consistent.
    total_area = sum(g["area_pct"] for g in gems)
    if total_area > 0:
        rho_stone = sum(
            STONE_DENSITIES.get(g["hue_class"], STONE_DENSITIES["other"]) * g["area_pct"]
            for g in gems
        ) / total_area
    else:
        rho_stone = STONE_DENSITIES["other"]

    f_photo = _clip(gem_area_pct / 100.0, 0.0, MAX_STONE_FRAC)
    sigma_f = PHOTO_FRAC_REL_SIGMA * f_photo + PHOTO_FRAC_ABS_SIGMA

    # Physics-implied stone volume fraction from the measured bulk density.
    # VALIDITY: the mixture model only applies while the measured density is
    # explainable by SOME stone fraction up to MAX_STONE_FRAC. Below that,
    # the item cannot be predominantly gold and a "gold mass" is meaningless.
    f_implied_raw = (rho_gold - measured_density) / (rho_gold - rho_stone)
    model_valid = -0.05 <= f_implied_raw <= MAX_STONE_FRAC
    f_implied = _clip(f_implied_raw, 0.0, MAX_STONE_FRAC)

    if model_valid:
        gold_volume = (1.0 - f_implied) * volume_cm3
        gold_mass   = rho_gold * gold_volume
        stone_mass  = max(0.0, weight_dry - gold_mass)
    else:
        gold_mass = stone_mass = None

    # Consistency test (not band conformity): the declared karat plus the
    # PHOTO-measured stone fraction predict a bulk density. Risk is how far
    # the measurement sits from that prediction, in units of the total
    # uncertainty (measurement + 2D->3D proxy + stone identity + alloy band
    # spread). Genuine stone-set items land near the prediction -> low risk;
    # the wide proxy uncertainty makes the test forgiving where it should be,
    # and the hidden-volume check below catches what this test cannot.
    adj_low  = (1 - f_photo) * ref["low"]  + f_photo * rho_stone
    adj_high = (1 - f_photo) * ref["high"] + f_photo * rho_stone
    rho_pred = (1 - f_photo) * rho_gold + f_photo * rho_stone
    band_sigma = (adj_high - adj_low) / 4.0            # alloy band as ±2σ spread
    sigma_total = math.sqrt(
        sigma_rho ** 2
        + ((rho_gold - rho_stone) * sigma_f) ** 2
        + (f_photo * STONE_DENSITY_SIGMA) ** 2
        + band_sigma ** 2
    )
    z = abs(measured_density - rho_pred) / sigma_total
    adjusted_risk = _phi((z - Z_CENTER) / Z_WIDTH)     # 2σ accept → 3σ+ reject

    # Direction safety: stones only explain LOW readings. Above the plain
    # band the correction must not soften anything.
    if measured_density > ref["high"]:
        plain_above = 1.0 - _phi((ref["high"] - measured_density) / sigma_rho)
        adjusted_risk = max(adjusted_risk, min(1.0, plain_above))

    # Hidden-volume check: physics needs more non-gold volume than the
    # camera can account for.
    excess = f_implied - f_photo
    hidden_severity = _clip(excess / max(2.0 * sigma_f, 1e-6) / 2.0, 0.0, 1.0)
    hidden_volume_flag = excess > 2.0 * sigma_f and f_implied_raw > 0

    rho_at_max = (1 - MAX_STONE_FRAC) * rho_gold + MAX_STONE_FRAC * rho_stone
    if not model_valid:
        note = (
            f"Measured density ({measured_density:.2f} g/cm³) is below what any realistic stone "
            f"content could explain — even {MAX_STONE_FRAC*100:.0f}% stones would still read "
            f"≥ {rho_at_max:.1f} g/cm³. The gold/stone mixture model does not apply: "
            "the item is not predominantly gold."
        )
    elif hidden_volume_flag:
        note = (
            f"Physics implies {f_implied*100:.0f}% non-gold volume but the photo accounts for "
            f"only {f_photo*100:.0f}% as stones — {excess*100:.0f}% of the volume is unexplained. "
            "A concealed core of lighter metal under the surface would produce exactly this pattern."
        )
    elif f_photo > 0:
        note = (
            f"Detected stones ({f_photo*100:.0f}% of visible area, assumed ~{rho_stone:.2f} g/cm³) "
            f"are consistent with the measured bulk density. Estimated gold content: "
            f"{gold_mass:.1f} g of {weight_dry:.1f} g total."
        )
    else:
        note = "No stones detected — plain-metal density analysis applies unchanged."

    return {
        "mode":                 "stone_corrected",
        "model_valid":          model_valid,
        "rho_gold_nominal":     rho_gold,
        "rho_stone_assumed":    round(rho_stone, 2),
        "stone_frac_photo":     round(f_photo, 4),
        "stone_frac_implied":   round(f_implied, 4),
        "stone_frac_sigma":     round(sigma_f, 4),
        "gold_mass_g":          round(gold_mass, 2) if gold_mass is not None else None,
        "stone_mass_g":         round(stone_mass, 2) if stone_mass is not None else None,
        "gold_mass_fraction":   round(gold_mass / weight_dry, 4) if gold_mass is not None and weight_dry > 0 else None,
        "adjusted_band":        {"low": round(adj_low, 2), "high": round(adj_high, 2)},
        "rho_predicted":        round(rho_pred, 2),
        "consistency_z":        round(z, 2),
        "sigma_total":          round(sigma_total, 4),
        "adjusted_density_risk": round(adjusted_risk, 4),
        "hidden_volume_flag":   hidden_volume_flag,
        "hidden_severity":      round(hidden_severity, 4),
        "gems":                 gems,
        "note":                 note,
    }
