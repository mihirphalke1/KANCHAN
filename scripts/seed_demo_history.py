#!/usr/bin/env python3
"""
Seed data/case_history.json with 8 realistic demo cases.
Run from repo root:  python3 scripts/seed_demo_history.py
"""
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

HISTORY_PATH = Path("data/case_history.json")

def ts(offset_hours: float) -> str:
    dt = datetime.utcnow() - timedelta(hours=offset_hours)
    return dt.isoformat() + "Z"

CASES = [
    # ── 1. Genuine 22K necklace ──────────────────────────────────────────────
    {
        "item_description": "22K gold necklace with antique finish",
        "declared_karat": 22,
        "branch_id": "BLR-001",
        "customer": {
            "name": "Meera Raghunathan",
            "account_no": "CNR0012345678",
            "loan_app_no": "GL/2025/BLR001/00142",
            "officer_name": "Suresh Kumar (EMP-4421)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.07,
                "measured_density": 17.38,
                "expected_low": 17.2,
                "expected_high": 17.9,
                "weight_dry": 10.0,
                "weight_submerged": 9.42,
                "mode": "computed",
            },
            "visual": {
                "risk_score": 0.08,
                "confidence": "high",
                "mode": "no_images",
            },
            "acoustic": {
                "risk_score": 0.11,
                "confidence": "medium",
                "mode": "no_audio",
            },
            "streak": {
                "risk_score": 0.09,
                "confidence": "medium",
                "mode": "heuristic",
            },
        },
        "contradiction": {
            "contradiction_score": 0.05,
            "flags": [],
        },
        "fusion": {
            "risk_score": 0.09,
            "mode": "heuristic",
            "shap_values": None,
        },
        "benford": {
            "status": "insufficient_data",
            "n_samples": 1,
            "p_value": None,
            "alert": False,
        },
        "verdict": {
            "risk_level": "GENUINE",
            "confidence": "HIGH",
            "loan_action": "APPROVE",
            "fusion_risk": 0.09,
            "plain_english": (
                "All four analysis signals are consistent with genuine 22-karat gold. "
                "The measured density of 17.38 g/cm³ falls squarely within the expected "
                "range of 17.2–17.9 g/cm³ for 22K gold. No contradictions detected between "
                "modalities. The item can be accepted with high confidence."
            ),
            "action": (
                "Proceed with loan approval. Item may be accepted as collateral at "
                "the declared purity of 22K. Retain a copy of this report in the loan file."
            ),
            "llm_provider": "heuristic",
        },
        "_hours_ago": 0.5,
    },

    # ── 2. Genuine 18K ring ──────────────────────────────────────────────────
    {
        "item_description": "18K gold ring, diamond-set solitaire",
        "declared_karat": 18,
        "branch_id": "MUM-003",
        "customer": {
            "name": "Anjali Sharma",
            "account_no": "CNR0087654321",
            "loan_app_no": "GL/2025/MUM003/00389",
            "officer_name": "Priya Nair (EMP-2203)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.10,
                "measured_density": 15.61,
                "expected_low": 15.2,
                "expected_high": 15.9,
                "weight_dry": 15.0,
                "weight_submerged": 14.03,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.12, "confidence": "medium", "mode": "no_images"},
            "acoustic": {"risk_score": 0.14, "confidence": "medium", "mode": "no_audio"},
            "streak": {"risk_score": 0.10, "confidence": "medium", "mode": "heuristic"},
        },
        "contradiction": {"contradiction_score": 0.06, "flags": []},
        "fusion": {"risk_score": 0.12, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 2, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "GENUINE",
            "confidence": "MEDIUM",
            "loan_action": "APPROVE",
            "fusion_risk": 0.12,
            "plain_english": (
                "The measured density of 15.61 g/cm³ is consistent with 18-karat gold (expected 15.2–15.9). "
                "All modality signals are low-risk and internally consistent. The item is assessed as genuine "
                "18K gold with medium confidence."
            ),
            "action": (
                "Loan may be approved. Item accepted as collateral at declared purity of 18K. "
                "Standard physical verification by the branch appraiser is still recommended."
            ),
            "llm_provider": "heuristic",
        },
        "_hours_ago": 2.1,
    },

    # ── 3. Genuine 22K bangles (visual + acoustic also ran) ──────────────────
    {
        "item_description": "22K gold bangles (set of 4), plain polished",
        "declared_karat": 22,
        "branch_id": "HYD-002",
        "customer": {
            "name": "Kavitha Reddy",
            "account_no": "CNR0056123890",
            "loan_app_no": "GL/2025/HYD002/00211",
            "officer_name": "Rajesh Naidu (EMP-3301)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.08,
                "measured_density": 17.52,
                "expected_low": 17.2,
                "expected_high": 17.9,
                "weight_dry": 28.5,
                "weight_submerged": 26.82,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.06, "confidence": "high", "mode": "efficientnet"},
            "acoustic": {"risk_score": 0.09, "confidence": "high", "mode": "svm"},
            "streak": {"risk_score": 0.07, "confidence": "high", "mode": "heuristic"},
        },
        "contradiction": {"contradiction_score": 0.04, "flags": []},
        "fusion": {"risk_score": 0.08, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 3, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "GENUINE",
            "confidence": "HIGH",
            "loan_action": "APPROVE",
            "fusion_risk": 0.08,
            "plain_english": (
                "All four modalities — density, visual, acoustic, and streak — independently confirm genuine "
                "22-karat gold. The density reading of 17.52 g/cm³ is well within the expected range. "
                "The acoustic ring pattern matches hallmarked gold and no visual anomalies were detected. "
                "This is a textbook genuine item."
            ),
            "action": (
                "Approve the gold loan at full assessed value. All signals pass. "
                "Document this case report in the borrower's loan file."
            ),
            "llm_provider": "groq",
        },
        "_hours_ago": 5.5,
    },

    # ── 4. Borderline — slightly under-karat ─────────────────────────────────
    {
        "item_description": "22K gold chain — borderline density reading",
        "declared_karat": 22,
        "branch_id": "CHN-007",
        "customer": {
            "name": "Venkatesh Subramanian",
            "account_no": "CNR0071234560",
            "loan_app_no": "GL/2025/CHN007/00078",
            "officer_name": "Deepa Krishnamurthy (EMP-2987)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.52,
                "measured_density": 16.41,
                "expected_low": 17.2,
                "expected_high": 17.9,
                "weight_dry": 12.0,
                "weight_submerged": 11.18,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.31, "confidence": "medium", "mode": "efficientnet"},
            "acoustic": {"risk_score": 0.44, "confidence": "medium", "mode": "svm"},
            "streak": {"risk_score": 0.28, "confidence": "medium", "mode": "heuristic"},
        },
        "contradiction": {
            "contradiction_score": 0.18,
            "flags": ["Density risk is moderate while visual appears normal — item may be under-karat"],
        },
        "fusion": {"risk_score": 0.40, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 4, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "BORDERLINE",
            "confidence": "MEDIUM",
            "loan_action": "HOLD",
            "fusion_risk": 0.40,
            "plain_english": (
                "The measured density of 16.41 g/cm³ is below the expected range for 22K gold (17.2–17.9). "
                "This may indicate that the item is actually 18K gold declared as 22K, or that it contains "
                "a small proportion of base metal alloy. Acoustic and visual signals are mildly elevated. "
                "The item is borderline and requires physical expert appraisal before a loan decision."
            ),
            "action": (
                "Do not approve the loan at this stage. Refer the item to the branch gold appraiser "
                "for hallmark verification and acid test. Return a decision within 2 working days."
            ),
            "llm_provider": "groq",
        },
        "_hours_ago": 9.3,
    },

    # ── 5. Borderline — mixed signals ────────────────────────────────────────
    {
        "item_description": "24K gold coin — acoustic anomaly",
        "declared_karat": 24,
        "branch_id": "DEL-005",
        "customer": {
            "name": "Sunita Malhotra",
            "account_no": "CNR0094561230",
            "loan_app_no": "GL/2025/DEL005/00503",
            "officer_name": "Amit Verma (EMP-1855)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.13,
                "measured_density": 19.08,
                "expected_low": 19.1,
                "expected_high": 19.5,
                "weight_dry": 30.0,
                "weight_submerged": 28.47,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.16, "confidence": "medium", "mode": "no_images"},
            "acoustic": {"risk_score": 0.67, "confidence": "high", "mode": "svm"},
            "streak": {"risk_score": 0.14, "confidence": "medium", "mode": "heuristic"},
        },
        "contradiction": {
            "contradiction_score": 0.42,
            "flags": [
                "Acoustic risk (67%) is significantly higher than density risk (13%) — possible internal void",
                "Acoustic and visual signals are inconsistent",
            ],
        },
        "fusion": {"risk_score": 0.49, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 5, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "BORDERLINE",
            "confidence": "LOW",
            "loan_action": "HOLD",
            "fusion_risk": 0.49,
            "plain_english": (
                "While the density reading is consistent with 24K gold, the acoustic fingerprint is "
                "significantly anomalous — the ring pattern does not match hallmarked solid gold coins. "
                "This contradiction between density and acoustic signals may indicate internal voids or "
                "a layered construction. The item cannot be cleared without further investigation."
            ),
            "action": (
                "Hold loan decision. Recommend X-ray fluorescence (XRF) or sectional testing of the coin "
                "to rule out a filled or plated item. Do not accept as collateral until tests are complete."
            ),
            "llm_provider": "gemini",
        },
        "_hours_ago": 18.0,
    },

    # ── 6. REJECT — density override (impossible density) ────────────────────
    {
        "item_description": "22K gold bangle — density anomaly",
        "declared_karat": 22,
        "branch_id": "CHN-007",
        "customer": {
            "name": "Rajan Pillai",
            "account_no": "CNR0056781234",
            "loan_app_no": "GL/2025/CHN007/00061",
            "officer_name": "Anand Krishnan (EMP-3317)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.97,
                "measured_density": 40.0,
                "expected_low": 17.2,
                "expected_high": 17.9,
                "weight_dry": 10.0,
                "weight_submerged": 9.75,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.18, "confidence": "medium", "mode": "no_images"},
            "acoustic": {"risk_score": 0.22, "confidence": "medium", "mode": "no_audio"},
            "streak": {"risk_score": 0.15, "confidence": "medium", "mode": "heuristic"},
        },
        "contradiction": {
            "contradiction_score": 0.72,
            "flags": [
                "CRITICAL: Density of 40.00 g/cm³ is physically impossible for gold — heavier than osmium",
                "Visual and acoustic signals appear normal but density makes this item impossible to verify",
            ],
        },
        "fusion": {"risk_score": 0.18, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 6, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "REJECT",
            "confidence": "HIGH",
            "loan_action": "DECLINE",
            "fusion_risk": 0.18,
            "plain_english": (
                "The Archimedes density test returned a critically anomalous reading of 40.00 g/cm³ — "
                "a value that is physically impossible for any gold alloy (maximum 19.3 for pure 24K). "
                "This indicates that either the weight measurement was tampered with, the item contains "
                "an extremely dense core material (e.g. iridium or osmium), or the item is fraudulent. "
                "This is an automatic physics-level REJECT regardless of other signals."
            ),
            "action": (
                "Do not approve the loan. Escalate to the branch gold appraiser and the fraud team. "
                "Retain the item for further investigation. File an incident report per bank policy. "
                "Do not return the item to the customer until the case is resolved."
            ),
            "llm_provider": "heuristic",
        },
        "_hours_ago": 24.0,
    },

    # ── 7. REJECT — tungsten core (density passes, contradiction catches it) ──
    {
        "item_description": "24K gold bar — tungsten-core forgery",
        "declared_karat": 24,
        "branch_id": "HYD-002",
        "customer": {
            "name": "Vijay Bhattacharya",
            "account_no": "CNR0033219876",
            "loan_app_no": "GL/2025/HYD002/00215",
            "officer_name": "Kavitha Reddy (EMP-1108)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.38,
                "measured_density": 19.23,
                "expected_low": 19.1,
                "expected_high": 19.5,
                "weight_dry": 20.0,
                "weight_submerged": 18.96,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.82, "confidence": "high", "mode": "efficientnet"},
            "acoustic": {"risk_score": 0.89, "confidence": "high", "mode": "svm"},
            "streak": {"risk_score": 0.91, "confidence": "high", "mode": "heuristic"},
        },
        "contradiction": {
            "contradiction_score": 0.78,
            "flags": [
                "Visual risk (82%) strongly contradicts normal density — plating suspected",
                "Acoustic fingerprint matches tungsten, not gold",
                "Streak colour is inconsistent with declared 24K purity",
                "Visual, acoustic, and streak signals all flag HIGH risk while density appears borderline",
            ],
        },
        "fusion": {"risk_score": 0.86, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 7, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "REJECT",
            "confidence": "HIGH",
            "loan_action": "DECLINE",
            "fusion_risk": 0.86,
            "plain_english": (
                "This case exhibits the classic signature of a tungsten-core gold-plated forgery. "
                "The density of 19.23 g/cm³ appears plausible for 24K gold because tungsten (19.25 g/cm³) "
                "has nearly identical density to gold — this is precisely why such forgeries exist. "
                "However, all three other modalities (visual analysis, acoustic ring test, and touchstone streak) "
                "independently flag extreme risk. The contradiction score of 78% is a strong fraud signal."
            ),
            "action": (
                "Reject the loan immediately. Do not return the item. Escalate to the branch manager "
                "and the Fraud Investigation Unit. Conduct sectional drilling or XRF analysis to confirm "
                "tungsten core. File a police complaint if forgery is confirmed."
            ),
            "llm_provider": "groq",
        },
        "_hours_ago": 36.0,
    },

    # ── 8. REJECT — known fake with visual AI catch ───────────────────────────
    {
        "item_description": "18K gold earrings — suspected gold-plated brass",
        "declared_karat": 18,
        "branch_id": "BLR-001",
        "customer": {
            "name": "Pooja Iyer",
            "account_no": "CNR0019876543",
            "loan_app_no": "GL/2025/BLR001/00158",
            "officer_name": "Suresh Kumar (EMP-4421)",
        },
        "modality_scores": {
            "density": {
                "risk_score": 0.71,
                "measured_density": 11.30,
                "expected_low": 15.2,
                "expected_high": 15.9,
                "weight_dry": 5.0,
                "weight_submerged": 4.40,
                "mode": "computed",
            },
            "visual": {"risk_score": 0.88, "confidence": "high", "mode": "efficientnet"},
            "acoustic": {"risk_score": 0.76, "confidence": "high", "mode": "svm"},
            "streak": {"risk_score": 0.84, "confidence": "high", "mode": "heuristic"},
        },
        "contradiction": {
            "contradiction_score": 0.31,
            "flags": [
                "Density of 11.30 g/cm³ is consistent with brass/bronze, not 18K gold",
                "Visual appearance inconsistent with hallmarked gold — surface texture anomaly detected",
            ],
        },
        "fusion": {"risk_score": 0.81, "mode": "heuristic", "shap_values": None},
        "benford": {"status": "insufficient_data", "n_samples": 8, "p_value": None, "alert": False},
        "verdict": {
            "risk_level": "REJECT",
            "confidence": "HIGH",
            "loan_action": "DECLINE",
            "fusion_risk": 0.81,
            "plain_english": (
                "The measured density of 11.30 g/cm³ is far below the expected range for 18K gold (15.2–15.9), "
                "and is consistent with brass or bronze. Visual AI analysis confirms surface texture anomalies "
                "typical of gold plating over a base metal substrate. The acoustic ring test and streak test "
                "both independently confirm this assessment. This item is gold-plated brass, not solid gold."
            ),
            "action": (
                "Decline the loan immediately. Inform the customer that the item does not meet "
                "the minimum purity standard for gold loans. Retain a record of this assessment. "
                "If the customer disputes the finding, refer to an independent assay office."
            ),
            "llm_provider": "gemini",
        },
        "_hours_ago": 48.0,
    },
]


def build_case(raw: dict) -> dict:
    hours = raw.pop("_hours_ago", 0)
    case_id = uuid.uuid4().hex[:8]
    # Rename modality_scores keys to match backend schema
    ms = raw.get("modality_scores", {})
    if "visual" in ms and "image" not in ms:
        ms["image"] = ms.pop("visual")
    return {
        "case_id":          case_id,
        "timestamp":        ts(hours),
        **raw,
    }


def main():
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases = [build_case(dict(c)) for c in CASES]
    HISTORY_PATH.write_text(json.dumps(cases, indent=2))
    print(f"Seeded {len(cases)} demo cases to {HISTORY_PATH}")
    for c in cases:
        rl = c["verdict"]["risk_level"]
        print(f"  {c['case_id']}  {rl:12s}  {c['item_description'][:55]}")


if __name__ == "__main__":
    main()
