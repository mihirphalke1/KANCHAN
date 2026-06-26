"""
LLM-generated plain-English verdict using Groq or Google Gemini.
Falls back gracefully when no API key is set.
"""
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)


def _build_prompt(payload: dict) -> str:
    scores = payload.get("modality_scores", {})
    verdict = payload.get("fusion_risk", 0.5)
    contra = payload.get("contradiction", {})
    density = payload.get("density_details", {})
    karat = payload.get("declared_karat", "unknown")
    desc = payload.get("item_description", "gold item")

    flags_text = "; ".join(contra.get("flags", [])) or "none"
    contra_score = contra.get("contradiction_score", 0.0)

    return f"""You are a senior gold appraiser at a bank helping a junior officer understand an AI analysis result.

Item: {desc} (declared {karat}K gold)

Analysis results:
- Overall risk score: {verdict:.2f} (0=genuine, 1=fake)
- Image analysis risk: {scores.get('image', {}).get('risk_score', 'N/A')}
- Density analysis risk: {scores.get('density', {}).get('risk_score', 'N/A')} (measured density: {density.get('measured_density', 'N/A')} g/cm³, expected {density.get('expected_low', 'N/A')}-{density.get('expected_high', 'N/A')} g/cm³)
- Acoustic analysis risk: {scores.get('acoustic', {}).get('risk_score', 'N/A')}
- Streak analysis risk: {scores.get('streak', {}).get('risk_score', 'N/A')}
- Cross-modal contradiction score: {contra_score:.2f}
- Contradiction flags: {flags_text}

Write TWO short paragraphs for a bank officer with no ML background:
1. What these results mean in plain English (2-3 sentences). Mention which signals are concerning.
2. What action the officer should take next (1-2 sentences).

Be direct and specific. Do not use jargon. Do not repeat the numbers verbatim."""


def _heuristic_verdict(fusion_risk: float, contradiction_flags: list) -> tuple[str, str]:
    """Fallback plain-English verdict when no LLM is available."""
    has_flags = len(contradiction_flags) > 0

    if fusion_risk < 0.25:
        explanation = (
            "All four analysis signals — visual appearance, density, sound, and streak test — "
            "are consistent with genuine gold at the declared karat. No suspicious patterns were detected."
        )
        action = "The item appears genuine. You may proceed with loan approval subject to standard documentation checks."
    elif fusion_risk < 0.55:
        if has_flags:
            flag_summary = contradiction_flags[0] if contradiction_flags else ""
            explanation = (
                f"The analysis detected a conflict between signals: {flag_summary}. "
                "This pattern can indicate a composite item (e.g., tungsten core with gold plating) "
                "that passes some tests but fails others."
            )
            action = "Hold the item for secondary physical testing. Consider XRF spectrometry or acid test before approving."
        else:
            explanation = (
                "Most signals are in the genuine range but one or more are borderline. "
                "The item cannot be confidently classified without additional verification."
            )
            action = "Recommend a secondary physical verification (acid test or XRF) before proceeding."
    else:
        explanation = (
            "Multiple analysis signals indicate this item is likely not genuine gold at the declared karat. "
            + (f"Key concern: {contradiction_flags[0]}" if contradiction_flags else "High overall risk score.")
        )
        action = "Do not approve the loan for this item. Escalate to the branch gold appraiser for manual inspection."

    return explanation, action


async def generate_verdict(payload: dict) -> dict:
    """
    Generate LLM verdict. Tries Groq first, then Gemini, then heuristic.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    fusion_risk = payload.get("fusion_risk", 0.5)
    contra = payload.get("contradiction", {})
    flags  = contra.get("flags", [])

    prompt = _build_prompt(payload)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                )
                text = response.choices[0].message.content.strip()
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                explanation = paragraphs[0] if paragraphs else text
                action = paragraphs[1] if len(paragraphs) > 1 else ""
                return {
                    "plain_english":  explanation,
                    "action":         action,
                    "llm_provider":   "groq:llama3-70b-8192",
                }
            except Exception as e:
                logger.warning("Groq LLM failed (%s), trying Gemini", e)

    if provider in ("gemini", "google") or (provider == "groq" and not os.getenv("GROQ_API_KEY")):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model   = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                text = response.text.strip()
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                explanation = paragraphs[0] if paragraphs else text
                action = paragraphs[1] if len(paragraphs) > 1 else ""
                return {
                    "plain_english":  explanation,
                    "action":         action,
                    "llm_provider":   "gemini-1.5-flash",
                }
            except Exception as e:
                logger.warning("Gemini LLM failed (%s), using heuristic", e)

    explanation, action = _heuristic_verdict(fusion_risk, flags)
    return {
        "plain_english":  explanation,
        "action":         action,
        "llm_provider":   "heuristic",
    }
