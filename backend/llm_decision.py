import json
import os

import openai
from dotenv import load_dotenv

from risk_engine import calculate_risk

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

    if os.getenv("OPENROUTER_API_KEY") and "openrouter" not in getattr(openai, "api_base", ""):
        openai.api_base = "https://openrouter.ai/v1"


def get_llm_decision(face_match: float, liveness_score: float, blink_detected: bool,
                     head_movement: str, deepfake_score: float) -> dict:
    if not OPENAI_API_KEY:
        return _rule_based_fallback(face_match, liveness_score, blink_detected, head_movement, deepfake_score)

    prompt = (
        "You are a KYC risk assessment assistant. "
        "Evaluate the risk based on the following values and return only valid JSON with two keys: "
        "risk_score (0-100) and reasoning."
        f" face_match={face_match}, liveness_score={liveness_score}, "
        f"blink_detected={blink_detected}, head_movement={head_movement}, "
        f"deepfake_score={deepfake_score}."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful risk assessment assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200,
        )
        content = response.choices[0].message["content"].strip()
        parsed = json.loads(content)

        return {
            "risk_score": parsed.get("risk_score"),
            "reasoning": parsed.get("reasoning")
        }
    except Exception:
        return _rule_based_fallback(face_match, liveness_score, blink_detected, head_movement, deepfake_score)


def _rule_based_fallback(face_match, liveness_score, blink_detected, head_movement, deepfake_score):
    face_risk = 100 - face_match
    liveness_risk = 100 - liveness_score
    deepfake_risk = deepfake_score

    behavior_risk = 0
    if not blink_detected:
        behavior_risk += 5
    if head_movement == "no_face":
        behavior_risk += 10
    elif head_movement == "center":
        behavior_risk += 5

    risk = calculate_risk(face_match, liveness_score, deepfake_score,
                          blink_detected=blink_detected,
                          head_movement=head_movement)

    reason = (
        f"Face risk contributed {face_risk * 0.4:.2f}, "
        f"liveness risk contributed {liveness_risk * 0.25:.2f}, "
        f"deepfake risk contributed {deepfake_risk * 0.25:.2f}, "
        f"and behavior risk contributed {behavior_risk * 0.10:.2f}."
    )

    return {
        "risk_score": risk,
        "reasoning": reason
    }
