import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SYSTEM_PROMPT = """You are a highly reliable AI-based KYC Decision Engine designed for a banking-grade identity verification system.

Your role is to act as an expert fraud detection system that analyzes outputs from multiple deep learning (CNN-based) models and produces a final decision.

You will receive structured inputs in JSON format:

{
  "face_match_score": number (0-100),
  "liveness_score": number (0-100),
  "blink_detected": true/false,
  "head_movement": "left" / "right" / "center" / "no_face",
  "deepfake_score": number (0-100)
}

---

### YOUR TASK:

1. Analyze all inputs logically and consistently
2. Detect inconsistencies or suspicious patterns
3. Assign:
   - Final Decision: GENUINE / SUSPICIOUS / FRAUDULENT
   - Risk Score: 0-100
   - Risk Level: LOW / MEDIUM / HIGH
4. Provide a short but precise reasoning

---

### DECISION RULES (STRICT):

FACE MATCH:
- > 80 → Strong identity match
- 60-80 → Moderate match
- < 60 → Weak match

LIVENESS:
- > 80 → Strong liveness
- 50-80 → Partial liveness
- < 50 → Weak or failed liveness

DEEPFAKE:
- < 30 → Likely real
- 30-60 → Suspicious
- > 60 → Likely fake

---

### LOGIC:

- If face_match_score < 50 → FRAUDULENT
- If deepfake_score > 70 → FRAUDULENT
- If liveness_score < 40 → FRAUDULENT

- If moderate values in multiple categories → SUSPICIOUS
- If all scores are strong → GENUINE

- If blink_detected = false AND head_movement = "center"
  → Increase suspicion (possible spoof)

- If head_movement = "no_face"
  → Immediately HIGH RISK

---

### OUTPUT FORMAT (STRICT JSON ONLY):

{
  "final_decision": "GENUINE / SUSPICIOUS / FRAUDULENT",
  "risk_score": number,
  "risk_level": "LOW / MEDIUM / HIGH",
  "reasoning": "Explain decision clearly based on scores and detected patterns"
}

---

### IMPORTANT RULES:

- Be deterministic (same input = same output)
- Do NOT hallucinate
- Do NOT ignore any field
- Always justify decision using actual values
- Be strict like a real banking fraud system
- No extra text outside JSON"""


def get_llm_decision(face_match: float, liveness_score: float, blink_detected: bool,
                     head_movement: str, deepfake_score: float) -> dict:

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())

    user_input = json.dumps({
        "face_match_score": face_match,
        "liveness_score": liveness_score,
        "blink_detected": blink_detected,
        "head_movement": head_movement,
        "deepfake_score": deepfake_score
    })

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_input}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return _rule_based_fallback(face_match, liveness_score, blink_detected, head_movement, deepfake_score)


def _rule_based_fallback(face_match, liveness_score, blink_detected, head_movement, deepfake_score):
    if face_match < 50 or deepfake_score > 70 or liveness_score < 40 or head_movement == "no_face":
        decision, level = "FRAUDULENT", "HIGH"
    elif face_match >= 80 and liveness_score >= 80 and deepfake_score < 30:
        decision, level = "GENUINE", "LOW"
    else:
        decision, level = "SUSPICIOUS", "MEDIUM"

    risk = round((100 - face_match) * 0.4 + (100 - liveness_score) * 0.3 + deepfake_score * 0.3, 2)
    spoof_note = " No blink and centered head detected — possible spoof." if not blink_detected and head_movement == "center" else ""

    return {
        "final_decision": decision,
        "risk_score": risk,
        "risk_level": level,
        "reasoning": f"[Fallback] Face match: {face_match}%, Liveness: {liveness_score}%, Deepfake: {deepfake_score}%, Head: {head_movement}.{spoof_note}"
    }
