from risk_engine import calculate_risk

def get_llm_decision(face_match: float, liveness_score: float, blink_detected: bool,
                     head_movement: str, deepfake_score: float) -> dict:
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
