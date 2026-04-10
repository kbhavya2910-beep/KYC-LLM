def calculate_risk(face_score, liveness_score, deepfake_score,
                   blink_detected=True, head_movement="center"):

    face_risk = 100 - face_score
    liveness_risk = 100 - liveness_score
    deepfake_risk = deepfake_score

    behavior_risk = 0
    if not blink_detected:
        behavior_risk += 5
    if head_movement == "no_face":
        behavior_risk += 10
    elif head_movement == "center":
        behavior_risk += 5

    risk = (
        face_risk * 0.4 +
        liveness_risk * 0.25 +
        deepfake_risk * 0.25 +
        behavior_risk * 0.10
    )

    risk = max(0, min(100, risk))
    return round(risk, 2)