def calculate_risk(face_score, liveness_score, deepfake_score):

    risk = (
        (100 - face_score) * 0.4 +
        (100 - liveness_score) * 0.3 +
        deepfake_score * 0.3
    )

    if risk < 30:
        level = "LOW"
    elif risk < 70:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return round(risk, 2), level