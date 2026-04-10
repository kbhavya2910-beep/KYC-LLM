import cv2
import numpy as np

def detect_deepfake(frame, face_match_score=50):
    """
    Deepfake risk based on image quality signals + face match context.
    High face match = lower deepfake risk.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpness — blurry = suspicious
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = max(0, min(100, 100 - (blur / 5)))

        # Face match context — if face matches well, deepfake risk is lower
        match_penalty = max(0, (100 - face_match_score) * 0.3)

        score = (blur_score * 0.4) + match_penalty
        return round(min(100, score), 2)

    except Exception:
        return round(max(0, (100 - face_match_score) * 0.3), 2)
