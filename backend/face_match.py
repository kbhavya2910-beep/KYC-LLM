import cv2
import tempfile
import os
from deepface import DeepFace

def get_face_match_score(id_img, live_img):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            cv2.imwrite(f1.name, id_img)
            cv2.imwrite(f2.name, live_img)
            result = DeepFace.verify(f1.name, f2.name, enforce_detection=False)
        os.unlink(f1.name)
        os.unlink(f2.name)
        distance = result["distance"]
        score = max(0, (1 - distance) * 100)
        return round(score, 2)
    except Exception:
        return 0
