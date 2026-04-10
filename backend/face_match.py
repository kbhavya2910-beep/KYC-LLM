import cv2
import numpy as np
import tempfile, os
from deepface import DeepFace

_id_embedding = None

def set_id_embedding(id_img):
    global _id_embedding
    try:
        # Resize to 160x160 — Facenet native size, faster processing
        img_resized = cv2.resize(id_img, (160, 160))
        result = DeepFace.represent(
            img_resized,
            model_name="Facenet",
            detector_backend="skip",   # skip detection on ID — already a face photo
            enforce_detection=False
        )
        _id_embedding = np.array(result[0]["embedding"])
        norm = np.linalg.norm(_id_embedding)
        if norm > 0:
            _id_embedding = _id_embedding / norm  # pre-normalize
        print("[FaceMatch] ID embedding cached")
        return True
    except Exception as e:
        print(f"[FaceMatch] Embedding error: {e}")
        return False

def get_face_match_score(live_img):
    global _id_embedding
    if _id_embedding is None:
        return 0
    try:
        # Detect and crop face first for accuracy
        gray  = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Add padding around face
            pad = int(w * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(live_img.shape[1], x + w + pad)
            y2 = min(live_img.shape[0], y + h + pad)
            face_crop = live_img[y1:y2, x1:x2]
        else:
            face_crop = live_img

        img_resized = cv2.resize(face_crop, (160, 160))

        result = DeepFace.represent(
            img_resized,
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False
        )
        live_emb = np.array(result[0]["embedding"])
        norm = np.linalg.norm(live_emb)
        if norm > 0:
            live_emb = live_emb / norm

        # Cosine similarity (both pre-normalized = just dot product)
        cosine_sim = float(np.dot(_id_embedding, live_emb))
        score = max(0.0, min(100.0, cosine_sim * 100.0))
        print(f"[FaceMatch] cosine={cosine_sim:.4f} score={score:.2f}%")
        return round(score, 2)
    except Exception as e:
        print(f"[FaceMatch] Error: {e}")
        return 0
