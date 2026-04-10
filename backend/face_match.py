import cv2
import tempfile
import os
import numpy as np
from deepface import DeepFace

# Cache the ID image embedding so it's only computed once on upload
_id_embedding = None

def set_id_embedding(id_img):
    global _id_embedding
    try:
        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f.close()
        cv2.imwrite(f.name, id_img)
        result = DeepFace.represent(
            f.name,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        os.unlink(f.name)
        _id_embedding = np.array(result[0]["embedding"])
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
        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f.close()
        cv2.imwrite(f.name, live_img)
        result = DeepFace.represent(
            f.name,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        os.unlink(f.name)
        live_emb = np.array(result[0]["embedding"])

        # Cosine similarity → score
        cosine_sim = np.dot(_id_embedding, live_emb) / (
            np.linalg.norm(_id_embedding) * np.linalg.norm(live_emb) + 1e-6
        )
        score = max(0.0, min(100.0, cosine_sim * 100.0))
        print(f"[FaceMatch] cosine={cosine_sim:.4f} score={score:.2f}%")
        return round(score, 2)
    except Exception as e:
        print(f"[FaceMatch] Error: {e}")
        return 0
