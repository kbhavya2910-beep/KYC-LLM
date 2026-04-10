import cv2
import tempfile
import os
from deepface import DeepFace

def get_face_match_score(id_img, live_img):
    try:
        f1 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f2 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        f1.close()
        f2.close()

        cv2.imwrite(f1.name, id_img)
        cv2.imwrite(f2.name, live_img)

        result = DeepFace.verify(
            f1.name, f2.name,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        os.unlink(f1.name)
        os.unlink(f2.name)

        distance  = result["distance"]
        threshold = result["threshold"]

        # 100% when distance=0 (perfect match), 0% when distance >= threshold
        score = max(0.0, min(100.0, (1.0 - distance / threshold) * 100.0))
        print(f"[FaceMatch] distance={distance:.4f} threshold={threshold:.4f} score={score:.2f}%")
        return round(score, 2)

    except Exception as e:
        print(f"[FaceMatch] Error: {e}")
        return 0
