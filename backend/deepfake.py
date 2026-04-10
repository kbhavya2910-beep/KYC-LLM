import cv2
import numpy as np

def detect_deepfake(frame):
    """
    Estimates deepfake likelihood using:
    1. Laplacian blur score  — real faces are sharp
    2. Noise level           — synthetic faces have less natural noise
    3. Color channel variance — real faces have natural skin tone variation
    Higher score = more likely fake
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Blur detection — low sharpness = suspicious
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = max(0, min(100, 100 - (laplacian_var / 10)))

        # 2. Noise level — too clean = suspicious (synthetic)
        noise = np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5,5), 0))
        noise_score = max(0, min(100, 100 - (noise * 5)))

        # 3. Color variance — unnatural uniform skin = suspicious
        b, g, r = cv2.split(frame.astype(np.float32))
        color_var = np.mean([np.std(b), np.std(g), np.std(r)])
        color_score = max(0, min(100, 100 - (color_var * 2)))

        # Weighted combination
        deepfake_score = (blur_score * 0.4) + (noise_score * 0.3) + (color_score * 0.3)
        return round(deepfake_score, 2)

    except Exception:
        return 50.0
