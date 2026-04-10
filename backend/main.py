from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import base64

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from face_match import get_face_match_score, set_id_embedding
from blink import detect_blink
from head_pose import detect_head_movement
from deepfake import detect_deepfake
from llm_decision import get_llm_decision

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
id_img_store = {}
executor     = ThreadPoolExecutor(max_workers=4)

@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/{filename}")
def static_files(filename: str):
    path = os.path.join(FRONTEND_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/upload-id")
async def upload_id(id_image: UploadFile = File(...)):
    id_bytes = await id_image.read()
    img = cv2.imdecode(np.frombuffer(id_bytes, np.uint8), 1)
    if img is None:
        return {"error": "Invalid image"}
    id_img_store["img"] = img
    set_id_embedding(img)
    return {"status": "ID uploaded"}

class FrameData(BaseModel):
    frame: str

@app.post("/verify-frame")
async def verify_frame(data: FrameData):
    if "img" not in id_img_store:
        return {"error": "No ID image uploaded"}

    img_bytes = base64.b64decode(data.frame.split(",")[-1])
    live_img  = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
    if live_img is None:
        return {"error": "Invalid frame"}

    # Run face_match, blink, head in parallel
    f_face  = executor.submit(get_face_match_score, live_img)
    f_blink = executor.submit(detect_blink, live_img)
    f_head  = executor.submit(detect_head_movement, live_img)

    face_score = f_face.result()
    blink      = f_blink.result()
    head       = f_head.result()

    deepfake_score = detect_deepfake(live_img, face_score)

    liveness_score = 0
    if blink:
        liveness_score += 50
    if head in ["left", "right"]:
        liveness_score += 50

    llm = get_llm_decision(face_score, liveness_score, blink, head, deepfake_score)

    return {
        "face_match":     face_score,
        "blink_detected": blink,
        "head_movement":  head,
        "liveness_score": liveness_score,
        "deepfake_score": deepfake_score,
        "risk_score":     llm.get("risk_score"),
        "risk_level":     llm.get("risk_level"),
        "final_decision": llm.get("final_decision"),
        "reasoning":      llm.get("reasoning")
    }
