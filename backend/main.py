from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import cv2
import base64

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from face_match import get_face_match_score
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

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "frontend")), name="static")

# Store uploaded ID image in memory
id_img_store = {}

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html"))

@app.post("/upload-id")
async def upload_id(id_image: UploadFile = File(...)):
    id_bytes = await id_image.read()
    img = cv2.imdecode(np.frombuffer(id_bytes, np.uint8), 1)
    if img is None:
        return {"error": "Invalid image"}
    id_img_store["img"] = img
    return {"status": "ID uploaded"}

class FrameData(BaseModel):
    frame: str  # base64 encoded JPEG

@app.post("/verify-frame")
async def verify_frame(data: FrameData):
    if "img" not in id_img_store:
        return {"error": "No ID image uploaded"}

    # Decode base64 webcam frame
    img_bytes = base64.b64decode(data.frame.split(",")[-1])
    live_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
    if live_img is None:
        return {"error": "Invalid frame"}

    id_img = id_img_store["img"]

    face_score     = get_face_match_score(id_img, live_img)
    blink          = detect_blink(live_img)
    head           = detect_head_movement(live_img)
    deepfake_score = detect_deepfake(live_img)

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