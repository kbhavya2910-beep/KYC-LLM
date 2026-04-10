import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_blink(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return False
    (x, y, w, h) = faces[0]
    roi  = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 10)
    return len(eyes) < 2

def get_eye_openness(frame):
    """Returns 0-100 score of how open the eyes are — higher = more open = more live"""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return 0
    (x, y, w, h) = faces[0]
    roi  = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 10)
    if len(eyes) == 0:
        return 0
    # Use average eye area relative to face area as openness score
    face_area = w * h
    eye_area  = sum(ew * eh for (_, _, ew, eh) in eyes)
    score     = min(100, (eye_area / face_area) * 1000)
    return round(score, 2)
