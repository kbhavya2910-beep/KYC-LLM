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
    roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi, 1.1, 10)

    # If fewer than 2 eyes detected, likely blinking
    return len(eyes) < 2
