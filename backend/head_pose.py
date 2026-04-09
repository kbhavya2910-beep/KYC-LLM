import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_head_movement(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "no_face"

    (x, y, w, h) = faces[0]

    center_x = x + w // 2
    frame_center = frame.shape[1] // 2

    if center_x < frame_center - 50:
        return "left"
    elif center_x > frame_center + 50:
        return "right"
    else:
        return "center"