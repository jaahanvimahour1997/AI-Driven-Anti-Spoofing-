import os
import cv2
import numpy as np
import face_recognition

# ----------------------------
# CHANGE THESE PATHS
# ----------------------------
DRIVER_PATH = r"C:\Users\hp\PycharmProjects\PythonProject21\roles\driver"
PARAMEDIC_PATH = r"C:\Users\hp\PycharmProjects\PythonProject21\roles\paramedic"

driver_encodings = []
paramedic_encodings = []

print("Loading stored role images...")

# ----------------------------
# LOAD DRIVER ENCODINGS
# ----------------------------
for img_name in os.listdir(DRIVER_PATH):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img = face_recognition.load_image_file(os.path.join(DRIVER_PATH, img_name))
        enc = face_recognition.face_encodings(img)
        if enc:
            driver_encodings.append(enc[0])

# ----------------------------
# LOAD PARAMEDIC ENCODINGS
# ----------------------------
for img_name in os.listdir(PARAMEDIC_PATH):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img = face_recognition.load_image_file(os.path.join(PARAMEDIC_PATH, img_name))
        enc = face_recognition.face_encodings(img)
        if enc:
            paramedic_encodings.append(enc[0])

print(f"Loaded {len(driver_encodings)} driver encodings.")
print(f"Loaded {len(paramedic_encodings)} paramedic encodings.")

# ----------------------------
# ROLE IDENTIFICATION FUNCTION
# ----------------------------

def identify_role(face_image):
    """
    Input: Cropped face image (BGR)
    Output: Role label string
    """

    rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)

    if not enc:
        return "Unknown"

    encoding = enc[0]

    # Compare with drivers
    driver_match = face_recognition.compare_faces(driver_encodings, encoding, tolerance=0.45)

    # Compare with paramedics
    paramedic_match = face_recognition.compare_faces(paramedic_encodings, encoding, tolerance=0.45)

    if True in driver_match:
        return "Real Driver"

    elif True in paramedic_match:
        return "Real Paramedic"

    else:
        return "Real - Unregistered"
