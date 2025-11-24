import cv2
import os
import numpy as np
import face_recognition
import tensorflow as tf
from ultralytics import YOLO


# ----------------------------------------------------------
# PATH SETTINGS
# ----------------------------------------------------------
DRIVER_DIR = r"C:\Users\hp\PycharmProjects\PythonProject21\roles\driver"
PARAMEDIC_DIR = r"C:\Users\hp\PycharmProjects\PythonProject21\roles\paramedic"

SPOOF_MODEL_PATH = "anti_spoof_mobilenet.h5"
YOLO_FACE_PATH = "model.pt"

IMG_SIZE = 96
LABELS = ["Real", "Spoof"]


# ----------------------------------------------------------
# LOAD YOLO FACE DETECTOR
# ----------------------------------------------------------
print("[INFO] Loading YOLOv8 Face Detector...")
face_model = YOLO(YOLO_FACE_PATH)


# ----------------------------------------------------------
# LOAD ANTI-SPOOFING MODEL
# ----------------------------------------------------------
print("[INFO] Loading Anti-Spoofing Model...")
spoof_model = tf.keras.models.load_model(SPOOF_MODEL_PATH)


# ----------------------------------------------------------
# LOAD ROLE ENCODINGS
# ----------------------------------------------------------
def load_encodings(folder):
    enc_list = []
    for img_name in os.listdir(folder):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = face_recognition.load_image_file(os.path.join(folder, img_name))
            enc = face_recognition.face_encodings(img)
            if enc:
                enc_list.append(enc[0])
    return enc_list


print("[INFO] Loading Driver Encodings...")
driver_encodings = load_encodings(DRIVER_DIR)

print("[INFO] Loading Paramedic Encodings...")
paramedic_encodings = load_encodings(PARAMEDIC_DIR)


# ----------------------------------------------------------
# ROLE IDENTIFICATION FUNCTION
# ----------------------------------------------------------
def identify_role(encoding):
    """
    Input: face encoding vector
    Output: Driver / Paramedic / Unregistered
    """

    # Compare with drivers
    driver_dist = face_recognition.face_distance(driver_encodings, encoding)
    paramedic_dist = face_recognition.face_distance(paramedic_encodings, encoding)

    if len(driver_dist) == 0 or len(paramedic_dist) == 0:
        return "Real - Unregistered"

    best_driver = np.min(driver_dist)
    best_paramedic = np.min(paramedic_dist)

    if best_driver < best_paramedic and best_driver < 0.45:
        return "Real Driver"

    elif best_paramedic < best_driver and best_paramedic < 0.45:
        return "Real Paramedic"

    else:
        return "Real - Unregistered"


# ----------------------------------------------------------
# START WEBCAM
# ----------------------------------------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Starting Real-Time Anti-Spoof + Role Identification...")
print("Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model.predict(frame, conf=0.5)

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # --------------------------
            # ANTI-SPOOFING
            # --------------------------
            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            pred = spoof_model.predict(face_resized)
            class_id = np.argmax(pred)
            spoof_label = LABELS[class_id]

            # If SPOOF → do not check role
            if spoof_label == "Spoof":
                final_label = "SPOOF"
                color = (0, 0, 255)

            else:
                # --------------------------
                # ROLE IDENTIFICATION
                # --------------------------
                rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                enc = face_recognition.face_encodings(rgb_face)

                if enc:
                    role = identify_role(enc[0])
                    final_label = role
                else:
                    final_label = "Real - Unregistered"

                color = (0, 255, 0)

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, final_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Anti-Spoof + Role Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
