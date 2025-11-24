from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import face_recognition
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# -------------------------------
# LOAD MODELS
# -------------------------------
yolo_model = YOLO("model.pt")
spoof_model = load_model("anti_spoof_mobilenet.h5")

# -------------------------------
# LOAD ROLE (DRIVER / PARAMEDIC) DATASET
# -------------------------------
role_data_path = r"C:\Users\hp\PycharmProjects\PythonProject21\roles"

driver_encodings = []
paramedic_encodings = []

def load_role_images():
    global driver_encodings, paramedic_encodings

    for root, dirs, files in os.walk(role_data_path):

        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):

                img_path = os.path.join(root, file)
                img = face_recognition.load_image_file(img_path)

                encoding = face_recognition.face_encodings(img)
                if not encoding:
                    continue

                if "driver" in root.lower():
                    driver_encodings.append(encoding[0])

                elif "paramedic" in root.lower():
                    paramedic_encodings.append(encoding[0])


load_role_images()
print("Role encodings loaded:")
print("Drivers:", len(driver_encodings))
print("Paramedics:", len(paramedic_encodings))


# -------------------------------
# IMAGE DECODER
# -------------------------------
def decode_image(img_string):
    img_data = base64.b64decode(img_string)
    np_img = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)


# -------------------------------
# ROLE IDENTIFICATION FUNCTION
# -------------------------------
def identify_role(face_crop):
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)

    if len(enc) == 0:
        return "Unknown", 0.0

    face_enc = enc[0]

    # Compare with driver encodings
    driver_distances = face_recognition.face_distance(driver_encodings, face_enc)
    paramedic_distances = face_recognition.face_distance(paramedic_encodings, face_enc)

    min_driver = np.min(driver_distances) if len(driver_distances) else 0.9
    min_paramedic = np.min(paramedic_distances) if len(paramedic_distances) else 0.9

    if min_driver < 0.45:
        return "Driver", 1 - min_driver

    if min_paramedic < 0.45:
        return "Paramedic", 1 - min_paramedic

    return "Unknown", 0.0


# -------------------------------
# API ENDPOINT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # get image
    data = request.json
    img_b64 = data.get("image")

    if img_b64 is None:
        return jsonify({"error": "No image found"}), 400

    frame = decode_image(img_b64)

    # YOLO FACE DETECTION
    results = yolo_model(frame)[0]
    detections = results.boxes.xyxy.cpu().numpy()

    if len(detections) == 0:
        return jsonify({"label": "No Face Detected"})

    x1, y1, x2, y2 = map(int, detections[0])
    face_crop = frame[y1:y2, x1:x2]

    # PREPARE FOR SPOOF MODEL
    face_resized = cv2.resize(face_crop, (96, 96))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)

    spoof_prob = spoof_model.predict(face_resized)[0][0]
    is_real = spoof_prob > 0.5

    if not is_real:
        return jsonify({"label": "Spoof", "confidence": float(spoof_prob)})

    # ROLE IDENTIFICATION
    role, conf = identify_role(face_crop)

    return jsonify({
        "label": f"Real {role}",
        "confidence": float(conf),
        "spoof_score": float(spoof_prob)
    })


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
