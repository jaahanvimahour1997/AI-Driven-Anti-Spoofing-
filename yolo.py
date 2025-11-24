import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Load YOLOv8 face detector
face_model = YOLO("model.pt")   # make sure this file exists!

# Load your anti-spoofing model
spoof_model = tf.keras.models.load_model("anti_spoof_mobilenet.h5")

IMG_SIZE = 96

# Label mapping
labels = ["Real", "Spoof"]

# Open webcam
cap = cv2.VideoCapture(0)

print("✔ Real-time Anti-Spoofing System Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model.predict(frame, conf=0.50)

    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            face_resized = face_resized.astype("float32") / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            # Spoof prediction
            pred = spoof_model.predict(face_resized)
            class_id = np.argmax(pred)
            label = labels[class_id]

            # Color: Real = Green, Spoof = Red
            color = (0, 255, 0) if label == "Real" else (0, 0, 255)

            # Draw box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Anti-Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
