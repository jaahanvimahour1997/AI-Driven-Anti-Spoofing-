import cv2
import base64
import requests
import json
import os
from datetime import datetime

# -------------------------
# API URL
# -------------------------
url = "http://127.0.0.1:5000/predict"

# -------------------------
# Create Results Folder
# -------------------------
RESULT_FOLDER = "api_results"
SPOOF_FOLDER = os.path.join(RESULT_FOLDER, "spoof_images")
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(SPOOF_FOLDER, exist_ok=True)

LOG_FILE = os.path.join(RESULT_FOLDER, "results_log.json")

# If JSON log file does not exist, create empty list
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        json.dump([], f, indent=4)

# -------------------------
# Open Webcam
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode()

    payload = {"image": img_str}

    try:
        response = requests.post(url, json=payload).json()
        label = response.get("label", "No Response")
        confidence = response.get("confidence", 0)
        role = response.get("role", "Unknown")
    except:
        label = "API Error"
        confidence = 0
        role = "Unknown"

    # -------------------------
    # DRAW RESULT ON SCREEN
    # -------------------------
    if "Real" in label:
        color = (0, 255, 0)
    elif "Spoof" in label:
        color = (0, 0, 255)
    else:
        color = (0, 255, 255)

    text = f"{label} | {role} | {confidence}"
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Anti-Spoofing + Role Detection", frame)

    # -------------------------
    # SAVE RESULTS TO JSON FILE
    # -------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = {
        "timestamp": timestamp,
        "label": label,
        "confidence": confidence,
        "role": role
    }

    # Append to JSON log file
    with open(LOG_FILE, "r+") as f:
        data = json.load(f)
        data.append(log_entry)
        f.seek(0)
        json.dump(data, f, indent=4)

    # -------------------------
    # IF SPOOF → SAVE IMAGE
    # -------------------------
    if "Spoof" in label:
        img_name = f"spoof_{timestamp.replace(':','-').replace(' ','_')}.jpg"
        save_path = os.path.join(SPOOF_FOLDER, img_name)
        cv2.imwrite(save_path, frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
