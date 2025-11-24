import cv2
import os
import time

# Load Haarcascade face detector (comes with OpenCV)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create dataset folders
os.makedirs("dataset/real", exist_ok=True)
os.makedirs("dataset/spoof", exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
time.sleep(1)

real_count = len(os.listdir("dataset/real"))
spoof_count = len(os.listdir("dataset/spoof"))

print("Press 'R' to capture REAL faces")
print("Press 'S' to capture SPOOF faces")
print("Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles for preview
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, "R=Real | S=Spoof | Q=Quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Cropped Dataset Capture", frame)
    key = cv2.waitKey(1)

    for (x, y, w, h) in faces:

        # Crop face
        face_crop = frame[y:y + h, x:x + w]

        # Save REAL
        if key == ord('r'):
            filename = f"dataset/real/real_{real_count}.jpg"
            cv2.imwrite(filename, face_crop)
            real_count += 1
            print("Saved:", filename)

        # Save SPOOF
        elif key == ord('s'):
            filename = f"dataset/spoof/spoof_{spoof_count}.jpg"
            cv2.imwrite(filename, face_crop)
            spoof_count += 1
            print("Saved:", filename)

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
