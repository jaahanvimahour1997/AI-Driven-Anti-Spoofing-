 Real-Time Face Anti-Spoofing & Role Classification
A real-time liveness detection and role-based face recognition system built using YOLOv8 Face, MobileNet CNN, and Face Recognition. The system detects faces, identifies spoof attempts, and classifies real users as Driver or Paramedic based on dataset folders.

🚀 Features
Real-Time Face Detection using YOLOv8n-face

Spoof Detection (Liveness Check) using a fine-tuned MobileNet model

Role Classification using face recognition (Driver / Paramedic datasets)

Works with Webcam or API input

Detects common spoof attacks:

Printed photo
Mobile screen
Zoomed-in fake faces
Screen edges & glare patterns
Generates clear labels: ✔️ Real Driver ✔️ Real Paramedic ❌ Spoof

🗂️ Project Structure
project/
│── app.py                # Flask API for prediction
│── webcam_script.py      # Real-time webcam spoof detection
│── yolov8n-face.pt       # Face detection model
│── spoof_model.h5        # Trained MobileNet anti-spoofing model
│── data_role/
│    ├── driver/
│    └── paramedic/
│── utils/
│    ├── preprocessing.py
│    ├── face_matcher.py
│── requirements.txt
│── README.md
🔧 Tech Stack
Python
Flask API
OpenCV
Ultralytics YOLOv8
TensorFlow / Keras
face_recognition
NumPy
⚙️ Installation
Clone the repo:
git clone https://github.com/yourusername/antispoofing.git
cd antispoofing
Install dependencies:
pip install -r requirements.txt
Place models:
yolov8n-face.pt      → root folder
spoof_model.h5       → root folder
Add role datasets:
data_role/driver/
data_role/paramedic/
▶️ Run the API
python app.py
API will run at:

http://127.0.0.1:5000/predict
Send base64 image to this endpoint.

▶️ Run Real-Time Webcam Detection
python webcam_script.py
🤖 How the System Works
1. Face Detection
YOLOv8 detects all faces in the frame along with bounding boxes.

2. Liveness Prediction
MobileNet CNN analyzes texture & depth cues to classify:

Real
Spoof
3. Role Identification
If the face is Real, the system compares it with stored encodings:

driver/
paramedic/
The final label is formed as:

Real Driver
Real Paramedic
Spoof
📈 Model Training (Summary)
MobileNet trained on real vs spoof dataset

Strong augmentations applied:

Zoom
Brightness change
Blur
Noise
Helps detect zoomed-in mobile attacks and screen glare

📌 Future Improvements
Add blink detection / head movement challenge
Replace face_recognition with Dlib or ArcFace for better accuracy
Add anti-deepfake analysis
Integrate multi-user attendance or logging system
📝 Conclusion
This project successfully performs real-time anti-spoofing with accurate role classification, making it suitable for security, attendance systems, and controlled access environments. The combination of YOLO, CNN-based spoof detection, and face recognition makes the system both fast and reliable.

👤 Author
Jaahanvi B.Tech CSE (AI & ML)

