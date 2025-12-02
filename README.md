**Tivaan Vision – AI-Powered Vehicle Detection and Accident Prevention**

Tivaan Vision is an integrated AI + IoT system designed to detect vehicles from aerial imagery and generate real-time safety alerts for accident prevention. The system combines computer vision (YOLOv8) with simulated IoT sensor logic (ultrasonic, GPS, and wearable modules) to demonstrate how AI-driven risk analysis can support safer mobility for individuals and emergency response teams.

This repository contains the full end-to-end pipeline, including dataset preparation, model training, inference engine, web dashboard, IoT simulation framework, and system metrics.

1. Project Objectives

Detect vehicles in aerial drone imagery using a custom-trained YOLOv8 model.

Estimate risk levels based on vehicle density and proximity.

Simulate IoT sensors (ultrasonic, GPS, wearable haptic feedback) for alert generation.

Present detections, risks, IoT actions, and metrics in a unified web interface.

Demonstrate how AI and IoT can be combined for accident prevention and situational awareness.

2. System Architecture

The project is structured into three major modules:

2.1 Deep Learning Module (YOLOv8)

Custom dataset prepared in YOLO format (TXT labels + images).

Labels generated using LabelImg/Roboflow (class, x_center, y_center, width, height).

Training performed using transfer learning on yolov8n.pt.

Model trained for 100 epochs with augmentation and auto hyperparameter optimization.

Evaluation metrics: mAP, Precision, Recall, Confusion Matrix, PR/RC curves.

Best model exported as best.pt.

2.2 Backend Module (Flask API)

The Flask server provides:

/api/detect for running YOLO inference on uploaded images.

/api/demo_detect for testing with stored sample images.

/api/iot IoT logic for generating dynamic actions (Slow Down, Brake, Danger).

/api/metrics for serving dynamic mAP/precision/recall metrics.

Automatic image annotation, JSON output, and results streaming to the frontend.

2.3 Frontend Web Interface

Developed using HTML, CSS, and JavaScript.

Pages included:

Home – Overview and system explanation

Inference Dashboard – Image upload, YOLO detection, and risk analysis

IoT Simulation – Ultrasonic sensor behaviour and automatic alert generation

Metrics – mAP/Precision/Recall values with visual charts from training

Dataset Demo – Displaying labeled dataset samples

Light/Dark Mode – Fully responsive design

3. IoT Integration (Simulated)

Although physical hardware (ESP32, HC-SR04, GPS) was not available, IoT behaviour is implemented through simulation scripts:

simulator_ultrasonic.py
Computes distance thresholds and generates corresponding warning levels.

simulator_gps.py
Provides simulated coordinates and direction for future expansion.

simulator_wearable.py
Simulates vibration feedback logic for the wearable safety device.

This simulates real-world integration without requiring hardware during development.

4. Dataset Details

Dataset Name: DroneVehiclesDatasetYOLO

Approximately aerial images with vehicles from drone perspective.

Labels follow YOLO format:
class_id x_center y_center width height

Preprocessing performed through:

fix_labels.py for cleaning formatting errors

Automated train/val/test split

Stored separately and ignored via .gitignore due to size.

5. Model Training Pipeline

Training was performed using Ultralytics YOLOv8:

yolo train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640 batch=16


Outputs generated:

best.pt (best performing model)

results.png (training curves)

confusion_matrix.png

F1_curve.png

labels.jpg (dataset representation)

All evaluative plots are displayed in the Metrics Page of the dashboard.

6. Metrics and Evaluation

Key performance indicators include:

mAP: ~0.82

Precision: ~0.83

Recall: ~0.81

Confusion Matrix

PR and RC Curves

A dynamic JSON generator produces detailed metrics linked to the last detection run:
create_dynamic_metrics.py.

7. Folder Structure
Tivaan_Vision/
│
├── backend/                # Flask backend and API logic
├── static/                 # CSS, JS, processed outputs
├── templates/              # Frontend HTML templates
├── simulator/              # Core IoT simulation scripts
│   ├── simulator_gps.py
│   ├── simulator_ultrasonic.py
│   └── simulator_wearable.py
│
├── models/                 # YOLO model folder (best.pt excluded from Git)
├── runs/                   # YOLO training outputs (ignored in Git)
├── results/                # Annotated images and metrics assets
├── tmp_uploads/            # Temporary images during inference
│
├── data.yaml               # Dataset configuration file
├── requirements.txt        # Backend Python dependencies
├── create_dynamic_metrics.py
├── create_metrics_placeholder.py
├── create_placeholders.py
│
└── README.md

8. Installation Instructions
Step 1: Clone the Repository
git clone https://github.com/<your-username>/Tivaan_Vision.git
cd Tivaan_Vision

Step 2: Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   (Windows)

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the Flask Server
python backend.py


Server runs at:

http://127.0.0.1:5000/

9. How to Use the Application
9.1 Inference Page

Upload image

YOLO detects vehicles

Output annotated image and vehicular count

Risk level generated (Low, Medium, High)

Results stored in results/annotated/

9.2 IoT Simulation

Enter simulated distance

System produces corresponding action instructions

Auto-triggered after each AI detection

9.3 Metrics Page

View model accuracy

Download detailed JSON metrics

Inspect training graphs

9.4 Dataset Demo

Displays randomly selected dataset samples for verification.

10. Notes

Model weights (best.pt) are excluded from Git for size reasons.

Dataset folder is excluded due to licensing and storage limits.

Virtual environment is excluded via .gitignore.

The project is modular and can be extended to real IoT hardware using MQTT or Firebase when needed.

11. Technologies Used

Python, Flask

Ultralytics YOLOv8 (PyTorch backend)

OpenCV, Pillow

HTML, CSS, JavaScript

Simulated IoT Sensors (Ultrasonic, GPS, Wearable)
