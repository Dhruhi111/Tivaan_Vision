# models/export_model.py
from ultralytics import YOLO
import os

weights = r"D:/Dhruhi Nuv/College/Tivaan_Vision/results/tivaan_yolov8_train/weights/best.pt"
if not os.path.exists(weights):
    raise FileNotFoundError("best.pt not found. Train first!")

model = YOLO(weights)
model.export(format="onnx")
print("✅ Model exported as ONNX successfully!")
