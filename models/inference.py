# models/inference.py
from ultralytics import YOLO
from pathlib import Path
import sys

ROOT = Path("D:/Dhruhi Nuv/College/Tivaan_Vision")
WEIGHTS = ROOT / "results" / "tivaan_yolov8_train" / "weights" / "best.pt"
VAL_IMAGES = ROOT / "DroneVehiclesDatasetYOLO" / "val" / "images"
OUT_DIR = ROOT / "results" / "inference_outputs"

if not WEIGHTS.exists():
    print("ERROR: Weights not found. Please train first.")
    sys.exit(1)
if not VAL_IMAGES.exists():
    print("ERROR: Validation images not found at:", VAL_IMAGES)
    sys.exit(1)

OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Running inference on:", VAL_IMAGES)
model = YOLO(str(WEIGHTS))
model.predict(source=str(VAL_IMAGES), save=True, project=str(OUT_DIR), name="predictions")
print("✅ Inference done. Outputs in:", OUT_DIR)
