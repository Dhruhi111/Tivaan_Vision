# models/evaluate.py
from ultralytics import YOLO
from pathlib import Path
import sys

ROOT = Path("D:/Dhruhi Nuv/College/Tivaan_Vision")
WEIGHTS = ROOT / "results" / "tivaan_yolov8_train" / "weights" / "best.pt"
DATA_YAML = ROOT / "data.yaml"

if not WEIGHTS.exists():
    print("ERROR: best.pt not found. Train the model first!")
    sys.exit(1)

print("Evaluating using weights:", WEIGHTS)
model = YOLO(str(WEIGHTS))
metrics = model.val(data=str(DATA_YAML))
print("✅ Evaluation complete.")
print(metrics)
