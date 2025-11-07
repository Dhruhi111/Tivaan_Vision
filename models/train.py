# models/train.py
import os
from pathlib import Path
import torch

# Safe allowlist for ultralytics DetectionModel when torch.load is strict
try:
    import ultralytics.nn.tasks as tasks
    torch.serialization.add_safe_globals([tasks.DetectionModel])
except Exception:
    pass

from ultralytics import YOLO

ROOT = Path("D:/Dhruhi Nuv/College/Tivaan_Vision")
DATA_YAML = ROOT / "data.yaml"  # absolute path to root data.yaml

def main():
    if not DATA_YAML.exists():
        raise SystemExit(f"data.yaml not found at {DATA_YAML}. Make sure it's at project root.")
    data_path = str(DATA_YAML.resolve())
    print("Using dataset YAML:", data_path)

    # Load pre-trained yolov8n (it will download if not present)
    model = YOLO("yolov8n.pt")

    # Quick test: change epochs to 2 for a quick run; for final training set higher (e.g., 30-50)
    epochs = int(os.environ.get("TV_EPOCHS", "2"))  # for test run default=2
    batch = int(os.environ.get("TV_BATCH", "8"))

    print(f"Starting training (epochs={epochs}, batch={batch}) ...")
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device="cpu",  # change to "cuda" if you have GPU
        name="tivaan_yolov8_train",
        project=str(ROOT / "results"),
        exist_ok=True
    )
    print("✅ Training finished. Check the 'results' folder for outputs.")

if __name__ == "__main__":
    main()
