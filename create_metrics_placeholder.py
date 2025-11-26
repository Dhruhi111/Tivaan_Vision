#!/usr/bin/env python3
"""
Safe helper: create static/results/metrics.json if it does not already exist.

- Will NOT overwrite an existing metrics.json
- Will include placeholders for mAP/precision/recall/pr_curve plus last_detection info (if available)
- Run from project root: python create_metrics_placeholder.py
"""

import json
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("static") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_FILE = RESULTS_DIR / "metrics.json"
LAST_DET_FILE = RESULTS_DIR / "last_detection.json"

if METRICS_FILE.exists():
    print(f"✅ metrics.json already exists at {METRICS_FILE}. Nothing changed.")
    exit(0)

# Try to read last_detection.json if present
last_detection = None
if LAST_DET_FILE.exists():
    try:
        last_detection = json.loads(LAST_DET_FILE.read_text(encoding="utf8"))
        print("Found last_detection.json — will include last_vehicle_count in metrics.")
    except Exception as e:
        print("Warning: could not read last_detection.json:", e)

# Compose a safe placeholder metrics structure
metrics = {
    "created_at": datetime.now().isoformat(),
    "notes": "This is an auto-generated placeholder metrics.json. Replace with real metrics after evaluation.",
    # placeholders - replace with real values if you compute them later
    "mAP": None,
    "precision": None,
    "recall": None,
    # Optional PR curve placeholder (recall, precision pairs) - empty so charts don't break
    "pr_curve": [],
    # Add some training summary placeholders
    "epochs": None,
    "train_images": None,
    "val_images": None,
    # last detection snapshot (if available)
    "last_detection": last_detection,
    # a small example of a summary table the frontend/chart can use
    "summary": {
        "total_detections_logged": 0,
        "notes": "Replace with computed metrics such as mAP@0.5, precision@0.5, recall@0.5"
    }
}

# Save safely
try:
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(f"✅ Created placeholder metrics.json at {METRICS_FILE}")
    print("You can now open the Metrics page — it will show placeholders and allow downloads.")
except Exception as e:
    print("✖ Failed to write metrics.json:", e)
    raise
