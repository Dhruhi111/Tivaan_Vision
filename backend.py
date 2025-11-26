# backend.py
"""
Flask backend for Tivaan Vision (complete, defensive).
Place in project root (e.g. D:\Dhruhi Nuv\College\Tivaan_Vision\backend.py)
Run with your venv activated:
    python backend.py
Dependencies:
    pip install flask flask-cors pillow ultralytics
"""

import os
import time
import json
import traceback
from pathlib import Path
from glob import glob
from urllib.parse import unquote
from io import BytesIO

from flask import Flask, request, jsonify, send_file, render_template, abort
from flask_cors import CORS
from PIL import Image

# Try import ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as e:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# ============= Configuration =============
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_RESULTS = STATIC_DIR / "results"
STATIC_RESULTS.mkdir(parents=True, exist_ok=True)

# IMPORTANT: set your dataset folder here (the one you said you have)
# Use raw string to avoid escape issues on Windows.
DATASET_DIR = Path(r"D:\Dhruhi Nuv\College\Tivaan_Vision\DroneVehiclesDatasetYOLO")

# Model weights default locations (we try known paths)
DEFAULT_WEIGHTS = BASE_DIR / "results" / "tivaan_yolov8_train" / "weights" / "best.pt"
FALLBACK_WEIGHTS = BASE_DIR / "yolov8n.pt"

# ============= App init =============
app = Flask(__name__, static_folder=str(STATIC_DIR), template_folder=str(BASE_DIR / "templates"))
CORS(app)

# ============= Model loading =============
MODEL = None
MODEL_LOADED = False
MODEL_ERR = None
MODEL_PATH = None

def try_load_model():
    global MODEL, MODEL_LOADED, MODEL_ERR, MODEL_PATH
    if not ULTRALYTICS_AVAILABLE:
        MODEL_ERR = "ultralytics not installed in this venv."
        MODEL_LOADED = False
        print("[backend] ultralytics not installed.")
        return

    # pick the best candidate
    candidates = [DEFAULT_WEIGHTS, FALLBACK_WEIGHTS]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break

    if not found:
        MODEL_ERR = f"No model weights found. Checked: {candidates}"
        MODEL_LOADED = False
        print("[backend] model weights not found. Expected default at:", DEFAULT_WEIGHTS)
        return

    try:
        print(f"[backend] Loading model from: {found}")
        MODEL = YOLO(str(found))
        MODEL_LOADED = True
        MODEL_ERR = None
        MODEL_PATH = str(found)
        print("[backend] Model loaded successfully.")
    except Exception as e:
        MODEL = None
        MODEL_LOADED = False
        MODEL_ERR = str(e) + "\n" + traceback.format_exc()
        print("[backend] Failed to load model:", MODEL_ERR)

# Try load at startup
try_load_model()

# ============= Helper functions =============
def safe_dataset_image_path(rel_path: str):
    """
    Ensure rel_path is inside DATASET_DIR and return Path object.
    rel_path is expected to be a path relative to DATASET_DIR, e.g. 'val/images/00570.jpg'
    """
    if not rel_path:
        return None
    # decode URL encoded
    rel = unquote(rel_path)
    candidate = (DATASET_DIR / rel).resolve()
    try:
        if DATASET_DIR.resolve() in candidate.parents or DATASET_DIR.resolve() == candidate.parent:
            return candidate
    except Exception:
        return None
    return None

def latest_annotated_url():
    files = sorted(glob(str(STATIC_RESULTS / "annotated_*.jpg")), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    return "/static/results/" + Path(files[0]).name

# ============= Routes =============

@app.route("/")
def home():
    # homepage (templates/home.html)
    return render_template("home.html")

@app.route("/inference")
def inference_page():
    return render_template("inference.html")

@app.route("/metrics")
def metrics_page():
    return render_template("metrics.html")

@app.route("/iot")
def iot_page():
    return render_template("iot.html")

@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded": MODEL_LOADED,
        "model_err": MODEL_ERR,
        "model_path": MODEL_PATH,
        "ultralytics_available": ULTRALYTICS_AVAILABLE,
        "dataset_dir": str(DATASET_DIR) if DATASET_DIR.exists() else None
    })

# List dataset images (train/val/test). We return relative paths that can be passed back to /dataset_image
@app.route("/api/dataset_images")
def api_dataset_images():
    """
    Returns JSON:
    {
      "train": ["train/images/0001.jpg", ...],
      "val": [...],
      "test": [...]
    }
    Note: For large datasets we return first N images only to avoid huge payloads.
    """
    out = {}
    cap = 200  # max images per split to return (you can change)
    for split in ("train", "val", "test"):
        images_dir = DATASET_DIR / split / "images"
        if not images_dir.exists():
            out[split] = []
            continue
        images = sorted([p for p in images_dir.glob("**/*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        # convert to relative paths relative to DATASET_DIR: e.g., "val/images/00570.jpg"
        rels = [str(p.relative_to(DATASET_DIR)).replace("\\","/") for p in images[:cap]]
        out[split] = rels
    return jsonify(out)

# Serve a dataset image by relative path (safe check)
@app.route("/dataset_image")
def dataset_image():
    """
    Query param: ?img=<relative-path>
    Example: /dataset_image?img=val/images/00570_jpg.rf.1ccac6fe20948b2e9d2783b8a751f138.jpg
    """
    rel = request.args.get("img", "")
    safe = safe_dataset_image_path(rel)
    if not safe or not safe.exists():
        return jsonify({"error":"image not found or invalid path", "path": rel}), 404
    # send file
    return send_file(str(safe), mimetype="image/jpeg")

# POST /api/detect - accept file upload ('file' field) and run model
@app.route("/api/detect", methods=["POST"])
def api_detect():
    if not ULTRALYTICS_AVAILABLE:
        return jsonify({"error":"ultralytics not installed"}), 500
    if not MODEL_LOADED:
        return jsonify({"error":"model not loaded", "details": MODEL_ERR}), 500

    if "file" not in request.files:
        return jsonify({"error":"no file uploaded. Use 'file' field"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error":"empty filename"}), 400

    # save temporarily
    tmp_dir = BASE_DIR / "tmp_uploads"
    tmp_dir.mkdir(exist_ok=True)
    ts = int(time.time() * 1000)
    suffix = Path(f.filename).suffix or ".jpg"
    tmp_path = tmp_dir / f"upload_{ts}{suffix}"
    try:
        f.save(tmp_path)
    except Exception as e:
        return jsonify({"error":"failed saving uploaded file", "details": str(e)}), 500

    # run model
    try:
        results = MODEL(str(tmp_path), imgsz=640, conf=0.25, device="cpu")
    except Exception as e:
        # try alternative call
        try:
            results = MODEL.predict(str(tmp_path), imgsz=640, conf=0.25, device="cpu")
        except Exception as ex:
            tb = traceback.format_exc()
            return jsonify({"error":"model inference error", "details": str(ex), "trace": tb}), 500

    vehicle_count = 0
    out_url = None
    try:
        if len(results) > 0:
            r = results[0]
            # robust count
            try:
                vehicle_count = int(len(r.boxes))
            except Exception:
                try:
                    vehicle_count = int(r.boxes.data.shape[0])
                except Exception:
                    vehicle_count = 0

            # annotate (plot)
            try:
                arr = r.plot()
                img = Image.fromarray(arr)
                out_name = f"annotated_{int(time.time()*1000)}.jpg"
                out_path = STATIC_RESULTS / out_name
                img.save(out_path, quality=90)
                out_url = "/static/results/" + out_name
            except Exception as e:
                print("[backend] annotated save failed:", e)
                out_url = latest_annotated_url()
    except Exception as e:
        print("[backend] result parsing error:", e)

    # cleanup tmp file
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    if out_url is None:
        out_url = latest_annotated_url()

    # heuristic risk
    risk = "low" if vehicle_count < 15 else "medium" if vehicle_count < 40 else "high"
    return jsonify({"vehicle_count": vehicle_count, "risk_level": risk, "output_image": out_url})

# GET /api/demo_detect?img=<rel_path> -> run detection on dataset image directly (no upload)
@app.route("/api/demo_detect", methods=["GET"])
def api_demo_detect():
    if not ULTRALYTICS_AVAILABLE:
        return jsonify({"error":"ultralytics not installed"}), 500
    if not MODEL_LOADED:
        return jsonify({"error":"model not loaded", "details": MODEL_ERR}), 500

    rel = request.args.get("img", "")
    if not rel:
        # if no image provided, try a sample inside dataset: first val image found
        imgs = []
        for split in ("val","test","train"):
            d = DATASET_DIR / split / "images"
            if d.exists():
                imgs = sorted([p for p in d.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
                if imgs:
                    rel = str(imgs[0].relative_to(DATASET_DIR)).replace("\\","/")
                    break
    safe = safe_dataset_image_path(rel)
    if not safe or not safe.exists():
        return jsonify({"error":"demo image not found", "requested": rel}), 404

    try:
        results = MODEL(str(safe), imgsz=640, conf=0.25, device="cpu")
    except Exception as e:
        try:
            results = MODEL.predict(str(safe), imgsz=640, conf=0.25, device="cpu")
        except Exception as ex:
            tb = traceback.format_exc()
            return jsonify({"error":"inference error", "details": str(ex), "trace": tb}), 500

    vehicle_count = 0
    out_url = None
    if len(results) > 0:
        r = results[0]
        try:
            vehicle_count = int(len(r.boxes))
        except Exception:
            vehicle_count = 0
        # create annotated image
        try:
            arr = r.plot()
            img = Image.fromarray(arr)
            out_name = f"annotated_demo_{int(time.time()*1000)}.jpg"
            out_path = STATIC_RESULTS / out_name
            img.save(out_path, quality=90)
            out_url = "/static/results/" + out_name
        except Exception as e:
            print("[backend] demo annotated save failed:", e)
            out_url = latest_annotated_url()

    if out_url is None:
        out_url = latest_annotated_url()
    risk = "low" if vehicle_count < 15 else "medium" if vehicle_count < 40 else "high"
    return jsonify({"vehicle_count": vehicle_count, "risk_level": risk, "output_image": out_url})

# IoT endpoint (same as earlier)
@app.route("/api/iot", methods=["GET","POST"])
def api_iot():
    try:
        data = request.get_json(force=True, silent=True) or {}
        dist = data.get("distance") if data else request.args.get("distance")
        try:
            d = float(dist) if dist is not None else 100.0
        except Exception:
            d = 100.0
        if d < 0:
            d = 100.0
        if d < 20:
            alert = "DANGER"
            action = "EMERGENCY_STOP"
        elif d < 50:
            alert = "WARN"
            action = "slow_down"
        else:
            alert = "SAFE"
            action = "none"
        return jsonify({"distance": int(d), "alert": alert, "recommended_action": action})
    except Exception as e:
        return jsonify({"error":"iot error", "details": str(e)}), 500

@app.route("/api/latest_annotated")
def api_latest_annotated():
    url = latest_annotated_url()
    return jsonify({"latest_annotated": url})

@app.route("/api/metrics")
def api_metrics():
    """
    Return links to results files if present, and basic metrics.json if present.
    """
    metrics_file = STATIC_RESULTS / "metrics.json"
    metrics = None
    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf8"))
        except Exception:
            metrics = None
    def safe_rel(path: Path):
        return f"/static/results/{path.name}" if path.exists() else None
    return jsonify({
        "metrics": metrics,
        "labels": safe_rel(STATIC_RESULTS / "labels.jpg"),
        "confusion_matrix": safe_rel(STATIC_RESULTS / "confusion_matrix.png"),
        "results_plot": safe_rel(STATIC_RESULTS / "results.png")
    })

# API to serve your uploaded project PDF (developer note: path provided)
@app.route("/docs/capstone_pdf")
def serve_capstone_pdf():
    pdf_path = Path("/mnt/data/Deeplearning_Capstone_22000381&22000401.pdf")
    if pdf_path.exists():
        return send_file(str(pdf_path), mimetype="application/pdf")
    return jsonify({"error":"pdf not found"}), 404

# ============= Run server =============
if __name__ == "__main__":
    port = 8501
    print(f"[backend] Starting backend. Model loaded: {MODEL_LOADED}, model_path: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)