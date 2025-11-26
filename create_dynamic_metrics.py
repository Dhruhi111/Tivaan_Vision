# create_dynamic_metrics.py
"""
Create a detailed per-image metrics JSON using the most-recent annotated image
or using the explicit --image argument.

Behavior:
 - If --image is provided, use that image (absolute or relative path).
 - Else, read static/results/last_detection.json (if present) and the latest
   annotated_*.jpg in static/results.
 - If latest annotated image mtime > last_detection timestamp (or timestamp missing),
   and ultralytics is available, re-run the model ON THAT IMAGE to produce fresh detections.
 - Writes static/results/metrics_detailed.json and static/results/metrics_summary_small.png
"""

import json
import time
import argparse
import math
from pathlib import Path
from datetime import datetime
from statistics import mean, median, pstdev

# Optional image & plotting libs
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Optional ultralytics (only used if available and we need to re-run model)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

BASE = Path(__file__).resolve().parent
STATIC_RESULTS = BASE / "static" / "results"
STATIC_RESULTS.mkdir(parents=True, exist_ok=True)

LAST_DET = STATIC_RESULTS / "last_detection.json"
METRICS_DETAILED = STATIC_RESULTS / "metrics_detailed.json"
SUMMARY_SMALL = STATIC_RESULTS / "metrics_summary_small.png"

# ---------- Helpers ----------
def read_json_safe(p: Path):
    if not p.exists(): 
        return None
    try:
        return json.loads(p.read_text(encoding="utf8"))
    except Exception as e:
        print("⚠️ Failed to parse JSON:", p, e)
        return None

def latest_annotated_file():
    files = sorted(STATIC_RESULTS.glob("annotated_*.jpg"), key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None

def ts_from_last_detection(d):
    # Accept various timestamp fields
    if not isinstance(d, dict):
        return None
    for k in ("timestamp", "time", "ts", "created_at"):
        v = d.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            try:
                # maybe ISO format
                dt = datetime.fromisoformat(v.replace("Z",""))
                return int(dt.timestamp())
            except Exception:
                pass
    return None

def bbox_area(b):
    x1,y1,x2,y2 = b
    w = max(0.0, x2-x1)
    h = max(0.0, y2-y1)
    return w*h

def centroid(b):
    x1,y1,x2,y2 = b
    return [(x1+x2)/2.0, (y1+y2)/2.0]

def safe_number(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

# ---------- Model-run helper (defensive) ----------
def run_model_on_image(image_path, weights_path=None):
    """
    Run YOLO on image_path if ultralytics is available.
    Returns list of detections: dicts with bbox [x1,y1,x2,y2], conf, cls.
    """
    if not ULTRALYTICS_AVAILABLE:
        print("ℹ️ ultralytics not available; skipping model run.")
        return None
    try:
        model = YOLO(str(weights_path)) if weights_path else YOLO()
        results = model(str(image_path), imgsz=640, conf=0.25, device="cpu")
        if len(results) == 0:
            return []
        r = results[0]
        detections = []
        # Try robust access to boxes
        try:
            # new-style API
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else [0]*len(boxes)
        except Exception:
            try:
                data = r.boxes.data.cpu().numpy()  # fallback layout
                boxes = data[:, :4]
                confs = data[:, 4]
                clss = data[:, 5].astype(int) if data.shape[1] > 5 else [0]*len(boxes)
            except Exception as ex:
                print("⚠️ Unable to parse r.boxes:", ex)
                return []
        for i,b in enumerate(boxes):
            x1,y1,x2,y2 = [float(x) for x in b]
            conf = float(confs[i]) if i < len(confs) else 0.0
            cls = int(clss[i]) if i < len(clss) else 0
            detections.append({"bbox":[x1,y1,x2,y2], "conf": round(conf,4), "cls": int(cls)})
        return detections
    except Exception as e:
        print("❌ Model run failed:", e)
        return None

# ---------- Build metrics ----------
def build_for_image(image_path: Path, last_detection: dict = None, model_weights=None):
    """
    Build detailed metrics for this image path.
    If last_detection is provided and includes detections, we'll prefer that
    (but if it is stale and model can run, we prefer re-running).
    """
    # Try to get detections list from last_detection (if it points to this image)
    detections = None
    if isinstance(last_detection, dict):
        # If last_detection references this image explicitly, and contains 'detections', use it.
        od = last_detection.get("output_image") or last_detection.get("image_name") or last_detection.get("image_path")
        if od:
            od_name = Path(od).name
            if od_name == image_path.name and "detections" in last_detection and isinstance(last_detection["detections"], list):
                detections = last_detection["detections"]

    # If no detections or detections are stale, try to run model on the image
    if (not detections or len(detections)==0) and ULTRALYTICS_AVAILABLE:
        print("ℹ️ Running model on image (fresh) ->", image_path)
        run_dets = run_model_on_image(image_path, weights_path=model_weights)
        if run_dets is not None:
            detections = run_dets

    # Normalize detections
    norm = []
    for d in detections or []:
        bbox=None; conf=None; cls=0
        if isinstance(d, dict):
            bbox = d.get("bbox") or d.get("xyxy")
            conf = safe_number(d.get("conf") or d.get("confidence") or d.get("score"), 0.0)
            cls = int(d.get("cls", d.get("class",0)))
        elif isinstance(d, (list,tuple)):
            if len(d) >= 4:
                bbox = list(map(float, d[:4]))
            if len(d) >= 5:
                conf = safe_number(d[4], 0.0)
            if len(d) >= 6:
                cls = int(d[5])
        if bbox is None:
            continue
        area = bbox_area(bbox)
        cx, cy = centroid(bbox)
        norm.append({
            "bbox":[round(float(x),2) for x in bbox],
            "conf": round(conf if conf is not None else 0.0,4),
            "cls": int(cls),
            "area_px": round(area,2),
            "center":[round(cx,2), round(cy,2)]
        })

    vehicle_count = None
    if last_detection and isinstance(last_detection, dict):
        # prefer explicit vehicle_count if matches this image
        od = last_detection.get("output_image") or last_detection.get("image_name") or last_detection.get("image_path")
        if od and Path(od).name == image_path.name:
            vc = last_detection.get("vehicle_count")
            try:
                vehicle_count = int(vc) if vc is not None else None
            except Exception:
                vehicle_count = None

    if vehicle_count is None:
        # fallback to current detection length
        vehicle_count = len(norm)

    # confidence stats
    confs = [x["conf"] for x in norm if x.get("conf") is not None]
    conf_stats = {}
    if confs:
        conf_stats["mean_conf"] = round(mean(confs),4)
        conf_stats["median_conf"] = round(median(confs),4)
        conf_stats["std_conf"] = round(pstdev(confs),4) if len(confs) > 1 else 0.0
        conf_stats["min_conf"] = round(min(confs),4)
        conf_stats["max_conf"] = round(max(confs),4)
    else:
        conf_stats.update({"mean_conf":None,"median_conf":None,"std_conf":None,"min_conf":None,"max_conf":None})

    # image size & density
    image_size = None
    density = None
    if Image and image_path.exists():
        try:
            with Image.open(image_path) as im:
                w,h = im.size
                image_size = {"width": w, "height": h}
                if w*h > 0:
                    density = round((vehicle_count / (w*h)) * 100000, 4)
        except Exception:
            image_size = None

    # risk heuristics (same thresholds as your backend)
    vc_int = vehicle_count if isinstance(vehicle_count, int) else None
    if isinstance(vc_int, int):
        risk = "low" if vc_int < 15 else "medium" if vc_int < 40 else "high"
        if vc_int < 15:
            recommended = {"action":"none","message":"Monitor. No immediate action required."}
        elif vc_int < 40:
            recommended = {"action":"slow_down","message":"Moderate congestion — advise slow down / caution."}
        else:
            recommended = {"action":"reroute_or_stop","message":"High density — suggest reroute or emergency stop."}
    else:
        risk = "unknown"
        recommended = {"action":"no_data","message":"Insufficient data"}

    # Estimated metrics (heuristic - no ground truth)
    mean_conf = conf_stats.get("mean_conf")
    precision_est = round(mean_conf,3) if mean_conf is not None else None
    recall_est = None
    if isinstance(vc_int,int):
        recall_est = round(max(0.3, min(0.99, 0.9 - (vc_int/1000.0))), 3)
    mAP_est = round((precision_est + recall_est)/2.0,3) if (precision_est is not None and recall_est is not None) else None

    out = {
        "generated_at": int(time.time()),
        "generated_iso": datetime.utcfromtimestamp(time.time()).isoformat()+"Z",
        "image": {
            "name": image_path.name,
            "relative": "/static/results/" + image_path.name if str(image_path).startswith(str(STATIC_RESULTS)) else str(image_path)
        },
        "vehicle_count": int(vehicle_count) if isinstance(vehicle_count,(int,float)) else vehicle_count,
        "detections": norm,
        "confidence_stats": conf_stats,
        "image_size": image_size,
        "density_per_100k_px": density,
        "estimated_metrics": {
            "precision": precision_est,
            "recall": recall_est,
            "mAP": mAP_est
        },
        "risk_level": risk,
        "recommended_iot_action": recommended,
        "source_last_detection": last_detection if isinstance(last_detection, dict) else None
    }

    # Save json
    try:
        METRICS_DETAILED.write_text(json.dumps(out, indent=2), encoding="utf8")
        print("✅ Wrote:", METRICS_DETAILED)
    except Exception as e:
        print("❌ Failed to write metrics JSON:", e)

    # Small visual summary
    try:
        if plt:
            vals = [out["estimated_metrics"].get(k) or 0.0 for k in ("precision","recall","mAP")]
            labels = ["Precision","Recall","mAP"]
            plt.close("all")
            fig, ax = plt.subplots(figsize=(4,2.2), dpi=120)
            ax.bar(labels, vals, color=["#2777ff","#1f77b4","#2ca02c"])
            ax.set_ylim(0,1)
            ax.set_title("Est. Metrics")
            for i,v in enumerate(vals):
                ax.text(i, v+0.02, f"{v:.3f}" if v else "—", ha="center", fontsize=8)
            plt.tight_layout()
            fig.savefig(SUMMARY_SMALL, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            print("✅ Saved summary:", SUMMARY_SMALL.name)
        elif Image:
            w,h = 640,200
            im = Image.new("RGB",(w,h),(18,24,33))
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.text((12,12), f"Vehicles: {out['vehicle_count']}  Risk: {out['risk_level']}", fill=(255,255,255), font=font)
            im.save(SUMMARY_SMALL)
            print("✅ Saved small summary (Pillow)")
    except Exception as e:
        print("⚠️ Failed to create summary image:", e)

    return out

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Create detailed metrics for last detection / given image")
    parser.add_argument("--image", "-i", help="Absolute or relative image path to analyze (overrides last_detection)", default=None)
    parser.add_argument("--weights", "-w", help="Optional weights path for model run", default=None)
    args = parser.parse_args()

    last = read_json_safe(LAST_DET) if LAST_DET.exists() else None
    latest_ann = latest_annotated_file()

    image_to_use = None

    # If CLI image provided, use that
    if args.image:
        image_to_use = Path(args.image)
        if not image_to_use.exists():
            print("❌ Provided image does not exist:", image_to_use)
            return
    else:
        # If last_detection exists and references an image, prefer it
        if last and isinstance(last, dict):
            od = last.get("output_image") or last.get("image_name") or last.get("image_path")
            if od:
                # if it's a static results reference, build local path
                if str(od).startswith("/static/"):
                    p = STATIC_RESULTS / Path(od).name
                else:
                    p = Path(od)
                if p.exists():
                    image_to_use = p
                else:
                    image_to_use = None

        # If we don't have an image from last_detection, use latest annotated
        if image_to_use is None and latest_ann:
            image_to_use = latest_ann

        # If both exist, check timestamps and possibly re-run on latest_ann if it's newer
        if image_to_use and latest_ann and image_to_use.exists() and latest_ann.exists():
            last_ts = ts_from_last_detection(last) or 0
            ann_ts = int(latest_ann.stat().st_mtime)
            # If annotated image is newer than last_detection timestamp -> use latest_ann and prefer re-run
            if ann_ts > last_ts:
                print("ℹ️ Latest annotated image is newer than last_detection. Using annotated:", latest_ann.name)
                image_to_use = latest_ann

    if image_to_use is None:
        print("❌ No image to analyze. Place an annotated_*.jpg in static/results or pass --image.")
        return

    # Build metrics
    result = build_for_image(image_to_use, last_detection=last, model_weights=args.weights)
    if result:
        print("✅ Metrics created for:", image_to_use.name)
        print("-> metrics_detailed.json: /static/results/metrics_detailed.json")
    else:
        print("❌ Failed to produce metrics.")

if __name__ == "__main__":
    main()
