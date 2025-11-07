# models/fix_labels_robust.py
"""
Robust label fixer for YOLO datasets.

What it does:
- For each subset (train/val/test) it pairs label .txt files with their images.
- Repairs common issues:
  - non-integer class names -> set to 0
  - class ids != 0 (for single-class dataset) -> set to 0
  - pixel coordinates -> normalize using image size
  - bbox given as x1,y1,x2,y2 (pixels) -> converts to center,w,h normalized
  - wrong token counts handled where possible
  - removes completely broken files
- Writes fixed labels with 6 decimal places.
- Produces a summary report file fix_report.txt in project root.
"""
from pathlib import Path
from PIL import Image
import math

ROOT = Path("D:/Dhruhi Nuv/College/Tivaan_Vision")
DATASET = ROOT / "DroneVehiclesDatasetYOLO"
SUBSETS = ["train", "val", "test"]
REPORT = ROOT / "fix_report.txt"

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def convert_xywh_pixels_to_norm(x, y, w, h, iw, ih):
    # x,y may be either top-left (x,y) with w,h OR center-based; if w and h are large values, we treat as pixels
    cx = x + w/2.0
    cy = y + h/2.0
    return cx/iw, cy/ih, w/iw, h/ih

def convert_xyxy_to_xywh_norm(x1, y1, x2, y2, iw, ih):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return cx/iw, cy/ih, w/iw, h/ih

def fix_label_file(txt_path, img_path, nc=1):
    """
    Return: (fixed_lines_list, status_msg)
    fixed_lines_list: list of strings to write (or empty)
    status_msg: explanation (fixed / deleted / skipped)
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if text == "":
        return ([], "empty_deleted")

    lines = text.splitlines()
    fixed_lines = []
    iw, ih = None, None
    if img_path.exists():
        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
        except:
            iw, ih = None, None

    for li, line in enumerate(lines):
        orig = line.strip()
        if orig == "":
            continue
        parts = orig.replace(",", " ").split()
        # handle classname text
        # if first token is non-numeric, set class = 0
        cls_token = parts[0]
        cls_val = safe_float(cls_token)
        if cls_val is None:
            cls = 0
        else:
            # if float, cast to int
            cls = int(cls_val)

        # for single class project, ensure class is 0
        if nc == 1:
            cls = 0

        coords = parts[1:]
        # if there are more than 4 coords, attempt to trim to 4
        if len(coords) > 4:
            coords = coords[:4]
        # if fewer than 4 coords, cannot fix reliably (skip)
        if len(coords) < 4:
            # try to salvage if there are 3 coords and all <=1 (maybe missing width/height)
            # but safest is to skip this line
            continue

        # parse floats
        coords_f = [safe_float(c) for c in coords]
        if any(v is None for v in coords_f):
            # cannot parse -> skip
            continue

        # coords_f now has 4 floats: c0,c1,c2,c3
        a,b,c,d = coords_f

        # If image size available and any coordinate >1 -> treat as pixel format (various possibilities)
        if iw and ih and (a > 1 or b > 1 or c > 1 or d > 1):
            # Two common pixel formats:
            # 1) x_center_px, y_center_px, w_px, h_px
            #    detect: w_px or h_px > 1 and x_center_px < iw and y_center_px < ih
            # 2) x1_px, y1_px, x2_px, y2_px
            #    detect: (c > a) and (d > b) and c<=iw and d<=ih
            if (c > 1 and d > 1) and (a <= iw and b <= ih):
                # Guess case 1: center_x, center_y, w_px, h_px
                cx_px, cy_px, w_px, h_px = a,b,c,d
                cxn, cyn, wn, hn = (cx_px/iw, cy_px/ih, w_px/iw, h_px/ih)
            elif (c > a and d > b) and (c <= iw and d <= ih):
                # case 2: x1,y1,x2,y2
                cxn, cyn, wn, hn = convert_xyxy_to_xywh_norm(a,b,c,d, iw, ih)
            else:
                # fallback: treat as center,w,h in pixels
                cxn, cyn, wn, hn = convert_xywh_pixels_to_norm(a,b,c,d, iw, ih)
        else:
            # assume coords are already normalized center x,y,w,h in [0,1]
            cxn, cyn, wn, hn = a, b, c, d

        # validate normalized numbers
        if not (0 <= cxn <= 1 and 0 <= cyn <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
            # attempt to clamp to [1e-6, 1]
            cxn = min(max(cxn, 1e-6), 1.0)
            cyn = min(max(cyn, 1e-6), 1.0)
            wn = min(max(wn, 1e-6), 1.0)
            hn = min(max(hn, 1e-6), 1.0)

        fixed_lines.append(f"{cls} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f}")

    if fixed_lines:
        return (fixed_lines, "fixed")
    else:
        return ([], "deleted")

def main():
    total_fixed = 0
    total_deleted = 0
    total_checked = 0
    report_lines = []
    for subset in SUBSETS:
        label_dir = DATASET / subset / "labels"
        img_dir = DATASET / subset / "images"
        if not label_dir.exists():
            report_lines.append(f"Missing label dir: {label_dir}")
            continue
        for txt in label_dir.glob("*.txt"):
            total_checked += 1
            # find corresponding image (try common extensions)
            base = txt.stem
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
                cand = img_dir / (base + ext)
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                # try without suffix modifications (some datasets have different naming) - try any file starting with stem
                matches = list(img_dir.glob(base + "*"))
                if matches:
                    img_path = matches[0]
            fixed_lines, status = fix_label_file(txt, img_path)
            if status == "fixed":
                txt.write_text("\n".join(fixed_lines))
                total_fixed += 1
            elif status == "deleted":
                try:
                    txt.unlink()
                    total_deleted += 1
                except Exception:
                    pass
            report_lines.append(f"{subset} {txt.name} -> {status} (img={'yes' if img_path else 'no'})")

    summary = [
        f"Dataset fix report",
        f"Total label files checked: {total_checked}",
        f"Total fixed files (rewritten): {total_fixed}",
        f"Total deleted files (unfixable): {total_deleted}",
        ""
    ]
    summary.extend(report_lines[:200])  # include first 200 details to keep report reasonable
    REPORT.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary[:10]))
    print(f"\nWrote detailed report to {REPORT.resolve()}")
    print("✅ Done. Now run verify_labels.py again to confirm.")

if __name__ == "__main__":
    main()
