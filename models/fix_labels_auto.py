from pathlib import Path

# Update this path for your dataset
DATASET = Path("D:/Dhruhi Nuv/College/Tivaan_Vision/DroneVehiclesDatasetYOLO")

def normalize(x, y, w, h, img_w, img_h):
    """Convert absolute coords to normalized YOLO format"""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def try_fix_line(parts, img_w=1280, img_h=720):
    """Try to fix a malformed label line."""
    # Case 1: if has 4 values → assume missing class, add class 0
    if len(parts) == 4:
        parts = ["0"] + parts
    # Case 2: if has >5 values, take first 5 only
    if len(parts) > 5:
        parts = parts[:5]
    # Case 3: if coordinates look like >1, assume they’re pixel coords
    try:
        cls, *coords = map(float, parts)
        if any(c > 1 for c in coords):
            # convert from absolute to normalized
            x, y, w, h = coords
            coords = normalize(x, y, w, h, img_w, img_h)
        return f"{int(cls)} " + " ".join(f"{c:.6f}" for c in coords)
    except:
        return None

def fix_labels():
    subsets = ["train", "val", "test"]
    fixed, deleted = 0, 0
    for subset in subsets:
        label_dir = DATASET / subset / "labels"
        for txt in label_dir.glob("*.txt"):
            lines = txt.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.strip().replace(",", " ").split()
                fixed_line = try_fix_line(parts)
                if fixed_line:
                    new_lines.append(fixed_line)
            if new_lines:
                txt.write_text("\n".join(new_lines))
                fixed += 1
            else:
                txt.unlink()  # delete bad file
                deleted += 1
    print(f"✅ Fixed {fixed} label files, deleted {deleted} broken ones.")

if __name__ == "__main__":
    fix_labels()
