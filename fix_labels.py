import os, glob

root = r"D:\Dhruhi Nuv\College\Tivaan_Vision\DroneVehiclesDatasetYOLO"
count_fixed = 0
count_deleted = 0

for subset in ["train", "val", "test"]:
    label_dir = os.path.join(root, subset, "labels")
    for path in glob.glob(os.path.join(label_dir, "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls, x, y, w, h = map(float, parts)
                cls = 0.0  # force single class
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue
                new_lines.append(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            except Exception:
                continue

        if new_lines:
            with open(path, "w") as f:
                f.write("\n".join(new_lines))
            count_fixed += 1
        else:
            os.remove(path)
            count_deleted += 1

print(f"✅ Labels fixed: {count_fixed}, Deleted: {count_deleted}")
