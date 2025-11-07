from pathlib import Path

DATASET = Path("D:/Dhruhi Nuv/College/Tivaan_Vision/DroneVehiclesDatasetYOLO")
subsets = ["train", "val", "test"]

def check_labels():
    bad_files = []
    for subset in subsets:
        label_dir = DATASET / subset / "labels"
        txt_files = list(label_dir.glob("*.txt"))
        print(f"\nChecking {subset} ({len(txt_files)} label files)")
        for txt in txt_files:
            try:
                lines = txt.read_text().strip().splitlines()
                if not lines:
                    bad_files.append((txt, "empty"))
                    continue
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        bad_files.append((txt, "wrong number of values"))
                        break
                    cls, x, y, w, h = map(float, parts)
                    if not (0 <= cls < 1):
                        # If cls is 1 but nc=1, this is wrong
                        bad_files.append((txt, f"invalid class {cls}"))
                        break
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        bad_files.append((txt, "non-normalized coords"))
                        break
            except Exception as e:
                bad_files.append((txt, f"error: {e}"))
    if bad_files:
        print("\n⚠️ Found problematic label files:")
        for f, msg in bad_files[:20]:  # show first 20
            print(f"  {f} → {msg}")
        print(f"Total bad files: {len(bad_files)}")
    else:
        print("\n✅ All label files look fine!")

if __name__ == "__main__":
    check_labels()
