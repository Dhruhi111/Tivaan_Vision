# models/verify_dataset_paths.py
from pathlib import Path
import yaml

ROOT = Path("D:/Dhruhi Nuv/College/Tivaan_Vision")
DATA_YAML_ROOT = ROOT / "data.yaml"
DATASET_FOLDER = ROOT / "DroneVehiclesDatasetYOLO"

def print_exists(p):
    print(f"{p} -> {'EXISTS' if Path(p).exists() else 'MISSING'}")

def load_and_check_yaml(yaml_path):
    print("Loading YAML:", yaml_path)
    with open(yaml_path, 'r') as f:
        d = yaml.safe_load(f)
    print("YAML content:", d)
    # ensure absolute paths
    for k in ['train','val','test']:
        if k in d:
            p = Path(d[k])
            print_exists(p)
    return d

def check_dataset_folder():
    print("Checking dataset folder:", DATASET_FOLDER)
    print_exists(DATASET_FOLDER)
    for sub in ["train","val","test"]:
        img_dir = DATASET_FOLDER / sub / "images"
        lbl_dir = DATASET_FOLDER / sub / "labels"
        print(f"\n{sub.upper()} images:", img_dir)
        print_exists(img_dir)
        print(f"{sub.upper()} labels:", lbl_dir)
        print_exists(lbl_dir)

if __name__ == "__main__":
    if DATA_YAML_ROOT.exists():
        d = load_and_check_yaml(DATA_YAML_ROOT)
    else:
        print("Root data.yaml not found at:", DATA_YAML_ROOT)
    print("\n--- Dataset folder checks ---")
    check_dataset_folder()
    print("\nIf any path is MISSING, fix the path or move files accordingly.")
