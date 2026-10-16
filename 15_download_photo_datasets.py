"""
15_download_photo_datasets.py
Downloads and sets up:
1. AlphaDent dataset (DSLR photos with 9 pathology/restoration classes)
   - Code & Pretrained YOLOv8-seg weights from https://github.com/ZFTurbo/AlphaDent
   - Dataset archive from Hugging Face / GitHub release v1.1
2. DentalAI dataset (Intraoral photos covering Tooth, Caries, Cavity, Crack)
   - Downloaded via kagglehub (pawanvalluri/dental-segmentation)
"""

import os
import sys
import json
import zipfile
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WEIGHTS_DIR = BASE_DIR / "alphadent_weights"
DATA_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

WEIGHTS_URL_640 = "https://github.com/ZFTurbo/AlphaDent/releases/download/v1.0/yolov8x_AlphaDent_9_classes_640px.pt"
ALPHADENT_ZIP_URL = "https://huggingface.co/datasets/ZFTurbo/AlphaDent/resolve/main/AlphaDent.zip"
DENTALAI_KAGGLE_HANDLE = "pawanvalluri/dental-segmentation"

def setup_weights():
    target_pt = WEIGHTS_DIR / "yolov8x_AlphaDent_9_classes_640px.pt"
    if target_pt.exists() and target_pt.stat().st_size > 100 * 1024 * 1024:
        print(f"AlphaDent pretrained weights already present at: {target_pt} ({target_pt.stat().st_size / 1e6:.1f} MB)")
        return target_pt

    print(f"Downloading AlphaDent YOLOv8x weights from {WEIGHTS_URL_640}...")
    cmd = ["curl", "-L", "-o", str(target_pt), WEIGHTS_URL_640]
    subprocess.run(cmd, check=True)
    print(f"Weights downloaded to: {target_pt}")
    return target_pt

def setup_dentalai():
    print(f"Checking/Downloading DentalAI via kagglehub ({DENTALAI_KAGGLE_HANDLE})...")
    import kagglehub
    path = kagglehub.dataset_download(DENTALAI_KAGGLE_HANDLE)
    print(f"DentalAI dataset downloaded to: {path}")

    # Symlink or note path in data/dentalai
    target_link = DATA_DIR / "dentalai"
    if not target_link.exists():
        try:
            target_link.symlink_to(path)
            print(f"Created symlink {target_link} -> {path}")
        except Exception as e:
            print(f"Symlink failed ({e}), using direct path: {path}")
    return str(path)

def setup_alphadent():
    extract_dir = DATA_DIR / "alphadent"
    zip_path = DATA_DIR / "AlphaDent.zip"

    if (extract_dir / "images").exists() or (extract_dir / "AlphaDent" / "images").exists():
        print(f"AlphaDent dataset already extracted at: {extract_dir}")
        return str(extract_dir)

    if zip_path.exists() and not (DATA_DIR / "AlphaDent.zip.aria2").exists():
        print(f"Extracting {zip_path} to {extract_dir}...")
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("AlphaDent extracted successfully.")
        return str(extract_dir)
    else:
        print(f"AlphaDent zip is still downloading or not yet unpacked.")
        print(f"Target zip path: {zip_path}")
        return str(extract_dir)

def main():
    print("=== Step 1: Photo Datasets Setup ===")
    weights_path = setup_weights()
    dentalai_path = setup_dentalai()
    alphadent_path = setup_alphadent()

    summary = {
        "alphadent_weights": str(weights_path),
        "dentalai_dir": str(dentalai_path),
        "alphadent_dir": str(alphadent_path)
    }

    with open(BASE_DIR / "photo_datasets_paths.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset paths saved to photo_datasets_paths.json:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
