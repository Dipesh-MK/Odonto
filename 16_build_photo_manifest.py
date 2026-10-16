"""
16_build_photo_manifest.py
Builds the unified manifest CSV for photo datasets:
1. AlphaDent (Caries 1-6, Abrasion; tags Filling/Crown separately as non-pathology)
2. DentalAI (Tooth, Caries, Cavity, and especially Crack)

Columns:
- image_path: absolute path to the intraoral photo
- mask_path: absolute path to the ground truth mask or label file
- source_dataset: 'AlphaDent' or 'DentalAI'
- anomaly_class: specific class name(s)
- is_pathology: boolean flag (False for restorations like Filling/Crown or healthy Tooth)
- split: train / valid / test

Also includes Step 1 validation:
Loads 10 random images with their masks, overlays the mask semi-transparently,
and saves PNGs to validation_step1/ for visual inspection.
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from mask_utils import yolo_poly_to_mask, overlay_mask_on_image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VALIDATION_DIR = BASE_DIR / "validation_step1"
VALIDATION_DIR.mkdir(exist_ok=True)

# AlphaDent 9 classes
ALPHADENT_CLASSES = {
    0: 'Abrasion',
    1: 'Filling',
    2: 'Crown',
    3: 'Caries 1 class',
    4: 'Caries 2 class',
    5: 'Caries 3 class',
    6: 'Caries 4 class',
    7: 'Caries 5 class',
    8: 'Caries 6 class'
}

DENTALAI_CLASSES = {
    0: 'background',
    1: 'Caries',
    2: 'Cavity',
    3: 'Crack',
    4: 'Tooth'
}

def parse_alphadent():
    records = []
    alphadent_dir = DATA_DIR / "alphadent"
    if not alphadent_dir.exists():
        print("AlphaDent directory does not exist yet.")
        return records

    # Search for images in train/valid/test
    img_patterns = [
        str(alphadent_dir / "**" / "*.jpg"),
        str(alphadent_dir / "**" / "*.png")
    ]
    all_images = []
    for pat in img_patterns:
        all_images.extend(glob.glob(pat, recursive=True))

    # Filter out anything that is inside a labels or annotations folder
    image_paths = [p for p in all_images if "/images/" in p and not p.endswith("_mask.png")]
    print(f"AlphaDent images located: {len(image_paths)}")

    for img_path in sorted(image_paths):
        split = "train"
        if "/valid/" in img_path:
            split = "valid"
        elif "/test/" in img_path:
            split = "test"

        # Corresponding label file
        label_path = img_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        
        classes_present = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in ALPHADENT_CLASSES:
                            classes_present.append(cls_id)
        
        unique_classes = sorted(list(set(classes_present)))
        if not unique_classes:
            # No annotations
            class_str = "None"
            is_pathology = False
        else:
            class_names = [ALPHADENT_CLASSES[c] for c in unique_classes]
            class_str = "; ".join(class_names)
            # Pathology check: Filling (1) and Crown (2) are restorations, not pathology
            pathology_ids = [c for c in unique_classes if c not in (1, 2)]
            is_pathology = len(pathology_ids) > 0

        records.append({
            "image_path": os.path.abspath(img_path),
            "mask_path": os.path.abspath(label_path) if os.path.exists(label_path) else "",
            "source_dataset": "AlphaDent",
            "anomaly_class": class_str,
            "is_pathology": is_pathology,
            "split": split
        })

    return records

def parse_dentalai():
    records = []
    dentalai_dir = DATA_DIR / "dentalai"
    if not dentalai_dir.exists():
        print("DentalAI directory does not exist yet.")
        return records

    for split in ["train", "valid", "test"]:
        split_dir = dentalai_dir / split
        if not split_dir.exists():
            continue
        images = sorted(glob.glob(str(split_dir / "*.jpg")))
        for img_path in images:
            mask_path = img_path.replace(".jpg", "_mask.png")
            if not os.path.exists(mask_path):
                continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            unique_vals = set(np.unique(mask))
            
            classes_present = []
            for v in [3, 1, 2, 4]:  # Prioritize Crack, then Caries, Cavity, Tooth
                if v in unique_vals:
                    classes_present.append(DENTALAI_CLASSES[v])
            
            if not classes_present:
                class_str = "background"
                is_pathology = False
            else:
                class_str = "; ".join(classes_present)
                # Pathology if Crack (3), Caries (1), or Cavity (2)
                is_pathology = any(v in unique_vals for v in [1, 2, 3])

            records.append({
                "image_path": os.path.abspath(img_path),
                "mask_path": os.path.abspath(mask_path),
                "source_dataset": "DentalAI",
                "anomaly_class": class_str,
                "is_pathology": is_pathology,
                "split": split
            })

    print(f"DentalAI images located: {len(records)}")
    return records

def run_validation_overlay(manifest_df, num_samples=10):
    """
    Selects 10 random pathology samples, renders masks on images,
    and saves to validation_step1/ for inspection.
    """
    print(f"\n--- Running Step 1 Validation: Overlaying {num_samples} sample masks ---")
    pathology_df = manifest_df[manifest_df["is_pathology"]].copy()
    if len(pathology_df) == 0:
        print("No pathology images found to validate.")
        return

    # Sample ensuring diverse classes
    np.random.seed(42)
    sample_indices = np.random.choice(len(pathology_df), size=min(num_samples, len(pathology_df)), replace=False)
    samples = pathology_df.iloc[sample_indices]

    saved_images = []
    for idx, (_, row) in enumerate(samples.iterrows()):
        img_bgr = cv2.imread(row["image_path"])
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        binary_mask = np.zeros((h, w), dtype=np.uint8)

        if row["source_dataset"] == "AlphaDent":
            # Parse YOLO format polygon
            if os.path.exists(row["mask_path"]):
                with open(row["mask_path"], "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) > 5:
                            cls_id = int(parts[0])
                            # Only render pathology masks (exclude Filling/Crown)
                            if cls_id not in (1, 2):
                                coords = [float(x) for x in parts[1:]]
                                poly_mask = yolo_poly_to_mask(coords, h, w)
                                binary_mask = np.maximum(binary_mask, poly_mask)
        elif row["source_dataset"] == "DentalAI":
            # Read mask png: 3 = Crack, 1 = Caries, 2 = Cavity
            m = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
            if m is not None:
                # Prioritize crack if present, else caries/cavity
                if 3 in m:
                    binary_mask = (m == 3).astype(np.uint8)
                else:
                    binary_mask = np.isin(m, [1, 2]).astype(np.uint8)

        # Overlay semi-transparent mask (Green)
        overlay = overlay_mask_on_image(img_bgr, binary_mask, color=(0, 255, 0), alpha=0.55)

        # Add text label
        label_text = f"[{row['source_dataset']}] {row['anomaly_class']}"
        cv2.putText(overlay, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        out_name = f"sample_{idx+1:02d}_{row['source_dataset']}_{row['anomaly_class'].replace(' ', '_').replace(';', '_')[:30]}.png"
        out_path = VALIDATION_DIR / out_name
        cv2.imwrite(str(out_path), overlay)
        saved_images.append(str(out_path))
        print(f"  [Sample {idx+1:02d}] {row['source_dataset']} | {row['anomaly_class']} | Mask pixels: {int(binary_mask.sum())} -> Saved to: {out_name}")

    print(f"Validation images successfully saved to: {VALIDATION_DIR}")
    return saved_images

def main():
    print("=== Step 1: Building Photo Manifest ===")
    records = []
    records.extend(parse_alphadent())
    records.extend(parse_dentalai())

    df = pd.DataFrame(records)
    out_csv = BASE_DIR / "photo_manifest.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nManifest saved to: {out_csv}")
    print(f"Total entries: {len(df)}")
    print("\nDataset breakdown:")
    print(df["source_dataset"].value_counts())
    print("\nPathology flag breakdown:")
    print(df.groupby(["source_dataset", "is_pathology"]).size())
    print("\nTop anomaly classes:")
    print(df["anomaly_class"].value_counts().head(15))

    # Run validation
    run_validation_overlay(df, num_samples=10)

if __name__ == "__main__":
    main()
