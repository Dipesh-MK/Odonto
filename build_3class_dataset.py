"""
build_3class_dataset.py

Prepares unified 3-class dental pathology dataset for YOLOv8-seg fine-tuning:
- Class 0: Abrasion (from AlphaDent)
- Class 1: Caries (collapsed from AlphaDent Caries 1-6 + DentalAI Caries/Cavity)
- Class 2: Crack (from DentalAI)

Restorations (Filling, Crown) and healthy tooth annotations are strictly excluded.
A strict patient-level split (70% train / 15% val / 15% test) is enforced:
- AlphaDent: patient ID extracted from filename prefix (pXXX)
- DentalAI: patient ID extracted from case number prefix (<case_id>_jpg.rf...)
No patient appears in more than one split.
"""

import os
import re
import glob
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALPHADENT_DIR = DATA_DIR / "alphadent"
DENTALAI_DIR = DATA_DIR / "dentalai"

OUTPUT_DIR = BASE_DIR / "dataset_3class_seg"

# Target classes
CLASS_NAMES = ["Abrasion", "Caries", "Crack"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# AlphaDent 9 classes mapping:
# 0: Abrasion -> Target 0
# 1: Filling -> EXCLUDE
# 2: Crown -> EXCLUDE
# 3: Caries 1 class -> Target 1
# 4: Caries 2 class -> Target 1
# 5: Caries 3 class -> Target 1
# 6: Caries 4 class -> Target 1
# 7: Caries 5 class -> Target 1
# 8: Caries 6 class -> Target 1

ALPHADENT_CLASS_REMAP = {
    0: 0,  # Abrasion -> Abrasion
    3: 1,  # Caries 1 -> Caries
    4: 1,  # Caries 2 -> Caries
    5: 1,  # Caries 3 -> Caries
    6: 1,  # Caries 4 -> Caries
    7: 1,  # Caries 5 -> Caries
    8: 1,  # Caries 6 -> Caries
}

# DentalAI classes:
# 0: background -> EXCLUDE
# 1: Caries -> Target 1 (Caries)
# 2: Cavity -> Target 1 (Caries)
# 3: Crack -> Target 2 (Crack)
# 4: Tooth -> EXCLUDE

DENTALAI_CLASS_REMAP = {
    1: 1,  # Caries -> Caries
    2: 1,  # Cavity -> Caries
    3: 2,  # Crack -> Crack
}


def mask_to_yolo_polygons(binary_mask: np.ndarray, min_area: int = 15) -> list:
    """
    Converts binary mask to normalized YOLO polygon coordinates [x1, y1, x2, y2, ...].
    """
    h, w = binary_mask.shape[:2]
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        # Approximate contour for compact representation
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2)
        norm_coords = []
        for x, y in pts:
            norm_x = min(max(float(x) / w, 0.0), 1.0)
            norm_y = min(max(float(y) / h, 0.0), 1.0)
            norm_coords.extend([round(norm_x, 6), round(norm_y, 6)])
        if len(norm_coords) >= 6:
            polygons.append(norm_coords)
    return polygons


def collect_alphadent_samples():
    """Parses all labeled AlphaDent images and extracts patient IDs and remapped annotations."""
    samples = []
    # Train and valid have labels
    for split_folder in ["train", "valid"]:
        img_dir = ALPHADENT_DIR / "images" / split_folder
        lbl_dir = ALPHADENT_DIR / "labels" / split_folder
        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Extract patient ID from filename e.g. p001_F_51_001.jpg -> p001
            m = re.match(r"^(p\d+)", img_path.name)
            patient_id = m.group(1) if m else f"alpha_{img_path.stem}"

            # Parse and remap annotations
            remapped_lines = []
            has_pathology = False
            classes_present = set()

            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:  # class + at least 3 points (6 coords)
                        continue
                    src_cls = int(parts[0])
                    if src_cls in ALPHADENT_CLASS_REMAP:
                        tgt_cls = ALPHADENT_CLASS_REMAP[src_cls]
                        coords = parts[1:]
                        remapped_lines.append(f"{tgt_cls} " + " ".join(coords))
                        classes_present.add(tgt_cls)
                        has_pathology = True

            # If image had ONLY Filling/Crown or no pathology, exclude to prevent pure non-pathology pollution
            if has_pathology:
                samples.append({
                    "sample_id": f"alpha_{img_path.stem}",
                    "patient_id": patient_id,
                    "dataset": "AlphaDent",
                    "img_path": str(img_path),
                    "annotation_lines": remapped_lines,
                    "classes": classes_present,
                })

    print(f"AlphaDent valid pathology samples: {len(samples)} across {len(set(s['patient_id'] for s in samples))} patients")
    return samples


def collect_dentalai_samples():
    """Parses DentalAI images and _mask.png masks, extracting polygons for Crack and Caries."""
    samples = []
    for split_folder in ["train", "valid", "test"]:
        split_dir = DENTALAI_DIR / split_folder
        if not split_dir.exists():
            continue

        for img_path in sorted(split_dir.glob("*.jpg")):
            mask_path = split_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                continue

            # Extract case / patient ID from prefix (e.g. 1000_jpg.rf.xxx.jpg -> 1000)
            m = re.match(r"^(\d+)", img_path.name)
            patient_id = f"dentalai_{m.group(1)}" if m else f"dentalai_{img_path.stem}"

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue

            unique_vals = set(np.unique(mask))
            # Check for pathology: Crack (3) or Caries/Cavity (1, 2)
            pathology_vals = [v for v in unique_vals if v in DENTALAI_CLASS_REMAP]
            if not pathology_vals:
                continue

            remapped_lines = []
            classes_present = set()

            for src_val, tgt_cls in DENTALAI_CLASS_REMAP.items():
                if src_val in unique_vals:
                    bin_mask = (mask == src_val).astype(np.uint8) * 255
                    polygons = mask_to_yolo_polygons(bin_mask)
                    for poly in polygons:
                        coords_str = " ".join(f"{c:.6f}" for c in poly)
                        remapped_lines.append(f"{tgt_cls} {coords_str}")
                        classes_present.add(tgt_cls)

            if remapped_lines:
                samples.append({
                    "sample_id": f"dental_{img_path.stem}",
                    "patient_id": patient_id,
                    "dataset": "DentalAI",
                    "img_path": str(img_path),
                    "annotation_lines": remapped_lines,
                    "classes": classes_present,
                })

    print(f"DentalAI valid pathology samples: {len(samples)} across {len(set(s['patient_id'] for s in samples))} patients")
    return samples


def split_patients(all_samples, seed=42):
    """
    Splits samples at the patient level:
    - 70% Train
    - 15% Val
    - 15% Test
    Ensures Crack, Abrasion, and Caries instances are distributed into all splits.
    """
    np.random.seed(seed)
    patient_to_samples = {}
    for s in all_samples:
        pid = s["patient_id"]
        patient_to_samples.setdefault(pid, []).append(s)

    # Classify patients by dominant class to ensure stratified patient splitting
    crack_patients = []
    abrasion_patients = []
    caries_only_patients = []

    for pid, s_list in patient_to_samples.items():
        all_cls = set()
        for s in s_list:
            all_cls.update(s["classes"])
        if 2 in all_cls:  # Has Crack
            crack_patients.append(pid)
        elif 0 in all_cls:  # Has Abrasion
            abrasion_patients.append(pid)
        else:
            caries_only_patients.append(pid)

    def split_group(p_list):
        p_shuffled = np.random.permutation(p_list)
        n = len(p_shuffled)
        n_val = max(1, int(round(n * 0.15)))
        n_test = max(1, int(round(n * 0.15)))
        val_p = set(p_shuffled[:n_val])
        test_p = set(p_shuffled[n_val : n_val + n_test])
        train_p = set(p_shuffled[n_val + n_test :])
        return train_p, val_p, test_p

    tr_crack, val_crack, ts_crack = split_group(crack_patients)
    tr_abr, val_abr, ts_abr = split_group(abrasion_patients)
    tr_car, val_car, ts_car = split_group(caries_only_patients)

    train_pids = tr_crack | tr_abr | tr_car
    val_pids = val_crack | val_abr | val_car
    test_pids = ts_crack | ts_abr | ts_car

    # Verify zero overlap
    assert len(train_pids & val_pids) == 0, "Patient overlap between train and val!"
    assert len(train_pids & test_pids) == 0, "Patient overlap between train and test!"
    assert len(val_pids & test_pids) == 0, "Patient overlap between val and test!"

    split_map = {}
    for pid in train_pids:
        split_map[pid] = "train"
    for pid in val_pids:
        split_map[pid] = "val"
    for pid in test_pids:
        split_map[pid] = "test"

    for s in all_samples:
        s["split"] = split_map[s["patient_id"]]

    return all_samples


def build_yolo_dataset(all_samples):
    """Writes images and label files to dataset_3class_seg/."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": {0: 0, 1: 0, 2: 0}, "val": {0: 0, 1: 0, 2: 0}, "test": {0: 0, 1: 0, 2: 0}}
    img_counts = {"train": 0, "val": 0, "test": 0}
    patient_counts = {"train": set(), "val": set(), "test": set()}

    manifest_records = []

    for s in all_samples:
        split = s["split"]
        s_id = s["sample_id"]
        img_src = Path(s["img_path"])
        ext = img_src.suffix

        dst_img_name = f"{s_id}{ext}"
        dst_img_path = OUTPUT_DIR / "images" / split / dst_img_name
        dst_lbl_name = f"{s_id}.txt"
        dst_lbl_path = OUTPUT_DIR / "labels" / split / dst_lbl_name

        # Copy image or create symlink (copying avoids broken paths)
        shutil.copy2(str(img_src), str(dst_img_path))

        with open(dst_lbl_path, "w") as f:
            for line in s["annotation_lines"]:
                f.write(line.strip() + "\n")
                cls_id = int(line.split()[0])
                counts[split][cls_id] += 1

        img_counts[split] += 1
        patient_counts[split].add(s["patient_id"])

        manifest_records.append({
            "sample_id": s_id,
            "patient_id": s["patient_id"],
            "dataset": s["dataset"],
            "split": split,
            "classes": list(s["classes"]),
            "image_path": str(dst_img_path),
            "label_path": str(dst_lbl_path)
        })

    # Write data.yaml for YOLOv8
    yaml_content = f"""# Odonto Unified 3-Class Dental Pathology Dataset
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: Abrasion
  1: Caries
  2: Crack
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    # Save manifest
    pd.DataFrame(manifest_records).to_csv(OUTPUT_DIR / "dataset_manifest.csv", index=False)

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    for sp in ["train", "val", "test"]:
        p_cnt = len(patient_counts[sp])
        i_cnt = img_counts[sp]
        c_abr = counts[sp][0]
        c_car = counts[sp][1]
        c_crk = counts[sp][2]
        print(f"[{sp.upper():5s}] Patients: {p_cnt:3d} | Images: {i_cnt:4d} | Abrasion: {c_abr:4d} | Caries: {c_car:4d} | Crack: {c_crk:3d}")
    print("=" * 70)
    print(f"YOLO configuration written to: {OUTPUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    print("Starting dataset construction...")
    alpha_samples = collect_alphadent_samples()
    dental_samples = collect_dentalai_samples()
    total_samples = alpha_samples + dental_samples
    print(f"Total candidate samples: {len(total_samples)}")

    split_samples = split_patients(total_samples)
    build_yolo_dataset(split_samples)
