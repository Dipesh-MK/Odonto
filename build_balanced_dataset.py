"""
build_balanced_dataset.py

Constructs an oversampled, class-balanced dataset directory for YOLOv8-seg training:
- Preserves the EXACT patient-level train/val/test split from dataset_3class_seg.
- Oversamples the 67 Crack-containing training images by 8x (1 original + 7 copies).
- Copies/symlinks images and labels with unique names (e.g. {stem}_crackrep{i}.jpg / .txt).
- Leaves val and test splits 100% identical and untouched via symlinks.
- Generates dataset_3class_seg_balanced/data.yaml.
"""

import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "dataset_3class_seg"
DST_DIR = BASE_DIR / "dataset_3class_seg_balanced"

OVERSAMPLE_FACTOR = 8  # 1 original + 7 duplicates

def build():
    if DST_DIR.exists():
        print(f"Cleaning existing directory {DST_DIR} ...")
        shutil.rmtree(DST_DIR)

    for split in ["train", "val", "test"]:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Val and Test: Link identical files
    for split in ["val", "test"]:
        src_img_dir = SRC_DIR / "images" / split
        src_lbl_dir = SRC_DIR / "labels" / split
        for img_p in src_img_dir.glob("*.*"):
            dst_img = DST_DIR / "images" / split / img_p.name
            dst_lbl = DST_DIR / "labels" / split / f"{img_p.stem}.txt"
            os.symlink(img_p.resolve(), dst_img)
            src_lbl = src_lbl_dir / f"{img_p.stem}.txt"
            if src_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)

    print(f"Linked Val ({len(list((DST_DIR / 'images' / 'val').glob('*.*')))}) and Test ({len(list((DST_DIR / 'images' / 'test').glob('*.*')))}) images.")

    # 2. Train split: Link originals + duplicate crack images
    src_train_img = SRC_DIR / "images" / "train"
    src_train_lbl = SRC_DIR / "labels" / "train"

    train_images = sorted(list(src_train_img.glob("*.*")))
    crack_images = []
    normal_images = []

    for img_p in train_images:
        lbl_p = src_train_lbl / f"{img_p.stem}.txt"
        has_crack = False
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7 and int(parts[0]) == 2:
                        has_crack = True
                        break
        if has_crack:
            crack_images.append(img_p)
        else:
            normal_images.append(img_p)

    print(f"Found {len(crack_images)} crack-containing train images and {len(normal_images)} normal train images.")

    # Add all original train images
    for img_p in train_images:
        dst_img = DST_DIR / "images" / "train" / img_p.name
        dst_lbl = DST_DIR / "labels" / "train" / f"{img_p.stem}.txt"
        os.symlink(img_p.resolve(), dst_img)
        src_lbl = src_train_lbl / f"{img_p.stem}.txt"
        if src_lbl.exists():
            os.symlink(src_lbl.resolve(), dst_lbl)

    # Add oversampled copies for crack images
    for rep in range(1, OVERSAMPLE_FACTOR):
        for img_p in crack_images:
            rep_name = f"{img_p.stem}_crackrep{rep}{img_p.suffix}"
            rep_lbl = f"{img_p.stem}_crackrep{rep}.txt"
            dst_img = DST_DIR / "images" / "train" / rep_name
            dst_lbl = DST_DIR / "labels" / "train" / rep_lbl
            os.symlink(img_p.resolve(), dst_img)
            src_lbl = src_train_lbl / f"{img_p.stem}.txt"
            if src_lbl.exists():
                os.symlink(src_lbl.resolve(), dst_lbl)

    total_train = len(list((DST_DIR / "images" / "train").glob("*.*")))
    print(f"Total train images after {OVERSAMPLE_FACTOR}x crack oversampling: {total_train} (added {total_train - len(train_images)} copies)")

    # 3. Create data.yaml
    data_yaml_content = f"""# Odonto 3-Class Segmentation Dataset (Crack-Balanced 8x Oversampling)
path: {DST_DIR.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: Abrasion
  1: Caries
  2: Crack
"""
    with open(DST_DIR / "data.yaml", "w") as f:
        f.write(data_yaml_content)

    print(f"Saved dataset config to {DST_DIR / 'data.yaml'}")

if __name__ == "__main__":
    build()
