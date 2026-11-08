"""
eval_old_pretrained_baseline.py

Evaluates the original AlphaDent pretrained YOLOv8-seg weights
(alphadent_weights/yolov8x_AlphaDent_9_classes_640px.pt) on the newly constructed
held-out test set (dataset_3class_seg/images/test).

Evaluates:
- Abrasion (Class 0 in old model -> Class 0 in test set)
- Caries (Classes 3-8 in old model merged -> Class 1 in test set)
- Crack (Not present in old model -> Class 2 in test set, expected 0.00)

Computes:
- Per-class Mean IoU
- Overall Mean IoU across all test images
- Saves detailed metrics to baseline_test_results.csv and baseline_summary.json
"""

import os
import cv2
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "alphadent_weights" / "yolov8x_AlphaDent_9_classes_640px.pt"
DATASET_DIR = BASE_DIR / "dataset_3class_seg"
TEST_IMG_DIR = DATASET_DIR / "images" / "test"
TEST_LBL_DIR = DATASET_DIR / "labels" / "test"
OUT_DIR = BASE_DIR / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)


def parse_ground_truth(lbl_path, h, w):
    """Parses ground truth polygons for classes 0 (Abrasion), 1 (Caries), 2 (Crack)."""
    gt_masks = {0: np.zeros((h, w), dtype=np.uint8),
                1: np.zeros((h, w), dtype=np.uint8),
                2: np.zeros((h, w), dtype=np.uint8)}
    if not os.path.exists(lbl_path):
        return gt_masks

    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
            # Denormalize
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = np.round(coords).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            if cls_id in gt_masks:
                gt_masks[cls_id] = np.maximum(gt_masks[cls_id], mask)
    return gt_masks


def extract_old_predictions(result, h, w):
    """
    Extracts predictions from 9-class model and collapses Caries 1-6 (classes 3-8).
    Class 0 -> Abrasion
    Classes 3-8 -> Caries
    Crack -> empty (not in 9-class model)
    """
    pred_masks = {0: np.zeros((h, w), dtype=np.uint8),
                  1: np.zeros((h, w), dtype=np.uint8),
                  2: np.zeros((h, w), dtype=np.uint8)}

    if result.masks is None or len(result.masks) == 0:
        return pred_masks

    classes = result.boxes.cls.tolist()
    polygons = result.masks.xy

    for cls_idx, poly in zip(classes, polygons):
        cid = int(cls_idx)
        if len(poly) < 3:
            continue
        pts = np.round(poly).astype(np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)

        if cid == 0:  # Abrasion
            pred_masks[0] = np.maximum(pred_masks[0], m)
        elif cid in (3, 4, 5, 6, 7, 8):  # Caries 1-6
            pred_masks[1] = np.maximum(pred_masks[1], m)

    return pred_masks


def compute_iou(mask_a, mask_b):
    """Computes Intersection over Union."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return np.nan
    return float(intersection) / float(union)


def run_baseline_eval():
    print(f"Loading pretrained 9-class model: {WEIGHTS_PATH} ...")
    model = YOLO(str(WEIGHTS_PATH))

    test_images = sorted(list(TEST_IMG_DIR.glob("*.*")))
    print(f"Found {len(test_images)} test images in {TEST_IMG_DIR}")

    records = []
    class_ious = {0: [], 1: [], 2: []}

    for idx, img_p in enumerate(test_images):
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_masks = parse_ground_truth(lbl_p, h, w)

        # Run inference (conf=0.25 standard default)
        results = model.predict(source=str(img_p), conf=0.25, verbose=False)
        pred_masks = extract_old_predictions(results[0], h, w)

        row = {"sample_id": img_p.stem}
        for cid, cname in [(0, "Abrasion"), (1, "Caries"), (2, "Crack")]:
            gt_has = gt_masks[cid].sum() > 0
            pred_has = pred_masks[cid].sum() > 0
            if gt_has or pred_has:
                iou = compute_iou(gt_masks[cid], pred_masks[cid])
                row[f"iou_{cname}"] = round(iou, 4)
                class_ious[cid].append(iou)
            else:
                row[f"iou_{cname}"] = np.nan

        records.append(row)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_images):
            print(f"Evaluated {idx + 1}/{len(test_images)} test images...")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "baseline_test_results.csv", index=False)

    summary = {
        "model_name": "yolov8x_AlphaDent_9_classes_640px.pt",
        "num_test_images": len(test_images),
        "mean_iou_abrasion": round(float(np.nanmean(class_ious[0])), 4) if class_ious[0] else 0.0,
        "mean_iou_caries": round(float(np.nanmean(class_ious[1])), 4) if class_ious[1] else 0.0,
        "mean_iou_crack": round(float(np.nanmean(class_ious[2])), 4) if class_ious[2] else 0.0,
        "support_abrasion": len(class_ious[0]),
        "support_caries": len(class_ious[1]),
        "support_crack": len(class_ious[2]),
    }
    with open(OUT_DIR / "baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("OLD PRETRAINED BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"Abrasion Mean IoU: {summary['mean_iou_abrasion']:.4f} (on {summary['support_abrasion']} test images)")
    print(f"Caries   Mean IoU: {summary['mean_iou_caries']:.4f} (on {summary['support_caries']} test images)")
    print(f"Crack    Mean IoU: {summary['mean_iou_crack']:.4f} (on {summary['support_crack']} test images)")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run_baseline_eval()
