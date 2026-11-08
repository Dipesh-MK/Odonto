"""
eval_finetuned_model.py

Evaluates the fine-tuned 3-class YOLOv8x-seg model on the held-out test set:
1. Computes per-class Mean IoU for:
   - Abrasion (Class 0)
   - Caries (Class 1)
   - Crack (Class 2)
2. Loads baseline metrics from evaluation_outputs/baseline_summary.json.
3. Compares Before (Pretrained 9-class) vs After (Fine-Tuned 3-class).
4. Generates a 4-panel visual comparison grid:
   [Original Image] | [Old Pretrained Model] | [New Fine-Tuned Model] | [Ground Truth]
   for representative test samples of each class.
5. Saves comparison report to evaluation_outputs/finetuned_summary.json and visual grids.
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
DATASET_DIR = BASE_DIR / "dataset_3class_seg"
TEST_IMG_DIR = DATASET_DIR / "images" / "test"
TEST_LBL_DIR = DATASET_DIR / "labels" / "test"
OUT_DIR = BASE_DIR / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)

OLD_WEIGHTS = BASE_DIR / "alphadent_weights" / "yolov8x_AlphaDent_9_classes_640px.pt"
NEW_WEIGHTS = BASE_DIR / "runs" / "fine_tune_3class" / "yolov8x_3class_model" / "weights" / "best.pt"

CLASS_NAMES = {0: "Abrasion", 1: "Caries", 2: "Crack"}
CLASS_COLORS = {
    0: (255, 165, 0),   # Orange for Abrasion
    1: (0, 0, 255),     # Red for Caries
    2: (255, 0, 255),   # Magenta for Crack
}


def parse_ground_truth(lbl_path, h, w):
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
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = np.round(coords).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            if cls_id in gt_masks:
                gt_masks[cls_id] = np.maximum(gt_masks[cls_id], mask)
    return gt_masks


def extract_3class_predictions(result, h, w):
    pred_masks = {0: np.zeros((h, w), dtype=np.uint8),
                  1: np.zeros((h, w), dtype=np.uint8),
                  2: np.zeros((h, w), dtype=np.uint8)}
    if result.masks is None or len(result.masks) == 0:
        return pred_masks

    classes = result.boxes.cls.tolist()
    polygons = result.masks.xy

    for cls_idx, poly in zip(classes, polygons):
        cid = int(cls_idx)
        if cid in pred_masks and len(poly) >= 3:
            pts = np.round(poly).astype(np.int32)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            pred_masks[cid] = np.maximum(pred_masks[cid], m)
    return pred_masks


def extract_old_predictions(result, h, w):
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
        if cid == 0:
            pred_masks[0] = np.maximum(pred_masks[0], m)
        elif cid in (3, 4, 5, 6, 7, 8):
            pred_masks[1] = np.maximum(pred_masks[1], m)
    return pred_masks


def compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return np.nan
    return float(intersection) / float(union)


def render_overlay(img_bgr, masks_dict, alpha=0.45):
    canvas = img_bgr.copy()
    overlay = np.zeros_like(img_bgr)
    has_any = False
    for cid, mask in masks_dict.items():
        if mask.sum() > 0:
            color = CLASS_COLORS.get(cid, (0, 255, 0))
            overlay[mask > 0] = color
            has_any = True
    if has_any:
        return cv2.addWeighted(canvas, 1.0, overlay, alpha, 0)
    return canvas


def evaluate_model(weights_path=None):
    w_path = Path(weights_path) if weights_path else NEW_WEIGHTS
    if not w_path.exists():
        # Fallback to last.pt if best.pt not found
        w_path = w_path.parent / "last.pt"
    if not w_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {w_path}")

    print(f"Loading Fine-Tuned Model: {w_path} ...")
    model_new = YOLO(str(w_path))
    model_old = YOLO(str(OLD_WEIGHTS))

    test_images = sorted(list(TEST_IMG_DIR.glob("*.*")))
    print(f"Evaluating {len(test_images)} test images...")

    records = []
    class_ious = {0: [], 1: [], 2: []}

    # Samples for visual comparison
    visual_candidates = {0: [], 1: [], 2: []}

    for idx, img_p in enumerate(test_images):
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_masks = parse_ground_truth(lbl_p, h, w)

        res_new = model_new.predict(source=str(img_p), conf=0.25, verbose=False)
        pred_new = extract_3class_predictions(res_new[0], h, w)

        res_old = model_old.predict(source=str(img_p), conf=0.25, verbose=False)
        pred_old = extract_old_predictions(res_old[0], h, w)

        row = {"sample_id": img_p.stem}
        for cid, cname in CLASS_NAMES.items():
            gt_has = gt_masks[cid].sum() > 0
            pred_has = pred_new[cid].sum() > 0
            if gt_has or pred_has:
                iou = compute_iou(gt_masks[cid], pred_new[cid])
                row[f"iou_{cname}"] = round(iou, 4)
                class_ious[cid].append(iou)
                if gt_has and len(visual_candidates[cid]) < 3:
                    visual_candidates[cid].append((img_p, img, gt_masks, pred_old, pred_new))
            else:
                row[f"iou_{cname}"] = np.nan

        records.append(row)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_images):
            print(f"Evaluated {idx + 1}/{len(test_images)} test images...")

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "finetuned_test_results.csv", index=False)

    # Load baseline summary if available
    base_sum_path = OUT_DIR / "baseline_summary.json"
    baseline = {}
    if base_sum_path.exists():
        with open(base_sum_path, "r") as f:
            baseline = json.load(f)

    new_summary = {
        "model_name": "fine_tuned_yolov8x_3class",
        "weights_path": str(w_path),
        "num_test_images": len(test_images),
        "mean_iou_abrasion": round(float(np.nanmean(class_ious[0])), 4) if class_ious[0] else 0.0,
        "mean_iou_caries": round(float(np.nanmean(class_ious[1])), 4) if class_ious[1] else 0.0,
        "mean_iou_crack": round(float(np.nanmean(class_ious[2])), 4) if class_ious[2] else 0.0,
        "support_abrasion": len(class_ious[0]),
        "support_caries": len(class_ious[1]),
        "support_crack": len(class_ious[2]),
        "baseline_comparison": {
            "abrasion": {
                "before": baseline.get("mean_iou_abrasion", 0.4900),
                "after": round(float(np.nanmean(class_ious[0])), 4) if class_ious[0] else 0.0,
                "delta": round(float(np.nanmean(class_ious[0])) - baseline.get("mean_iou_abrasion", 0.49), 4),
            },
            "caries": {
                "before": baseline.get("mean_iou_caries", 0.2362),
                "after": round(float(np.nanmean(class_ious[1])), 4) if class_ious[1] else 0.0,
                "delta": round(float(np.nanmean(class_ious[1])) - baseline.get("mean_iou_caries", 0.2362), 4),
            },
            "crack": {
                "before": 0.0000,
                "after": round(float(np.nanmean(class_ious[2])), 4) if class_ious[2] else 0.0,
                "delta": round(float(np.nanmean(class_ious[2])), 4) if class_ious[2] else 0.0,
            }
        }
    }

    with open(OUT_DIR / "finetuned_summary.json", "w") as f:
        json.dump(new_summary, f, indent=2)

    # Render Visual Comparison Grids
    grid_img_paths = []
    for cid, samples in visual_candidates.items():
        cname = CLASS_NAMES[cid]
        for s_idx, (p, img_bgr, gt_m, p_old, p_new) in enumerate(samples):
            h_orig, w_orig = img_bgr.shape[:2]
            target_w = 400
            target_h = int(h_orig * (target_w / w_orig))

            def prep(im):
                return cv2.resize(im, (target_w, target_h))

            p1 = prep(img_bgr)
            p2 = prep(render_overlay(img_bgr, p_old))
            p3 = prep(render_overlay(img_bgr, p_new))
            p4 = prep(render_overlay(img_bgr, gt_m))

            # Add headers
            def add_header(im, text, col=(255, 255, 255)):
                cv2.rectangle(im, (0, 0), (target_w, 28), (20, 20, 20), -1)
                cv2.putText(im, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                return im

            p1 = add_header(p1, f"Original: {p.stem[:18]}")
            p2 = add_header(p2, "Old 9-Class Pretrained", (0, 255, 255))
            p3 = add_header(p3, "New 3-Class Fine-Tuned", (0, 255, 0))
            p4 = add_header(p4, f"Ground Truth: {cname}", (255, 200, 0))

            row = np.hstack([p1, p2, p3, p4])
            grid_fname = f"visual_comparison_{cname.lower()}_sample_{s_idx+1}.png"
            grid_p = OUT_DIR / grid_fname
            cv2.imwrite(str(grid_p), row)
            grid_img_paths.append(str(grid_p))

    print("\n" + "=" * 70)
    print("FINE-TUNED MODEL EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Abrasion Mean IoU : {new_summary['mean_iou_abrasion']:.4f} (Baseline: {baseline.get('mean_iou_abrasion', 0.4900):.4f})")
    print(f"Caries   Mean IoU : {new_summary['mean_iou_caries']:.4f} (Baseline: {baseline.get('mean_iou_caries', 0.2362):.4f})")
    print(f"Crack    Mean IoU : {new_summary['mean_iou_crack']:.4f} (Baseline: 0.0000)")
    print("=" * 70)
    print(f"Visual comparison grids saved ({len(grid_img_paths)} grids in {OUT_DIR})")
    return new_summary


if __name__ == "__main__":
    evaluate_model()
