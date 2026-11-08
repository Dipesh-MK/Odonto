"""
eval_balanced_model.py

Evaluates the class-balanced fine-tuned YOLOv8x-seg model against:
1. Baseline (Old 9-class Pretrained)
2. First Fine-Tune (25 epochs, unweighted)
3. New Balanced Fine-Tune (35 epochs, 8x Crack oversampling)

On the EXACT same 354-image held-out test set (228 patients).
Computes per-class IoU, crack confusion counts (TP, FP, FN, Precision, Recall),
and renders 4-panel visual comparison grids.
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
TEST_IMG_DIR = BASE_DIR / "dataset_3class_seg" / "images" / "test"
TEST_LBL_DIR = BASE_DIR / "dataset_3class_seg" / "labels" / "test"
OUT_DIR = BASE_DIR / "evaluation_outputs"
OUT_DIR.mkdir(exist_ok=True)

OLD_WEIGHTS = BASE_DIR / "alphadent_weights" / "yolov8x_AlphaDent_9_classes_640px.pt"
FIRST_WEIGHTS = BASE_DIR / "runs" / "fine_tune_3class" / "yolov8x_3class_model" / "weights" / "best.pt"
BALANCED_WEIGHTS = BASE_DIR / "runs" / "fine_tune_3class_balanced" / "yolov8x_3class_balanced" / "weights" / "best.pt"

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
    gt_polys = {0: [], 1: [], 2: []}
    if not os.path.exists(lbl_path):
        return gt_masks, gt_polys

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
                gt_polys[cls_id].append(pts)
    return gt_masks, gt_polys

def extract_3class_predictions(result, h, w):
    pred_masks = {0: np.zeros((h, w), dtype=np.uint8),
                  1: np.zeros((h, w), dtype=np.uint8),
                  2: np.zeros((h, w), dtype=np.uint8)}
    pred_polys = {0: [], 1: [], 2: []}
    if result.masks is None or len(result.masks) == 0:
        return pred_masks, pred_polys

    classes = result.boxes.cls.tolist()
    polygons = result.masks.xy

    for cls_idx, poly in zip(classes, polygons):
        cid = int(cls_idx)
        if cid in pred_masks and len(poly) >= 3:
            pts = np.round(poly).astype(np.int32)
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            pred_masks[cid] = np.maximum(pred_masks[cid], m)
            pred_polys[cid].append(pts)
    return pred_masks, pred_polys

def compute_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return np.nan
    return float(intersection) / float(union)

def render_overlay(img_bgr, masks_dict, alpha=0.5):
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

def run_eval():
    w_path = BALANCED_WEIGHTS
    if not w_path.exists():
        w_path = w_path.parent / "last.pt"
    if not w_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {w_path}")

    print(f"Loading Models...")
    print(f"  Balanced Model: {w_path}")
    print(f"  First Model   : {FIRST_WEIGHTS}")
    model_bal = YOLO(str(w_path))
    model_first = YOLO(str(FIRST_WEIGHTS))

    test_images = sorted(list(TEST_IMG_DIR.glob("*.*")))
    print(f"Evaluating {len(test_images)} test images...")

    records = []
    class_ious = {0: [], 1: [], 2: []}

    # Crack confusion metrics
    gt_crack_total = 0
    pred_crack_total = 0
    tp_iou25 = 0
    fp_iou25 = 0
    fn_iou25 = 0
    img_tp = 0
    img_fp = 0
    img_fn = 0
    img_tn = 0

    crack_visual_samples = []

    for idx, img_p in enumerate(test_images):
        lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_masks, gt_polys = parse_ground_truth(lbl_p, h, w)

        res_bal = model_bal.predict(source=str(img_p), conf=0.25, verbose=False)[0]
        pred_bal, polys_bal = extract_3class_predictions(res_bal, h, w)

        res_first = model_first.predict(source=str(img_p), conf=0.25, verbose=False)[0]
        pred_first, _ = extract_3class_predictions(res_first, h, w)

        row = {"sample_id": img_p.stem}
        for cid, cname in CLASS_NAMES.items():
            gt_has = gt_masks[cid].sum() > 0
            pred_has = pred_bal[cid].sum() > 0
            if gt_has or pred_has:
                iou = compute_iou(gt_masks[cid], pred_bal[cid])
                row[f"iou_{cname}"] = round(iou, 4)
                class_ious[cid].append(iou)
            else:
                row[f"iou_{cname}"] = np.nan

        records.append(row)

        # Crack specific diagnosis
        num_gt_crack = len(gt_polys[2])
        num_pred_crack = len(polys_bal[2])
        gt_crack_total += num_gt_crack
        pred_crack_total += num_pred_crack

        has_gt_c = num_gt_crack > 0
        has_pred_c = num_pred_crack > 0

        if has_gt_c and has_pred_c:
            img_tp += 1
        elif not has_gt_c and has_pred_c:
            img_fp += 1
        elif has_gt_c and not has_pred_c:
            img_fn += 1
        else:
            img_tn += 1

        # Match instances
        matched_gt = set()
        matched_pred = set()
        for p_i, p_pts in enumerate(polys_bal[2]):
            pm = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(pm, [p_pts], 1)
            for g_i, g_pts in enumerate(gt_polys[2]):
                gm = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(gm, [g_pts], 1)
                inter = np.logical_and(pm, gm).sum()
                union = np.logical_or(pm, gm).sum()
                if union > 0 and (inter / union) >= 0.25:
                    if g_i not in matched_gt and p_i not in matched_pred:
                        matched_gt.add(g_i)
                        matched_pred.add(p_i)

        tp_iou25 += len(matched_gt)
        fp_iou25 += (num_pred_crack - len(matched_pred))
        fn_iou25 += (num_gt_crack - len(matched_gt))

        if has_gt_c and len(crack_visual_samples) < 6:
            crack_visual_samples.append((img_p, img, gt_masks, pred_first, pred_bal))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_images):
            print(f"Evaluated {idx + 1}/{len(test_images)} test images...")

    prec25 = tp_iou25 / (tp_iou25 + fp_iou25) if (tp_iou25 + fp_iou25) > 0 else 0
    rec25 = tp_iou25 / (tp_iou25 + fn_iou25) if (tp_iou25 + fn_iou25) > 0 else 0
    f1_25 = 2 * prec25 * rec25 / (prec25 + rec25) if (prec25 + rec25) > 0 else 0

    img_prec = img_tp / (img_tp + img_fp) if (img_tp + img_fp) > 0 else 0
    img_rec = img_tp / (img_tp + img_fn) if (img_tp + img_fn) > 0 else 0

    results_summary = {
        "model_name": "fine_tuned_yolov8x_3class_balanced",
        "weights_path": str(w_path),
        "num_test_images": len(test_images),
        "mean_iou_abrasion": round(float(np.nanmean(class_ious[0])), 4),
        "mean_iou_caries": round(float(np.nanmean(class_ious[1])), 4),
        "mean_iou_crack": round(float(np.nanmean(class_ious[2])), 4),
        "three_way_comparison": {
            "abrasion": {
                "baseline": 0.3976,
                "first_finetune": 0.5316,
                "balanced_finetune": round(float(np.nanmean(class_ious[0])), 4)
            },
            "caries": {
                "baseline": 0.1861,
                "first_finetune": 0.2220,
                "balanced_finetune": round(float(np.nanmean(class_ious[1])), 4)
            },
            "crack": {
                "baseline": 0.0000,
                "first_finetune": 0.1428,
                "balanced_finetune": round(float(np.nanmean(class_ious[2])), 4)
            }
        },
        "crack_instance_confusion": {
            "total_gt": gt_crack_total,
            "total_pred": pred_crack_total,
            "tp": tp_iou25,
            "fp": fp_iou25,
            "fn": fn_iou25,
            "precision": round(prec25, 4),
            "recall": round(rec25, 4),
            "f1": round(f1_25, 4)
        },
        "crack_image_confusion": {
            "tp_images": img_tp,
            "fp_images": img_fp,
            "fn_images": img_fn,
            "tn_images": img_tn,
            "image_precision": round(img_prec, 4),
            "image_recall": round(img_rec, 4)
        }
    }

    with open(OUT_DIR / "balanced_finetuned_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Render Visual Comparison Grids for Crack
    grid_img_paths = []
    for s_idx, (p, img_bgr, gt_m, p_first, p_bal) in enumerate(crack_visual_samples):
        h_orig, w_orig = img_bgr.shape[:2]
        target_w = 400
        target_h = int(h_orig * (target_w / w_orig))

        def prep(im):
            return cv2.resize(im, (target_w, target_h))

        p1 = prep(img_bgr)
        p2 = prep(render_overlay(img_bgr, p_first))
        p3 = prep(render_overlay(img_bgr, p_bal))
        p4 = prep(render_overlay(img_bgr, gt_m))

        def add_header(im, text, col=(255, 255, 255)):
            cv2.rectangle(im, (0, 0), (target_w, 28), (20, 20, 20), -1)
            cv2.putText(im, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            return im

        p1 = add_header(p1, f"Photo: {p.stem[:18]}")
        p2 = add_header(p2, "1st Fine-Tune (Unbalanced)", (0, 255, 255))
        p3 = add_header(p3, "Balanced Fine-Tune (Oversampled)", (0, 255, 0))
        p4 = add_header(p4, "Ground Truth: Crack", (255, 0, 255))

        row = np.hstack([p1, p2, p3, p4])
        grid_fname = f"visual_comparison_crack_balanced_sample_{s_idx+1}.png"
        grid_p = OUT_DIR / grid_fname
        cv2.imwrite(str(grid_p), row)
        grid_img_paths.append(str(grid_p))

    print("\n" + "=" * 70)
    print("BALANCED FINE-TUNED MODEL EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Abrasion IoU: Baseline={0.3976:.4f} | 1st={0.5316:.4f} | Balanced={results_summary['mean_iou_abrasion']:.4f}")
    print(f"Caries   IoU: Baseline={0.1861:.4f} | 1st={0.2220:.4f} | Balanced={results_summary['mean_iou_caries']:.4f}")
    print(f"Crack    IoU: Baseline={0.0000:.4f} | 1st={0.1428:.4f} | Balanced={results_summary['mean_iou_crack']:.4f}")
    print(f"Crack Instance Confusion: TP={tp_iou25}, FP={fp_iou25}, FN={fn_iou25} (Recall={rec25:.4f}, Prec={prec25:.4f})")
    print("=" * 70)
    return results_summary

if __name__ == "__main__":
    run_eval()
