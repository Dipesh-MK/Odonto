"""
17_pretrained_segmentation_test.py
Evaluates AlphaDent's pretrained YOLOv8-seg weights (yolov8x_AlphaDent_9_classes_640px.pt)
on a held-out sample of 35 validation images from the AlphaDent valid split.

Computes:
1. Overall mean IoU between predicted and ground-truth pathology masks.
2. Per-class IoU breakdown (Abrasion, Caries 1, Caries 2, Caries 3, Caries 4, Caries 5, Caries 6).
3. Produces a side-by-side visual comparison grid (Image | Ground Truth | Prediction) for 6-8 examples.
4. Generates segmentation_test_results.csv and summary metrics.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from mask_utils import yolo_poly_to_mask, compute_mask_iou, overlay_mask_on_image

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "alphadent_weights" / "yolov8x_AlphaDent_9_classes_640px.pt"
MANIFEST_PATH = BASE_DIR / "photo_manifest.csv"
OUT_DIR = BASE_DIR / "validation_step2"
OUT_DIR.mkdir(exist_ok=True)

# AlphaDent 9 classes mapping
CLASS_MAP = {
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

PATHOLOGY_CLASS_IDS = [0, 3, 4, 5, 6, 7, 8]

def load_gt_masks(label_path, height, width):
    """
    Parses YOLO label txt and returns:
    - overall pathology mask
    - dict of class_id -> binary mask
    """
    overall_mask = np.zeros((height, width), dtype=np.uint8)
    class_masks = {cid: np.zeros((height, width), dtype=np.uint8) for cid in PATHOLOGY_CLASS_IDS}
    
    if not os.path.exists(label_path):
        return overall_mask, class_masks

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 5:
                cls_id = int(parts[0])
                if cls_id in PATHOLOGY_CLASS_IDS:
                    coords = [float(x) for x in parts[1:]]
                    poly_mask = yolo_poly_to_mask(coords, height, width)
                    overall_mask = np.maximum(overall_mask, poly_mask)
                    class_masks[cls_id] = np.maximum(class_masks[cls_id], poly_mask)

    return overall_mask, class_masks

def extract_pred_masks(result, height, width):
    """
    Extracts predicted masks from ultralytics Result:
    - overall predicted pathology mask
    - dict of class_id -> predicted binary mask
    """
    overall_mask = np.zeros((height, width), dtype=np.uint8)
    class_masks = {cid: np.zeros((height, width), dtype=np.uint8) for cid in PATHOLOGY_CLASS_IDS}

    if result.masks is None or len(result.masks) == 0:
        return overall_mask, class_masks

    classes = result.boxes.cls.tolist()
    polygons = result.masks.xy

    for cls_idx, poly in zip(classes, polygons):
        cid = int(cls_idx)
        if cid in PATHOLOGY_CLASS_IDS and len(poly) >= 3:
            pts = np.round(poly).astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            overall_mask = np.maximum(overall_mask, mask)
            class_masks[cid] = np.maximum(class_masks[cid], mask)

    return overall_mask, class_masks

def create_comparison_panel(img_bgr, gt_mask, pred_mask, title, iou, target_width=600):
    """
    Creates a 3-panel horizontal comparison:
    [Original Image] | [Ground Truth (Green)] | [Model Prediction (Cyan)]
    """
    h, w = img_bgr.shape[:2]
    scale = target_width / w
    target_h = int(h * scale)

    # Downscale for crisp side-by-side rendering
    img_small = cv2.resize(img_bgr, (target_width, target_h), interpolation=cv2.INTER_AREA)
    gt_small = cv2.resize(gt_mask, (target_width, target_h), interpolation=cv2.INTER_NEAREST)
    pred_small = cv2.resize(pred_mask, (target_width, target_h), interpolation=cv2.INTER_NEAREST)

    overlay_gt = overlay_mask_on_image(img_small, gt_small, color=(0, 220, 0), alpha=0.55)
    overlay_pred = overlay_mask_on_image(img_small, pred_small, color=(255, 200, 0), alpha=0.55)

    # Add text labels on panels
    cv2.putText(img_small, "Clinical Photo", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay_gt, "Ground Truth", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay_pred, f"YOLOv8-seg (IoU: {iou:.3f})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    panel = np.hstack([img_small, overlay_gt, overlay_pred])
    
    # Add top header bar
    header = np.zeros((50, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    
    return np.vstack([header, panel])

def main():
    print("=== Step 2: Pretrained YOLOv8-seg Segmentation Test ===")
    assert WEIGHTS_PATH.exists(), f"Weights not found at {WEIGHTS_PATH}"
    print(f"Loading pretrained weights from: {WEIGHTS_PATH}")
    model = YOLO(str(WEIGHTS_PATH))

    manifest = pd.read_csv(MANIFEST_PATH)
    # Select held-out images from valid split that contain pathology
    valid_pathology = manifest[
        (manifest["source_dataset"] == "AlphaDent") &
        (manifest["split"] == "valid") &
        (manifest["is_pathology"] == True)
    ].copy()

    print(f"Total held-out validation pathology images available: {len(valid_pathology)}")
    
    # We evaluate all available held-out validation images for full rigor (82 images)
    sample_df = valid_pathology.reset_index(drop=True)
    sample_size = len(sample_df)
    print(f"Evaluating model on all {sample_size} held-out validation pathology images...")

    results_records = []
    class_ious = {cid: [] for cid in PATHOLOGY_CLASS_IDS}
    comparison_panels = []

    for idx, row in sample_df.iterrows():
        img_path = row["image_path"]
        label_path = row["mask_path"]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        # Safeguard: verify source dataset is compatible with AlphaDent model
        if row["source_dataset"] != "AlphaDent":
            print(f"SKIPPED: {img_path} - source dataset '{row['source_dataset']}' not in AlphaDent vocabulary.")
            continue

        gt_overall, gt_by_class = load_gt_masks(label_path, h, w)

        # Run inference using AlphaDent's exact parameters (conf=0.1, iou=0.6, imgsz=640)
        res = model.predict(img_path, imgsz=640, conf=0.1, iou=0.6, device='cpu', verbose=False)[0]
        pred_overall, pred_by_class = extract_pred_masks(res, h, w)

        overall_iou = compute_mask_iou(gt_overall, pred_overall)

        record = {
            "image_id": Path(img_path).name,
            "anomaly_class": row["anomaly_class"],
            "overall_iou": overall_iou,
            "gt_pixels": int(gt_overall.sum()),
            "pred_pixels": int(pred_overall.sum()),
        }

        # Per-class IoU for classes present in GT or Prediction
        for cid in PATHOLOGY_CLASS_IDS:
            cname = CLASS_MAP[cid]
            gt_c = gt_by_class[cid]
            pred_c = pred_by_class[cid]
            if gt_c.sum() > 0 or pred_c.sum() > 0:
                c_iou = compute_mask_iou(gt_c, pred_c)
                class_ious[cid].append(c_iou)
                record[f"iou_{cname.replace(' ', '_')}"] = c_iou
            else:
                record[f"iou_{cname.replace(' ', '_')}"] = np.nan

        results_records.append(record)

        # Generate comparison panels for the first 8 samples
        if idx < 8:
            short_title = f"Sample {idx+1:02d}: {Path(img_path).name} | {row['anomaly_class']}"
            panel = create_comparison_panel(img_bgr, gt_overall, pred_overall, short_title, overall_iou)
            panel_path = OUT_DIR / f"comparison_{idx+1:02d}.png"
            cv2.imwrite(str(panel_path), panel)
            comparison_panels.append(panel)
            print(f"  [Sample {idx+1:02d}] {Path(img_path).name} | IoU: {overall_iou:.3f} | {row['anomaly_class']}")

    # Save structured results table
    results_df = pd.DataFrame(results_records)
    results_csv = BASE_DIR / "segmentation_test_results.csv"
    results_df.to_csv(results_csv, index=False)

    mean_overall_iou = results_df["overall_iou"].mean()

    # Per-class summary
    class_summary = []
    for cid in PATHOLOGY_CLASS_IDS:
        cname = CLASS_MAP[cid]
        ious = class_ious[cid]
        mean_c_iou = float(np.mean(ious)) if len(ious) > 0 else np.nan
        class_summary.append({
            "class_id": cid,
            "class_name": cname,
            "sample_count": len(ious),
            "mean_iou": mean_c_iou
        })

    class_summary_df = pd.DataFrame(class_summary)
    class_summary_csv = BASE_DIR / "segmentation_class_breakdown.csv"
    class_summary_df.to_csv(class_summary_csv, index=False)

    print("\n" + "="*60)
    print("=== SEGMENTATION TEST SUMMARY ===")
    print("="*60)
    print(f"Overall Mean IoU (All Pathology Masks): {mean_overall_iou:.4f}")
    print("\nPer-Class IoU Breakdown:")
    print(class_summary_df.to_string(index=False))

    print("\nDecision Gate Analysis:")
    if mean_overall_iou >= 0.5:
        print(f"✅ PASS: Mean IoU ({mean_overall_iou:.4f}) meets the >0.5 threshold for MVP.")
    else:
        print(f"⚠️ CAUTION: Mean IoU ({mean_overall_iou:.4f}) is below 0.5 target.")

    print("\nPer-class reliability notes:")
    for _, r in class_summary_df.iterrows():
        if pd.notna(r["mean_iou"]):
            status = "GOOD" if r["mean_iou"] >= 0.5 else "LOW / SCRUTINIZE"
            print(f"  - {r['class_name']}: Mean IoU = {r['mean_iou']:.4f} (N={r['sample_count']}) -> {status}")

    print(f"\nResults saved to:")
    print(f"  - CSV: {results_csv}")
    print(f"  - Class Breakdown: {class_summary_csv}")
    print(f"  - Visual Comparisons: {OUT_DIR}")

if __name__ == "__main__":
    main()
