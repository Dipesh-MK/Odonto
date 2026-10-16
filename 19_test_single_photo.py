#!/usr/bin/env python3
"""
19_test_single_photo.py

Interactive test tool for the photo-guided anomaly pipeline (Steps 1-3).
Takes any image path (or picks a random image from the manifest),
runs YOLOv8-seg prediction, extracts geometric shape parameters,
computes IoU if ground-truth mask is available, and displays/saves the visual result.

Usage:
  ./venv/bin/python 19_test_single_photo.py
  ./venv/bin/python 19_test_single_photo.py --class Crack
  ./venv/bin/python 19_test_single_photo.py --class Abrasion
  ./venv/bin/python 19_test_single_photo.py --image /path/to/any_photo.jpg
  ./venv/bin/python 19_test_single_photo.py --show
"""

import argparse
import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

from mask_utils import (
    yolo_poly_to_mask,
    extract_mask_shape_parameters,
    compute_mask_iou
)
import importlib
_extract_mod = importlib.import_module("18_extract_mask_shape_params")
load_mask_for_row = _extract_mod.load_mask_for_row
overlay_shape_annotations = _extract_mod.overlay_shape_annotations

WEIGHTS_PATH = "alphadent_weights/yolov8x_AlphaDent_9_classes_640px.pt"
MANIFEST_PATH = "photo_manifest.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Test photo segmentation and shape extraction on an image.")
    parser.add_argument("--image", type=str, default=None, help="Path to image file (optional, picks random if omitted)")
    parser.add_argument("--class", dest="anomaly_class", type=str, default=None, help="Filter random selection by anomaly class (e.g. Crack, Abrasion, Caries)")
    parser.add_argument("--output", type=str, default="test_result.png", help="Path to save visual output")
    parser.add_argument("--show", action="store_true", help="Display interactive matplotlib window (requires local GUI display)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: {MANIFEST_PATH} not found. Run 16_build_photo_manifest.py first.")
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    gt_record = None
    if args.image:
        image_path = os.path.abspath(args.image)
        matched = df[df["image_path"] == image_path]
        if not matched.empty:
            gt_record = matched.iloc[0].to_dict()
    else:
        candidates = df[df["is_pathology"] == True]
        if args.anomaly_class:
            candidates = candidates[candidates["anomaly_class"].str.contains(args.anomaly_class, case=False, na=False)]
            if candidates.empty:
                print(f"No samples found with class filter '{args.anomaly_class}'.")
                return
        gt_record = candidates.sample(1, random_state=random.randint(1, 999999)).iloc[0].to_dict()
        image_path = gt_record["image_path"]

    print(f"\n==================================================")
    print(f"Testing Photo Anomaly Pipeline on:")
    print(f"Image: {image_path}")
    if gt_record:
        print(f"Source Dataset: {gt_record.get('source_dataset')}")
        print(f"Annotated Class: {gt_record.get('anomaly_class')}")
    print(f"==================================================")

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read image from {image_path}")
        return
    h, w = img.shape[:2]

    # 2. Load ground-truth mask if available
    gt_mask = None
    if gt_record:
        try:
            gt_mask = load_mask_for_row(gt_record, h, w)
        except Exception as e:
            print(f"Could not load GT mask: {e}")

    # 3. Run YOLOv8-seg model inference
    pred_mask = np.zeros((h, w), dtype=np.uint8)
    model = YOLO(WEIGHTS_PATH)
    results = model.predict(source=image_path, imgsz=640, conf=0.1, iou=0.6, verbose=False)
    
    detected_classes = []
    if results and len(results) > 0 and results[0].masks is not None:
        boxes = results[0].boxes
        names = results[0].names
        for i, poly in enumerate(results[0].masks.xy):
            cls_id = int(boxes.cls[i].item())
            cls_name = names.get(cls_id, f"class_{cls_id}")
            # Skip restorations (Filling=1, Crown=2 in 9-class model)
            if cls_id in (1, 2):
                continue
            detected_classes.append(cls_name)
            pts = np.int32(poly)
            if len(pts) >= 3:
                cv2.fillPoly(pred_mask, [pts], 255)

    # Model's known pathology classes (AlphaDent 9-class model excludes restorations Filling and Crown)
    MODEL_KNOWN_PATHOLOGY = {
        'Abrasion',
        'Caries 1 class',
        'Caries 2 class',
        'Caries 3 class',
        'Caries 4 class',
        'Caries 5 class',
        'Caries 6 class'
    }

    print(f"\n[Model Prediction]")
    print(f"Detected pathology classes: {list(set(detected_classes)) if detected_classes else 'None'}")

    # Verify if ground truth class vocabulary matches model capabilities before computing IoU
    iou_status_str = None
    if gt_record and gt_mask is not None and gt_mask.sum() > 0:
        raw_classes = [c.strip() for c in str(gt_record.get('anomaly_class', '')).split(';')]
        # Filter out non-pathology like Filling, Crown, Tooth
        pathology_gt_classes = [c for c in raw_classes if c not in ('Filling', 'Crown', 'Tooth')]
        
        # Check vocabulary compatibility
        is_compatible = (gt_record.get('source_dataset') == 'AlphaDent') and any(
            c in MODEL_KNOWN_PATHOLOGY for c in pathology_gt_classes
        )

        print(f"\n[IoU Evaluation]")
        if not is_compatible:
            mismatched_str = ', '.join(pathology_gt_classes) if pathology_gt_classes else gt_record.get('anomaly_class', '')
            print(f"SKIPPED: ground truth class '{mismatched_str}' is not in this model's class list — comparison would be invalid.")
            if any('Crack' in c for c in pathology_gt_classes):
                print("NOTE: 'Crack' evaluation is not yet testable with current weights (AlphaDent model does not contain Crack, and DentalAI dataset ships no pretrained weights).")
            iou_status_str = f"IoU: SKIPPED\n(Class '{mismatched_str}' not in model)"
        else:
            if pred_mask.sum() > 0:
                iou = compute_mask_iou(pred_mask, gt_mask)
                print(f"IoU (Prediction vs Ground Truth): {iou:.4f}")
                iou_status_str = f"IoU: {iou:.4f}"
            else:
                print(f"IoU: 0.0000 (Model did not predict a mask for this pathology)")
                iou_status_str = "IoU: 0.0000 (No pred)"

    # 4. Extract shape parameters (prefer predicted if exists, otherwise fallback to GT for demonstration)
    eval_mask = pred_mask if pred_mask.sum() > 0 else (gt_mask if gt_mask is not None else None)
    eval_source = "YOLOv8 Predicted Mask" if pred_mask.sum() > 0 else "Ground Truth Mask"

    if eval_mask is not None and eval_mask.sum() > 0:
        params = extract_mask_shape_parameters(eval_mask)
        print(f"\n[Extracted Shape Parameters ({eval_source})]")
        print(f"  Centroid (X, Y):       ({params['cx']:.1f}, {params['cy']:.1f})")
        print(f"  Area:                  {int(params['area_pixels'])} px ({params['area_fraction']*100:.3f}% of image)")
        print(f"  Effective Radius (r):  {params['effective_radius']:.2f} px")
        print(f"  Major / Minor Axes:    {params['major_axis']:.1f} px / {params['minor_axis']:.1f} px")
        print(f"  Aspect Ratio:          {params['aspect_ratio']:.2f} {'(Elongated / Crack-like)' if params['aspect_ratio'] > 4 else '(Elliptical / Cavity-like)'}")
        print(f"  Orientation Angle:     {params['ellipse_angle']:.1f} deg")

        overlay_img = overlay_shape_annotations(img, eval_mask, params)
    else:
        print("\n[Extracted Shape Parameters]")
        print("  No mask available to extract parameters.")
        overlay_img = img.copy()

    # 5. Build side-by-side visualization
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_overlay = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original Photo")
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(gt_mask, cmap="inferno")
        gt_title = f"Ground Truth Mask\n({gt_record.get('anomaly_class', '')})"
        if iou_status_str:
            gt_title += f"\n[{iou_status_str.replace(chr(10), ' - ')}]"
        axes[1].set_title(gt_title, fontsize=10)
    else:
        axes[1].text(0.5, 0.5, "No GT Mask", ha="center", va="center")
        axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(rgb_overlay)
    title = f"Fitted Shape Parameters\n({eval_source})\n"
    if eval_mask is not None and eval_mask.sum() > 0:
        title += f"r={params['effective_radius']:.1f}px, aspect={params['aspect_ratio']:.2f}"
    axes[2].set_title(title, fontsize=10)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nVisual result saved to: {os.path.abspath(args.output)}")

    if args.show:
        print("Displaying interactive window (close window to exit)...")
        plt.show()

if __name__ == "__main__":
    main()
