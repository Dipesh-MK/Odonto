"""
18_extract_mask_shape_params.py
Extracts geometric shape parameters from segmentation masks:
- Centroid (cx, cy in pixel coordinates)
- Bounding ellipse (major axis, minor axis, angle)
- Aspect ratio (major / minor)
- Effective radius (equivalent circular radius = sqrt(area / pi))
- Area (pixel count and fraction of total image area)

Outputs:
- photo_manifest_with_shape_params.csv
- Validation: prints parameters for 10 sample masks and saves visual overlays
  (centroid marker + fitted bounding ellipse) to validation_step3/
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from mask_utils import yolo_poly_to_mask, extract_mask_shape_parameters

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "photo_manifest.csv"
OUT_CSV = BASE_DIR / "photo_manifest_with_shape_params.csv"
VALIDATION_DIR = BASE_DIR / "validation_step3"
VALIDATION_DIR.mkdir(exist_ok=True)

def load_mask_for_row(row, height, width):
    """
    Loads binary mask for a manifest row:
    - AlphaDent: parses YOLO label txt polygon(s) for pathology (excludes class 1, 2)
    - DentalAI: loads _mask.png, prioritizes Crack (3), then Caries (1) / Cavity (2)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    mask_path = row["mask_path"]
    if not isinstance(mask_path, str) or not os.path.exists(mask_path):
        return mask

    if row["source_dataset"] == "AlphaDent":
        with open(mask_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 5:
                    cls_id = int(parts[0])
                    # Only include pathology
                    if cls_id not in (1, 2):
                        coords = [float(x) for x in parts[1:]]
                        poly_mask = yolo_poly_to_mask(coords, height, width)
                        mask = np.maximum(mask, poly_mask)
    elif row["source_dataset"] == "DentalAI":
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is not None:
            if 3 in m:  # Crack class prioritized
                mask = (m == 3).astype(np.uint8)
            else:
                mask = np.isin(m, [1, 2]).astype(np.uint8)

    return mask

def overlay_shape_annotations(img_bgr, mask, params):
    """
    Overlays centroid (red cross/dot) and fitted ellipse (yellow) on image.
    """
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    # Draw semi-transparent green mask first
    mask_bool = mask > 0
    overlay[mask_bool] = (overlay[mask_bool] * 0.6 + np.array([0, 220, 0]) * 0.4).astype(np.uint8)

    cx = params.get("cx")
    cy = params.get("cy")
    major = params.get("major_axis")
    minor = params.get("minor_axis")
    angle = params.get("ellipse_angle")

    if pd.notna(cx) and pd.notna(cy) and np.isfinite(cx) and np.isfinite(cy):
        center_pt = (int(round(cx)), int(round(cy)))
        # Draw centroid marker
        marker_radius = max(5, int(min(h, w) * 0.008))
        cv2.circle(overlay, center_pt, marker_radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, center_pt, marker_radius + 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw fitted ellipse if valid
        if pd.notna(major) and pd.notna(minor) and major > 0 and minor > 0:
            axes = (int(round(major / 2)), int(round(minor / 2)))
            thickness = max(2, int(min(h, w) * 0.003))
            cv2.ellipse(overlay, center_pt, axes, angle, 0, 360, (0, 255, 255), thickness, cv2.LINE_AA)

    return overlay

def main():
    print("=== Step 3: Mask Shape Parameter Extraction ===")
    manifest = pd.read_csv(MANIFEST_PATH)
    pathology_manifest = manifest[manifest["is_pathology"]].copy().reset_index(drop=True)
    print(f"Total pathology records to extract shape parameters from: {len(pathology_manifest)}")

    all_params = []
    print("Extracting parameters across pathology manifest...")

    for idx, row in pathology_manifest.iterrows():
        img_path = row["image_path"]
        if not os.path.exists(img_path):
            all_params.append({k: np.nan for k in ["cx", "cy", "area_pixels", "area_fraction", "major_axis", "minor_axis", "aspect_ratio", "effective_radius", "ellipse_angle"]})
            continue

        # Read dimensions efficiently
        img = cv2.imread(img_path)
        if img is None:
            all_params.append({k: np.nan for k in ["cx", "cy", "area_pixels", "area_fraction", "major_axis", "minor_axis", "aspect_ratio", "effective_radius", "ellipse_angle"]})
            continue

        h, w = img.shape[:2]
        mask = load_mask_for_row(row, h, w)
        params = extract_mask_shape_parameters(mask)
        all_params.append(params)

        if (idx + 1) % 500 == 0 or (idx + 1) == len(pathology_manifest):
            print(f"  Processed {idx + 1}/{len(pathology_manifest)} images...")

    params_df = pd.DataFrame(all_params)
    combined_df = pd.concat([pathology_manifest, params_df], axis=1)
    combined_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved manifest with shape parameters to: {OUT_CSV}")

    # Step 3 Validation: 10 sample masks with printed parameters and visual overlays
    print("\n--- Running Step 3 Validation: 10 Samples Visual Overlays & Printed Stats ---")
    np.random.seed(42)
    # Ensure both AlphaDent and DentalAI (and specifically Crack) are represented
    crack_samples = combined_df[combined_df["anomaly_class"].str.contains("Crack", na=False)]
    alphadent_samples = combined_df[combined_df["source_dataset"] == "AlphaDent"]
    dentalai_samples = combined_df[(combined_df["source_dataset"] == "DentalAI") & (~combined_df["anomaly_class"].str.contains("Crack", na=False))]

    sample_picks = []
    if len(crack_samples) >= 3:
        sample_picks.extend(crack_samples.sample(3, random_state=42).index.tolist())
    if len(alphadent_samples) >= 4:
        sample_picks.extend(alphadent_samples.sample(4, random_state=42).index.tolist())
    if len(dentalai_samples) >= 3:
        sample_picks.extend(dentalai_samples.sample(3, random_state=42).index.tolist())

    sample_val_df = combined_df.loc[sample_picks].reset_index(drop=True)

    print(f"\n{'#':<3} {'Dataset':<10} {'Class':<30} {'Centroid (X,Y)':<18} {'Area (px)':<10} {'Area Frac':<10} {'Aspect':<8} {'Eff. Radius':<10}")
    print("-" * 115)

    for i, row in sample_val_df.iterrows():
        img = cv2.imread(row["image_path"])
        h, w = img.shape[:2]
        mask = load_mask_for_row(row, h, w)
        p = {
            "cx": row["cx"],
            "cy": row["cy"],
            "major_axis": row["major_axis"],
            "minor_axis": row["minor_axis"],
            "aspect_ratio": row["aspect_ratio"],
            "effective_radius": row["effective_radius"],
            "ellipse_angle": row["ellipse_angle"],
            "area_pixels": row["area_pixels"],
            "area_fraction": row["area_fraction"]
        }

        # Format stats row
        c_str = f"({row['cx']:.1f}, {row['cy']:.1f})" if pd.notna(row['cx']) else "N/A"
        print(f"{i+1:<3} {row['source_dataset']:<10} {row['anomaly_class'][:28]:<30} {c_str:<18} {int(row['area_pixels']):<10} {row['area_fraction']:<10.5f} {row['aspect_ratio']:<8.2f} {row['effective_radius']:<10.2f}")

        # Render overlay
        annotated = overlay_shape_annotations(img, mask, p)

        # Draw summary card on image
        banner_h = 70
        banner = np.zeros((banner_h, annotated.shape[1], 3), dtype=np.uint8)
        info_text_1 = f"[{row['source_dataset']}] {row['anomaly_class']} | Centroid: {c_str} | Area: {int(row['area_pixels'])} px ({row['area_fraction']*100:.2f}%)"
        info_text_2 = f"Aspect Ratio: {row['aspect_ratio']:.2f} | Eff. Radius: {row['effective_radius']:.1f} px | Ellipse Angle: {row['ellipse_angle']:.1f} deg"
        
        scale = max(0.5, min(1.0, annotated.shape[1] / 1200))
        cv2.putText(banner, info_text_1, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(banner, info_text_2, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)

        final_out = np.vstack([banner, annotated])

        out_name = f"shape_sample_{i+1:02d}_{row['source_dataset']}_{row['anomaly_class'].replace(' ', '_').replace(';', '_')[:25]}.png"
        out_path = VALIDATION_DIR / out_name
        cv2.imwrite(str(out_path), final_out)

    print(f"\nStep 3 validation overlays saved to: {VALIDATION_DIR}")

if __name__ == "__main__":
    main()
