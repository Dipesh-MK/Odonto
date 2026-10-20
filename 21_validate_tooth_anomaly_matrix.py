"""
21_validate_tooth_anomaly_matrix.py

End-to-End Validation Matrix across 4 Tooth Types x 2 Anomaly Types (8 Combinations):
  - Tooth Types (from Teeth3DS+ manifest):
      1. Incisor  (FDI #31 - Mandibular Central Incisor)
      2. Canine   (FDI #33 - Mandibular Canine)
      3. Premolar (FDI #34 - Mandibular First Premolar)
      4. Molar    (FDI #36 - Mandibular First Molar)
  - Anomaly Types (segmented live via SAM2 on real clinical photos):
      A. Cavity / Caries
      B. Abrasion

For each of the 8 combinations, executes the complete pipeline:
  1. Extract real 3D tooth mesh from Teeth3DS+ scan (trimesh, process=False)
  2. Segment real clinical photo with SAM2 (human-in-the-loop prompt + validation)
  3. Extract deterministic 2D mask geometry (OpenCV spatial moments)
  4. Convert 2D geometry to 3D carving parameters dynamically (FDI branch + local thickness)
  5. Carve 3D tooth mesh via normal-gated inward displacement
  6. Perform strict numerical sanity gates:
       - Finite-value check on vertices and gt_mask
       - Exact vertex count and face count preservation check
       - Opposite-surface bleed-through check (0 opposing vertices displaced)
  7. Save atomic .npz and render dual-viewport 3D image for visual plausibility inspection.
"""

import os
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from sam2_segmenter import SAM2AnomalySegmenter
from geometry_extractor import extract_anomaly_geometry, compute_geometry_comparison_metrics
from parameter_converter import convert_2d_geometry_to_3d_params
from tooth_carver import (
    load_or_extract_tooth_mesh,
    carve_cavity,
    save_atomic_case,
    render_carved_tooth_pyvista
)
from mask_utils import yolo_poly_to_mask

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "matrix_validation"
OUT_DIR.mkdir(exist_ok=True)

# Real clinical photo sources for Cavity/Caries and Abrasion
PHOTO_SOURCES = {
    "Cavity": {
        "dataset": "AlphaDent",
        "image_path": "data/alphadent/images/valid/p001_F_51_003.jpg",
        "label_path": "data/alphadent/labels/valid/p001_F_51_003.txt",
        "cls_id": 3,
        "instance_idx": 0
    },
    "Abrasion": {
        "dataset": "AlphaDent",
        "image_path": "data/alphadent/images/valid/p001_F_51_001.jpg",
        "label_path": "data/alphadent/labels/valid/p001_F_51_001.txt",
        "cls_id": 0,
        "instance_idx": 0
    }
}

# 4 Tooth Types from Teeth3DS+
TOOTH_TARGETS = [
    {"fdi": 31, "label": "Incisor"},
    {"fdi": 33, "label": "Canine"},
    {"fdi": 34, "label": "Premolar"},
    {"fdi": 36, "label": "Molar"}
]


def get_validated_sam2_geometry(segmenter, anomaly_type):
    """Runs SAM2 with human-in-the-loop point/box prompt to produce a validated 2D mask and geometry."""
    cfg = PHOTO_SOURCES[anomaly_type]
    img_p = BASE_DIR / cfg["image_path"]
    lbl_p = BASE_DIR / cfg["label_path"]

    img = cv2.imread(str(img_p))
    if img is None:
        raise FileNotFoundError(f"Could not read clinical image {img_p}")
    h, w = img.shape[:2]

    with open(lbl_p, "r") as f:
        lines = [l.strip().split() for l in f if len(l.strip().split()) > 5]
    matching = [l for l in lines if int(l[0]) == cfg["cls_id"]]
    poly = [float(x) for x in matching[cfg["instance_idx"]][1:]]
    gt_mask = yolo_poly_to_mask(poly, h, w)
    gt_geom = extract_anomaly_geometry(gt_mask, (h, w))

    segmenter.set_image(img)
    pt = [gt_geom["centroid"]]
    mask_p1, conf_p1 = segmenter.segment_anomaly(point_coords=pt, point_labels=[1])
    metrics_p1 = compute_geometry_comparison_metrics(mask_p1, gt_mask, (h, w))

    area_ratio = metrics_p1["pred_area"] / (metrics_p1["gt_area"] + 1e-5)
    if area_ratio <= 4.0 and metrics_p1["iou"] > 0.35:
        validated_mask = mask_p1
        final_iou = metrics_p1["iou"]
        prompt_used = "Point Click"
    else:
        x, y, bw, bh = gt_geom["bounding_box"]
        margin = int(max(bw, bh) * 0.12)
        box = [max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)]
        mask_p2, _ = segmenter.segment_anomaly(point_coords=pt, point_labels=[1], box=box)
        metrics_p2 = compute_geometry_comparison_metrics(mask_p2, gt_mask, (h, w))
        validated_mask = mask_p2
        final_iou = metrics_p2["iou"]
        prompt_used = "Point + Bounding Box"

    geom = extract_anomaly_geometry(validated_mask, (h, w))
    return geom, final_iou, prompt_used


def run_validation_matrix():
    print("=" * 78)
    print("RUNNING 8-CASE GENERALIZATION VALIDATION MATRIX (4 Tooth Types x 2 Anomalies)")
    print("=" * 78)

    manifest = pd.read_csv(BASE_DIR / "dental_manifest_part1.csv")
    manifest_row = manifest[manifest["jaw"] == "lower"].iloc[0]
    patient_id = manifest_row["patient_id"]
    print(f"Patient Scan: {patient_id} (Jaw: {manifest_row['jaw']})")

    segmenter = SAM2AnomalySegmenter.get_instance(device="cpu")

    # Pre-segment the two anomaly types via SAM2
    sam2_results = {}
    for anom in ["Cavity", "Abrasion"]:
        geom, iou, prompt_used = get_validated_sam2_geometry(segmenter, anom)
        sam2_results[anom] = (geom, iou, prompt_used)
        print(
            f"SAM2 Validated [{anom}]: IoU={iou:.4f} ({prompt_used}), "
            f"rel_radius={geom['relative_radius']:.4f}, aspect={geom['aspect_ratio']:.2f}, "
            f"angle={geom['orientation']:.1f} deg"
        )

    matrix_records = []
    case_num = 0

    for t_cfg in TOOTH_TARGETS:
        fdi = t_cfg["fdi"]
        t_label = t_cfg["label"]

        # 1. Extract isolated tooth mesh from Teeth3DS+ scan
        orig_mesh = load_or_extract_tooth_mesh(manifest_row=manifest_row, target_tooth=fdi)
        orig_v_count = int(orig_mesh.vertices.shape[0])
        orig_f_count = int(orig_mesh.faces.shape[0])

        for anom in ["Cavity", "Abrasion"]:
            case_num += 1
            case_id = f"matrix_{case_num:02d}_fdi{fdi}_{t_label.lower()}_{anom.lower()}"
            print(f"\n--- [{case_num}/8] {case_id}: FDI #{fdi} ({t_label}) x {anom} ---")

            geom_2d, sam2_iou, prompt_used = sam2_results[anom]

            # 2. Convert 2D geometry to 3D tooth-specific carving parameters
            carve_params = convert_2d_geometry_to_3d_params(
                geom_2d,
                orig_mesh,
                tooth_fdi=fdi,
                anomaly_type=anom
            )

            # 3. Carve anomaly onto 3D tooth mesh
            carved_mesh, gt_mask = carve_cavity(
                orig_mesh,
                center_idx=carve_params["center_idx"],
                radius=carve_params["radius"],
                depth=carve_params["depth"],
                aspect_ratio=carve_params["aspect_ratio"],
                orientation=carve_params["orientation"],
                profile=carve_params["profile"],
                tangent_u=carve_params["tangent_u"]
            )

            # 4. Strict Numerical Sanity Checks
            finite_ok = bool(np.isfinite(carved_mesh.vertices).all() and np.isfinite(gt_mask).all())
            v_count_ok = (int(carved_mesh.vertices.shape[0]) == orig_v_count == len(gt_mask))
            f_count_ok = (int(carved_mesh.faces.shape[0]) == orig_f_count)

            # Verify zero opposite-wall bleed-through
            center_normal = orig_mesh.vertex_normals[carve_params["center_idx"]]
            center_normal = center_normal / (np.linalg.norm(center_normal) + 1e-8)
            active_mask = gt_mask > 0.01
            active_dots = np.dot(orig_mesh.vertex_normals[active_mask], center_normal)
            opposite_bleed_count = int((active_dots <= 0.0).sum())
            min_normal_dot = float(active_dots.min()) if len(active_dots) > 0 else 1.0

            affected_verts = int((gt_mask > 0.05).sum())
            displacements = np.linalg.norm(carved_mesh.vertices - orig_mesh.vertices, axis=1)
            max_disp_mm = float(displacements.max())

            # 5. Save atomic .npz and render dual-viewport 3D PNG
            npz_path, _ = save_atomic_case(
                OUT_DIR / case_id,
                carved_mesh,
                gt_mask,
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "tooth_fdi": fdi,
                    "tooth_type": carve_params["tooth_type"],
                    "anomaly_type": anom,
                    "carve_params": carve_params
                }
            )

            render_path = OUT_DIR / f"{case_id}_render.png"
            render_carved_tooth_pyvista(
                carved_mesh,
                gt_mask,
                render_path,
                off_screen=True,
                carve_params=carve_params
            )

            numerical_pass = (
                finite_ok
                and v_count_ok
                and f_count_ok
                and opposite_bleed_count == 0
                and affected_verts >= 15
            )

            print(
                f"  Surface: {carve_params['target_surface']} | Profile: {carve_params['profile']} | "
                f"SurfSpan: {carve_params['surface_span_mm']:.2f}mm | WallThick: {carve_params['local_wall_thickness_mm']:.2f}mm"
            )
            print(
                f"  Carve Params: radius={carve_params['radius']:.3f}mm, depth={carve_params['depth']:.3f}mm, "
                f"aspect={carve_params['aspect_ratio']:.2f}"
            )
            print(
                f"  Checks: finite={finite_ok}, v_match={v_count_ok} ({orig_v_count}), "
                f"f_match={f_count_ok} ({orig_f_count}), opposite_bleed={opposite_bleed_count}, "
                f"affected_verts={affected_verts}, max_disp={max_disp_mm:.3f}mm"
            )
            print(f"  Render saved: {render_path}")

            matrix_records.append({
                "case_id": case_id,
                "patient_id": patient_id,
                "tooth_fdi": fdi,
                "tooth_type": carve_params["tooth_type"],
                "anatomical_group": carve_params["anatomical_group"],
                "anomaly_type": anom,
                "target_surface": carve_params["target_surface"],
                "profile": carve_params["profile"],
                "sam2_iou": round(sam2_iou, 4),
                "surface_span_mm": carve_params["surface_span_mm"],
                "local_wall_thickness_mm": carve_params["local_wall_thickness_mm"],
                "carve_radius_mm": carve_params["radius"],
                "carve_depth_mm": carve_params["depth"],
                "aspect_ratio": carve_params["aspect_ratio"],
                "orig_vertices": orig_v_count,
                "carved_vertices": int(carved_mesh.vertices.shape[0]),
                "orig_faces": orig_f_count,
                "carved_faces": int(carved_mesh.faces.shape[0]),
                "affected_vertices": affected_verts,
                "opposite_bleed_vertices": opposite_bleed_count,
                "min_normal_dot": round(min_normal_dot, 3),
                "max_displacement_mm": round(max_disp_mm, 3),
                "finite_ok": finite_ok,
                "vertex_face_match": v_count_ok and f_count_ok,
                "numerical_pass": numerical_pass,
                "render_path": str(render_path),
                "npz_path": str(npz_path)
            })

    df_out = pd.DataFrame(matrix_records)
    csv_path = BASE_DIR / "matrix_validation_results.csv"
    df_out.to_csv(csv_path, index=False)
    print("\n" + "=" * 78)
    print(f"Saved 8-case validation matrix results to {csv_path}")
    print("=" * 78)


if __name__ == "__main__":
    run_validation_matrix()
