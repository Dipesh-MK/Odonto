import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure root in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from whitelist import APPROVED_FDI_TEETH, validate_tooth_anomaly_combination
from teacher_agent_service import InteractiveSAMSegmenter, ClinicalDiagnosticsAgent
from depth_extractor import estimate_raw_depth_map, detrend_enamel_baseline, render_relief_heatmap_overlay
from tooth_carver import (
    load_or_extract_tooth_mesh,
    carve_contour_conforming_defect,
    carve_depth_heightmap_defect,
    render_carved_tooth_pyvista
)

OUT_DIR = BASE_DIR / "demo_shortlist_renders"
OUT_DIR.mkdir(exist_ok=True)

# Candidate list covering 10 Caries and 10 Abrasion
CANDIDATES = [
    # CARIES (Molar & Premolar occlusal/coronal)
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_01_molar_occlusal_table.jpg", "depth_mm": 0.85, "bbox": [0.35, 0.35, 0.65, 0.65]},
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_02_posterior_fossa_pit.jpg", "depth_mm": 0.75, "bbox": [0.35, 0.35, 0.65, 0.65]},
    {"anomaly": "Cavity", "fdi": 14, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_03_maxillary_premolar_cavity.jpg", "depth_mm": 0.70, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_04_cervical_margin_decay.jpg", "depth_mm": 0.80, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_05_deep_dentinal_lesion.jpg", "depth_mm": 1.10, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_06_adolescent_fissure_caries.jpg", "depth_mm": 0.75, "bbox": [0.35, 0.35, 0.65, 0.65]},
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_07_severe_carious_breakdown.png", "depth_mm": 1.15, "bbox": [0.25, 0.25, 0.75, 0.75]},
    {"anomaly": "Cavity", "fdi": 36, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_08_mandibular_molar_cavity.jpg", "depth_mm": 0.85, "bbox": [0.35, 0.35, 0.65, 0.65]},
    {"anomaly": "Cavity", "fdi": 15, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_09_premolar_occlusal_decay.jpg", "depth_mm": 0.70, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Cavity", "fdi": 16, "surface": "occlusal_table", "img_path": "clinical_test_images/caries/caries_10_interproximal_coronal_caries.jpg", "depth_mm": 0.80, "bbox": [0.30, 0.30, 0.70, 0.70]},

    # ABRASION (Anterior Incisors & Canines, labial/facial)
    {"anomaly": "Abrasion", "fdi": 11, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_01_anterior_labial_facet.jpg", "depth_mm": 0.38, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 13, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_02_canine_cervical_wear.jpg", "depth_mm": 0.42, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 11, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_03_attrition_incisal_edge.jpg", "depth_mm": 0.35, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 21, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_04_mechanical_brush_wear.jpg", "depth_mm": 0.40, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 11, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_05_central_incisor_notch.jpg", "depth_mm": 0.38, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 12, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_06_lateral_wear_facet.jpg", "depth_mm": 0.35, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 11, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_07_generalized_incisal_attrition.jpg", "depth_mm": 0.35, "bbox": [0.25, 0.25, 0.75, 0.75]},
    {"anomaly": "Abrasion", "fdi": 11, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_08_cervical_enamel_loss.jpg", "depth_mm": 0.40, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 13, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_09_canine_tip_flattening.jpg", "depth_mm": 0.38, "bbox": [0.30, 0.30, 0.70, 0.70]},
    {"anomaly": "Abrasion", "fdi": 31, "surface": "labial_facial", "img_path": "clinical_test_images/abrasion/abrasion_10_mandibular_anterior_abrasion.jpg", "depth_mm": 0.35, "bbox": [0.30, 0.30, 0.70, 0.70]},
]

def run_evaluation():
    has_gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    print(f"[*] Environment Check: GEMINI_API_KEY present = {has_gemini}")
    agent_mode = "gemini" if has_gemini else "local_expert"
    print(f"[*] Using Clinical Diagnostics Agent mode: {agent_mode}")

    print("\n[*] Initializing Segmenter and Diagnostic Agent...")
    segmenter = InteractiveSAMSegmenter(device="cpu")
    diag_agent = ClinicalDiagnosticsAgent()

    results = []
    
    for i, item in enumerate(CANDIDATES):
        c_id = f"{item['anomaly'].lower()}_{i+1:02d}"
        img_p = BASE_DIR / item["img_path"]
        fdi = item["fdi"]
        surf = item["surface"]
        anom = item["anomaly"]
        depth = item["depth_mm"]
        bbox = item["bbox"]
        
        print(f"\n[{i+1}/{len(CANDIDATES)}] Evaluating: {img_p.name} | FDI #{fdi} | {anom}...")
        
        record = {
            "id": c_id,
            "filename": img_p.name,
            "path": str(img_p),
            "anomaly": anom,
            "tooth_fdi": fdi,
            "surface": surf,
            "status": "INIT",
            "failure_reason": None
        }

        # 1. Image Check
        if not img_p.exists():
            record["status"] = "FAIL"
            record["failure_reason"] = f"Image file not found: {img_p}"
            results.append(record)
            print(f"  ❌ FAILED: {record['failure_reason']}")
            continue

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None or img_bgr.size == 0:
            record["status"] = "FAIL"
            record["failure_reason"] = f"Failed to decode image: {img_p}"
            results.append(record)
            print(f"  ❌ FAILED: {record['failure_reason']}")
            continue

        # 2. Whitelist Check
        is_allowed, wl_reason = validate_tooth_anomaly_combination(fdi, anom)
        if not is_allowed:
            record["status"] = "FAIL"
            record["failure_reason"] = f"Whitelist blocked: {wl_reason}"
            results.append(record)
            print(f"  ❌ FAILED (Whitelist): {record['failure_reason']}")
            continue

        # 3. FastSAM Segmentation
        try:
            sam_res = segmenter.segment_roi(img_bgr, bbox_norm=bbox)
            poly_norm = sam_res.get("polygon_norm", [])
            crop_bgr = sam_res.get("crop_bgr", np.zeros((0,), dtype=np.uint8))
            record["point_count"] = len(poly_norm)
            
            if len(poly_norm) < 3:
                record["status"] = "FAIL"
                record["failure_reason"] = f"FastSAM produced degenerate polygon ({len(poly_norm)} vertices)"
                results.append(record)
                print(f"  ❌ FAILED (FastSAM): {record['failure_reason']}")
                continue
        except Exception as e:
            record["status"] = "FAIL"
            record["failure_reason"] = f"FastSAM segmentation crashed: {e}"
            results.append(record)
            print(f"  ❌ FAILED (FastSAM Crash): {e}")
            continue

        # 4. Clinical Diagnostics Agent
        try:
            diag_res = diag_agent.diagnose_anomaly(
                crop_bgr=crop_bgr,
                tooth_fdi=fdi,
                target_surface=surf,
                force_agent="local_expert"
            )
            record["diagnosis"] = {
                "pathology": diag_res.get("pathology"),
                "severity_grade": diag_res.get("severity_grade"),
                "confidence": diag_res.get("confidence"),
                "recommended_depth_mm": diag_res.get("recommended_depth_mm"),
                "clinical_rationale": diag_res.get("clinical_rationale"),
                "source": diag_res.get("source", "local_expert")
            }
        except Exception as e:
            record["status"] = "FAIL"
            record["failure_reason"] = f"Diagnostics agent crashed: {e}"
            results.append(record)
            print(f"  ❌ FAILED (Diagnostics): {e}")
            continue

        # 5. Depth Relief Extraction
        try:
            ch, cw = crop_bgr.shape[:2]
            crop_mask = np.zeros((ch, cw), dtype=np.uint8)
            poly_arr = np.array(poly_norm, dtype=np.float32)
            min_x, max_x = poly_arr[:, 0].min(), poly_arr[:, 0].max()
            min_y, max_y = poly_arr[:, 1].min(), poly_arr[:, 1].max()
            span_x = max(1e-4, max_x - min_x)
            span_y = max(1e-4, max_y - min_y)
            px = np.clip((poly_arr[:, 0] - min_x) / span_x * (cw - 1), 0, cw - 1)
            py = np.clip((poly_arr[:, 1] - min_y) / span_y * (ch - 1), 0, ch - 1)
            pts = np.round(np.column_stack([px, py])).astype(np.int32)
            cv2.fillPoly(crop_mask, [pts], 255)

            raw_depth = estimate_raw_depth_map(crop_bgr)
            depth_relief, baseline, metrics_depth = detrend_enamel_baseline(raw_depth, crop_mask)
        except Exception as e:
            record["status"] = "FAIL"
            record["failure_reason"] = f"Depth extraction crashed: {e}"
            results.append(record)
            print(f"  ❌ FAILED (Depth): {e}")
            continue

        # 6. 3D Carve Execution (Method A vs Method B)
        try:
            clean_mesh = load_or_extract_tooth_mesh(target_tooth=fdi)
            
            # Method A
            carved_a, inf_a, met_a = carve_contour_conforming_defect(
                mesh=clean_mesh.copy(),
                polygon_norm=poly_norm,
                target_surface=surf,
                depth=depth,
                tooth_fdi=fdi
            )
            
            # Method B
            carved_b, inf_b, met_b = carve_depth_heightmap_defect(
                mesh=clean_mesh.copy(),
                polygon_norm=poly_norm,
                depth_relief_map=depth_relief,
                target_surface=surf,
                depth=depth,
                tooth_fdi=fdi
            )

            v_a = met_a.get("num_displaced_vertices", 0)
            d_a = met_a.get("max_depth_reached_mm", 0.0)
            v_b = met_b.get("num_displaced_vertices", 0)
            d_b = met_b.get("max_depth_reached_mm", 0.0)

            record["method_a"] = {"displaced_vertices": v_a, "max_depth_mm": round(d_a, 3)}
            record["method_b"] = {"displaced_vertices": v_b, "max_depth_mm": round(d_b, 3)}
            
            # Divergence computation
            v_diff_pct = abs(v_a - v_b) / max(1, max(v_a, v_b)) * 100.0
            d_diff = abs(d_a - d_b)
            record["divergence"] = {
                "vertex_diff_pct": round(v_diff_pct, 1),
                "depth_diff_mm": round(d_diff, 3),
                "large_divergence": bool(v_diff_pct > 35.0 or d_diff > 0.3)
            }
        except Exception as e:
            record["status"] = "FAIL"
            record["failure_reason"] = f"3D Carving crashed: {e}"
            results.append(record)
            print(f"  ❌ FAILED (Carving): {e}")
            continue

        # 7. Render 3D Previews (PyVista)
        render_stem = f"preview_{anom.lower()}_{img_p.stem}"
        render_path_a = OUT_DIR / f"{render_stem}_method_a.png"
        render_path_b = OUT_DIR / f"{render_stem}_method_b.png"
        
        try:
            render_carved_tooth_pyvista(carved_a, inf_a, str(render_path_a), off_screen=True, carve_params={"anomaly_type": anom, "depth": depth})
            render_carved_tooth_pyvista(carved_b, inf_b, str(render_path_b), off_screen=True, carve_params={"anomaly_type": anom, "depth": depth})
            
            record["render_method_a"] = str(render_path_a)
            record["render_method_b"] = str(render_path_b)
            record["status"] = "SUCCESS"
            print(f"  ✅ SUCCESS: SAM pts={len(poly_norm)} | A_verts={v_a} | B_verts={v_b} | Div={v_diff_pct:.1f}%")
        except Exception as e:
            record["status"] = "FAIL"
            record["failure_reason"] = f"3D rendering crashed: {e}"
            print(f"  ❌ FAILED (Rendering): {e}")

        results.append(record)

    # Save summary json
    summary_path = OUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Evaluation complete! Processed {len(results)} candidate images.")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'SUCCESS')} / {len(results)}")
    print(f"Summary saved to: {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_evaluation()
