"""
dashboard.py

Standalone Dev-Facing Model Testing & 3D Carving QA Dashboard for Odonto CV Pipeline.
- Checkpoint: runs/fine_tune_3class/yolov8x_3class_model/weights/best.pt (25 ep unweighted)
- Connected Pipelines:
  1. 2D Segmentation: YOLOv8x-seg prediction & mask extraction
  2. Whitelist Gate: Strict validation of Tooth x Anomaly combinations (whitelist.py)
  3. 2D-to-3D Geometry & Parameter Conversion: geometry_extractor.py & parameter_converter.py
  4. 3D Tooth Carving & Multi-View Rendering: tooth_carver.py & PyVista dual-viewport renderer
- Explicit Constraints:
  - Caries maps to Cavity carve profile
  - Abrasion maps to Abrasion carve profile
  - Crack is handled strictly as 2D-only (no 3D carve profile, 11.1% recall warning)
  - Completely isolated from student-facing app.py
"""

import os
import io
import cv2
import json
import base64
import tempfile
import numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from geometry_extractor import extract_anomaly_geometry
from parameter_converter import convert_2d_geometry_to_3d_params, compute_anatomical_frame, raycast_surface_landmark
from tooth_carver import (
    load_or_extract_tooth_mesh,
    carve_cavity,
    carve_v_groove_path,
    carve_chipped_cusp,
    carve_contour_conforming_defect,
    carve_depth_heightmap_defect,
    render_carved_tooth_pyvista
)
from anomaly_profiles import list_profiles, get_profile, ANOMALY_REGISTRY
from whitelist import validate_tooth_anomaly_combination, APPROVED_FDI_TEETH, SUPPORTED_ANOMALIES
from teacher_agent_service import InteractiveSAMSegmenter, ClinicalDiagnosticsAgent, ClinicalTransferAgent
from depth_extractor import estimate_raw_depth_map, detrend_enamel_baseline, render_relief_heatmap_overlay

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "runs" / "fine_tune_3class" / "yolov8x_3class_model" / "weights" / "best.pt"
TEST_IMG_DIR = BASE_DIR / "dataset_3class_seg" / "images" / "test"
TEST_LBL_DIR = BASE_DIR / "dataset_3class_seg" / "labels" / "test"
NOVEL_DIR = BASE_DIR / "novel_samples"
TEMP_DIR = BASE_DIR / "test_3d_renders"
TEMP_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {0: "Abrasion", 1: "Caries", 2: "Crack"}
CLASS_COLORS_BGR = {
    0: (0, 165, 255),   # Orange for Abrasion (BGR)
    1: (38, 38, 220),   # Red for Caries (BGR)
    2: (239, 70, 217),  # Magenta for Crack (BGR)
}
CLASS_COLORS_HEX = {
    0: "#F97316",  # Orange
    1: "#EF4444",  # Red
    2: "#D946EF",  # Magenta
}

# Explicit mapping from 2D YOLO detection class to 3D carve profile
CLASS_TO_CARVE_MAPPING = {
    0: {"name": "Abrasion", "canonical": "Abrasion", "profile": "abrasion", "3d_supported": True},
    1: {"name": "Caries",   "canonical": "Cavity",   "profile": "cavity",   "3d_supported": True},
    2: {"name": "Crack",    "canonical": "Crack",    "profile": None,       "3d_supported": False,
        "reason": "Crack is an experimental 2D-only diagnostic class (11.1% validated recall). It has no registered 3D volumetric carve profile in the clinical whitelist."}
}

app = FastAPI(title="Odonto CV Model & 3D Carve QA Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Model checkpoint not found at: {WEIGHTS_PATH}")

print(f"Loading YOLOv8x-seg checkpoint: {WEIGHTS_PATH} ...")
model = YOLO(str(WEIGHTS_PATH))
print("Model loaded successfully into QA Dashboard.")


def encode_image_to_base64(img_bgr: np.ndarray, format: str = ".jpg") -> str:
    success, buffer = cv2.imencode(format, img_bgr)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format == ".jpg" else "image/png"
    return f"data:{mime};base64,{b64}"


def parse_ground_truth(lbl_path: Path, h: int, w: int):
    gt_masks = {0: np.zeros((h, w), dtype=np.uint8),
                1: np.zeros((h, w), dtype=np.uint8),
                2: np.zeros((h, w), dtype=np.uint8)}
    gt_instances = []
    if not lbl_path.exists():
        return gt_masks, gt_instances

    with open(lbl_path, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cid = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = np.round(coords).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            if cid in gt_masks:
                gt_masks[cid] = np.maximum(gt_masks[cid], mask)
                gt_instances.append({
                    "id": idx + 1,
                    "class_id": cid,
                    "class_name": CLASS_NAMES.get(cid, f"Class {cid}"),
                    "color": CLASS_COLORS_HEX.get(cid, "#FFFFFF"),
                    "area_pixels": int(mask.sum())
                })
    return gt_masks, gt_instances


def run_inference_on_cv2(
    img_bgr: np.ndarray,
    conf_thresh: float = 0.25,
    overlay_alpha: float = 0.5,
    visible_classes: List[int] = None
):
    h, w = img_bgr.shape[:2]
    if visible_classes is None:
        visible_classes = [0, 1, 2]

    res = model.predict(source=img_bgr, conf=conf_thresh, imgsz=640, verbose=False)[0]

    pred_masks = {0: np.zeros((h, w), dtype=np.uint8),
                  1: np.zeros((h, w), dtype=np.uint8),
                  2: np.zeros((h, w), dtype=np.uint8)}
    instances = []

    overlay_bgr = img_bgr.copy()
    mask_layer = np.zeros_like(img_bgr)
    has_any_mask = False

    if res.boxes is not None and len(res.boxes) > 0 and res.masks is not None:
        classes = res.boxes.cls.tolist()
        confidences = res.boxes.conf.tolist()
        xyxy_boxes = res.boxes.xyxy.tolist()
        polygons = res.masks.xy

        for idx, (c, conf, box, poly) in enumerate(zip(classes, confidences, xyxy_boxes, polygons)):
            cid = int(c)
            if len(poly) < 3 or cid not in visible_classes:
                continue

            pts = np.round(poly).astype(np.int32)
            inst_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(inst_mask, [pts], 1)
            pred_masks[cid] = np.maximum(pred_masks[cid], inst_mask)

            color_bgr = CLASS_COLORS_BGR.get(cid, (0, 255, 0))
            mask_layer[inst_mask > 0] = color_bgr
            has_any_mask = True

            cv2.polylines(overlay_bgr, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

            bx1, by1, bx2, by2 = [int(v) for v in box]
            label = f"{CLASS_NAMES.get(cid, str(cid))} {conf:.1%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay_bgr, (bx1, max(0, by1 - 20)), (bx1 + lw + 8, max(0, by1)), color_bgr, -1)
            cv2.putText(overlay_bgr, label, (bx1 + 4, max(14, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            is_crack = (cid == 2)
            mapping = CLASS_TO_CARVE_MAPPING.get(cid, {})
            instances.append({
                "id": idx + 1,
                "class_id": cid,
                "class_name": CLASS_NAMES.get(cid, f"Class {cid}"),
                "canonical_anomaly": mapping.get("canonical"),
                "can_carve_3d": mapping.get("3d_supported", False),
                "confidence": round(float(conf), 4),
                "box": [bx1, by1, bx2, by2],
                "area_pixels": int(inst_mask.sum()),
                "color": CLASS_COLORS_HEX.get(cid, "#FFFFFF"),
                "is_crack": is_crack,
                "warning": "⚠️ LOW VALIDATED RECALL (11.1%) — Experimental hypothesis only. High risk of false positive / missed fracture." if is_crack else None
            })

    if has_any_mask:
        overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, mask_layer, overlay_alpha, 0)

    return overlay_bgr, pred_masks, instances, res


def render_ground_truth_overlay(img_bgr: np.ndarray, gt_masks: dict, overlay_alpha: float = 0.5):
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()
    mask_layer = np.zeros_like(img_bgr)
    has_any = False

    for cid, m in gt_masks.items():
        if m.sum() > 0:
            color = CLASS_COLORS_BGR.get(cid, (0, 255, 0))
            mask_layer[m > 0] = color
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
            has_any = True

    if has_any:
        return cv2.addWeighted(canvas, 1.0, mask_layer, overlay_alpha, 0)
    return canvas


def calculate_per_class_iou(gt_masks: dict, pred_masks: dict):
    results = {}
    for cid in [0, 1, 2]:
        cname = CLASS_NAMES[cid]
        gt_sum = gt_masks[cid].sum()
        pred_sum = pred_masks[cid].sum()
        if gt_sum == 0 and pred_sum == 0:
            results[cname.lower()] = None
        else:
            inter = np.logical_and(gt_masks[cid], pred_masks[cid]).sum()
            union = np.logical_or(gt_masks[cid], pred_masks[cid]).sum()
            iou = float(inter) / float(union) if union > 0 else 0.0
            results[cname.lower()] = round(iou, 4)
    return results


# API Endpoints
@app.get("/api/samples")
def get_sample_presets():
    """Returns curated presets and test images."""
    curated = [
        # Crack presets
        {"stem": "dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7", "name": "DentalAI Crack #1 (Verified TP, IoU=0.748)", "category": "Crack"},
        {"stem": "dental_1282_jpg.rf.e0b8c616bbb750f4e4d63d90475ad90d", "name": "DentalAI Crack #2 (Vertical Fracture Line)", "category": "Crack"},
        {"stem": "dental_1350_jpg.rf.a52d104621129d7616f64dcaea88a506", "name": "DentalAI Crack #3 (Coronal Hairline Crack)", "category": "Crack"},
        {"stem": "dental_1400_jpg.rf.2fd49afa2c11b123a730872e1f714d09", "name": "DentalAI Crack #4 (Subtle Enamel Fissure)", "category": "Crack"},
        # Abrasion presets
        {"stem": "alpha_p006_F_15_002", "name": "AlphaDent Abrasion #1 (Cervical Notch)", "category": "Abrasion"},
        {"stem": "alpha_p008_M_26_001", "name": "AlphaDent Abrasion #2 (Multi-Tooth Wear)", "category": "Abrasion"},
        {"stem": "alpha_p008_M_26_002", "name": "AlphaDent Abrasion #3 (Gingival Enamel Loss)", "category": "Abrasion"},
        {"stem": "alpha_p008_M_26_003", "name": "AlphaDent Abrasion #4 (Severe Cervical Wear)", "category": "Abrasion"},
        # Caries presets
        {"stem": "alpha_p006_F_15_001", "name": "AlphaDent Caries #1 (Deep Occlusal Cavity)", "category": "Caries"},
        {"stem": "alpha_p022_F_28_002", "name": "AlphaDent Caries #2 (Molar Pit & Fissure)", "category": "Caries"},
        {"stem": "alpha_p022_F_28_003", "name": "AlphaDent Caries #3 (Interproximal Shadowing)", "category": "Caries"},
        {"stem": "alpha_p025_F_32_001", "name": "AlphaDent Caries #4 (Gross Carious Breakdown)", "category": "Caries"},
        # Novel out-of-dataset presets
        {"stem": "novel_01_alphadent_filling_case", "name": "Novel: Dental Filling (Excluded from training)", "category": "Novel / Out-of-Dataset"},
        {"stem": "novel_02_alphadent_crown_case", "name": "Novel: Dental Crown (Excluded from training)", "category": "Novel / Out-of-Dataset"},
        {"stem": "novel_03_dentalai_healthy_case", "name": "Novel: Healthy Dentition (Excluded from training)", "category": "Novel / Out-of-Dataset"},
        {"stem": "novel_04_dentalai_healthy_anterior", "name": "Novel: Anterior Teeth (Excluded from training)", "category": "Novel / Out-of-Dataset"},
    ]
    return {"curated": curated}


@app.get("/api/teeth")
def get_approved_teeth():
    """Returns the 28 approved FDI permanent teeth from the whitelist."""
    teeth_list = []
    for fdi, meta in sorted(APPROVED_FDI_TEETH.items()):
        teeth_list.append({
            "fdi": fdi,
            "name": meta["name"],
            "short_name": meta["short_name"],
            "quadrant": meta["quadrant"],
            "arch": meta.get("arch", ""),
            "tooth_type": meta["tooth_type"],
            "anatomical_group": meta["anatomical_group"],
            "surface": meta["surface"],
            "allowed_anomalies": meta["allowed_anomalies"]
        })
    return {"teeth": teeth_list}


@app.get("/api/inspect/preset/{stem}")
def inspect_preset_sample(
    stem: str,
    conf: float = 0.25,
    opacity: float = 0.5,
    show_abrasion: bool = True,
    show_caries: bool = True,
    show_crack: bool = True,
):
    visible = []
    if show_abrasion: visible.append(0)
    if show_caries: visible.append(1)
    if show_crack: visible.append(2)

    img_p = NOVEL_DIR / f"{stem}.jpg"
    is_novel = img_p.exists()
    lbl_p = None

    if not is_novel:
        img_p = TEST_IMG_DIR / f"{stem}.jpg"
        if not img_p.exists():
            img_p = TEST_IMG_DIR / f"{stem}.png"
        lbl_p = TEST_LBL_DIR / f"{stem}.txt"

    if not img_p.exists():
        raise HTTPException(status_code=404, detail=f"Image {stem} not found")

    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to decode image from disk")
    h, w = img_bgr.shape[:2]

    overlay_bgr, pred_masks, instances, _ = run_inference_on_cv2(img_bgr, conf, opacity, visible)

    has_gt = (lbl_p is not None and lbl_p.exists())
    gt_masks = {0: np.zeros((h, w), dtype=np.uint8), 1: np.zeros((h, w), dtype=np.uint8), 2: np.zeros((h, w), dtype=np.uint8)}
    gt_instances = []
    gt_overlay_b64 = None
    ious = {"abrasion": None, "caries": None, "crack": None}

    if has_gt:
        gt_masks, gt_instances = parse_ground_truth(lbl_p, h, w)
        gt_overlay_bgr = render_ground_truth_overlay(img_bgr, gt_masks, opacity)
        gt_overlay_b64 = encode_image_to_base64(gt_overlay_bgr)
        ious = calculate_per_class_iou(gt_masks, pred_masks)

    has_crack_pred = any(inst["is_crack"] for inst in instances)

    # Suggest recommended default tooth for 3D carve based on detected anomaly
    recommended_fdi = 36
    if instances:
        first_cid = instances[0]["class_id"]
        if first_cid == 0:  # Abrasion -> Incisor #11 or Canine #33
            recommended_fdi = 11
        elif first_cid == 1:  # Caries -> Molar #36 or Incisor #11
            recommended_fdi = 36

    return {
        "stem": stem,
        "is_novel": is_novel,
        "dimensions": {"width": w, "height": h},
        "original_image": encode_image_to_base64(img_bgr),
        "prediction_overlay": encode_image_to_base64(overlay_bgr),
        "ground_truth_overlay": gt_overlay_b64,
        "has_ground_truth": has_gt,
        "ious": ious,
        "instances": instances,
        "recommended_fdi": recommended_fdi,
        "has_crack_prediction": has_crack_pred,
        "crack_warning_active": has_crack_pred or ("crack" in stem.lower()),
        "status_message": "Benchmark test sample with ground truth comparison." if has_gt else "Novel out-of-dataset dental sample. Ground truth unavailable."
    }


@app.post("/api/inspect/upload")
async def inspect_uploaded_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    opacity: float = Form(0.5),
    show_abrasion: bool = Form(True),
    show_caries: bool = Form(True),
    show_crack: bool = Form(True),
):
    visible = []
    if show_abrasion: visible.append(0)
    if show_caries: visible.append(1)
    if show_crack: visible.append(2)

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    h, w = img_bgr.shape[:2]

    overlay_bgr, pred_masks, instances, _ = run_inference_on_cv2(img_bgr, conf, opacity, visible)
    has_crack_pred = any(inst["is_crack"] for inst in instances)

    recommended_fdi = 36
    if instances:
        first_cid = instances[0]["class_id"]
        if first_cid == 0:
            recommended_fdi = 11
        elif first_cid == 1:
            recommended_fdi = 36

    return {
        "filename": file.filename,
        "is_novel": True,
        "dimensions": {"width": w, "height": h},
        "original_image": encode_image_to_base64(img_bgr),
        "prediction_overlay": encode_image_to_base64(overlay_bgr),
        "ground_truth_overlay": None,
        "has_ground_truth": False,
        "ious": {"abrasion": None, "caries": None, "crack": None},
        "instances": instances,
        "recommended_fdi": recommended_fdi,
        "has_crack_prediction": has_crack_pred,
        "crack_warning_active": has_crack_pred,
        "status_message": "Custom user upload: No ground truth available. Model is operating on arbitrary novel input."
    }


@app.post("/api/carve")
async def carve_3d_anomaly(
    stem: Optional[str] = Form(None),
    tooth_fdi: int = Form(...),
    instance_idx: int = Form(0),
    conf: float = Form(0.25),
    show_abrasion: bool = Form(True),
    show_caries: bool = Form(True),
    show_crack: bool = Form(True),
    file: Optional[UploadFile] = File(None)
):
    """
    Executes the 2D-to-3D carving pipeline:
    1. Obtains 2D prediction mask from model.
    2. Enforces Whitelist: checks (tooth_fdi, canonical_anomaly).
    3. Handles Crack explicitly as 2D-only.
    4. Converts 2D geometry into 3D carving parameters.
    5. Carves base mesh and generates PyVista dual-viewport render.
    """
    img_bgr = None
    if file is not None and file.filename:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif stem:
        img_p = NOVEL_DIR / f"{stem}.jpg"
        if not img_p.exists():
            img_p = TEST_IMG_DIR / f"{stem}.jpg"
            if not img_p.exists():
                img_p = TEST_IMG_DIR / f"{stem}.png"
        if img_p.exists():
            img_bgr = cv2.imread(str(img_p))

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No valid image supplied for 3D carving")

    h, w = img_bgr.shape[:2]

    # Run inference to retrieve polygon and masks
    res = model.predict(source=img_bgr, conf=conf, imgsz=640, verbose=False)[0]

    if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
        return {
            "status": "no_anomaly",
            "is_blocked": True,
            "message": "No anomaly detected in the image at the selected confidence threshold. Cannot carve 3D tooth.",
            "tooth_fdi": tooth_fdi
        }

    classes = res.boxes.cls.tolist()
    confidences = res.boxes.conf.tolist()
    polygons = res.masks.xy

    visible = []
    if show_abrasion: visible.append(0)
    if show_caries: visible.append(1)
    if show_crack: visible.append(2)

    filtered_classes = []
    filtered_confidences = []
    filtered_polygons = []
    for c, cval, poly in zip(classes, confidences, polygons):
        if int(c) in visible:
            filtered_classes.append(c)
            filtered_confidences.append(cval)
            filtered_polygons.append(poly)

    if filtered_classes:
        classes = filtered_classes
        confidences = filtered_confidences
        polygons = filtered_polygons

    # Pick target instance
    idx = max(0, min(instance_idx, len(classes) - 1))
    cid = int(classes[idx])
    inst_conf = float(confidences[idx])
    poly = polygons[idx]

    cname = CLASS_NAMES.get(cid, "Unknown")
    mapping = CLASS_TO_CARVE_MAPPING.get(cid, {})
    canonical_anomaly = mapping.get("canonical", "Cavity")

    # REQUIREMENT 5: Handle Crack explicitly as 2D-only
    if cid == 2 or not mapping.get("3d_supported", False):
        return {
            "status": "crack_2d_only",
            "is_blocked": True,
            "error_code": "CRACK_2D_ONLY",
            "class_name": "Crack",
            "tooth_fdi": tooth_fdi,
            "message": "3D rendering is not available for Crack.",
            "advisory": (
                "Crack has no registered volumetric carve profile in the Odonto clinical whitelist and was never part of Phase 0 3D validation. "
                "Per clinical safety rules, Crack is strictly maintained as an experimental 2D diagnostic hypothesis (11.1% validated recall) "
                "and is never forced through cavity or abrasion carving profiles."
            )
        }

    # REQUIREMENT 4: Enforce Whitelist and show clear message when blocked
    is_valid, wl_details = validate_tooth_anomaly_combination(tooth_fdi, canonical_anomaly)
    if not is_valid:
        return {
            "status": "whitelist_blocked",
            "is_blocked": True,
            "error_code": wl_details.get("code", "WHITELIST_BLOCKED"),
            "class_name": cname,
            "canonical_anomaly": canonical_anomaly,
            "tooth_fdi": tooth_fdi,
            "tooth_name": wl_details.get("tooth_name", f"Tooth {tooth_fdi}"),
            "allowed_anomalies": wl_details.get("allowed_anomalies_for_tooth", []),
            "message": f"Whitelist Blocked: {wl_details.get('error', 'Combination is not clinically approved.')}",
            "advisory": "The Odonto whitelist prohibits this combination to prevent non-anatomical saucer defects on posterior wear or unvalidated 3rd molars."
        }

    # Extract 2D Mask & Geometry
    inst_mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) >= 3:
        pts = np.round(poly).astype(np.int32)
        cv2.fillPoly(inst_mask, [pts], 1)

    geom = extract_anomaly_geometry(inst_mask, (h, w))

    # Load 3D base mesh
    base_mesh = load_or_extract_tooth_mesh(target_tooth=tooth_fdi)

    # Convert 2D geometry to 3D carve params
    carve_params = convert_2d_geometry_to_3d_params(
        geom,
        base_mesh,
        tooth_fdi=tooth_fdi,
        anomaly_type=canonical_anomaly
    )

    # Execute 3D Carve
    carve_kwargs = {
        "center_idx": carve_params["center_idx"],
        "radius": carve_params["radius"],
        "depth": carve_params["depth"],
        "aspect_ratio": carve_params.get("aspect_ratio", 1.0),
        "orientation": carve_params.get("orientation", 0.0),
        "profile": carve_params.get("profile", "cavity"),
        "tangent_u": carve_params.get("tangent_u"),
        "mask_2d": carve_params.get("mask_2d"),
    }

    carved_mesh, gt_mask = carve_cavity(base_mesh, **carve_kwargs)

    # Render via PyVista off-screen
    render_filename = f"render_carve_{tooth_fdi}_{canonical_anomaly.lower()}_{os.getpid()}.png"
    render_path = TEMP_DIR / render_filename
    render_carved_tooth_pyvista(
        carved_mesh=carved_mesh,
        gt_mask=gt_mask,
        output_img_path=str(render_path),
        off_screen=True,
        carve_params=carve_params
    )

    if not render_path.exists():
        raise HTTPException(status_code=500, detail="Failed to render 3D carved mesh via PyVista")

    render_bgr = cv2.imread(str(render_path))
    render_b64 = encode_image_to_base64(render_bgr, format=".png")

    try:
        render_path.unlink()  # Clean up temp file
    except:
        pass

    influenced_count = int((gt_mask > 0.05).sum())

    return {
        "status": "carved_success",
        "is_blocked": False,
        "tooth_fdi": tooth_fdi,
        "tooth_name": wl_details.get("tooth_name"),
        "tooth_type": wl_details.get("tooth_type"),
        "anatomical_group": wl_details.get("anatomical_group"),
        "target_surface": wl_details.get("target_surface"),
        "canonical_anomaly": canonical_anomaly,
        "detected_class": cname,
        "confidence": round(inst_conf, 4),
        "profile": carve_params.get("profile"),
        "render_image": render_b64,
        "metrics": {
            "radius_mm": round(float(carve_params["radius"]), 2),
            "depth_mm": round(float(carve_params["depth"]), 3),
            "aspect_ratio": round(float(carve_params["aspect_ratio"]), 2),
            "orientation_deg": round(float(carve_params["orientation"]), 1),
            "influenced_vertices": influenced_count,
            "total_vertices": len(carved_mesh.vertices),
            "center_vertex_idx": int(carve_params["center_idx"])
        },
        "notes": carve_params.get("notes", ""),
        "message": f"Successfully mapped 2D {cname} detection onto 3D FDI #{tooth_fdi} ({wl_details.get('tooth_name')}) on {wl_details.get('target_surface')} surface."
    }


@app.get("/api/profiles")
def get_registered_profiles():
    return {"profiles": list_profiles()}


@app.post("/api/carve_manual")
async def carve_manual_anomaly(
    profile_id: str = Form(...),
    tooth_fdi: int = Form(...),
    crack_depth: float = Form(0.35),
    crack_width: float = Form(0.45),
    crack_preset: str = Form("occlusal_fissure"),
    points_2d_json: Optional[str] = Form(None),
    cusp_target: str = Form("mesiobuccal"),
    chip_height: float = Form(1.4),
    chip_angle: float = Form(35.0)
):
    """
    Executes developer-facing manual 3D anomaly sculpting for non-whitelisted or experimental
    profiles (v_groove_path for Crack, face_removal_recap for Chipped Cusp).
    Carries explicit 'Manual Placement (Experimental)' metadata and badge.
    """
    from scipy.spatial import cKDTree

    base_mesh = load_or_extract_tooth_mesh(target_tooth=tooth_fdi)
    badge = "Manual Placement (Experimental) — Not Whitelisted for Autonomous Student Mode"

    if profile_id == "v_groove_path":
        frame = compute_anatomical_frame(base_mesh)
        v = base_mesh.vertices
        tree = cKDTree(v)
        c = frame["center"]
        e_md = frame["e_md"]
        e_facial = frame["e_facial"]
        e_crown = frame["e_crown"]

        pts_custom = []
        is_photo_derived = False
        if points_2d_json:
            try:
                raw_pts = json.loads(points_2d_json)
                if isinstance(raw_pts, list) and len(raw_pts) >= 2:
                    raw_arr = np.array(raw_pts, dtype=np.float64)
                    xs, ys = raw_arr[:, 0], raw_arr[:, 1]
                    x_min, x_max = float(xs.min()), float(xs.max())
                    y_min, y_max = float(ys.min()), float(ys.max())
                    x_span = x_max - x_min
                    y_span = y_max - y_min

                    # If points are localized to a tooth subregion on photo (span < 0.45),
                    # scale relative progression across the tooth crown [0.15, 0.85]
                    if x_span < 0.45 and y_span < 0.45 and (x_span > 1e-4):
                        norm_xs = 0.15 + 0.70 * (xs - x_min) / x_span
                        norm_ys = 0.20 + 0.60 * (ys - y_min) / max(y_span, 1e-4) if y_span > 1e-4 else np.full_like(ys, 0.5)
                    else:
                        norm_xs = np.clip(xs, 0.05, 0.95)
                        norm_ys = np.clip(ys, 0.05, 0.95)

                    surface = "occlusal" if (tooth_fdi % 10 >= 4) else "buccal"
                    for rx, ry in zip(norm_xs, norm_ys):
                        res = raycast_surface_landmark(base_mesh, surface=surface, rel_coords=(float(rx), float(ry)))
                        pts_custom.append(v[res["center_idx"]])
                    is_photo_derived = True
            except Exception:
                pts_custom = []

        if is_photo_derived and len(pts_custom) >= 2:
            path_pts = np.array(pts_custom)
            path_source = f"🟢 Photo-Derived ({len(pts_custom)} Traced Points)"
        else:
            is_photo_derived = False
            if crack_preset == "labial_fracture" or (tooth_fdi % 10 <= 3):
                p1 = c + 0.3 * e_md + 2.5 * e_facial + 3.8 * e_crown
                p2 = c + 0.1 * e_md + 2.6 * e_facial + 1.2 * e_crown
                p3 = c - 0.2 * e_md + 2.4 * e_facial - 1.5 * e_crown
                test_pts = [p1, p2, p3]
                path_source = "🟡 [DEMO PRESET — Not Photo-Derived] Coronal Labial Hairline Fracture"
            elif crack_preset == "marginal_ridge":
                p1 = c + 2.8 * e_md + 1.2 * e_facial + 2.0 * e_crown
                p2 = c + 2.5 * e_md + 0.0 * e_facial + 2.0 * e_crown
                p3 = c + 2.2 * e_md - 1.2 * e_facial + 2.0 * e_crown
                test_pts = [p1, p2, p3]
                path_source = "🟡 [DEMO PRESET — Not Photo-Derived] Marginal Ridge Stress Fracture"
            else:
                p1 = c + 2.5 * e_md + 0.2 * e_facial + 2.5 * e_crown
                p2 = c + 1.0 * e_md + 0.0 * e_facial + 2.5 * e_crown
                p3 = c - 0.8 * e_md - 0.2 * e_facial + 2.5 * e_crown
                p4 = c - 2.5 * e_md + 0.1 * e_facial + 2.5 * e_crown
                test_pts = [p1, p2, p3, p4]
                path_source = "🟡 [DEMO PRESET — Not Photo-Derived] Occlusal Fissure Structural Crack"

            snapped = [tree.query(p)[1] for p in test_pts]
            path_pts = v[snapped]

        carved_mesh, influence = carve_v_groove_path(base_mesh, path_pts, depth=crack_depth, width=crack_width)
        profile_info = get_profile("v_groove_path")
        target_surface = "occlusal" if (tooth_fdi % 10 >= 4) else "labial"
        metrics = {
            "depth_mm": round(float(crack_depth), 3),
            "width_mm": round(float(crack_width), 3),
            "path_type": path_source,
            "is_photo_derived": is_photo_derived,
            "traced_points_count": len(pts_custom) if is_photo_derived else 0,
            "source_type": f"Photo-Derived ({len(pts_custom)} pts)" if is_photo_derived else "Demo Preset (Generic)",
            "influenced_vertices": int((influence > 0.05).sum()),
            "total_vertices": len(carved_mesh.vertices)
        }

    elif profile_id == "face_removal_recap":
        carved_mesh, influence = carve_chipped_cusp(
            base_mesh,
            cusp_target=cusp_target,
            chip_height=chip_height,
            chip_angle=chip_angle
        )
        profile_info = get_profile("face_removal_recap")
        target_surface = "occlusal / cusp tip"
        metrics = {
            "chip_height_mm": round(float(chip_height), 2),
            "chip_angle_deg": round(float(chip_angle), 1),
            "target_cusp": cusp_target,
            "is_photo_derived": False,
            "source_type": "Manual Cusp Selection",
            "influenced_vertices": int((influence > 0.05).sum()),
            "original_vertices": len(base_mesh.vertices),
            "capped_vertices": len(carved_mesh.vertices)
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported manual profile: {profile_id}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        render_tmp_path = Path(tmp.name)

    render_carved_tooth_pyvista(carved_mesh, influence, render_tmp_path, off_screen=True)
    render_bgr = cv2.imread(str(render_tmp_path))
    render_b64 = encode_image_to_base64(render_bgr, format=".png")
    try:
        render_tmp_path.unlink()
    except Exception:
        pass

    return {
        "status": "carved_success",
        "is_manual": True,
        "is_photo_derived": is_photo_derived if profile_id == "v_groove_path" else False,
        "source_label": (f"🟢 Photo-Derived ({len(pts_custom)} Traced Points)" if (profile_id == "v_groove_path" and is_photo_derived) else "🟡 Demo Preset (Generic — Not Photo-Derived)"),
        "badge": badge,
        "profile_id": profile_id,
        "profile_name": profile_info.display_name if profile_info else profile_id,
        "tooth_fdi": tooth_fdi,
        "target_surface": target_surface,
        "render_image": render_b64,
        "metrics": metrics,
        "message": f"Executed manual {profile_info.display_name if profile_info else profile_id} on FDI #{tooth_fdi}."
    }


@app.post("/api/inspect/batch")
async def batch_inference_presets(
    stems_json: str = Form("[]"),
    conf: float = Form(0.25)
):
    try:
        stems = json.loads(stems_json)
    except:
        stems = []

    if not stems:
        stems = [
            "dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7",
            "dental_1282_jpg.rf.e0b8c616bbb750f4e4d63d90475ad90d",
            "alpha_p006_F_15_002",
            "alpha_p008_M_26_001",
            "alpha_p006_F_15_001",
            "alpha_p022_F_28_002",
            "novel_01_alphadent_filling_case",
            "novel_03_dentalai_healthy_case"
        ]

    batch_results = []
    for stem in stems:
        try:
            res = inspect_preset_sample(stem, conf=conf, opacity=0.5)
            batch_results.append({
                "stem": stem,
                "is_novel": res["is_novel"],
                "prediction_overlay": res["prediction_overlay"],
                "instances": res["instances"],
                "ious": res["ious"],
                "has_crack": res["has_crack_prediction"],
                "status": res["status_message"]
            })
        except Exception as e:
            continue

    return {"results": batch_results}


# ============================================================================
# PHASE 1 & 2: TEACHER-GUIDED LOCALIZATION & CLINICAL AGENT ENDPOINTS
# ============================================================================

sam_segmenter = InteractiveSAMSegmenter(device="cpu", imgsz=640)
clinical_agent = ClinicalDiagnosticsAgent()
transfer_agent = ClinicalTransferAgent()


@app.post("/api/teacher/extract_depth")
async def teacher_extract_depth(request: Request):
    """
    Method B: Monocular Depth Estimation & Local Enamel Detrending.
    Extracts dense heightmap relief Δz(u, v) using Depth Anything V2.
    """
    body = await request.json()
    crop_base64 = body.get("crop_base64")
    polygon_norm = body.get("polygon_norm")
    if not crop_base64:
        raise HTTPException(status_code=400, detail="crop_base64 is required.")

    data_str = crop_base64.split(",", 1)[1] if "," in crop_base64 else crop_base64
    img_bytes = base64.b64decode(data_str)
    crop_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if crop_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid crop image data.")

    ch, cw = crop_bgr.shape[:2]
    crop_mask = np.zeros((ch, cw), dtype=np.uint8)
    if polygon_norm and len(polygon_norm) >= 3:
        poly_arr = np.array(polygon_norm, dtype=np.float32)
        min_x, max_x = poly_arr[:, 0].min(), poly_arr[:, 0].max()
        min_y, max_y = poly_arr[:, 1].min(), poly_arr[:, 1].max()
        span_x = max(1e-4, max_x - min_x)
        span_y = max(1e-4, max_y - min_y)
        px = np.clip((poly_arr[:, 0] - min_x) / span_x * (cw - 1), 0, cw - 1)
        py = np.clip((poly_arr[:, 1] - min_y) / span_y * (ch - 1), 0, ch - 1)
        pts = np.round(np.column_stack([px, py])).astype(np.int32)
        cv2.fillPoly(crop_mask, [pts], 255)
    else:
        cv2.ellipse(crop_mask, (cw // 2, ch // 2), (int(cw * 0.35), int(ch * 0.35)), 0, 0, 360, 255, -1)

    raw_depth = estimate_raw_depth_map(crop_bgr)
    delta_z, baseline, metrics = detrend_enamel_baseline(raw_depth, crop_mask)
    heatmap_overlay = render_relief_heatmap_overlay(crop_bgr, delta_z, crop_mask)

    # Downsample delta_z to 96x96 for fast JSON transfer
    delta_z_thumb = cv2.resize(delta_z, (96, 96), interpolation=cv2.INTER_AREA)

    return {
        "success": True,
        "heatmap_base64": encode_image_to_base64(heatmap_overlay),
        "depth_relief_grid": delta_z_thumb.tolist(),
        "metrics": metrics
    }


@app.post("/api/teacher/agent_recommend")
async def teacher_agent_recommend(request: Request):
    """
    Phase 2 (Transfer Agent): Evaluates anomaly pathology and recommends
    whitelist-approved target teeth for simulation placement.
    """
    body = await request.json()
    pathology = body.get("pathology", "Caries")
    severity = body.get("severity_grade", "")
    current_fdi = body.get("current_fdi")
    rec = transfer_agent.recommend_target_teeth(pathology, severity, current_fdi)
    return {"success": True, **rec}


@app.post("/api/teacher/carve")
async def teacher_contour_carve(request: Request):
    """
    Phase 3: Executes 3D depth mesh deformation.
    Supports both Method A (Contour-Conforming Parametric) and
    Method B (Dense Depth-Relief Conformal Pasting).
    """
    body = await request.json()
    polygon_norm = body.get("polygon_norm")
    tooth_fdi = int(body.get("tooth_fdi", 16))
    target_surface = body.get("target_surface", "occlusal_table")
    depth_mm = float(body.get("depth_mm", 0.85))
    margin_steepness = float(body.get("margin_steepness", 0.70))
    floor_morphology = body.get("floor_morphology", "irregular_concave")
    pathology = body.get("pathology", "Caries")
    method = body.get("method", "contour_conforming")
    depth_relief_grid = body.get("depth_relief_grid")

    if not polygon_norm or len(polygon_norm) < 3:
        raise HTTPException(status_code=400, detail="Valid 2D SAM polygon coordinates required.")

    # 1. Whitelist Verification
    is_valid, whitelist_meta = validate_tooth_anomaly_combination(tooth_fdi, pathology)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail=f"Clinical Whitelist Violation: {whitelist_meta.get('reason', 'Not approved for this tooth.')}"
        )

    # 2. Load Real 3D Tooth Mesh
    try:
        tooth_mesh = load_or_extract_tooth_mesh(target_tooth=tooth_fdi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tooth mesh FDI #{tooth_fdi}: {e}")

    # 3. Execute Selected Method
    try:
        if method == "depth_heightmap" and depth_relief_grid is not None:
            depth_map = np.array(depth_relief_grid, dtype=np.float32)
            carved_mesh, influence, metrics = carve_depth_heightmap_defect(
                mesh=tooth_mesh,
                polygon_norm=polygon_norm,
                depth_relief_map=depth_map,
                target_surface=target_surface,
                depth=depth_mm,
                tooth_fdi=tooth_fdi
            )
        else:
            carved_mesh, influence, metrics = carve_contour_conforming_defect(
                mesh=tooth_mesh,
                polygon_norm=polygon_norm,
                target_surface=target_surface,
                depth=depth_mm,
                margin_steepness=margin_steepness,
                floor_morphology=floor_morphology,
                tooth_fdi=tooth_fdi
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D Carving algorithm failed: {e}")

    # 4. Render 2-Viewport PyVista Visualization
    render_img_name = f"teacher_carve_fdi_{tooth_fdi}_{os.getpid()}.png"
    render_path = TEMP_DIR / render_img_name
    render_carved_tooth_pyvista(
        carved_mesh=carved_mesh,
        gt_mask=influence,
        output_img_path=str(render_path),
        carve_params={"anomaly_type": pathology, "depth": depth_mm}
    )

    render_b64 = ""
    if render_path.exists():
        img_bgr = cv2.imread(str(render_path))
        if img_bgr is not None:
            render_b64 = encode_image_to_base64(img_bgr, format=".png")

    tooth_meta = APPROVED_FDI_TEETH.get(tooth_fdi, {})
    tooth_name = tooth_meta.get("name", f"Tooth #{tooth_fdi}")

    return {
        "success": True,
        "render_base64": render_b64,
        "tooth_fdi": tooth_fdi,
        "tooth_name": tooth_name,
        "pathology": pathology,
        "depth_mm": depth_mm,
        "method": method,
        "metrics": metrics,
        "whitelist_meta": whitelist_meta,
    }


@app.post("/api/teacher/compare_methods")
async def teacher_compare_methods(request: Request):
    """
    Executes BOTH Method A (Parametric Contour-Conforming) and
    Method B (Dense Monocular Depth-Relief Transfer) on the target tooth mesh,
    and returns PyVista 3D renders side by side with comparison metrics.
    """
    body = await request.json()
    polygon_norm = body.get("polygon_norm")
    tooth_fdi = int(body.get("tooth_fdi", 16))
    target_surface = body.get("target_surface", "occlusal_table")
    depth_mm = float(body.get("depth_mm", 0.85))
    margin_steepness = float(body.get("margin_steepness", 0.70))
    floor_morphology = body.get("floor_morphology", "irregular_concave")
    pathology = body.get("pathology", "Caries")
    depth_relief_grid = body.get("depth_relief_grid")

    if not polygon_norm or len(polygon_norm) < 3:
        raise HTTPException(status_code=400, detail="Valid 2D SAM polygon coordinates required.")

    is_valid, whitelist_meta = validate_tooth_anomaly_combination(tooth_fdi, pathology)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail=f"Clinical Whitelist Violation: {whitelist_meta.get('reason', 'Not approved for this tooth.')}"
        )

    tooth_mesh = load_or_extract_tooth_mesh(target_tooth=tooth_fdi)

    # 1. Execute Method A (Parametric)
    carved_mesh_a, inf_a, metrics_a = carve_contour_conforming_defect(
        mesh=tooth_mesh.copy(),
        polygon_norm=polygon_norm,
        target_surface=target_surface,
        depth=depth_mm,
        margin_steepness=margin_steepness,
        floor_morphology=floor_morphology,
        tooth_fdi=tooth_fdi
    )

    path_a = TEMP_DIR / f"compare_method_a_{tooth_fdi}_{os.getpid()}.png"
    render_carved_tooth_pyvista(carved_mesh_a, inf_a, str(path_a), carve_params={"anomaly_type": pathology, "depth": depth_mm})
    render_a_b64 = encode_image_to_base64(cv2.imread(str(path_a)), format=".png") if path_a.exists() else ""

    # 2. Execute Method B (Dense Monocular Depth Relief)
    if depth_relief_grid is not None:
        depth_map = np.array(depth_relief_grid, dtype=np.float32)
    else:
        # Default synthetic grid if none provided
        depth_map = np.ones((64, 64), dtype=np.float32)

    carved_mesh_b, inf_b, metrics_b = carve_depth_heightmap_defect(
        mesh=tooth_mesh.copy(),
        polygon_norm=polygon_norm,
        depth_relief_map=depth_map,
        target_surface=target_surface,
        depth=depth_mm,
        tooth_fdi=tooth_fdi
    )

    path_b = TEMP_DIR / f"compare_method_b_{tooth_fdi}_{os.getpid()}.png"
    render_carved_tooth_pyvista(carved_mesh_b, inf_b, str(path_b), carve_params={"anomaly_type": pathology, "depth": depth_mm})
    render_b_b64 = encode_image_to_base64(cv2.imread(str(path_b)), format=".png") if path_b.exists() else ""

    tooth_meta = APPROVED_FDI_TEETH.get(tooth_fdi, {})
    tooth_name = tooth_meta.get("name", f"Tooth #{tooth_fdi}")

    return {
        "success": True,
        "tooth_fdi": tooth_fdi,
        "tooth_name": tooth_name,
        "pathology": pathology,
        "depth_mm": depth_mm,
        "method_a": {
            "title": "Method A: Synthetic Contour-Conforming (Parametric Floor)",
            "render_base64": render_a_b64,
            "metrics": metrics_a,
            "floor_type": floor_morphology,
        },
        "method_b": {
            "title": "Method B: Monocular Depth Conformal Transfer (Depth Anything V2 Organic Floor)",
            "render_base64": render_b_b64,
            "metrics": metrics_b,
            "floor_type": "depth_heightmap_conformal",
        },
        "comparison_highlights": {
            "method_a_advantage": "Smooth, continuous mathematical floor; ideal for stylized student training cases.",
            "method_b_advantage": "Photorealistic organic roughness, pit micro-ripples, and authentic demineralization textures transferred directly from photo.",
            "topology_guarantee": "Both methods 100% preserve watertight manifold vertex count and face connectivity."
        }
    }


@app.post("/api/teacher/segment")
async def teacher_sam_segment(
    file: Optional[UploadFile] = File(None),
    stem: Optional[str] = Form(None),
    image_b64: Optional[str] = Form(None),
    bbox_norm: Optional[str] = Form(None),
    points_norm: Optional[str] = Form(None),
):
    """
    Phase 1: Interactive SAM boundary snapping.
    Converts teacher bounding box or point prompt into exact lesion contour polygon.
    """
    img_bgr = None
    if file is not None:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif stem:
        img_p = NOVEL_DIR / f"{stem}.jpg"
        if not img_p.exists():
            img_p = TEST_IMG_DIR / f"{stem}.jpg"
            if not img_p.exists():
                img_p = TEST_IMG_DIR / f"{stem}.png"
        if img_p.exists():
            img_bgr = cv2.imread(str(img_p))
    elif image_b64:
        data_str = image_b64
        if data_str.startswith("data:"):
            data_str = data_str.split(",", 1)[1]
        img_bytes = base64.b64decode(data_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No valid image input provided.")

    parsed_bbox = None
    if bbox_norm:
        try:
            parsed_bbox = json.loads(bbox_norm) if isinstance(bbox_norm, str) else bbox_norm
        except Exception as e:
            pass

    parsed_points = None
    if points_norm:
        try:
            parsed_points = json.loads(points_norm) if isinstance(points_norm, str) else points_norm
        except Exception as e:
            pass

    if not parsed_bbox and not parsed_points:
        raise HTTPException(status_code=400, detail="Must provide at least a bounding box or point prompt.")

    res = sam_segmenter.segment_roi(img_bgr, bbox_norm=parsed_bbox, points_norm=parsed_points)
    if not res.get("success"):
        raise HTTPException(status_code=422, detail=res.get("error", "Failed to segment lesion boundary."))

    return {
        "success": True,
        "polygon_norm": res["polygon_norm"],
        "polygon_px": res["polygon_px"],
        "centroid_norm": res["centroid_norm"],
        "bbox_norm": res["bbox_norm"],
        "area_px": res["area_px"],
        "points_count": len(res["polygon_norm"]),
        "overlay_base64": res["overlay_base64"],
        "crop_base64": res["crop_base64"],
    }


@app.post("/api/teacher/diagnose")
async def teacher_clinical_diagnose(request: Request):
    """
    Phase 2: Multimodal VLM Clinical Diagnosis Agent.
    Evaluates lesion crop -> Returns structured ICDAS grade, tissue involvement, and millimeter depth.
    Supports both JSON body and multipart form data.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    crop_base64 = body.get("crop_base64")
    stem = body.get("stem")
    bbox_norm = body.get("bbox_norm")
    tooth_fdi = int(body.get("tooth_fdi", 16))
    target_surface = body.get("target_surface", "occlusal_table")
    instructor_notes = body.get("instructor_notes", "")
    api_key = body.get("api_key")

    crop_bgr = None
    if crop_base64:
        data_str = crop_base64
        if data_str.startswith("data:"):
            data_str = data_str.split(",", 1)[1]
        img_bytes = base64.b64decode(data_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        crop_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif stem and bbox_norm:
        img_p = NOVEL_DIR / f"{stem}.jpg"
        if not img_p.exists():
            img_p = TEST_IMG_DIR / f"{stem}.jpg"
            if not img_p.exists():
                img_p = TEST_IMG_DIR / f"{stem}.png"
        if img_p.exists():
            full_img = cv2.imread(str(img_p))
            if full_img is not None:
                h, w = full_img.shape[:2]
                bx = json.loads(bbox_norm) if isinstance(bbox_norm, str) else bbox_norm
                x1, y1, x2, y2 = int(bx[0] * w), int(bx[1] * h), int(bx[2] * w), int(bx[3] * h)
                crop_bgr = full_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    if crop_bgr is None or crop_bgr.size == 0:
        raise HTTPException(status_code=400, detail="Valid lesion crop image is required for diagnosis.")

    agent = ClinicalDiagnosticsAgent(api_key=api_key) if api_key else clinical_agent
    diagnosis = agent.diagnose_anomaly(crop_bgr, tooth_fdi=tooth_fdi, target_surface=target_surface, additional_notes=instructor_notes or "")

    # Validate against Whitelist
    pathology = diagnosis.get("pathology", "Caries")
    is_valid, whitelist_meta = validate_tooth_anomaly_combination(tooth_fdi, pathology)

    diagnosis["whitelist_valid"] = is_valid
    diagnosis["whitelist_meta"] = whitelist_meta
    diagnosis["tooth_fdi"] = tooth_fdi
    diagnosis["target_surface"] = target_surface

    return diagnosis


@app.post("/api/teacher/carve")
async def teacher_contour_carve(request: Request):
    """
    Phase 3: Executes contour-conforming 3D depth mesh deformation.
    Directly carves the exact SAM polygon into the real patient 3D tooth mesh,
    enforces Whitelist compliance, and renders off-screen PyVista visualization.
    """
    body = await request.json()
    polygon_norm = body.get("polygon_norm")
    tooth_fdi = int(body.get("tooth_fdi", 16))
    target_surface = body.get("target_surface", "occlusal_table")
    depth_mm = float(body.get("depth_mm", 0.85))
    margin_steepness = float(body.get("margin_steepness", 0.70))
    floor_morphology = body.get("floor_morphology", "irregular_concave")
    pathology = body.get("pathology", "Caries")

    if not polygon_norm or len(polygon_norm) < 3:
        raise HTTPException(status_code=400, detail="Valid 2D SAM polygon coordinates required.")

    # 1. Whitelist Verification
    is_valid, whitelist_meta = validate_tooth_anomaly_combination(tooth_fdi, pathology)
    if not is_valid:
        raise HTTPException(
            status_code=422,
            detail=f"Clinical Whitelist Violation: {whitelist_meta.get('reason', 'Not approved for this tooth.')}"
        )

    # 2. Load Real 3D Tooth Mesh
    try:
        tooth_mesh = load_or_extract_tooth_mesh(target_tooth=tooth_fdi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tooth mesh FDI #{tooth_fdi}: {e}")

    # 3. Execute Contour-Conforming Carving
    try:
        carved_mesh, influence, metrics = carve_contour_conforming_defect(
            mesh=tooth_mesh,
            polygon_norm=polygon_norm,
            target_surface=target_surface,
            depth=depth_mm,
            margin_steepness=margin_steepness,
            floor_morphology=floor_morphology,
            tooth_fdi=tooth_fdi
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D Carving algorithm failed: {e}")

    # 4. Render 2-Viewport PyVista Visualization
    render_img_name = f"teacher_carve_fdi_{tooth_fdi}_{os.getpid()}.png"
    render_path = TEMP_DIR / render_img_name
    render_carved_tooth_pyvista(
        carved_mesh=carved_mesh,
        gt_mask=influence,
        output_img_path=str(render_path),
        carve_params={"anomaly_type": pathology, "depth": depth_mm}
    )

    render_b64 = ""
    if render_path.exists():
        img_bgr = cv2.imread(str(render_path))
        if img_bgr is not None:
            render_b64 = encode_image_to_base64(img_bgr, format=".png")

    tooth_meta = APPROVED_FDI_TEETH.get(tooth_fdi, {})
    tooth_name = tooth_meta.get("name", f"Tooth #{tooth_fdi}")

    return {
        "success": True,
        "render_base64": render_b64,
        "tooth_fdi": tooth_fdi,
        "tooth_name": tooth_name,
        "pathology": pathology,
        "depth_mm": depth_mm,
        "metrics": metrics,
        "whitelist_meta": whitelist_meta,
    }


# HTML Single-Page UI
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Odonto CV Model Testing & 3D Carving QA Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#3B82F6', 600: '#2563EB' },
            abrasion: '#F97316',
            caries: '#EF4444',
            crack: '#D946EF',
          }
        }
      }
    }
  </script>
  <style>
    .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: #1f2937; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">

  <!-- Top Header -->
  <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur px-6 py-4 sticky top-0 z-50 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-blue-500/20">
        🦷
      </div>
      <div>
        <div class="flex items-center space-x-2">
          <h1 class="text-lg font-bold text-slate-100 tracking-wide">Odonto CV Model Testing & 3D Carve QA Dashboard</h1>
          <span class="px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">DEV-ONLY</span>
          <span class="px-2 py-0.5 rounded text-xs font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/30">2D &rarr; 3D CARVE</span>
        </div>
        <p class="text-xs text-slate-400">Photo-guided segmentation connected end-to-end with whitelist-enforced 3D tooth carving</p>
      </div>
    </div>
    <div class="flex items-center space-x-3">
      <div class="flex items-center space-x-2 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700 text-xs">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="text-slate-300 font-mono">RTX 4060 CUDA + PyVista</span>
      </div>
      <div class="flex space-x-1 bg-slate-800 p-1 rounded-lg border border-slate-700 text-xs font-medium">
        <button id="tab-single-btn" onclick="switchTab('single')" class="px-3 py-1.5 rounded bg-blue-600 text-white font-semibold">Single Inspector</button>
        <button id="tab-teacher-btn" onclick="switchTab('teacher')" class="px-3 py-1.5 rounded text-slate-300 hover:text-white flex items-center space-x-1.5"><span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>Teacher Studio (SAM + Agent)</span></button>
        <button id="tab-batch-btn" onclick="switchTab('batch')" class="px-3 py-1.5 rounded text-slate-300 hover:text-white">Batch Grid</button>
        <button id="tab-bench-btn" onclick="switchTab('bench')" class="px-3 py-1.5 rounded text-slate-300 hover:text-white">Benchmark Reference</button>
      </div>
    </div>
  </header>

  <!-- Global Crack Warning Notice -->
  <div class="bg-amber-950/40 border-b border-amber-500/40 px-6 py-2.5 flex items-center justify-between text-xs text-amber-200">
    <div class="flex items-center space-x-2">
      <span class="text-base">⚠️</span>
      <span class="font-bold uppercase tracking-wider text-amber-300">Crack Detection Advisory:</span>
      <span>Validated instance recall is only <strong>11.1% (3/27 TP)</strong> due to training data scarcity (67 unique cases). Crack is strictly <strong>2D-only</strong> and has no 3D carve profile.</span>
    </div>
    <span class="px-2 py-0.5 bg-amber-500/20 text-amber-300 rounded border border-amber-500/40 font-mono text-[11px]">2D-Only / Unreliable</span>
  </div>

  <!-- Main Container -->
  <main class="flex-1 p-6 max-w-7xl w-full mx-auto">

    <!-- TAB 1: Single Image Inspector -->
    <div id="tab-single" class="space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">

        <!-- Controls Column -->
        <div class="lg:col-span-1 space-y-5 bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div>
            <h2 class="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3 flex items-center justify-between">
              <span>Input Photo</span>
              <span class="text-xs text-blue-400 font-normal">Select or Upload</span>
            </h2>

            <!-- Preset Dropdown -->
            <label class="block text-xs font-medium text-slate-400 mb-1">Benchmark / Novel Preset</label>
            <select id="preset-select" onchange="loadPresetSample()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500 mb-3">
              <option value="">Loading presets...</option>
            </select>

            <div class="relative flex py-2 items-center">
              <div class="flex-grow border-t border-slate-800"></div>
              <span class="flex-shrink mx-2 text-[10px] text-slate-500 uppercase">OR CUSTOM UPLOAD</span>
              <div class="flex-grow border-t border-slate-800"></div>
            </div>

            <!-- Upload Area -->
            <div id="drop-zone" onclick="document.getElementById('file-input').click()" class="border-2 border-dashed border-slate-700 hover:border-blue-500 rounded-xl p-4 text-center cursor-pointer bg-slate-800/40 hover:bg-slate-800/80 transition-all">
              <input type="file" id="file-input" accept="image/*" class="hidden" onchange="handleFileUpload(event)">
              <div class="text-2xl mb-1">📁</div>
              <p class="text-xs font-semibold text-slate-200">Click or Drag & Drop Photo</p>
              <p class="text-[10px] text-slate-400 mt-0.5">JPEG, PNG, WebP (Real Patient Photos)</p>
            </div>
          </div>

          <!-- 2D Detection Settings -->
          <div class="border-t border-slate-800 pt-4 space-y-4">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">Detection Parameters</h3>
            
            <div>
              <div class="flex justify-between text-xs text-slate-400 mb-1">
                <span>Confidence Threshold</span>
                <span id="conf-val" class="font-mono text-blue-400 font-bold">0.25</span>
              </div>
              <input type="range" id="conf-slider" min="0.05" max="0.95" step="0.05" value="0.25" oninput="updateConf(this.value)" class="w-full accent-blue-500 cursor-pointer">
            </div>

            <div>
              <div class="flex justify-between text-xs text-slate-400 mb-1">
                <span>Mask Opacity</span>
                <span id="opacity-val" class="font-mono text-blue-400 font-bold">50%</span>
              </div>
              <input type="range" id="opacity-slider" min="0.1" max="0.9" step="0.05" value="0.5" oninput="updateOpacity(this.value)" class="w-full accent-blue-500 cursor-pointer">
            </div>

            <div>
              <label class="block text-xs font-medium text-slate-400 mb-2">Filter Classes</label>
              <div class="space-y-2">
                <label class="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                  <input type="checkbox" id="toggle-abrasion" checked onchange="rerunInspection()" class="rounded bg-slate-800 border-slate-700 text-orange-500 focus:ring-0">
                  <span class="w-3 h-3 rounded-full bg-orange-500 inline-block"></span>
                  <span>Abrasion (3D Whitelist: Anterior)</span>
                </label>
                <label class="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                  <input type="checkbox" id="toggle-caries" checked onchange="rerunInspection()" class="rounded bg-slate-800 border-slate-700 text-red-500 focus:ring-0">
                  <span class="w-3 h-3 rounded-full bg-red-500 inline-block"></span>
                  <span>Caries &rarr; Cavity (3D Whitelist: All)</span>
                </label>
                <label class="flex items-center space-x-2 text-xs text-slate-300 cursor-pointer">
                  <input type="checkbox" id="toggle-crack" checked onchange="rerunInspection()" class="rounded bg-slate-800 border-slate-700 text-fuchsia-500 focus:ring-0">
                  <span class="w-3 h-3 rounded-full bg-fuchsia-500 inline-block"></span>
                  <span>Crack (⚠️ 2D-Only, No 3D Carve)</span>
                </label>
              </div>
            </div>
          </div>

          <!-- 3D Target Tooth Selector Box -->
          <div class="border-t border-slate-800 pt-4 space-y-3">
            <h3 class="text-xs font-bold uppercase tracking-wider text-purple-300 flex items-center justify-between">
              <span>3D Target Tooth</span>
              <span class="text-[10px] text-purple-400 bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/30">Whitelist Governed</span>
            </h3>

            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Select FDI Permanent Tooth</label>
              <select id="fdi-select" onchange="handleToothSelectionChange()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500">
                <option value="">Loading FDI teeth...</option>
              </select>
            </div>

            <!-- Anatomical Breakdown Self-Check Badge (Fix 2) -->
            <div id="tooth-anatomical-badge" class="bg-slate-800/80 border border-slate-700/80 rounded-lg p-2.5 text-xs space-y-1.5">
              <div class="flex justify-between items-center text-[11px]">
                <span class="text-slate-400">Jaw / Arch:</span>
                <span id="anat-arch" class="font-bold text-slate-100 font-mono">--</span>
              </div>
              <div class="flex justify-between items-center text-[11px]">
                <span class="text-slate-400">Tooth Type:</span>
                <span id="anat-type" class="font-bold text-purple-300 capitalize">--</span>
              </div>
              <div class="flex justify-between items-center text-[11px]">
                <span class="text-slate-400">Carve Surface:</span>
                <span id="anat-surf" class="font-mono text-slate-300 text-[10px]">--</span>
              </div>
            </div>

            <!-- Sanity Check Prompt Reminder (Fix 2) -->
            <div id="tooth-sanity-check-prompt" class="p-2 rounded-lg bg-blue-950/40 border border-blue-500/40 text-[11px] text-blue-200 flex items-start space-x-2">
              <span class="text-sm">🔍</span>
              <span><strong>Anatomy Check:</strong> Confirm selected tooth matches visible arch in photo (e.g. Upper Maxillary vs Lower Mandibular).</span>
            </div>

            <!-- Whitelist Status Indicator for current selection -->
            <div id="tooth-whitelist-indicator" class="p-2.5 rounded-lg border text-xs bg-slate-800/60 border-slate-700 text-slate-400">
              Select a tooth and detect an anomaly to test whitelist compatibility.
            </div>

            <!-- Action Button: Trigger 3D Carve -->
            <button id="carve-btn" onclick="executeCarve()" class="w-full py-2.5 px-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold shadow-lg shadow-purple-500/20 transition-all flex items-center justify-center space-x-2">
              <span>🚀</span>
              <span>Carve & Render on 3D Tooth</span>
            </button>
          </div>

          <!-- Manual 3D Sculpting Section (Experimental / Dev-Only) -->
          <div class="border-t border-slate-800 pt-4 space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="text-xs font-bold uppercase tracking-wider text-amber-300">Manual 3D Placement</h3>
              <span class="text-[9px] font-bold text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/30">EXPERIMENTAL</span>
            </div>
            
            <p class="text-[11px] text-slate-400">
              Manual placement for unwhitelisted profiles (Crack & Chipped Cusp).
            </p>

            <div>
              <label class="block text-xs font-medium text-slate-400 mb-1">Select Profile</label>
              <select id="manual-profile-select" onchange="toggleManualProfileUI()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-amber-200 focus:outline-none focus:border-amber-500">
                <option value="v_groove_path">Crack (v_groove_path)</option>
                <option value="face_removal_recap">Chipped Cusp (face_removal_recap)</option>
              </select>
            </div>

            <!-- Crack Controls -->
            <div id="manual-crack-controls" class="space-y-2.5">
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1">Crack Path Source</label>
                <select id="crack-preset-select" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200">
                  <option value="custom_canvas">✍️ [PHOTO TRACED] Canvas Clicked Polyline (Mark on photo)</option>
                  <option value="occlusal_fissure">📦 [DEMO PRESET — Not Photo-Derived] Occlusal Fissure Structural Crack (Molars)</option>
                  <option value="labial_fracture">📦 [DEMO PRESET — Not Photo-Derived] Coronal Labial Hairline Fracture (Incisors)</option>
                  <option value="marginal_ridge">📦 [DEMO PRESET — Not Photo-Derived] Marginal Ridge Stress Fracture</option>
                </select>
              </div>
              <div class="flex justify-between items-center text-[11px] bg-slate-800/60 p-2 rounded border border-slate-700/60">
                <span id="canvas-pts-count" class="text-slate-400">Photo Points: <strong class="text-amber-400 font-mono">0</strong> (Click photo to trace)</span>
                <button type="button" onclick="clearCanvasPoints()" class="text-red-400 hover:text-red-300 underline text-[10px]">Clear</button>
              </div>
              <div class="grid grid-cols-2 gap-2">
                <div>
                  <label class="block text-[10px] text-slate-400 mb-0.5">Depth (mm)</label>
                  <input type="number" id="crack-depth-input" value="0.35" step="0.05" min="0.1" max="1.2" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono">
                </div>
                <div>
                  <label class="block text-[10px] text-slate-400 mb-0.5">Width (mm)</label>
                  <input type="number" id="crack-width-input" value="0.45" step="0.05" min="0.15" max="1.2" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono">
                </div>
              </div>
            </div>

            <!-- Chipped Cusp Controls -->
            <div id="manual-chip-controls" class="hidden space-y-2.5">
              <div>
                <label class="block text-xs font-medium text-slate-400 mb-1">Target Cusp</label>
                <select id="chip-cusp-select" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200">
                  <option value="mesiobuccal">Mesiobuccal Cusp (Molars)</option>
                  <option value="distobuccal">Distobuccal Cusp (Molars)</option>
                  <option value="mesiolingual">Mesiolingual Cusp (Molars)</option>
                  <option value="distolingual">Distolingual Cusp (Molars)</option>
                  <option value="buccal">Buccal Cusp (Premolars/Canines)</option>
                  <option value="lingual">Lingual Cusp (Premolars)</option>
                  <option value="incisal_corner">Incisal Edge Corner (Incisors)</option>
                </select>
              </div>
              <div class="grid grid-cols-2 gap-2">
                <div>
                  <label class="block text-[10px] text-slate-400 mb-0.5">Chip Height (mm)</label>
                  <input type="number" id="chip-height-input" value="1.4" step="0.1" min="0.5" max="3.5" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono">
                </div>
                <div>
                  <label class="block text-[10px] text-slate-400 mb-0.5">Fracture Angle (°)</label>
                  <input type="number" id="chip-angle-input" value="35.0" step="5" min="15" max="65" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 font-mono">
                </div>
              </div>
            </div>

            <!-- Sanity Check Prompt for Manual Carve -->
            <div class="p-2 rounded-lg bg-amber-950/30 border border-amber-500/40 text-[11px] text-amber-200 flex items-start space-x-2">
              <span class="text-sm">⚠️</span>
              <span><strong>Sanity Check:</strong> Ensure selected tooth matches visible arch in photo (e.g. Upper Right #16 vs Lower #36).</span>
            </div>

            <!-- Action Button: Manual Carve -->
            <button id="manual-carve-btn" onclick="executeManualCarve()" class="w-full py-2 px-3 bg-amber-600/90 hover:bg-amber-500 text-white rounded-lg text-xs font-bold shadow-md shadow-amber-900/30 transition-all flex items-center justify-center space-x-2">
              <span>⚡</span>
              <span>Render Manual 3D Carve</span>
            </button>
          </div>

          <div id="loading-spinner" class="hidden text-center py-3">
            <div class="inline-block animate-spin rounded-full h-5 w-5 border-2 border-purple-500 border-t-transparent"></div>
            <p id="loading-text" class="text-xs text-slate-400 mt-2 font-mono">Processing...</p>
          </div>
        </div>

        <!-- Main Display Column (Visual Panels + 3D Render) -->
        <div class="lg:col-span-3 space-y-6">

          <!-- Dynamic Status Banner -->
          <div id="status-banner" class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex items-center justify-between text-xs">
            <div class="flex items-center space-x-2">
              <span id="status-icon" class="text-base">ℹ️</span>
              <span id="status-text" class="text-slate-300">Select a preset or upload an image to inspect 2D detections and generate 3D mesh carves.</span>
            </div>
            <div id="status-badges" class="flex space-x-2"></div>
          </div>

          <!-- Crack Warning Banner -->
          <div id="crack-alert-box" class="hidden bg-amber-950/50 border-2 border-amber-500/70 rounded-xl p-4 text-xs text-amber-200 shadow-lg shadow-amber-950/40">
            <div class="flex items-start space-x-3">
              <span class="text-2xl">⚠️</span>
              <div>
                <h4 class="font-bold text-amber-300 uppercase tracking-wider text-sm">Crack Anomaly Detected — 2D Only & Low Recall Warning</h4>
                <p class="mt-1 text-slate-300 leading-relaxed">
                  The model detected <strong>Crack (Class 2)</strong>. Real-world validation demonstrates an <strong>11.1% instance recall (3/27 TP)</strong> due to 67 training cases. 
                  <span class="text-amber-300 font-semibold">3D Carving is not supported for Crack</span>: it has no registered volumetric profile in the Odonto whitelist and cannot be projected to 3D.
                </p>
              </div>
            </div>
          </div>

          <!-- 2D Photo & Overlay Panels -->
          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <!-- 1. Original Photo -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
              <div class="px-3.5 py-2 bg-slate-800/80 border-b border-slate-800 flex justify-between items-center text-xs">
                <span class="font-bold text-slate-200">1. Original Photo (Click to trace crack)</span>
                <span id="img-dim" class="text-[11px] font-mono text-slate-400">---</span>
              </div>
              <div class="flex-1 min-h-[260px] bg-black/60 flex items-center justify-center p-2">
                <div class="relative inline-block max-h-[300px]">
                  <img id="orig-img" onclick="onPhotoClick(event)" class="cursor-crosshair max-h-[300px] w-auto object-contain rounded block" alt="Original photo" src="">
                  <svg id="orig-trace-svg" viewBox="0 0 100 100" preserveAspectRatio="none" class="absolute inset-0 pointer-events-none w-full h-full"></svg>
                </div>
                <div id="orig-placeholder" class="text-slate-500 text-xs">No image loaded</div>
              </div>
            </div>

            <!-- 2. Prediction Overlay -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
              <div class="px-3.5 py-2 bg-slate-800/80 border-b border-slate-800 flex justify-between items-center text-xs">
                <span class="font-bold text-blue-400">2. 2D Prediction Overlay (Click to trace crack)</span>
                <span id="pred-count-badge" class="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300">0 detections</span>
              </div>
              <div class="flex-1 min-h-[260px] bg-black/60 flex items-center justify-center p-2">
                <div class="relative inline-block max-h-[300px]">
                  <img id="pred-img" onclick="onPhotoClick(event)" class="cursor-crosshair max-h-[300px] w-auto object-contain rounded block" alt="Prediction Overlay" src="">
                  <svg id="pred-trace-svg" viewBox="0 0 100 100" preserveAspectRatio="none" class="absolute inset-0 pointer-events-none w-full h-full"></svg>
                </div>
                <div id="pred-placeholder" class="text-slate-500 text-xs">Awaiting inference</div>
              </div>
            </div>

            <!-- 3. Ground Truth Overlay -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
              <div class="px-3.5 py-2 bg-slate-800/80 border-b border-slate-800 flex justify-between items-center text-xs">
                <span class="font-bold text-emerald-400">3. Ground Truth Overlay</span>
                <span id="gt-status-badge" class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400">N/A</span>
              </div>
              <div class="flex-1 min-h-[260px] bg-black/60 flex items-center justify-center p-2 relative">
                <img id="gt-img" class="max-h-[300px] w-auto object-contain rounded" alt="Ground Truth Overlay" src="">
                <div id="gt-placeholder" class="text-slate-500 text-xs text-center p-4">
                  <p class="font-semibold text-slate-400 mb-1">No Ground Truth Available</p>
                  <p class="text-[11px] text-slate-500">Novel/upload sample has no benchmark mask.</p>
                </div>
              </div>
            </div>
          </div>

          <!-- Section: 3D Tooth Carve & Physical Mapping -->
          <div class="bg-slate-900 border border-purple-500/40 rounded-xl overflow-hidden shadow-xl shadow-purple-950/20">
            <div class="px-5 py-3.5 bg-gradient-to-r from-purple-950/70 to-slate-900 border-b border-slate-800 flex justify-between items-center">
              <div class="flex items-center space-x-2">
                <span class="text-purple-400 font-bold text-sm">4. 3D Tooth Carve & Multi-View Physical Render</span>
                <span id="carve-status-pill" class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700">AWAITING CARVE</span>
              </div>
              <div id="carve-surface-pill" class="text-xs text-slate-400 font-mono"></div>
            </div>

            <div class="p-5 space-y-4">

              <!-- Manual Experimental Badge (shown when manual placement is used) -->
              <div id="manual-experimental-badge" class="hidden bg-amber-950/70 border border-amber-500/80 p-3 rounded-xl text-xs text-amber-200 font-bold flex items-center justify-between shadow-lg shadow-amber-950/40">
                <div class="flex items-center space-x-2.5">
                  <span class="text-lg">⚠️</span>
                  <span>Manual Placement (Experimental) — Not Whitelisted for Autonomous Student Mode</span>
                </div>
                <span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono">DEV QA ONLY</span>
              </div>

              <!-- Whitelist Block Alert (if combination blocked) -->
              <div id="carve-block-alert" class="hidden bg-amber-950/50 border border-amber-500/70 rounded-xl p-4 text-xs text-amber-200">
                <div class="flex items-start space-x-3">
                  <span class="text-2xl">🚫</span>
                  <div>
                    <h5 class="font-bold text-amber-300 uppercase tracking-wider">Whitelist Blocked by Odonto Clinical Safety Protocol</h5>
                    <p id="carve-block-message" class="mt-1 text-slate-200 leading-relaxed"></p>
                    <p id="carve-block-advisory" class="mt-1 text-[11px] text-amber-300/80"></p>
                  </div>
                </div>
              </div>

              <!-- Crack 2D-Only Notice (if Crack detection) -->
              <div id="carve-crack-notice" class="hidden bg-blue-950/40 border border-blue-500/50 rounded-xl p-4 text-xs text-blue-200">
                <div class="flex items-start space-x-3">
                  <span class="text-2xl">ℹ️</span>
                  <div>
                    <h5 class="font-bold text-blue-300 uppercase tracking-wider">3D Carving Not Available for Crack</h5>
                    <p id="carve-crack-message" class="mt-1 text-slate-300 leading-relaxed"></p>
                  </div>
                </div>
              </div>

              <!-- 3D Render Image Container (1400x700 PyVista Dual Viewport) -->
              <div id="carve-render-container" class="hidden space-y-3">
                <!-- Prominent Carve Derivation Banner (Fix 1: Explicit Photo-Derived vs Preset Source) -->
                <div id="carve-derivation-banner" class="p-3 rounded-xl border text-xs font-bold flex items-center justify-between shadow-md">
                  <!-- Dynamically populated in JS -->
                </div>

                <div class="bg-black/80 rounded-xl p-2 border border-slate-800 flex items-center justify-center">
                  <img id="carve-render-img" class="w-full max-h-[450px] object-contain rounded" alt="3D Carved Tooth Multi-View Render" src="">
                </div>
                
                <!-- 3D Carve Parameter Metrics Strip (6 Columns including Carve Source) -->
                <div id="carve-metrics-strip" class="grid grid-cols-2 md:grid-cols-6 gap-2.5 pt-2">
                  <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 text-center">
                    <div class="text-[10px] text-slate-400 uppercase">Carve Source</div>
                    <div id="metric-source" class="text-xs font-bold text-emerald-300 font-mono mt-0.5 truncate" title="Carve Source">--</div>
                  </div>
                  <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 text-center">
                    <div class="text-[10px] text-slate-400 uppercase">Target Surface</div>
                    <div id="metric-surface" class="text-sm font-bold text-purple-300 font-mono mt-0.5">--</div>
                  </div>
                  <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 text-center">
                    <div class="text-[10px] text-slate-400 uppercase">Carve Radius/Width</div>
                    <div id="metric-radius" class="text-sm font-bold text-slate-100 font-mono mt-0.5">--</div>
                  </div>
                  <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 text-center">
                    <div class="text-[10px] text-slate-400 uppercase">Carve Depth/Height</div>
                    <div id="metric-depth" class="text-sm font-bold text-slate-100 font-mono mt-0.5">--</div>
                  </div>
                  <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 text-center">
                    <div class="text-[10px] text-slate-400 uppercase">Geometry/Aspect</div>
                    <div id="metric-ar" class="text-sm font-bold text-slate-100 font-mono mt-0.5">--</div>
                  </div>
                  <div class="bg-slate-800/60 p-2.5 rounded-lg border border-slate-700/60 text-center">
                    <div class="text-[10px] text-slate-400 uppercase">Influenced Vertices</div>
                    <div id="metric-verts" class="text-sm font-bold text-emerald-400 font-mono mt-0.5">--</div>
                  </div>
                </div>

                <div id="carve-notes" class="text-[11px] text-slate-400 italic bg-slate-800/30 p-2 rounded border border-slate-800"></div>
              </div>

              <!-- Placeholder when no 3D render yet -->
              <div id="carve-placeholder" class="py-10 text-center text-slate-500 text-xs">
                Select an FDI tooth number from the left panel and click <strong class="text-purple-400">"Carve & Render on 3D Tooth"</strong> to project the detected 2D anomaly onto the 3D tooth mesh.
              </div>

            </div>
          </div>

          <!-- Detected Instances Table -->
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex justify-between items-center">
              <span>Detected Anomaly Instances (Click Instance to Select for 3D Carve)</span>
              <span id="instance-summary-text" class="text-slate-500 font-normal">None detected</span>
            </h4>
            <div class="overflow-x-auto">
              <table class="w-full text-left text-xs">
                <thead class="text-slate-400 border-b border-slate-800 text-[11px] uppercase">
                  <tr>
                    <th class="py-2 px-3">#</th>
                    <th class="py-2 px-3">Detected Class</th>
                    <th class="py-2 px-3">Confidence</th>
                    <th class="py-2 px-3">3D Carve Profile</th>
                    <th class="py-2 px-3">Bounding Box</th>
                    <th class="py-2 px-3">Reliability Assessment</th>
                  </tr>
                </thead>
                <tbody id="instances-tbody" class="divide-y divide-slate-800 text-slate-300">
                  <tr>
                    <td colspan="6" class="py-4 text-center text-slate-500">No predictions to display</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

        </div>

      </div>
    </div>

    <!-- TAB 2: Batch Grid Testing -->
    <div id="tab-batch" class="hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
        <div>
          <h2 class="text-sm font-bold text-slate-200">Batch Inspection Grid</h2>
          <p class="text-xs text-slate-400 mt-0.5">Evaluate multiple benchmark and novel samples simultaneously.</p>
        </div>
        <div class="flex items-center space-x-3">
          <button onclick="runBatchPreset()" class="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg text-xs font-semibold shadow">Run 8-Sample Benchmark Grid</button>
        </div>
      </div>

      <div id="batch-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="col-span-full py-12 text-center text-slate-500 text-xs">
          Click "Run 8-Sample Benchmark Grid" to evaluate and render multiple samples concurrently.
        </div>
      </div>
    </div>

    <!-- TAB 3: Benchmark Reference -->
    <div id="tab-bench" class="hidden space-y-6">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 space-y-6">
        <div>
          <h2 class="text-base font-bold text-slate-100">Benchmark Reference & 3D Carve Integration</h2>
          <p class="text-xs text-slate-400 mt-1">Held-out test set performance (354 images, 228 patients) connected with the 28-tooth Phase 0 whitelist.</p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
            <div class="flex justify-between items-center mb-2">
              <span class="text-xs font-bold text-orange-400 uppercase tracking-wider">Abrasion &rarr; Abrasion Profile</span>
              <span class="px-2 py-0.5 bg-emerald-500/20 text-emerald-300 text-[10px] rounded font-bold">3D APPROVED (ANTERIOR)</span>
            </div>
            <div class="text-2xl font-bold font-mono text-slate-100">0.5316 <span class="text-xs text-slate-400 font-normal">Mean IoU</span></div>
            <p class="text-xs text-slate-400 mt-2">Whitelist approves on anterior teeth (#11-#13, #21-#23, #31-#33, #41-#43) on labial/facial surfaces. Posterior abrasion is blocked to prevent saucer artifacts.</p>
          </div>

          <div class="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
            <div class="flex justify-between items-center mb-2">
              <span class="text-xs font-bold text-red-400 uppercase tracking-wider">Caries &rarr; Cavity Profile</span>
              <span class="px-2 py-0.5 bg-blue-500/20 text-blue-300 text-[10px] rounded font-bold">3D APPROVED (ALL 28 TEETH)</span>
            </div>
            <div class="text-2xl font-bold font-mono text-slate-100">0.2220 <span class="text-xs text-slate-400 font-normal">Mean IoU</span></div>
            <p class="text-xs text-slate-400 mt-2">Explicitly maps to the Cavity carve profile. Whitelist approves across incisors, canines, premolars, and molars.</p>
          </div>

          <div class="bg-slate-800/40 border border-amber-500/40 rounded-xl p-4">
            <div class="flex justify-between items-center mb-2">
              <span class="text-xs font-bold text-fuchsia-400 uppercase tracking-wider">Crack (2D Diagnostic Only)</span>
              <span class="px-2 py-0.5 bg-amber-500/20 text-amber-300 text-[10px] rounded font-bold">⚠️ NO 3D PROFILE</span>
            </div>
            <div class="text-2xl font-bold font-mono text-slate-100">0.1428 <span class="text-xs text-slate-400 font-normal">Mean IoU</span></div>
            <p class="text-xs text-amber-200/90 mt-2">Validated instance recall is <strong>11.1%</strong> due to 67 training cases. 3D carving is strictly blocked to maintain clinical simulation integrity.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: Teacher Studio (Interactive SAM + Multimodal Clinical Agent) -->
    <div id="tab-teacher" class="hidden space-y-6">
      
      <!-- Studio Header Banner -->
      <div class="bg-gradient-to-r from-blue-950/60 via-slate-900 to-purple-950/60 border border-blue-500/30 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div class="flex items-center space-x-2">
              <span class="px-2.5 py-1 rounded text-xs font-bold bg-blue-500/20 text-blue-300 border border-blue-500/40">PHASE 1: SAM BOUNDARY</span>
              <span class="px-2.5 py-1 rounded text-xs font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40">PHASE 2: CLINICAL AGENT</span>
              <span class="px-2.5 py-1 rounded text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">TEACHER GATE</span>
            </div>
            <h2 class="text-xl font-bold text-white mt-2 flex items-center space-x-2">
              <span>🎯 Teacher-Guided Anomaly Identification & Feature Extraction</span>
            </h2>
            <p class="text-xs text-slate-300 mt-1 max-w-3xl leading-relaxed">
              Pinpoint suspicious lesions with interactive <strong>Box / Point prompts</strong>. SAM extracts the sub-pixel boundary polygon (<span class="font-mono text-amber-300">&gt;50 vertices</span>). The Clinical Specialist Agent then diagnoses ICDAS severity, tissue involvement, and recommends precise physical carve depths with full teacher review.
            </p>
          </div>
          <div class="flex items-center space-x-2">
            <button onclick="loadSampleTeacherPreset('alpha_p006_F_15_001')" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200">Preset: Occlusal Caries</button>
            <button onclick="loadSampleTeacherPreset('alpha_p006_F_15_002')" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200">Preset: Cervical Abrasion</button>
          </div>
        </div>
      </div>

      <!-- Main Studio Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

        <!-- Left Column: Photo & Interactive Prompt Canvas (5 cols) -->
        <div class="lg:col-span-5 space-y-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 class="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center space-x-2">
              <span>1. Localize Lesion</span>
            </h3>
            <div class="flex items-center space-x-1">
              <button id="t-mode-box" onclick="setTeacherMode('box')" class="px-2.5 py-1 rounded text-xs font-semibold bg-blue-600 text-white">🔲 Box Mode</button>
              <button id="t-mode-point" onclick="setTeacherMode('point')" class="px-2.5 py-1 rounded text-xs text-slate-300 hover:text-white bg-slate-800 border border-slate-700">📍 Point Mode</button>
              <button onclick="clearTeacherPrompts()" class="px-2 py-1 rounded text-xs text-rose-400 hover:bg-rose-950/40 border border-rose-900/50" title="Reset Prompts">🧹 Clear</button>
            </div>
          </div>

          <!-- Preset selector or custom upload -->
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div>
              <label class="block text-[11px] text-slate-400 mb-1">Select Sample Photo</label>
              <select id="teacher-preset-select" onchange="handleTeacherPresetChange()" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none">
                <option value="alpha_p006_F_15_001">AlphaDent #1 (Caries #16)</option>
                <option value="alpha_p006_F_15_002">AlphaDent #2 (Abrasion #11)</option>
                <option value="dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7">DentalAI #1 (Hairline Crack)</option>
                <option value="novel_01_alphadent_filling_case">Novel Filling Sample</option>
              </select>
            </div>
            <div>
              <label class="block text-[11px] text-slate-400 mb-1">Or Upload Custom Photo</label>
              <input type="file" id="teacher-file-input" accept="image/*" onchange="handleTeacherFileUpload(event)" class="w-full text-xs text-slate-400 file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-[11px] file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer">
            </div>
          </div>

          <!-- Interactive Canvas Container -->
          <div class="relative bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center min-h-[380px]">
            <img id="teacher-source-img" class="max-w-full max-h-[460px] object-contain select-none" alt="Teacher Inspection Source" style="display: block;">
            <canvas id="teacher-canvas" class="absolute top-0 left-0 w-full h-full cursor-crosshair"></canvas>
            <div id="teacher-canvas-hint" class="absolute bottom-2 left-2 px-2.5 py-1 bg-black/75 backdrop-blur rounded text-[11px] text-slate-300 pointer-events-none border border-slate-700 flex items-center space-x-1.5">
              <span class="w-2 h-2 rounded-full bg-blue-400 animate-ping"></span>
              <span id="teacher-hint-text">Click & Drag a box tightly around the lesion</span>
            </div>
          </div>

          <!-- Prompt readout & Snap SAM button -->
          <div class="bg-slate-950/60 rounded-lg p-3 border border-slate-800 space-y-2">
            <div class="flex items-center justify-between text-xs text-slate-400">
              <span class="font-medium">Active Prompt:</span>
              <span id="teacher-prompt-readout" class="font-mono text-blue-300">No prompt drawn yet</span>
            </div>
            <button id="snap-sam-btn" onclick="executeTeacherSAM()" class="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-lg shadow-lg shadow-blue-600/20 text-xs flex items-center justify-center space-x-2 transition-all">
              <span>⚡ Snap Lesion Boundary (SAM Phase 1)</span>
            </button>
          </div>
        </div>

        <!-- Middle Column: Snapped Boundary & High-Res Crop (3 cols) -->
        <div class="lg:col-span-3 space-y-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 class="text-sm font-bold text-slate-200 uppercase tracking-wider">
              <span>2. Snapped Boundary</span>
            </h3>
            <span id="sam-status-pill" class="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700">AWAITING SAM</span>
          </div>

          <!-- Snapped Overlay Image Display -->
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">Exact Contour Snapping Overlay</label>
            <div class="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden min-h-[180px] flex items-center justify-center relative">
              <img id="teacher-overlay-preview" class="max-w-full max-h-[220px] object-contain hidden" alt="SAM Snapped Boundary">
              <span id="teacher-overlay-placeholder" class="text-xs text-slate-500 italic p-4 text-center">Draw box or points and click "Snap Lesion Boundary"</span>
            </div>
          </div>

          <!-- Macro Crop Display -->
          <div>
            <label class="block text-[11px] text-slate-400 mb-1">High-Resolution Lesion Macro Crop</label>
            <div class="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden min-h-[120px] flex items-center justify-center p-2">
              <img id="teacher-crop-preview" class="max-h-[140px] rounded border border-slate-700 object-contain hidden shadow" alt="Lesion Crop">
              <span id="teacher-crop-placeholder" class="text-xs text-slate-500 italic text-center">Macro crop will appear here</span>
            </div>
          </div>

          <!-- 2.5D Depth Relief Heatmap (Depth Anything V2 - Method B) -->
          <div>
            <div class="flex items-center justify-between mb-1">
              <label class="block text-[11px] text-slate-400 font-semibold">2.5D Depth Relief (Depth Anything V2)</label>
              <span id="depth-status-pill" class="text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">DETRENDED RELIEF</span>
            </div>
            <div class="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden min-h-[130px] flex items-center justify-center p-2 relative">
              <img id="teacher-heatmap-preview" class="max-h-[140px] rounded border border-blue-500/40 object-contain hidden shadow" alt="Depth Anything Relief">
              <span id="teacher-heatmap-placeholder" class="text-xs text-slate-500 italic text-center">Depth Anything V2 relief map will render here</span>
            </div>
          </div>

          <!-- Geometric Metrics Card -->
          <div class="bg-slate-950/80 rounded-lg p-3 border border-slate-800 space-y-1.5 text-xs font-mono">
            <div class="flex justify-between text-slate-400">
              <span>Contour Points:</span>
              <span id="sam-metric-points" class="text-emerald-400 font-bold">--</span>
            </div>
            <div class="flex justify-between text-slate-400">
              <span>Centroid (cx, cy):</span>
              <span id="sam-metric-centroid" class="text-slate-200">--</span>
            </div>
            <div class="flex justify-between text-slate-400">
              <span>Pixel Area:</span>
              <span id="sam-metric-area" class="text-slate-200">--</span>
            </div>
            <div class="flex justify-between text-slate-400 border-t border-slate-800 pt-1">
              <span>Depth Model Samples:</span>
              <span id="depth-metric-samples" class="text-blue-300 font-bold">--</span>
            </div>
          </div>
        </div>

        <!-- Right Column: Clinical Specialist Agent (4 cols) -->
        <div class="lg:col-span-4 space-y-4 bg-slate-900 border border-slate-800 rounded-xl p-5">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 class="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center space-x-1.5">
              <span>3. Clinical Agent</span>
            </h3>
            <span id="agent-source-pill" class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">VLM / LOCAL EXPERT</span>
          </div>

          <!-- Clinical Transfer Agent Target Recommendation Banner -->
          <div id="agent-recommendation-box" class="hidden bg-gradient-to-r from-blue-950/70 to-purple-950/70 border border-blue-500/40 rounded-xl p-3.5 space-y-2">
            <div class="flex items-center justify-between">
              <span class="text-xs font-bold text-blue-300 flex items-center space-x-1.5">
                <span>🧠</span>
                <span>Transfer Agent Recommendation</span>
              </span>
              <span id="agent-rec-fdi-badge" class="px-2 py-0.5 rounded bg-blue-500/30 text-blue-200 border border-blue-400/50 text-[10px] font-mono font-bold">FDI #16</span>
            </div>
            <p id="agent-rec-rationale" class="text-[11px] text-slate-300 leading-relaxed italic">
              "Occlusal pit caries is clinically optimal on molar occlusal tables."
            </p>
            <button type="button" onclick="applyAgentRecommendedTooth()" class="w-full py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded text-[11px] transition shadow">
              &larr; Apply Agent Recommended Target Tooth
            </button>
          </div>

          <!-- Tooth & Surface context -->
          <div class="space-y-3">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Target FDI Tooth</label>
              <select id="teacher-fdi-select" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none">
                <!-- Dynamically populated from /api/teeth -->
              </select>
            </div>

            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-[11px] text-slate-400 mb-1">Target Surface</label>
                <select id="teacher-surface-select" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none">
                  <option value="occlusal_table">Occlusal Table</option>
                  <option value="labial_facial">Labial / Facial</option>
                  <option value="incisal_edge">Incisal Edge</option>
                  <option value="cervical">Cervical Margin</option>
                </select>
              </div>
              <div>
                <label class="block text-[11px] text-slate-400 mb-1">Optional API Key</label>
                <input type="password" id="teacher-api-key" placeholder="Server / Gemini Key" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none">
              </div>
            </div>

            <div>
              <label class="block text-[11px] text-slate-400 mb-1">Instructor Context / Clinical Notes</label>
              <input type="text" id="teacher-notes-input" placeholder="e.g. Discolored occlusal fissure, suspect ICDAS 4" class="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none">
            </div>

            <button id="run-diagnose-btn" onclick="executeTeacherDiagnosis()" disabled class="w-full py-2.5 bg-gradient-to-r from-purple-600 to-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed hover:from-purple-500 hover:to-indigo-500 text-white font-bold rounded-lg shadow-lg shadow-purple-600/20 text-xs flex items-center justify-center space-x-2 transition-all">
              <span>🤖 Run Clinical Diagnosis (Phase 2 Agent)</span>
            </button>
          </div>

          <!-- Structured Diagnostic Result Card -->
          <div id="diagnostic-card" class="hidden bg-slate-950 border border-purple-500/30 rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div>
                <div class="flex items-center space-x-2">
                  <span id="diag-pathology-badge" class="px-2.5 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30">Caries</span>
                  <span id="diag-confidence-text" class="text-xs text-slate-400 font-mono">Conf: 92%</span>
                </div>
                <h4 id="diag-severity-title" class="text-sm font-bold text-slate-100 mt-1">ICDAS 4 (Moderate Dentinal Cavitation)</h4>
              </div>
            </div>

            <!-- Tissues Involved -->
            <div class="flex items-center space-x-2 text-xs">
              <span class="text-slate-400">Tissues:</span>
              <div id="diag-tissues-container" class="flex items-center space-x-1">
                <span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700">Enamel</span>
                <span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40">Dentin</span>
              </div>
            </div>

            <!-- Recommended Depth with Teacher Override Slider -->
            <div class="border-t border-slate-800/80 pt-2 space-y-1">
              <div class="flex justify-between items-center text-xs">
                <span class="text-slate-300 font-semibold">Recommended Physical Depth:</span>
                <span id="diag-depth-readout" class="font-mono text-emerald-400 font-bold text-sm">0.75 mm</span>
              </div>
              <input type="range" id="teacher-depth-slider" min="0.10" max="2.00" step="0.05" value="0.75" oninput="updateTeacherDepth(this.value)" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500">
              <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                <span id="diag-min-depth">Min: 0.35mm</span>
                <span>Enamel Limit ~1.5mm</span>
                <span id="diag-max-depth">Max: 1.20mm</span>
              </div>
            </div>

            <!-- Method Selection: Method A vs Method B vs Side-by-Side Comparison -->
            <div class="border-t border-slate-800/80 pt-2.5 space-y-2">
              <div class="flex justify-between items-center text-xs">
                <span class="text-slate-300 font-semibold">Deformation Method:</span>
                <span id="active-method-label" class="font-mono text-purple-300 text-[11px] font-bold">Method B (Depth Model)</span>
              </div>
              <div class="grid grid-cols-3 gap-1.5 text-[11px]">
                <button type="button" id="btn-method-a" onclick="setCarveMethod('contour_conforming')" class="py-1.5 px-1.5 rounded bg-slate-800 text-slate-300 border border-slate-700 hover:text-white text-center font-medium">
                  Method A (Parametric)
                </button>
                <button type="button" id="btn-method-b" onclick="setCarveMethod('depth_heightmap')" class="py-1.5 px-1.5 rounded bg-purple-600 text-white font-semibold border border-purple-500 text-center">
                  Method B (Depth Model)
                </button>
                <button type="button" id="btn-method-compare" onclick="setCarveMethod('compare')" class="py-1.5 px-1.5 rounded bg-emerald-600/80 text-white font-bold border border-emerald-500 hover:bg-emerald-500 text-center">
                  ⚡ Side-by-Side
                </button>
              </div>
            </div>

            <!-- Margin Steepness & Floor -->
            <div class="grid grid-cols-2 gap-2 text-[11px] bg-slate-900/60 p-2 rounded border border-slate-800">
              <div>
                <span class="text-slate-400">Margin Wall:</span>
                <div id="diag-steepness-val" class="font-semibold text-slate-200">0.70 (Undermined)</div>
              </div>
              <div>
                <span class="text-slate-400">Floor Morphology:</span>
                <div id="diag-floor-val" class="font-semibold text-slate-200">Irregular Concave</div>
              </div>
            </div>

            <!-- Clinical Rationale -->
            <div class="bg-purple-950/30 border border-purple-500/20 rounded p-2.5 text-xs text-purple-200/90 leading-relaxed italic">
              "<span id="diag-rationale-text">Visible dark cavitation with extensive dentinal breakdown and undermined enamel margins.</span>"
            </div>

            <!-- Whitelist Status Banner -->
            <div id="diag-whitelist-banner" class="p-2 rounded text-xs border flex items-center space-x-2 bg-emerald-950/40 border-emerald-500/30 text-emerald-300">
              <span class="text-sm">✅</span>
              <span id="diag-whitelist-msg">APPROVED: Cavity profile compatible with FDI #16 Occlusal table.</span>
            </div>

            <!-- Confirmation Gate & 3D Carve Button -->
            <button id="approve-case-btn" onclick="confirmTeacherCase()" class="w-full py-2.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-lg shadow-lg shadow-emerald-600/20 text-xs flex items-center justify-center space-x-2 transition-all">
              <span id="approve-btn-text">⚡ Approve Case & Render 3D Carve</span>
            </button>
          </div>

        </div>

      </div>

      <!-- 4. PHASE 3: 3D CONTOUR-CONFORMING MESH RESULT CARD -->
      <div id="teacher-3d-result-card" class="hidden bg-slate-900 border border-blue-500/40 rounded-2xl p-6 shadow-2xl space-y-4">
        <div class="flex flex-col md:flex-row md:items-center justify-between pb-3 border-b border-slate-800 gap-3">
          <div class="flex items-center space-x-3">
            <div class="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center font-bold text-white shadow-lg">
              ✨
            </div>
            <div>
              <div class="flex items-center space-x-2">
                <h3 id="teacher-3d-main-title" class="text-base font-bold text-white tracking-wide">Phase 3: 3D Mesh Deformation</h3>
                <span id="teacher-method-badge" class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40">METHOD B (DEPTH MODEL)</span>
                <span class="px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/40">100% TOPOLOGY PRESERVED</span>
              </div>
              <p id="teacher-3d-tooth-title" class="text-xs text-slate-400">Maxillary Right First Molar (FDI #16)</p>
            </div>
          </div>
          <div id="single-metrics-bar" class="flex flex-wrap items-center gap-2 text-xs font-mono">
            <span id="teacher-3d-verts-badge" class="px-2.5 py-1 rounded bg-slate-800 text-emerald-400 border border-slate-700">-- Vertices Displaced</span>
            <span id="teacher-3d-depth-badge" class="px-2.5 py-1 rounded bg-slate-800 text-blue-400 border border-slate-700">Max Depth: -- mm</span>
            <span id="teacher-3d-span-badge" class="px-2.5 py-1 rounded bg-slate-800 text-amber-400 border border-slate-700">Span: -- mm</span>
          </div>
        </div>

        <!-- Single-Method Render Viewer Container -->
        <div id="single-carve-view">
          <div class="bg-slate-950 rounded-xl overflow-hidden border border-slate-800 relative flex items-center justify-center min-h-[350px]">
            <img id="teacher-3d-render-img" class="w-full max-h-[550px] object-contain select-none" alt="3D Carved Mesh">
          </div>
        </div>

        <!-- Side-by-Side Comparison Viewer Container (Method A vs Method B) -->
        <div id="side-by-side-comparison-view" class="hidden space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Left: Method A -->
            <div class="bg-slate-950 rounded-xl border border-blue-500/30 overflow-hidden flex flex-col">
              <div class="px-3.5 py-2.5 bg-slate-900 border-b border-slate-800 flex justify-between items-center text-xs">
                <span class="font-bold text-blue-300">Method A: Synthetic Contour-Conforming</span>
                <span class="px-2 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-300 font-mono">PARAMETRIC</span>
              </div>
              <div class="p-2 flex items-center justify-center min-h-[280px]">
                <img id="compare-render-a" class="max-h-[340px] w-auto object-contain rounded" alt="Method A Render">
              </div>
              <div class="p-3 bg-slate-900/80 border-t border-slate-800 text-[11px] font-mono space-y-1 text-slate-300">
                <div class="flex justify-between"><span>Displaced Vertices:</span><span id="compare-verts-a" class="text-emerald-400 font-bold">--</span></div>
                <div class="flex justify-between"><span>Max Depth:</span><span id="compare-depth-a" class="text-blue-300">--</span></div>
                <div class="flex justify-between"><span>Floor Profile:</span><span id="compare-floor-a" class="text-slate-400">Mathematical Falloff</span></div>
              </div>
            </div>

            <!-- Right: Method B -->
            <div class="bg-slate-950 rounded-xl border border-purple-500/30 overflow-hidden flex flex-col">
              <div class="px-3.5 py-2.5 bg-slate-900 border-b border-slate-800 flex justify-between items-center text-xs">
                <span class="font-bold text-purple-300">Method B: Monocular Depth Conformal Transfer</span>
                <span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 font-mono">DEPTH ANYTHING V2</span>
              </div>
              <div class="p-2 flex items-center justify-center min-h-[280px]">
                <img id="compare-render-b" class="max-h-[340px] w-auto object-contain rounded" alt="Method B Render">
              </div>
              <div class="p-3 bg-slate-900/80 border-t border-slate-800 text-[11px] font-mono space-y-1 text-slate-300">
                <div class="flex justify-between"><span>Displaced Vertices:</span><span id="compare-verts-b" class="text-emerald-400 font-bold">--</span></div>
                <div class="flex justify-between"><span>Max Depth:</span><span id="compare-depth-b" class="text-purple-300">--</span></div>
                <div class="flex justify-between"><span>Floor Profile:</span><span id="compare-floor-b" class="text-slate-400">Photographic Relief &Delta;z(u, v)</span></div>
              </div>
            </div>
          </div>

          <!-- Comparative Evaluation Card -->
          <div class="bg-slate-950/90 border border-slate-800 rounded-xl p-4 text-xs space-y-2.5">
            <h4 class="font-bold text-slate-200 uppercase tracking-wider text-[11px] flex items-center space-x-2">
              <span>⚖️</span>
              <span>Side-by-Side Clinical Comparative Assessment</span>
            </h4>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-300 text-[11px]">
              <div class="bg-slate-900/70 p-3 rounded-lg border border-blue-500/20">
                <div class="font-bold text-blue-300 mb-1">Method A (Synthetic Parametric):</div>
                <p class="text-slate-400 leading-relaxed">Smooth mathematical surface profile. Consistent and predictable crater curvature; well-suited for early-stage standardized dental simulation curricula.</p>
              </div>
              <div class="bg-slate-900/70 p-3 rounded-lg border border-purple-500/20">
                <div class="font-bold text-purple-300 mb-1">Method B (Depth Anything V2 Relief Transfer):</div>
                <p class="text-slate-400 leading-relaxed">Dense photographic heightmap transfer. Accurately preserves natural demineralized pits, irregular floor micro-ripples, and jagged fracture boundaries directly from patient photos.</p>
              </div>
            </div>
          </div>
        </div>

        <div class="flex items-center justify-between pt-2 text-xs text-slate-400">
          <span class="flex items-center space-x-1.5 text-emerald-400">
            <span>🛡️</span>
            <span>Opposite-wall perforation gate active &bull; Watertight manifold guaranteed &bull; Ready for student simulation room</span>
          </span>
          <button onclick="window.scrollTo({ top: 0, behavior: 'smooth' })" class="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">
            &uarr; Back to Top
          </button>
        </div>
      </div>

    </div>

  </main>

  <script>
    let currentStem = '';
    let currentUploadFile = null;
    let currentInstances = [];
    let selectedInstanceIdx = 0;
    let allTeeth = [];

    async function init() {
      await loadTeethRegistry();
      await loadPresets();
      initTeacherStudio();
    }

    async function loadTeethRegistry() {
      try {
        const resp = await fetch('/api/teeth');
        const data = await resp.json();
        allTeeth = data.teeth;
        const select = document.getElementById('fdi-select');
        select.innerHTML = '';

        let currentQuad = '';
        let groupElem = null;

        allTeeth.forEach(tooth => {
          if (tooth.quadrant !== currentQuad) {
            currentQuad = tooth.quadrant;
            groupElem = document.createElement('optgroup');
            groupElem.label = `${tooth.quadrant} Arch (${tooth.anatomical_group})`;
            select.appendChild(groupElem);
          }
          const opt = document.createElement('option');
          opt.value = tooth.fdi;
          opt.textContent = `FDI #${tooth.fdi} - ${tooth.short_name} (${tooth.tooth_type})`;
          groupElem.appendChild(opt);
        });

        // Default to #36 Lower Molar
        select.value = "36";
        handleToothSelectionChange();

        // Also populate Teacher Studio FDI selector
        const tSelect = document.getElementById('teacher-fdi-select');
        if (tSelect) {
          tSelect.innerHTML = '';
          let currentQuadT = '';
          let groupElemT = null;
          allTeeth.forEach(tooth => {
            if (tooth.quadrant !== currentQuadT) {
              currentQuadT = tooth.quadrant;
              groupElemT = document.createElement('optgroup');
              groupElemT.label = `${tooth.quadrant} Arch (${tooth.anatomical_group})`;
              tSelect.appendChild(groupElemT);
            }
            const opt = document.createElement('option');
            opt.value = tooth.fdi;
            opt.textContent = `FDI #${tooth.fdi} - ${tooth.short_name} (${tooth.tooth_type})`;
            groupElemT.appendChild(opt);
          });
          tSelect.value = "16"; // Default Upper Right 1st Molar
        }
      } catch (e) {
        console.error('Failed to load teeth registry:', e);
      }
    }

    async function loadPresets() {
      try {
        const resp = await fetch('/api/samples');
        const data = await resp.json();
        const select = document.getElementById('preset-select');
        select.innerHTML = '<option value="">-- Choose a benchmark sample --</option>';

        let currentGroup = '';
        let groupElem = null;

        data.curated.forEach(item => {
          if (item.category !== currentGroup) {
            currentGroup = item.category;
            groupElem = document.createElement('optgroup');
            groupElem.label = currentGroup;
            select.appendChild(groupElem);
          }
          const opt = document.createElement('option');
          opt.value = item.stem;
          opt.textContent = item.name;
          groupElem.appendChild(opt);
        });

        if (data.curated.length > 0) {
          select.value = data.curated[0].stem;
          loadPresetSample();
        }
      } catch (e) {
        console.error('Failed to load presets:', e);
      }
    }

    function switchTab(tab) {
      ['single', 'batch', 'bench', 'teacher'].forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        const btn = document.getElementById(`tab-${t}-btn`);
        if (el) el.classList.add('hidden');
        if (btn) btn.className = 'px-3 py-1.5 rounded text-slate-300 hover:text-white flex items-center space-x-1.5';
      });
      const activeEl = document.getElementById(`tab-${tab}`);
      const activeBtn = document.getElementById(`tab-${tab}-btn`);
      if (activeEl) activeEl.classList.remove('hidden');
      if (activeBtn) activeBtn.className = 'px-3 py-1.5 rounded bg-blue-600 text-white font-semibold flex items-center space-x-1.5';
      if (tab === 'teacher') {
        setTimeout(redrawTeacherCanvas, 60);
      }
    }

    // ========================================================================
    // TEACHER STUDIO JAVASCRIPT ENGINE (PHASE 1 & PHASE 2)
    // ========================================================================
    let teacherMode = 'box'; // 'box' or 'point'
    let teacherBbox = null; // [x1, y1, x2, y2] normalized
    let teacherPoints = []; // [[x, y], ...] normalized
    let teacherStem = 'alpha_p006_F_15_001';
    let teacherUploadFile = null;
    let teacherIsDrawing = false;
    let teacherStartPos = null;
    let teacherCurrentCropBase64 = null;
    let teacherLastDiagnosis = null;
    let teacherLastPolygonNorm = null;
    let teacherDepthReliefGrid = null;
    let teacherCarveMethod = 'depth_heightmap'; // 'contour_conforming' | 'depth_heightmap' | 'compare'
    let agentRecommendedFdi = 16;

    function setCarveMethod(method) {
      teacherCarveMethod = method;
      const btnA = document.getElementById('btn-method-a');
      const btnB = document.getElementById('btn-method-b');
      const btnC = document.getElementById('btn-method-compare');
      const lbl = document.getElementById('active-method-label');
      const approveText = document.getElementById('approve-btn-text');

      [btnA, btnB, btnC].forEach(b => {
        if (b) b.className = 'py-1.5 px-1.5 rounded bg-slate-800 text-slate-300 border border-slate-700 hover:text-white text-center font-medium';
      });

      if (method === 'contour_conforming') {
        if (btnA) btnA.className = 'py-1.5 px-1.5 rounded bg-blue-600 text-white font-semibold border border-blue-500 text-center';
        if (lbl) lbl.textContent = 'Method A (Parametric Mathematical Floor)';
        if (approveText) approveText.textContent = '⚡ Carve Tooth (Method A: Parametric)';
      } else if (method === 'depth_heightmap') {
        if (btnB) btnB.className = 'py-1.5 px-1.5 rounded bg-purple-600 text-white font-semibold border border-purple-500 text-center';
        if (lbl) lbl.textContent = 'Method B (Depth Anything V2 Organic Relief)';
        if (approveText) approveText.textContent = '⚡ Pasting 3D Anomaly (Method B: Depth Relief)';
      } else {
        if (btnC) btnC.className = 'py-1.5 px-1.5 rounded bg-emerald-600 text-white font-bold border border-emerald-500 text-center shadow';
        if (lbl) lbl.textContent = 'Side-by-Side Dual Comparison (A vs B)';
        if (approveText) approveText.textContent = '⚡ Render Side-by-Side Comparison (A vs B)';
      }
    }

    function setTeacherMode(mode) {
      teacherMode = mode;
      document.getElementById('t-mode-box').className = mode === 'box'
        ? 'px-2.5 py-1 rounded text-xs font-semibold bg-blue-600 text-white'
        : 'px-2.5 py-1 rounded text-xs text-slate-300 hover:text-white bg-slate-800 border border-slate-700';
      document.getElementById('t-mode-point').className = mode === 'point'
        ? 'px-2.5 py-1 rounded text-xs font-semibold bg-blue-600 text-white'
        : 'px-2.5 py-1 rounded text-xs text-slate-300 hover:text-white bg-slate-800 border border-slate-700';
      document.getElementById('teacher-hint-text').textContent = mode === 'box'
        ? 'Click & Drag a box tightly around the lesion'
        : 'Click 1 or more points on the lesion center';
    }

    function clearTeacherPrompts() {
      teacherBbox = null;
      teacherPoints = [];
      teacherIsDrawing = false;
      teacherDepthReliefGrid = null;
      document.getElementById('teacher-prompt-readout').textContent = 'No prompt drawn yet';
      const hm = document.getElementById('teacher-heatmap-preview');
      if (hm) hm.classList.add('hidden');
      const hmPh = document.getElementById('teacher-heatmap-placeholder');
      if (hmPh) hmPh.classList.remove('hidden');
      redrawTeacherCanvas();
    }

    function initTeacherStudio() {
      const canvas = document.getElementById('teacher-canvas');
      const img = document.getElementById('teacher-source-img');

      img.onload = () => {
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        redrawTeacherCanvas();
      };

      window.addEventListener('resize', () => {
        if (img.clientWidth > 0 && img.clientHeight > 0) {
          canvas.width = img.clientWidth;
          canvas.height = img.clientHeight;
          redrawTeacherCanvas();
        }
      });

      canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

        if (teacherMode === 'box') {
          teacherIsDrawing = true;
          teacherStartPos = { x: nx, y: ny };
          teacherBbox = null;
        } else {
          teacherPoints.push([nx, ny]);
          document.getElementById('teacher-prompt-readout').textContent = `${teacherPoints.length} Point(s) Placed`;
          redrawTeacherCanvas();
        }
      });

      canvas.addEventListener('mousemove', (e) => {
        if (!teacherIsDrawing || teacherMode !== 'box') return;
        const rect = canvas.getBoundingClientRect();
        const curX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const curY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

        const x1 = Math.min(teacherStartPos.x, curX);
        const y1 = Math.min(teacherStartPos.y, curY);
        const x2 = Math.max(teacherStartPos.x, curX);
        const y2 = Math.max(teacherStartPos.y, curY);

        teacherBbox = [x1, y1, x2, y2];
        redrawTeacherCanvas();
      });

      canvas.addEventListener('mouseup', () => {
        if (teacherIsDrawing && teacherMode === 'box' && teacherBbox) {
          teacherIsDrawing = false;
          const w = Math.round((teacherBbox[2] - teacherBbox[0]) * 100);
          const h = Math.round((teacherBbox[3] - teacherBbox[1]) * 100);
          document.getElementById('teacher-prompt-readout').textContent = `Bounding Box: ${w}% × ${h}% of ROI`;
          redrawTeacherCanvas();
        }
      });

      // Load initial preset
      loadSampleTeacherPreset('alpha_p006_F_15_001');
    }

    function redrawTeacherCanvas() {
      const canvas = document.getElementById('teacher-canvas');
      if (!canvas) return;
      if (canvas.clientWidth > 0 && canvas.clientHeight > 0) {
        if (canvas.width !== canvas.clientWidth || canvas.height !== canvas.clientHeight) {
          canvas.width = canvas.clientWidth;
          canvas.height = canvas.clientHeight;
        }
      }
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw Bounding Box if present
      if (teacherBbox && teacherBbox.length === 4) {
        const x = teacherBbox[0] * canvas.width;
        const y = teacherBbox[1] * canvas.height;
        const w = (teacherBbox[2] - teacherBbox[0]) * canvas.width;
        const h = (teacherBbox[3] - teacherBbox[1]) * canvas.height;

        ctx.fillStyle = 'rgba(59, 130, 246, 0.15)';
        ctx.fillRect(x, y, w, h);

        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        ctx.fillStyle = '#0284c7';
        const s = 6;
        ctx.fillRect(x - s/2, y - s/2, s, s);
        ctx.fillRect(x + w - s/2, y - s/2, s, s);
        ctx.fillRect(x - s/2, y + h - s/2, s, s);
        ctx.fillRect(x + w - s/2, y + h - s/2, s, s);
      }

      // Draw Points if present
      if (teacherPoints.length > 0) {
        teacherPoints.forEach((pt, idx) => {
          const px = pt[0] * canvas.width;
          const py = pt[1] * canvas.height;

          ctx.beginPath();
          ctx.arc(px, py, 9, 0, 2 * Math.PI);
          ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
          ctx.fill();

          ctx.beginPath();
          ctx.arc(px, py, 4, 0, 2 * Math.PI);
          ctx.fillStyle = '#ef4444';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          ctx.font = '10px monospace';
          ctx.fillStyle = '#ffffff';
          ctx.fillText(`#${idx + 1}`, px + 7, py - 4);
        });
      }
    }

    async function loadSampleTeacherPreset(stem) {
      teacherStem = stem;
      teacherUploadFile = null;
      clearTeacherPrompts();
      document.getElementById('teacher-preset-select').value = stem;

      try {
        const resp = await fetch(`/api/inspect/preset/${stem}`);
        const data = await resp.json();
        const img = document.getElementById('teacher-source-img');
        img.src = data.original_image;
      } catch (e) {
        console.error('Failed to load preset photo:', e);
      }
    }

    function handleTeacherPresetChange() {
      const stem = document.getElementById('teacher-preset-select').value;
      loadSampleTeacherPreset(stem);
    }

    function handleTeacherFileUpload(e) {
      const file = e.target.files[0];
      if (!file) return;
      teacherUploadFile = file;
      teacherStem = null;
      clearTeacherPrompts();

      const reader = new FileReader();
      reader.onload = (evt) => {
        const img = document.getElementById('teacher-source-img');
        img.src = evt.target.result;
      };
      reader.readAsDataURL(file);
    }

    async function executeTeacherSAM() {
      if (!teacherBbox && teacherPoints.length === 0) {
        alert('Please draw a bounding box or place at least one point prompt first.');
        return;
      }

      const btn = document.getElementById('snap-sam-btn');
      btn.innerHTML = '<span class="animate-spin mr-2">⚙️</span> Snapping Lesion Contour...';
      btn.disabled = true;

      const fd = new FormData();
      if (teacherUploadFile) {
        fd.append('file', teacherUploadFile);
      } else if (teacherStem) {
        fd.append('stem', teacherStem);
      }

      if (teacherBbox) {
        fd.append('bbox_norm', JSON.stringify(teacherBbox));
      }
      if (teacherPoints.length > 0) {
        fd.append('points_norm', JSON.stringify(teacherPoints));
      }

      try {
        const resp = await fetch('/api/teacher/segment', { method: 'POST', body: fd });
        const res = await resp.json();

        if (!resp.ok || !res.success) {
          alert('SAM Boundary Extraction Error: ' + (res.detail || res.error || 'Failed'));
          return;
        }

        const overlayImg = document.getElementById('teacher-overlay-preview');
        overlayImg.src = res.overlay_base64;
        overlayImg.classList.remove('hidden');
        document.getElementById('teacher-overlay-placeholder').classList.add('hidden');

        const cropImg = document.getElementById('teacher-crop-preview');
        cropImg.src = res.crop_base64;
        cropImg.classList.remove('hidden');
        document.getElementById('teacher-crop-placeholder').classList.add('hidden');
        teacherCurrentCropBase64 = res.crop_base64;
        teacherLastPolygonNorm = res.polygon_norm;

        document.getElementById('sam-metric-points').textContent = `${res.points_count} vertices`;
        document.getElementById('sam-metric-centroid').textContent = `(${res.centroid_norm[0].toFixed(2)}, ${res.centroid_norm[1].toFixed(2)})`;
        document.getElementById('sam-metric-area').textContent = `${res.area_px.toLocaleString()} px²`;

        const pill = document.getElementById('sam-status-pill');
        pill.textContent = 'BOUNDARY SNAPPED';
        pill.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';

        document.getElementById('run-diagnose-btn').disabled = false;

        // Auto-extract Depth Anything V2 Heightmap (Method B)
        await executeTeacherDepthExtraction(res.crop_base64, res.polygon_norm);

      } catch (e) {
        console.error('SAM error:', e);
        alert('Failed to connect to SAM service.');
      } finally {
        btn.innerHTML = '<span>⚡ Snap Lesion Boundary (SAM Phase 1)</span>';
        btn.disabled = false;
      }
    }

    async function executeTeacherDepthExtraction(cropBase64, polyNorm) {
      try {
        const resp = await fetch('/api/teacher/extract_depth', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            crop_base64: cropBase64,
            polygon_norm: polyNorm
          })
        });
        const res = await resp.json();
        if (res.success) {
          const hm = document.getElementById('teacher-heatmap-preview');
          hm.src = res.heatmap_base64;
          hm.classList.remove('hidden');
          document.getElementById('teacher-heatmap-placeholder').classList.add('hidden');
          teacherDepthReliefGrid = res.depth_relief_grid;
          if (res.metrics && res.metrics.enamel_ring_samples) {
            document.getElementById('depth-metric-samples').textContent = `${res.metrics.enamel_ring_samples.toLocaleString()} ring pts`;
          }
        }
      } catch (err) {
        console.error('Depth extraction error:', err);
      }
    }

    async function executeTeacherDiagnosis() {
      if (!teacherCurrentCropBase64) {
        alert('Please execute Phase 1 SAM boundary snapping first.');
        return;
      }

      const btn = document.getElementById('run-diagnose-btn');
      btn.innerHTML = '<span class="animate-spin mr-2">⚙️</span> Agent Evaluating Pathology...';
      btn.disabled = true;

      const fdi = parseInt(document.getElementById('teacher-fdi-select').value) || 16;
      const surface = document.getElementById('teacher-surface-select').value;
      const notes = document.getElementById('teacher-notes-input').value;
      const apiKey = document.getElementById('teacher-api-key').value;

      try {
        const resp = await fetch('/api/teacher/diagnose', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            crop_base64: teacherCurrentCropBase64,
            tooth_fdi: fdi,
            target_surface: surface,
            instructor_notes: notes,
            api_key: apiKey
          })
        });
        const diag = await resp.json();

        if (!resp.ok) {
          alert('Diagnosis Error: ' + (diag.detail || 'Evaluation failed.'));
          return;
        }

        teacherLastDiagnosis = diag;

        document.getElementById('diagnostic-card').classList.remove('hidden');

        const pBadge = document.getElementById('diag-pathology-badge');
        pBadge.textContent = diag.pathology;
        if (diag.pathology === 'Caries') {
          pBadge.className = 'px-2.5 py-0.5 rounded text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30';
        } else if (diag.pathology === 'Abrasion') {
          pBadge.className = 'px-2.5 py-0.5 rounded text-xs font-bold bg-orange-500/20 text-orange-400 border border-orange-500/30';
        } else {
          pBadge.className = 'px-2.5 py-0.5 rounded text-xs font-bold bg-fuchsia-500/20 text-fuchsia-400 border border-fuchsia-500/30';
        }

        document.getElementById('diag-confidence-text').textContent = `Conf: ${Math.round((diag.confidence || 0.9) * 100)}%`;
        document.getElementById('diag-severity-title').textContent = diag.severity_grade || diag.pathology;

        const sPill = document.getElementById('agent-source-pill');
        if (diag.source && diag.source.includes('gemini')) {
          sPill.textContent = 'GEMINI 2.0 FLASH';
          sPill.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30';
        } else {
          sPill.textContent = 'LOCAL EXPERT ENGINE';
          sPill.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/30';
        }

        const tBox = document.getElementById('diag-tissues-container');
        tBox.innerHTML = '';
        (diag.tissues_involved || ['enamel']).forEach(tis => {
          const sp = document.createElement('span');
          sp.textContent = tis.toUpperCase();
          sp.className = tis === 'enamel'
            ? 'px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700'
            : 'px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/40';
          tBox.appendChild(sp);
        });

        const depthVal = diag.recommended_depth_mm || 0.75;
        const depthSlider = document.getElementById('teacher-depth-slider');
        depthSlider.value = depthVal;
        document.getElementById('diag-depth-readout').textContent = `${depthVal.toFixed(2)} mm`;

        if (diag.depth_range_mm && diag.depth_range_mm.length === 2) {
          document.getElementById('diag-min-depth').textContent = `Min: ${diag.depth_range_mm[0].toFixed(2)}mm`;
          document.getElementById('diag-max-depth').textContent = `Max: ${diag.depth_range_mm[1].toFixed(2)}mm`;
          depthSlider.min = diag.depth_range_mm[0];
          depthSlider.max = diag.depth_range_mm[1];
        }

        document.getElementById('diag-steepness-val').textContent = `${(diag.margin_steepness || 0.7).toFixed(2)} (${diag.margin_steepness > 0.7 ? 'Undermined' : 'Beveled'})`;
        document.getElementById('diag-floor-val').textContent = (diag.floor_morphology || 'Irregular Concave').replace('_', ' ').toUpperCase();

        document.getElementById('diag-rationale-text').textContent = diag.clinical_rationale || 'Ground truth anatomical assessment.';

        const wBanner = document.getElementById('diag-whitelist-banner');
        if (diag.whitelist_valid) {
          wBanner.className = 'p-2 rounded text-xs border flex items-center space-x-2 bg-emerald-950/40 border-emerald-500/30 text-emerald-300';
          wBanner.innerHTML = `<span class="text-sm">✅</span><span>APPROVED: ${diag.pathology} is valid on FDI #${fdi} (${surface}).</span>`;
        } else {
          wBanner.className = 'p-2 rounded text-xs border flex items-center space-x-2 bg-rose-950/40 border-rose-500/30 text-rose-300';
          wBanner.innerHTML = `<span class="text-sm">⚠️</span><span>WHITELIST VIOLATION: ${diag.whitelist_meta ? diag.whitelist_meta.reason : 'Combination not approved.'}</span>`;
        }

        // Query Clinical Transfer Agent for simulation pasting recommendations
        await fetchAgentTargetRecommendation(diag.pathology, diag.severity_grade);

      } catch (e) {
        console.error('Diagnosis error:', e);
        alert('Failed to connect to Clinical Diagnostics Agent.');
      } finally {
        btn.innerHTML = '<span>🤖 Run Clinical Diagnosis (Phase 2 Agent)</span>';
        btn.disabled = false;
      }
    }

    async function fetchAgentTargetRecommendation(pathology, severity) {
      try {
        const curFdi = parseInt(document.getElementById('teacher-fdi-select').value) || 16;
        const resp = await fetch('/api/teacher/agent_recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pathology, severity_grade: severity, current_fdi: curFdi })
        });
        const rec = await resp.json();
        if (rec.success) {
          agentRecommendedFdi = rec.recommended_fdi;
          document.getElementById('agent-recommendation-box').classList.remove('hidden');
          document.getElementById('agent-rec-fdi-badge').textContent = `FDI #${rec.recommended_fdi}`;
          document.getElementById('agent-rec-rationale').textContent = `"${rec.recommendation_rationale}"`;
        }
      } catch (err) {
        console.error('Agent recommendation error:', err);
      }
    }

    function applyAgentRecommendedTooth() {
      if (agentRecommendedFdi) {
        document.getElementById('teacher-fdi-select').value = String(agentRecommendedFdi);
      }
    }

    function updateTeacherDepth(val) {
      document.getElementById('diag-depth-readout').textContent = `${parseFloat(val).toFixed(2)} mm`;
      if (teacherLastDiagnosis) {
        teacherLastDiagnosis.recommended_depth_mm = parseFloat(val);
      }
    }

    async function confirmTeacherCase() {
      if (!teacherLastDiagnosis) {
        alert('Please run clinical diagnosis first.');
        return;
      }
      if (!teacherLastPolygonNorm) {
        alert('No SAM boundary polygon available to carve.');
        return;
      }

      const btn = document.getElementById('approve-case-btn');
      btn.innerHTML = '<span class="animate-spin mr-2">⚙️</span> Sculpting 3D Defect...';
      btn.disabled = true;

      const fdi = parseInt(document.getElementById('teacher-fdi-select').value) || 16;
      const surface = document.getElementById('teacher-surface-select').value;
      const depth = parseFloat(document.getElementById('teacher-depth-slider').value) || 0.85;

      try {
        if (teacherCarveMethod === 'compare') {
          // SIDE-BY-SIDE COMPARISON PATH (METHOD A vs METHOD B)
          const resp = await fetch('/api/teacher/compare_methods', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              polygon_norm: teacherLastPolygonNorm,
              tooth_fdi: fdi,
              target_surface: surface,
              depth_mm: depth,
              margin_steepness: teacherLastDiagnosis.margin_steepness || 0.70,
              floor_morphology: teacherLastDiagnosis.floor_morphology || 'irregular_concave',
              pathology: teacherLastDiagnosis.pathology || 'Caries',
              depth_relief_grid: teacherDepthReliefGrid
            })
          });
          const res = await resp.json();
          if (!resp.ok) {
            alert('Comparison Carve Error: ' + (res.detail || 'Failed.'));
            return;
          }

          const card = document.getElementById('teacher-3d-result-card');
          card.classList.remove('hidden');
          document.getElementById('single-carve-view').classList.add('hidden');
          document.getElementById('side-by-side-comparison-view').classList.remove('hidden');

          document.getElementById('teacher-3d-main-title').textContent = 'Phase 3: Side-by-Side Method Comparison (A vs B)';
          document.getElementById('teacher-method-badge').textContent = 'METHOD A vs METHOD B DUAL-VIEW';
          document.getElementById('teacher-method-badge').className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';

          document.getElementById('compare-render-a').src = res.method_a.render_base64;
          document.getElementById('compare-verts-a').textContent = `${res.method_a.metrics.num_displaced_vertices.toLocaleString()} verts`;
          document.getElementById('compare-depth-a').textContent = `${res.method_a.metrics.max_depth_reached_mm} mm`;

          document.getElementById('compare-render-b').src = res.method_b.render_base64;
          document.getElementById('compare-verts-b').textContent = `${res.method_b.metrics.num_displaced_vertices.toLocaleString()} verts`;
          document.getElementById('compare-depth-b').textContent = `${res.method_b.metrics.max_depth_reached_mm} mm`;

          document.getElementById('teacher-3d-tooth-title').textContent = `${res.tooth_name} (FDI #${res.tooth_fdi}) • ${res.pathology} • ${surface.replace('_', ' ').toUpperCase()}`;
          card.scrollIntoView({ behavior: 'smooth' });

        } else {
          // SINGLE METHOD CARVE PATH (METHOD A OR METHOD B)
          const resp = await fetch('/api/teacher/carve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              polygon_norm: teacherLastPolygonNorm,
              tooth_fdi: fdi,
              target_surface: surface,
              depth_mm: depth,
              margin_steepness: teacherLastDiagnosis.margin_steepness || 0.70,
              floor_morphology: teacherLastDiagnosis.floor_morphology || 'irregular_concave',
              pathology: teacherLastDiagnosis.pathology || 'Caries',
              method: teacherCarveMethod,
              depth_relief_grid: teacherDepthReliefGrid
            })
          });
          const res = await resp.json();

          if (!resp.ok) {
            alert('3D Carving Error: ' + (res.detail || 'Carving failed.'));
            return;
          }

          const card = document.getElementById('teacher-3d-result-card');
          card.classList.remove('hidden');
          document.getElementById('single-carve-view').classList.remove('hidden');
          document.getElementById('side-by-side-comparison-view').classList.add('hidden');

          document.getElementById('teacher-3d-render-img').src = res.render_base64;
          document.getElementById('teacher-3d-tooth-title').textContent = `${res.tooth_name} (FDI #${res.tooth_fdi}) • ${res.pathology} • ${surface.replace('_', ' ').toUpperCase()}`;
          document.getElementById('teacher-3d-verts-badge').textContent = `${res.metrics.num_displaced_vertices} Vertices Displaced`;
          document.getElementById('teacher-3d-depth-badge').textContent = `Max Depth: ${res.metrics.max_depth_reached_mm} mm`;
          document.getElementById('teacher-3d-span-badge').textContent = `Span: ${res.metrics.lesion_span_mm[0]}mm × ${res.metrics.lesion_span_mm[1]}mm`;

          const mBadge = document.getElementById('teacher-method-badge');
          if (res.method === 'depth_heightmap') {
            mBadge.textContent = 'METHOD B (DEPTH MODEL RELIEF)';
            mBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40';
          } else {
            mBadge.textContent = 'METHOD A (SYNTHETIC PARAMETRIC)';
            mBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-blue-500/20 text-blue-300 border border-blue-500/40';
          }

          card.scrollIntoView({ behavior: 'smooth' });
        }

      } catch (e) {
        console.error('Carve error:', e);
        alert('Failed to execute 3D carving.');
      } finally {
        setCarveMethod(teacherCarveMethod);
        btn.disabled = false;
      }
    }

    function updateConf(val) {
      document.getElementById('conf-val').textContent = val;
      rerunInspection();
    }

    function updateOpacity(val) {
      document.getElementById('opacity-val').textContent = Math.round(val * 100) + '%';
      rerunInspection();
    }

    async function loadPresetSample() {
      const select = document.getElementById('preset-select');
      const stem = select.value;
      if (!stem) return;
      currentStem = stem;
      currentUploadFile = null;
      clearCanvasPoints();
      reset3DCarvePanel();
      rerunInspection();
    }

    async function handleFileUpload(evt) {
      const file = evt.target.files[0];
      if (!file) return;
      currentUploadFile = file;
      currentStem = '';
      document.getElementById('preset-select').value = '';
      clearCanvasPoints();
      reset3DCarvePanel();
      rerunInspection();
    }

    function reset3DCarvePanel() {
      document.getElementById('carve-render-container').classList.add('hidden');
      document.getElementById('carve-block-alert').classList.add('hidden');
      document.getElementById('carve-crack-notice').classList.add('hidden');
      document.getElementById('carve-placeholder').classList.remove('hidden');
      document.getElementById('carve-status-pill').textContent = 'AWAITING CARVE';
      document.getElementById('carve-status-pill').className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700';
    }

    async function rerunInspection() {
      const conf = document.getElementById('conf-slider').value;
      const opacity = document.getElementById('opacity-slider').value;
      const showAbrasion = document.getElementById('toggle-abrasion').checked;
      const showCaries = document.getElementById('toggle-caries').checked;
      const showCrack = document.getElementById('toggle-crack').checked;

      document.getElementById('loading-spinner').classList.remove('hidden');
      document.getElementById('loading-text').textContent = 'Running 2D segmentation...';

      try {
        let data;
        if (currentUploadFile) {
          const formData = new FormData();
          formData.append('file', currentUploadFile);
          formData.append('conf', conf);
          formData.append('opacity', opacity);
          formData.append('show_abrasion', showAbrasion);
          formData.append('show_caries', showCaries);
          formData.append('show_crack', showCrack);

          const resp = await fetch('/api/inspect/upload', { method: 'POST', body: formData });
          data = await resp.json();
        } else if (currentStem) {
          const url = `/api/inspect/preset/${currentStem}?conf=${conf}&opacity=${opacity}&show_abrasion=${showAbrasion}&show_caries=${showCaries}&show_crack=${showCrack}`;
          const resp = await fetch(url);
          data = await resp.json();
        } else {
          document.getElementById('loading-spinner').classList.add('hidden');
          return;
        }

        currentInstances = data.instances || [];
        selectedInstanceIdx = 0;

        if (data.recommended_fdi) {
          document.getElementById('fdi-select').value = String(data.recommended_fdi);
        }

        renderInspectionResults(data);
        handleToothSelectionChange();
      } catch (err) {
        console.error('Inspection error:', err);
      } finally {
        document.getElementById('loading-spinner').classList.add('hidden');
      }
    }

    function renderInspectionResults(data) {
      document.getElementById('orig-img').src = data.original_image;
      document.getElementById('orig-img').classList.remove('hidden');
      document.getElementById('orig-placeholder').classList.add('hidden');
      document.getElementById('img-dim').textContent = `${data.dimensions.width}x${data.dimensions.height}px`;

      document.getElementById('pred-img').src = data.prediction_overlay;
      document.getElementById('pred-img').classList.remove('hidden');
      document.getElementById('pred-placeholder').classList.add('hidden');
      document.getElementById('pred-count-badge').textContent = `${data.instances.length} detection(s)`;

      const gtImg = document.getElementById('gt-img');
      const gtPlaceholder = document.getElementById('gt-placeholder');
      const gtBadge = document.getElementById('gt-status-badge');

      if (data.has_ground_truth && data.ground_truth_overlay) {
        gtImg.src = data.ground_truth_overlay;
        gtImg.classList.remove('hidden');
        gtPlaceholder.classList.add('hidden');
        gtBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300';
        gtBadge.textContent = 'GROUND TRUTH AVAILABLE';
      } else {
        gtImg.src = '';
        gtImg.classList.add('hidden');
        gtPlaceholder.classList.remove('hidden');
        gtBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300';
        gtBadge.textContent = data.is_novel ? 'NOVEL / UNLABELED' : 'NO GROUND TRUTH';
      }

      document.getElementById('status-text').textContent = data.status_message;
      const statusBadges = document.getElementById('status-badges');
      statusBadges.innerHTML = '';
      if (data.is_novel) {
        statusBadges.innerHTML += '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">OUT-OF-DATASET</span>';
      }
      if (data.has_crack_prediction) {
        statusBadges.innerHTML += '<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">CRACK DETECTED (2D-ONLY)</span>';
      }

      const crackBox = document.getElementById('crack-alert-box');
      if (data.crack_warning_active) {
        crackBox.classList.remove('hidden');
      } else {
        crackBox.classList.add('hidden');
      }

      // Render Instances Table
      const tbody = document.getElementById('instances-tbody');
      tbody.innerHTML = '';
      document.getElementById('instance-summary-text').textContent = `${data.instances.length} instance(s) detected`;

      if (data.instances.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500">No anomalies detected above current confidence threshold</td></tr>';
      } else {
        data.instances.forEach((inst, idx) => {
          const row = document.createElement('tr');
          const isSelected = (idx === selectedInstanceIdx);
          row.className = isSelected 
            ? 'bg-purple-950/40 border border-purple-500/40 cursor-pointer' 
            : 'hover:bg-slate-800/40 cursor-pointer';

          row.onclick = () => {
            selectedInstanceIdx = idx;
            renderInspectionResults(data);
            updateWhitelistIndicator();
          };

          const carveProfileBadge = inst.can_carve_3d 
            ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">${inst.canonical_anomaly} Profile</span>`
            : `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">2D Only (No 3D Profile)</span>`;

          const reliabilityBadge = inst.is_crack 
            ? `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">⚠️ UNRELIABLE (11.1% recall)</span>`
            : `<span class="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300">Standard Confidence</span>`;

          row.innerHTML = `
            <td class="py-2.5 px-3 font-mono font-bold ${isSelected ? 'text-purple-400' : 'text-slate-400'}">${isSelected ? '▶ ' : ''}#${inst.id}</td>
            <td class="py-2.5 px-3 font-semibold flex items-center space-x-2">
              <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${inst.color}"></span>
              <span style="color: ${inst.color}">${inst.class_name}</span>
            </td>
            <td class="py-2.5 px-3 font-mono font-bold text-slate-100">${(inst.confidence * 100).toFixed(1)}%</td>
            <td class="py-2.5 px-3">${carveProfileBadge}</td>
            <td class="py-2.5 px-3 font-mono text-slate-400 text-[11px]">[${inst.box.join(', ')}]</td>
            <td class="py-2.5 px-3">${reliabilityBadge}</td>
          `;
          tbody.appendChild(row);
        });
      }
    }

    function handleToothSelectionChange() {
      const fdi = parseInt(document.getElementById('fdi-select').value, 10);
      const tooth = allTeeth.find(t => t.fdi === fdi);
      if (tooth) {
        const archLabel = tooth.arch || (tooth.quadrant === 'Q1' ? 'Maxillary Right (Q1)' : tooth.quadrant === 'Q2' ? 'Maxillary Left (Q2)' : tooth.quadrant === 'Q3' ? 'Mandibular Left (Q3)' : 'Mandibular Right (Q4)');
        document.getElementById('anat-arch').textContent = archLabel;
        document.getElementById('anat-type').textContent = `${tooth.short_name} (${tooth.tooth_type})`;
        document.getElementById('anat-surf').textContent = tooth.surface;
      }
      updateWhitelistIndicator();
    }

    function updateWhitelistIndicator() {
      const fdi = parseInt(document.getElementById('fdi-select').value, 10);
      const indicator = document.getElementById('tooth-whitelist-indicator');
      const tooth = allTeeth.find(t => t.fdi === fdi);

      if (!tooth) {
        indicator.innerHTML = '<span class="text-slate-400">Select an FDI tooth to check whitelist status.</span>';
        indicator.className = 'p-2.5 rounded-lg border text-xs bg-slate-800/60 border-slate-700 text-slate-400';
        return;
      }

      if (currentInstances.length === 0) {
        indicator.innerHTML = `<span class="text-slate-300">FDI #${tooth.fdi} (${tooth.name}): Allowed: <strong>${tooth.allowed_anomalies.join(', ')}</strong></span>`;
        indicator.className = 'p-2.5 rounded-lg border text-xs bg-slate-800/60 border-slate-700 text-slate-300';
        return;
      }

      const activeInst = currentInstances[selectedInstanceIdx] || currentInstances[0];
      const cid = activeInst.class_id;

      if (cid === 2) { // Crack
        indicator.innerHTML = `
          <div class="text-amber-300 font-bold flex items-center space-x-1.5">
            <span>⚠️</span>
            <span>Crack is 2D-Only</span>
          </div>
          <div class="text-[11px] text-slate-400 mt-1">No 3D carve profile exists for Crack. 3D button will display diagnostic notice.</div>
        `;
        indicator.className = 'p-2.5 rounded-lg border text-xs bg-amber-950/30 border-amber-500/50 text-amber-200';
        return;
      }

      const canonical = activeInst.canonical_anomaly;
      const isAllowed = tooth.allowed_anomalies.includes(canonical);

      if (isAllowed) {
        indicator.innerHTML = `
          <div class="text-emerald-400 font-bold flex items-center space-x-1.5">
            <span>✓</span>
            <span>Whitelist Approved for 3D Carve</span>
          </div>
          <div class="text-[11px] text-slate-300 mt-1">FDI #${tooth.fdi} (${tooth.short_name}) &times; <strong>${canonical}</strong> on <em>${tooth.surface}</em> surface.</div>
        `;
        indicator.className = 'p-2.5 rounded-lg border text-xs bg-emerald-950/30 border-emerald-500/50 text-emerald-200';
      } else {
        indicator.innerHTML = `
          <div class="text-amber-400 font-bold flex items-center space-x-1.5">
            <span>🚫</span>
            <span>Whitelist Blocked: ${canonical} not permitted on ${tooth.tooth_type}</span>
          </div>
          <div class="text-[11px] text-slate-300 mt-1">FDI #${tooth.fdi} only permits: ${tooth.allowed_anomalies.join(', ')}.</div>
        `;
        indicator.className = 'p-2.5 rounded-lg border text-xs bg-amber-950/40 border-amber-500/60 text-amber-200';
      }
    }

    async function executeCarve() {
      const fdi = parseInt(document.getElementById('fdi-select').value, 10);
      const conf = document.getElementById('conf-slider').value;

      document.getElementById('loading-spinner').classList.remove('hidden');
      document.getElementById('loading-text').textContent = 'Converting 2D geometry and carving 3D mesh...';

      reset3DCarvePanel();

      try {
        const formData = new FormData();
        formData.append('tooth_fdi', fdi);
        formData.append('instance_idx', selectedInstanceIdx);
        formData.append('conf', conf);
        formData.append('show_abrasion', document.getElementById('toggle-abrasion').checked);
        formData.append('show_caries', document.getElementById('toggle-caries').checked);
        formData.append('show_crack', document.getElementById('toggle-crack').checked);

        if (currentUploadFile) {
          formData.append('file', currentUploadFile);
        } else if (currentStem) {
          formData.append('stem', currentStem);
        }

        const resp = await fetch('/api/carve', { method: 'POST', body: formData });
        const data = await resp.json();

        handleCarveResult(data);
      } catch (err) {
        console.error('Carve execution error:', err);
        alert('Carve execution failed: ' + err.message);
      } finally {
        document.getElementById('loading-spinner').classList.add('hidden');
      }
    }

    function handleCarveResult(data) {
      document.getElementById('carve-placeholder').classList.add('hidden');

      if (data.status === 'crack_2d_only') {
        const crackNotice = document.getElementById('carve-crack-notice');
        crackNotice.classList.remove('hidden');
        document.getElementById('carve-crack-message').textContent = `${data.message} ${data.advisory}`;
        document.getElementById('carve-status-pill').textContent = '2D-ONLY (NO 3D PROFILE)';
        document.getElementById('carve-status-pill').className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40';
        return;
      }

      if (data.status === 'whitelist_blocked') {
        const blockAlert = document.getElementById('carve-block-alert');
        blockAlert.classList.remove('hidden');
        document.getElementById('carve-block-message').textContent = data.message;
        document.getElementById('carve-block-advisory').textContent = `${data.advisory} Allowed on this tooth: ${data.allowed_anomalies.join(', ')}.`;
        document.getElementById('carve-status-pill').textContent = `BLOCKED (${data.error_code})`;
        document.getElementById('carve-status-pill').className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-red-500/20 text-red-300 border border-red-500/40';
        return;
      }

      if (data.status === 'carved_success') {
        const container = document.getElementById('carve-render-container');
        container.classList.remove('hidden');

        document.getElementById('carve-render-img').src = data.render_image;
        const manualBadge = document.getElementById('manual-experimental-badge');
        const banner = document.getElementById('carve-derivation-banner');

        if (data.is_manual) {
          manualBadge.classList.remove('hidden');
          document.getElementById('carve-status-pill').textContent = 'MANUAL (EXPERIMENTAL)';
          document.getElementById('carve-status-pill').className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40';
          document.getElementById('carve-surface-pill').textContent = `Target: ${data.target_surface.toUpperCase()}`;

          // Prominent Fix 1 Badge on Carve Result Screen
          if (data.is_photo_derived) {
            banner.className = 'p-3 rounded-xl border text-xs font-bold flex items-center justify-between bg-emerald-950/70 border-emerald-500/80 text-emerald-300 shadow-md shadow-emerald-950/40';
            banner.innerHTML = `
              <div class="flex items-center space-x-2.5">
                <span class="text-lg">🟢</span>
                <span>Photo-Derived Crack Carve — Scaled & mapped directly from ${data.metrics.traced_points_count || 'photo'} traced points</span>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 border border-emerald-500/50 text-emerald-200 font-mono">PHOTO-DERIVED</span>
            `;
            document.getElementById('metric-source').textContent = `🟢 Photo (${data.metrics.traced_points_count} pts)`;
            document.getElementById('metric-source').className = 'text-xs font-bold text-emerald-300 font-mono mt-0.5 truncate';
          } else {
            banner.className = 'p-3 rounded-xl border text-xs font-bold flex items-center justify-between bg-amber-950/70 border-amber-500/80 text-amber-300 shadow-md shadow-amber-950/40';
            banner.innerHTML = `
              <div class="flex items-center space-x-2.5">
                <span class="text-lg">🟡</span>
                <span>Demo Preset Carve (Generic Path) — NOT derived from photo points (Synthetic preset: ${data.metrics.crack_preset || 'Preset'})</span>
              </div>
              <span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 border border-amber-500/50 text-amber-200 font-mono">DEMO PRESET</span>
            `;
            document.getElementById('metric-source').textContent = `🟡 Demo Preset`;
            document.getElementById('metric-source').className = 'text-xs font-bold text-amber-300 font-mono mt-0.5 truncate';
          }

          document.getElementById('metric-surface').textContent = data.target_surface;
          document.getElementById('metric-radius').textContent = data.metrics.width_mm ? `${data.metrics.width_mm} mm (W)` : (data.metrics.target_cusp || '--');
          document.getElementById('metric-depth').textContent = data.metrics.depth_mm ? `${data.metrics.depth_mm} mm (D)` : `${data.metrics.chip_height_mm} mm (H)`;
          document.getElementById('metric-ar').textContent = data.metrics.path_type ? 'Polyline' : `${data.metrics.chip_angle_deg}° Tilt`;
          document.getElementById('metric-verts').textContent = `${data.metrics.influenced_vertices.toLocaleString()} / ${(data.metrics.total_vertices || data.metrics.capped_vertices).toLocaleString()}`;
          document.getElementById('carve-notes').textContent = `${data.message} ${data.badge}`;
        } else {
          manualBadge.classList.add('hidden');
          document.getElementById('carve-status-pill').textContent = '3D CARVE SUCCESSFUL';
          document.getElementById('carve-status-pill').className = 'px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
          document.getElementById('carve-surface-pill').textContent = `Target: ${data.target_surface.toUpperCase()}`;

          banner.className = 'p-3 rounded-xl border text-xs font-bold flex items-center justify-between bg-purple-950/70 border-purple-500/80 text-purple-300 shadow-md shadow-purple-950/40';
          banner.innerHTML = `
            <div class="flex items-center space-x-2.5">
              <span class="text-lg">🚀</span>
              <span>Autonomous 2D-to-3D Projection — Segmented mask geometry extracted directly from photo</span>
            </div>
            <span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 border border-purple-500/50 text-purple-200 font-mono">AUTONOMOUS MASK</span>
          `;
          document.getElementById('metric-source').textContent = `🚀 Auto Mask`;
          document.getElementById('metric-source').className = 'text-xs font-bold text-purple-300 font-mono mt-0.5 truncate';

          document.getElementById('metric-surface').textContent = data.target_surface;
          document.getElementById('metric-radius').textContent = `${data.metrics.radius_mm} mm`;
          document.getElementById('metric-depth').textContent = `${data.metrics.depth_mm} mm`;
          document.getElementById('metric-ar').textContent = data.metrics.aspect_ratio;
          document.getElementById('metric-verts').textContent = `${data.metrics.influenced_vertices.toLocaleString()} / ${data.metrics.total_vertices.toLocaleString()}`;
          document.getElementById('carve-notes').textContent = `${data.message} Note: ${data.notes || 'Carve executed via parameter_converter & tooth_carver'}`;
        }
        return;
      }

      if (data.status === 'no_anomaly') {
        alert(data.message);
      }
    }

    let manualCanvasPoints = [];

    function toggleManualProfileUI() {
      const prof = document.getElementById('manual-profile-select').value;
      const crackBox = document.getElementById('manual-crack-controls');
      const chipBox = document.getElementById('manual-chip-controls');
      if (prof === 'v_groove_path') {
        crackBox.classList.remove('hidden');
        chipBox.classList.add('hidden');
      } else {
        crackBox.classList.add('hidden');
        chipBox.classList.remove('hidden');
      }
    }

    function onPhotoClick(event) {
      const img = event.target;
      const rect = img.getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      const relY = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
      manualCanvasPoints.push([relX, relY]);
      document.getElementById('canvas-pts-count').innerHTML = `Photo Points: <strong class="text-amber-400 font-mono">${manualCanvasPoints.length}</strong> (Tracing active)`;
      document.getElementById('crack-preset-select').value = 'custom_canvas';
      renderTraceSvg();
    }

    function clearCanvasPoints() {
      manualCanvasPoints = [];
      document.getElementById('canvas-pts-count').innerHTML = `Photo Points: <strong class="text-amber-400 font-mono">0</strong> (Click photo to trace)`;
      document.getElementById('crack-preset-select').value = 'custom_canvas';
      renderTraceSvg();
    }

    function renderTraceSvg() {
      const svgs = [document.getElementById('orig-trace-svg'), document.getElementById('pred-trace-svg')];
      svgs.forEach(svg => {
        if (!svg) return;
        svg.innerHTML = '';
        if (manualCanvasPoints.length === 0) return;

        if (manualCanvasPoints.length >= 2) {
          const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
          const ptsStr = manualCanvasPoints.map(p => `${(p[0]*100).toFixed(2)},${(p[1]*100).toFixed(2)}`).join(' ');
          poly.setAttribute('points', ptsStr);
          poly.setAttribute('fill', 'none');
          poly.setAttribute('stroke', '#fbbf24');
          poly.setAttribute('stroke-width', '2');
          poly.setAttribute('stroke-linecap', 'round');
          poly.setAttribute('stroke-linejoin', 'round');
          svg.appendChild(poly);
        }

        manualCanvasPoints.forEach((p, i) => {
          const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          circle.setAttribute('cx', `${(p[0]*100).toFixed(2)}`);
          circle.setAttribute('cy', `${(p[1]*100).toFixed(2)}`);
          circle.setAttribute('r', '2.5');
          circle.setAttribute('fill', '#f59e0b');
          circle.setAttribute('stroke', '#ffffff');
          circle.setAttribute('stroke-width', '0.8');
          svg.appendChild(circle);
        });
      });
    }

    function openPresetConfirmModal() {
      const presetSelect = document.getElementById('crack-preset-select');
      const presetText = presetSelect.options[presetSelect.selectedIndex].text;
      document.getElementById('modal-preset-name').textContent = presetText;
      document.getElementById('preset-confirm-modal').classList.remove('hidden');
    }

    function closePresetConfirmModal(proceed) {
      document.getElementById('preset-confirm-modal').classList.add('hidden');
      if (proceed) {
        executeManualCarve(true);
      }
    }

    async function executeManualCarve(forceProceed = false) {
      const fdi = parseInt(document.getElementById('fdi-select').value, 10);
      const profileId = document.getElementById('manual-profile-select').value;

      // FIX 1: Enforce explicit acknowledgment before rendering generic preset
      if (profileId === 'v_groove_path') {
        const preset = document.getElementById('crack-preset-select').value;
        const isPhotoTraced = (preset === 'custom_canvas' && manualCanvasPoints.length >= 2);
        if (!isPhotoTraced && !forceProceed) {
          openPresetConfirmModal();
          return;
        }
      }

      document.getElementById('loading-spinner').classList.remove('hidden');
      document.getElementById('loading-text').textContent = `Sculpting manual ${profileId}...`;
      reset3DCarvePanel();

      try {
        const formData = new FormData();
        formData.append('profile_id', profileId);
        formData.append('tooth_fdi', fdi);

        if (profileId === 'v_groove_path') {
          const preset = document.getElementById('crack-preset-select').value;
          formData.append('crack_preset', preset);
          formData.append('crack_depth', document.getElementById('crack-depth-input').value);
          formData.append('crack_width', document.getElementById('crack-width-input').value);
          if (preset === 'custom_canvas' && manualCanvasPoints.length >= 2) {
            formData.append('points_2d_json', JSON.stringify(manualCanvasPoints));
          }
        } else if (profileId === 'face_removal_recap') {
          formData.append('cusp_target', document.getElementById('chip-cusp-select').value);
          formData.append('chip_height', document.getElementById('chip-height-input').value);
          formData.append('chip_angle', document.getElementById('chip-angle-input').value);
        }

        const resp = await fetch('/api/carve_manual', { method: 'POST', body: formData });
        const data = await resp.json();
        handleCarveResult(data);
      } catch (err) {
        console.error('Manual carve execution error:', err);
        alert('Manual carve execution failed: ' + err.message);
      } finally {
        document.getElementById('loading-spinner').classList.add('hidden');
      }
    }

    async function runBatchPreset() {
      const grid = document.getElementById('batch-grid');
      grid.innerHTML = '<div class="col-span-full py-12 text-center text-slate-400"><div class="inline-block animate-spin rounded-full h-8 w-8 border-2 border-blue-500 border-t-transparent mb-2"></div><p>Evaluating 8 benchmark cases on GPU...</p></div>';

      try {
        const formData = new FormData();
        const resp = await fetch('/api/inspect/batch', { method: 'POST', body: formData });
        const data = await resp.json();

        grid.innerHTML = '';
        data.results.forEach((item, idx) => {
          const card = document.createElement('div');
          card.className = 'bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col';

          let crackBadge = item.has_crack 
            ? '<span class="px-1.5 py-0.5 bg-amber-500/20 text-amber-300 border border-amber-500/40 text-[9px] rounded font-bold">⚠️ CRACK DETECTED</span>'
            : '';

          card.innerHTML = `
            <div class="px-3 py-2 bg-slate-800/80 border-b border-slate-800 flex justify-between items-center text-[11px]">
              <span class="font-bold text-slate-300 truncate max-w-[160px]" title="${item.stem}">#${idx+1} ${item.stem.slice(0, 15)}...</span>
              ${crackBadge}
            </div>
            <div class="p-2 bg-black/60 flex items-center justify-center">
              <img src="${item.prediction_overlay}" class="max-h-[220px] w-auto object-contain rounded" alt="Batch sample">
            </div>
            <div class="p-3 bg-slate-900 border-t border-slate-800 text-[11px] space-y-1">
              <div class="flex justify-between text-slate-400">
                <span>Detections:</span>
                <span class="font-bold text-slate-200">${item.instances.length}</span>
              </div>
              <div class="flex justify-between text-slate-400">
                <span>Classes:</span>
                <span class="font-mono text-slate-300">${[...new Set(item.instances.map(i => i.class_name))].join(', ') || 'None'}</span>
              </div>
            </div>
          `;
          grid.appendChild(card);
        });
      } catch (err) {
        grid.innerHTML = `<div class="col-span-full py-12 text-center text-red-400">Batch inference error: ${err.message}</div>`;
      }
    }

    // Drag and drop handlers
    const dropZone = document.getElementById('drop-zone');
    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500', 'bg-slate-800'); }, false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => { e.preventDefault(); dropZone.classList.remove('border-blue-500', 'bg-slate-800'); }, false);
    });
    dropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        document.getElementById('file-input').files = files;
        handleFileUpload({ target: { files: files } });
      }
    });

    window.addEventListener('DOMContentLoaded', init);
  </script>

  <!-- Modal: Preset Confirmation when 0 traced points (Fix 1) -->
  <div id="preset-confirm-modal" class="hidden fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-slate-900 border-2 border-amber-500/80 rounded-2xl max-w-md w-full p-6 shadow-2xl shadow-amber-950/60 space-y-4">
      <div class="flex items-start space-x-3">
        <span class="text-3xl">⚠️</span>
        <div>
          <h4 class="text-sm font-bold text-amber-300 uppercase tracking-wide">Notice: Generic Demo Preset</h4>
          <p class="text-xs text-slate-300 mt-1 leading-relaxed">
            No points have been traced on the photo. You are about to carve a 3D tooth using a <strong class="text-amber-200">synthetic preset path</strong>.
          </p>
        </div>
      </div>
      <div class="bg-amber-950/40 border border-amber-500/40 rounded-lg p-3 text-xs text-slate-300 space-y-1">
        <p class="font-semibold text-amber-200">Selected Path Source:</p>
        <p id="modal-preset-name" class="font-mono text-amber-300">Occlusal Fissure Structural Crack (Molars)</p>
        <p class="text-[11px] text-amber-200/80 mt-1">This geometry is synthetic and does <strong class="text-red-400 font-semibold">NOT</strong> reflect the visible crack fracture path from your uploaded photo.</p>
      </div>
      <div class="flex items-center justify-end space-x-3 pt-2">
        <button type="button" onclick="closePresetConfirmModal(false)" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-semibold border border-slate-700">
          Cancel & Trace Photo
        </button>
        <button type="button" id="modal-proceed-btn" onclick="closePresetConfirmModal(true)" class="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-lg text-xs font-bold shadow-md shadow-amber-950/40">
          Proceed with Preset Path
        </button>
      </div>
    </div>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def get_dashboard_html():
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8055)
