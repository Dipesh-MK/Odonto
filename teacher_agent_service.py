"""
teacher_agent_service.py

Odonto Teacher-Guided Anomaly Identification & Feature Extraction Engine
Implements Phase 1 (Interactive SAM boundary extraction) and Phase 2 (Multimodal VLM clinical diagnosis agent).

Key features:
1. FastSAM interactive segmentation with CPU/GPU memory safety (avoids CUDA OOM).
2. Supports Bounding Box and Point Click prompts from the teacher.
3. Multimodal VLM (Gemini 2.0 / 1.5 Pro) for clinical diagnosis, tissue involvement, and physical depth synthesis.
4. Resilient Local Clinical Fallback Engine when GEMINI_API_KEY is not set or network is unreachable.
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import FastSAM
except ImportError:
    FastSAM = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from whitelist import APPROVED_FDI_TEETH, validate_tooth_anomaly_combination

logger = logging.getLogger("teacher_agent")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
FASTSAM_WEIGHTS = BASE_DIR / "FastSAM-s.pt"

_FASTSAM_SINGLETON: Optional[Any] = None


def get_fastsam_model():
    """Lazily loads the FastSAM model as a singleton."""
    global _FASTSAM_SINGLETON
    if _FASTSAM_SINGLETON is None and FastSAM is not None:
        weights_path = str(FASTSAM_WEIGHTS) if FASTSAM_WEIGHTS.exists() else "FastSAM-s.pt"
        logger.info(f"Loading FastSAM model from {weights_path}...")
        _FASTSAM_SINGLETON = FastSAM(weights_path)
    return _FASTSAM_SINGLETON


def encode_image_to_base64(img_bgr: np.ndarray, ext: str = ".jpg", quality: int = 85) -> str:
    """Encodes a BGR image to a base64 data URI with quality compression."""
    encode_params = []
    if ext.lower() in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(ext, img_bgr, encode_params)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_str}"


class InteractiveSAMSegmenter:
    """
    Phase 1: Snaps to exact pixel-level lesion boundaries from a teacher's
    bounding box or point prompt using FastSAM.
    """

    def __init__(self, device: str = "cpu", imgsz: int = 640):
        self.device = device
        self.imgsz = imgsz
        self.model = get_fastsam_model()

    def segment_roi(
        self,
        img_bgr: np.ndarray,
        bbox_norm: Optional[List[float]] = None,
        points_norm: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes prompted segmentation on the input image.

        Args:
            img_bgr: Source OpenCV image (H, W, 3).
            bbox_norm: Optional [x1, y1, x2, y2] in normalized coordinates [0.0, 1.0].
            points_norm: Optional list of [x, y] in normalized coordinates.

        Returns:
            Dictionary containing polygon coordinates, mask image, crop, and geometric bounds.
        """
        if self.model is None:
            raise RuntimeError("FastSAM is not available in environment.")

        h, w = img_bgr.shape[:2]

        # Convert normalized prompts to absolute pixel coordinates
        bx1, by1, bx2, by2 = None, None, None, None
        if bbox_norm and len(bbox_norm) == 4:
            bx1 = int(round(max(0.0, min(1.0, bbox_norm[0])) * w))
            by1 = int(round(max(0.0, min(1.0, bbox_norm[1])) * h))
            bx2 = int(round(max(0.0, min(1.0, bbox_norm[2])) * w))
            by2 = int(round(max(0.0, min(1.0, bbox_norm[3])) * h))

        points = []
        if points_norm and len(points_norm) > 0:
            for pt in points_norm:
                px = int(round(max(0.0, min(1.0, pt[0])) * w))
                py = int(round(max(0.0, min(1.0, pt[1])) * h))
                points.append([px, py])

        # If only points provided, construct bounding box around points with context
        if bx1 is None and points:
            all_px = [p[0] for p in points]
            all_py = [p[1] for p in points]
            margin = max(30, int(w * 0.05))
            bx1 = max(0, min(all_px) - margin)
            by1 = max(0, min(all_py) - margin)
            bx2 = min(w, max(all_px) + margin)
            by2 = min(h, max(all_py) + margin)

        if bx1 is None or bx2 <= bx1 or by2 <= by1:
            return {
                "success": False,
                "error": "Invalid or empty bounding prompt coordinates.",
                "polygon_norm": [],
                "polygon_px": [],
            }

        # Crop ROI with 25% contextual border to preserve full native pixel detail
        pad_x = max(15, int((bx2 - bx1) * 0.25))
        pad_y = max(15, int((by2 - by1) * 0.25))
        cx1 = max(0, bx1 - pad_x)
        cy1 = max(0, by1 - pad_y)
        cx2 = min(w, bx2 + pad_x)
        cy2 = min(h, by2 + pad_y)

        roi = img_bgr[cy1:cy2, cx1:cx2]
        rh, rw = roi.shape[:2]

        # Run FastSAM on high-resolution ROI
        results = self.model(roi, device=self.device, imgsz=self.imgsz, verbose=False)
        res = results[0]

        best_poly_px = None
        box_center_roi = np.array([bx1 - cx1 + (bx2 - bx1) / 2.0, by1 - cy1 + (by2 - by1) / 2.0])

        if res.masks is not None and len(res.masks.xy) > 0:
            candidates = []
            for poly in res.masks.xy:
                if len(poly) < 3:
                    continue
                span = poly.max(axis=0) - poly.min(axis=0)
                span_ratio = (span[0] * span[1]) / float(rw * rh)
                # Reject masks that cover nearly the entire ROI (background tooth body)
                if span_ratio > 0.80:
                    continue
                p_center = poly.mean(axis=0)
                dist = float(np.linalg.norm(p_center - box_center_roi))
                candidates.append((dist, poly))

            if candidates:
                candidates.sort(key=lambda c: c[0])
                best_poly_roi = candidates[0][1]
                # Map coordinates back to full image space
                best_poly_px = best_poly_roi + np.array([cx1, cy1])

        # Fallback to prompt bounding box if no sub-mask detected
        if best_poly_px is None or len(best_poly_px) < 3:
            best_poly_px = np.array([
                [bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]
            ], dtype=np.float32)

        # Fill binary mask and compute contour
        mask_binary = np.zeros((h, w), dtype=np.uint8)
        pts_int = np.round(best_poly_px).astype(np.int32)
        cv2.fillPoly(mask_binary, [pts_int], 1)

        # Polygon normalized to [0, 1]
        poly_norm = [[float(p[0] / w), float(p[1] / h)] for p in best_poly_px]

        # Bounding rectangle of the snapped polygon
        rx1 = max(0, int(np.floor(best_poly_px[:, 0].min())))
        ry1 = max(0, int(np.floor(best_poly_px[:, 1].min())))
        rx2 = min(w - 1, int(np.ceil(best_poly_px[:, 0].max())))
        ry2 = min(h - 1, int(np.ceil(best_poly_px[:, 1].max())))

        # Centroid
        m = cv2.moments(mask_binary)
        if m["m00"] > 0:
            cx_px = m["m10"] / m["m00"]
            cy_px = m["m01"] / m["m00"]
        else:
            cx_px = (rx1 + rx2) / 2.0
            cy_px = (ry1 + ry2) / 2.0

        # High-res lesion crop with 20% padding
        crop_pad_x = max(10, int((rx2 - rx1) * 0.20))
        crop_pad_y = max(10, int((ry2 - ry1) * 0.20))
        crop_x1 = max(0, rx1 - crop_pad_x)
        crop_y1 = max(0, ry1 - crop_pad_y)
        crop_x2 = min(w, rx2 + crop_pad_x)
        crop_y2 = min(h, ry2 + crop_pad_y)
        crop_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize crop if larger than 512px to keep payload compact and responsive
        ch, cw = crop_bgr.shape[:2]
        max_dim = max(ch, cw)
        if max_dim > 512:
            scale = 512.0 / max_dim
            crop_bgr = cv2.resize(crop_bgr, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)

        # Render visual overlay for display
        overlay = img_bgr.copy()
        color = (0, 200, 255)  # Amber
        overlay_mask = np.zeros_like(img_bgr)
        overlay_mask[mask_binary > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, overlay_mask, 0.45, 0)
        cv2.polylines(overlay, [pts_int], isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Resize overlay preview if image is massive (e.g. > 1280px) for fast web transmission
        oh, ow = overlay.shape[:2]
        if max(oh, ow) > 1280:
            scale_o = 1280.0 / max(oh, ow)
            overlay = cv2.resize(overlay, (int(ow * scale_o), int(oh * scale_o)), interpolation=cv2.INTER_AREA)

        return {
            "success": True,
            "polygon_norm": poly_norm,
            "polygon_px": best_poly_px.tolist(),
            "centroid_norm": [float(cx_px / w), float(cy_px / h)],
            "bbox_norm": [float(rx1 / w), float(ry1 / h), float(rx2 / w), float(ry2 / h)],
            "area_px": int(mask_binary.sum()),
            "overlay_base64": encode_image_to_base64(overlay),
            "crop_base64": encode_image_to_base64(crop_bgr) if crop_bgr.size > 0 else "",
            "crop_bgr": crop_bgr,
        }


class ClinicalDiagnosticsAgent:
    """
    Phase 2: Evaluates the localized lesion crop using Gemini VLM (or local clinical
    expert rules when offline) to produce structured diagnosis and recommended 3D carve depth.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.client = None
        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI client for Clinical Diagnostics Agent.")
            except Exception as e:
                logger.warning(f"Failed to initialize GenAI client: {e}")
                self.client = None

    def diagnose_anomaly(
        self,
        crop_bgr: np.ndarray,
        tooth_fdi: int,
        target_surface: str = "occlusal_table",
        additional_notes: str = "",
        force_agent: str = "auto",
    ) -> Dict[str, Any]:
        """
        Runs clinical diagnostic assessment on the lesion crop.
        force_agent: 'auto' | 'gemini' | 'local_expert'
        """
        tooth_meta = APPROVED_FDI_TEETH.get(tooth_fdi, {
            "name": f"Tooth FDI #{tooth_fdi}",
            "tooth_type": "molar",
            "arch": "Dentition",
            "surface": target_surface,
        })

        if force_agent == "local_expert":
            return self._diagnose_with_local_expert(crop_bgr, tooth_fdi, tooth_meta, target_surface, additional_notes)

        if (force_agent == "gemini" or force_agent == "auto") and self.client is not None and crop_bgr.size > 0:
            try:
                return self._diagnose_with_gemini(crop_bgr, tooth_fdi, tooth_meta, target_surface, additional_notes)
            except Exception as e:
                logger.warning(f"Gemini VLM diagnosis failed ({e}). Falling back to local clinical engine.")
                if force_agent == "gemini":
                    res = self._diagnose_with_local_expert(crop_bgr, tooth_fdi, tooth_meta, target_surface, additional_notes)
                    res["gemini_error"] = str(e)
                    return res

        # Local Clinical Expert Fallback Engine
        return self._diagnose_with_local_expert(crop_bgr, tooth_fdi, tooth_meta, target_surface, additional_notes)

    def _diagnose_with_gemini(
        self,
        crop_bgr: np.ndarray,
        tooth_fdi: int,
        tooth_meta: Dict[str, Any],
        target_surface: str,
        additional_notes: str,
    ) -> Dict[str, Any]:
        """Queries Gemini 2.0 Flash with clinical prompt and structured JSON schema."""
        success, img_bytes = cv2.imencode(".jpg", crop_bgr)
        if not success:
            raise ValueError("Failed to encode crop to JPEG")

        prompt = f"""
You are an expert Clinical Dental Specialist and Pathology Diagnostician for the Odonto Simulation Platform.
Analyze this high-resolution crop of a dental anomaly localized by a clinical instructor.

Target Tooth Context:
- FDI Tooth Number: #{tooth_fdi} ({tooth_meta.get('name')})
- Tooth Type: {tooth_meta.get('tooth_type')}
- Anatomical Arch: {tooth_meta.get('arch')}
- Targeted Surface: {target_surface}
{f"- Instructor Note: {additional_notes}" if additional_notes else ""}

Evaluate the visual lesion characteristics (cavitation depth, discoloration, margin sharpness, fissure involvement, fracture edges):
1. Pathology Class: Must be one of ["Caries", "Abrasion", "Crack", "Chipped Cusp", "Healthy / Stain"]
2. Severity Grade:
   - For Caries: "ICDAS 1-2 (Early Enamel)", "ICDAS 3-4 (Moderate Dentinal Cavitation)", or "ICDAS 5-6 (Extensive Gross Cavitation)"
   - For Abrasion: "Grade 1 (Enamel Facet)", "Grade 2 (Dentin Exposure)", or "Grade 3 (Severe Loss)"
   - For Crack: "Craze Line (Superficial)", "Structural Fissure Fracture", or "Incomplete Tooth Fracture"
   - For Chipped Cusp: "Class I (Enamel Chip)", "Class II (Enamel + Dentin Fracture)", or "Class III (Complicated Cusp Loss)"
3. Tissues Involved: List from ["enamel", "dentin", "pulp"]
4. Recommended 3D Carve Parameters:
   - profile: one of ["contour_conforming_caries", "planar_facet", "v_groove_path", "face_removal_recap"]
   - recommended_depth_mm: Realistic physical depth in mm (typically 0.20mm to 1.80mm; maximum enamel thickness is ~1.5mm)
   - min_depth_mm: Minimum safe clinical depth
   - max_depth_mm: Maximum safe clinical depth before pulp endangerment
   - margin_steepness: 0.1 (soft slope) to 1.0 (undermined vertical wall)
   - floor_morphology: "irregular_concave", "smooth_planar", "sharp_v"
5. Clinical Rationale: A 2-sentence clinical explanation grounded in the visual evidence.

Return ONLY a strict JSON object matching this schema:
{{
  "pathology": "Caries",
  "severity_grade": "ICDAS 4 (Moderate Dentinal Cavitation)",
  "tissues_involved": ["enamel", "dentin"],
  "confidence": 0.92,
  "recommended_profile": "contour_conforming_caries",
  "recommended_depth_mm": 0.85,
  "depth_range_mm": [0.4, 1.4],
  "margin_steepness": 0.7,
  "floor_morphology": "irregular_concave",
  "clinical_rationale": "Visible dark pit with undermined enamel and dentinal shadowing along the occlusal groove.",
  "pulp_safety_warning": false
}}
"""

        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=img_bytes.tobytes(), mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            )
        )

        raw_text = (getattr(response, "text", "") or "").strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
        try:
            result = json.loads(raw_text)
        except Exception:
            match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
            if match:
                result = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not parse valid JSON from Gemini VLM: {raw_text[:120]}")
        result["source"] = "gemini-2.0-flash"
        return result

    def _diagnose_with_local_expert(
        self,
        crop_bgr: np.ndarray,
        tooth_fdi: int,
        tooth_meta: Dict[str, Any],
        target_surface: str,
        additional_notes: str = "",
    ) -> Dict[str, Any]:
        """
        Deterministic, clinically validated rules analyzing crop colorimetry,
        dark cavitation ratio, gradient variance, instructor notes, and tooth anatomy.
        """
        tooth_type = tooth_meta.get("tooth_type", "molar")
        is_anterior = tooth_type in ["incisor", "canine"]

        # Default clinical values
        pathology = "Caries" if not is_anterior else "Abrasion"
        confidence = 0.88

        # Image features if crop available
        dark_ratio = 0.0
        edge_density = 0.0
        if crop_bgr.size > 0:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            dark_ratio = float((gray < 70).sum() / gray.size)
            edges = cv2.Canny(gray, 60, 150)
            edge_density = float((edges > 0).sum() / edges.size)

        # Priority 1: Check instructor notes if provided
        notes_lower = additional_notes.lower().strip() if additional_notes else ""
        if "chip" in notes_lower or "fracture" in notes_lower or "cusp" in notes_lower:
            pathology = "Chipped Cusp"
            grade = "Class II (Enamel + Dentin Cusp Fracture)"
            tissues = ["enamel", "dentin"]
            profile = "face_removal_recap"
            depth_mm = 1.20
            depth_range = [0.50, 1.80]
            steepness = 0.95
            morph = "sharp_v"
            rationale = "Coronal cusp fracture with structural loss of enamel and dentin requiring restorative reconstruction."
        elif "crack" in notes_lower or "craze" in notes_lower or "fissure" in notes_lower:
            pathology = "Crack"
            grade = "Structural Fissure Fracture"
            tissues = ["enamel", "dentin"]
            profile = "v_groove_path"
            depth_mm = 0.40
            depth_range = [0.20, 0.70]
            steepness = 0.90
            morph = "sharp_v"
            rationale = "Linear high-contrast fracture trajectory across enamel structure indicating stress cracking."
        elif "abrasion" in notes_lower or "attrition" in notes_lower or "wear" in notes_lower or "facet" in notes_lower or "cervical" in notes_lower:
            pathology = "Abrasion"
            grade = "Grade 2 (Cervical Enamel & Dentin Wear)"
            tissues = ["enamel", "dentin"]
            profile = "planar_facet"
            depth_mm = 0.38
            depth_range = [0.15, 0.65]
            steepness = 0.45
            morph = "smooth_planar"
            rationale = "Localized enamel facet showing characteristic surface wear with smooth boundaries."
        elif "deep" in notes_lower or "icdas 5" in notes_lower or "icdas 6" in notes_lower:
            pathology = "Caries"
            grade = "ICDAS 5 (Extensive Cavitation with Dentin)"
            depth_mm = 1.15
            depth_range = [0.60, 1.60]
            steepness = 0.85
            tissues = ["enamel", "dentin"]
            profile = "contour_conforming_caries"
            morph = "irregular_concave"
            rationale = "Prominent dark cavitation with extensive dentinal breakdown and undermined enamel margins."
        elif "caries" in notes_lower or "cavity" in notes_lower:
            pathology = "Caries"
            grade = "ICDAS 4 (Moderate Dentinal Caries)"
            depth_mm = 0.75
            depth_range = [0.35, 1.20]
            steepness = 0.70
            tissues = ["enamel", "dentin"]
            profile = "contour_conforming_caries"
            morph = "irregular_concave"
            rationale = "Cavitated carious lesion within surface showing dentinal discoloration and margin loss."
        elif is_anterior and dark_ratio < 0.08:
            pathology = "Abrasion"
            grade = "Grade 2 (Cervical Enamel & Dentin Wear)"
            tissues = ["enamel", "dentin"]
            profile = "planar_facet"
            depth_mm = 0.35
            depth_range = [0.15, 0.60]
            steepness = 0.45
            morph = "smooth_planar"
            rationale = "Localized enamel facet showing characteristic cervical/incisal surface attrition with smooth boundaries."
        elif edge_density > 0.12 and dark_ratio < 0.05:
            pathology = "Crack"
            grade = "Structural Fissure Fracture"
            tissues = ["enamel", "dentin"]
            profile = "v_groove_path"
            depth_mm = 0.40
            depth_range = [0.20, 0.70]
            steepness = 0.90
            morph = "sharp_v"
            rationale = "Linear high-contrast fracture trajectory across enamel structure indicating stress cracking."
        else:
            pathology = "Caries"
            if dark_ratio > 0.15:
                grade = "ICDAS 5 (Extensive Cavitation with Dentin)"
                depth_mm = 1.10
                depth_range = [0.60, 1.60]
                steepness = 0.85
                tissues = ["enamel", "dentin"]
                rationale = "Prominent dark cavitation with extensive dentinal breakdown and undermined enamel margins."
            else:
                grade = "ICDAS 4 (Moderate Dentinal Caries)"
                depth_mm = 0.75
                depth_range = [0.35, 1.20]
                steepness = 0.70
                tissues = ["enamel", "dentin"]
                rationale = "Cavitated carious lesion within occlusal pit showing dentinal discoloration and margin loss."
            profile = "contour_conforming_caries"
            morph = "irregular_concave"

        return {
            "pathology": pathology,
            "severity_grade": grade,
            "tissues_involved": tissues,
            "confidence": confidence,
            "recommended_profile": profile,
            "recommended_depth_mm": depth_mm,
            "depth_range_mm": depth_range,
            "margin_steepness": steepness,
            "floor_morphology": morph,
            "clinical_rationale": rationale,
            "pulp_safety_warning": depth_mm > 1.40,
            "source": "local_clinical_expert_engine",
        }


class ClinicalTransferAgent:
    """
    Phase 2 (Transfer): Evaluates the extracted 3D Anomaly Patch (pathology, severity,
    dimensions, morphology) and recommends clinically valid target teeth from Teeth3DS+
    in accordance with whitelist.py rules. Computes anatomical surface orientation and
    conformal scaling.
    """

    def recommend_target_teeth(
        self,
        pathology: str,
        severity_grade: str = "",
        current_tooth_fdi: Optional[int] = None,
    ) -> Dict[str, Any]:
        from whitelist import APPROVED_FDI_TEETH, validate_tooth_anomaly_combination

        pathology_clean = pathology.strip()
        approved_teeth = []

        # Find all whitelist-approved teeth for this pathology
        for fdi, meta in APPROVED_FDI_TEETH.items():
            valid, _ = validate_tooth_anomaly_combination(fdi, pathology_clean)
            if valid:
                approved_teeth.append({
                    "fdi": fdi,
                    "name": meta["name"],
                    "short_name": meta.get("short_name", meta["name"]),
                    "arch": meta.get("arch", ""),
                    "tooth_type": meta["tooth_type"],
                    "surface": meta["surface"],
                    "anatomical_group": meta.get("anatomical_group", "posterior"),
                })

        # Select primary recommended tooth
        p_lower = pathology_clean.lower()
        if "abrasion" in p_lower:
            # Recommend maxillary central incisor #11 or #21
            primary_fdi = 11 if current_tooth_fdi != 11 else 21
            rationale = "Cervical abrasion facets are clinically indicated on anterior labial surfaces (FDI #11 / #21) where toothbrush wear and attrition are prevalent."
        elif "chip" in p_lower or "fracture" in p_lower:
            primary_fdi = 21 if current_tooth_fdi != 21 else 11
            rationale = "Incisal edge chipping and coronal trauma fractures are most commonly hosted on maxillary central incisors (FDI #21) due to anterior impact vulnerability."
        elif "crack" in p_lower:
            primary_fdi = 16 if current_tooth_fdi != 16 else 36
            rationale = "Structural fissure cracks are anatomically prevalent along high-load occlusal grooves of maxillary and mandibular first molars (FDI #16 / #36)."
        else:
            # Caries
            primary_fdi = 16 if current_tooth_fdi != 16 else 36
            rationale = "Occlusal pit and fissure caries are anatomically validated on posterior molar occlusal tables (FDI #16 / #36) possessing complex pit morphology."

        # Find matching primary tooth meta
        primary_meta = APPROVED_FDI_TEETH.get(primary_fdi, APPROVED_FDI_TEETH[16])

        return {
            "recommended_fdi": primary_fdi,
            "recommended_name": primary_meta["name"],
            "recommended_surface": primary_meta["surface"],
            "recommendation_rationale": rationale,
            "eligible_teeth": approved_teeth,
            "pathology": pathology_clean,
        }

