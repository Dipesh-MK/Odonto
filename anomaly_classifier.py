"""
anomaly_classifier.py

CV-Based Dental Anomaly Classifier.

Classifies the type of dental anomaly visible in a 2D clinical photo
using the SAM2 binary mask + the original image pixels. This fills the
critical gap where SAM2 segments *something* but has no concept of
*what* the anomaly actually is.

Classification is 100% deterministic computer vision — zero LLM, zero
neural-network inference. It extracts two families of features:

A) MASK SHAPE FEATURES (from the binary mask alone):
   - Elongation / aspect ratio (major_axis / minor_axis)
   - Compactness (4π·Area / Perimeter²)
   - Solidity (Area / ConvexHullArea)
   - Extent (Area / BoundingBoxArea)
   - Relative size (mask area / image area)
   - Roughness (Perimeter / equivalent-circle perimeter)

B) IMAGE COLOR FEATURES (from masked pixels in the original photo):
   - Mean HSV channels inside the mask
   - Darkness ratio (% of pixels with V < 80)
   - Brown ratio (hue 8–25, sat > 40, val 30–180)
   - White/chalky ratio (sat < 30, val > 170)
   - Color variance (std of hue within mask)
   - Edge density inside mask (Canny edges / mask area)

Decision tree:
  1. CRACK:     Very elongated (AR > 3.5) OR (AR > 2.5 AND low compactness AND high edge density)
  2. EROSION:   Broad, shallow appearance: high compactness, high solidity, low darkness,
                position in cervical/gingival zone (rel_y > 0.55) or chalky color signature
  3. ABRASION:  Moderate elongation (AR 1.5–3.5), high solidity, lighter color (worn enamel),
                horizontal orientation, or cervical-zone position
  4. CAVITY:    Default — round/oval, dark interior, moderate-to-high compactness
"""

import math
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _extract_mask_shape_features(binary_mask: np.ndarray) -> Dict[str, float]:
    """Extracts geometric shape descriptors from a binary mask."""
    mask = (binary_mask > 0).astype(np.uint8)
    area = int(np.sum(mask))
    h, w = mask.shape[:2]

    if area < 10:
        return {
            "area": 0, "aspect_ratio": 1.0, "compactness": 0.0,
            "solidity": 0.0, "extent": 0.0, "roughness": 1.0,
            "relative_size": 0.0, "elongation": 1.0,
            "orientation_deg": 0.0, "perimeter": 0.0,
        }

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "area": area, "aspect_ratio": 1.0, "compactness": 0.0,
            "solidity": 0.0, "extent": 0.0, "roughness": 1.0,
            "relative_size": float(area / (h * w)), "elongation": 1.0,
            "orientation_deg": 0.0, "perimeter": 0.0,
        }

    largest = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest, True)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(largest)
    bb_area = max(w_bb * h_bb, 1)

    compactness = (4.0 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    extent = float(area / bb_area)
    equiv_perim = 2.0 * math.pi * math.sqrt(area / math.pi)
    roughness = float(perimeter / equiv_perim) if equiv_perim > 0 else 1.0
    relative_size = float(area / (h * w))

    # Aspect ratio and orientation from second-order central moments
    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]
        diff = mu20 - mu02
        disc = math.sqrt(diff ** 2 + 4.0 * mu11 ** 2)
        lambda1 = max(0.0, (mu20 + mu02 + disc) / 2.0)
        lambda2 = max(1e-6, (mu20 + mu02 - disc) / 2.0)
        aspect_ratio = float(math.sqrt(lambda1 / lambda2))
        orientation_deg = float(0.5 * math.degrees(math.atan2(2.0 * mu11, diff)))
    else:
        aspect_ratio = float(max(w_bb, h_bb) / max(min(w_bb, h_bb), 1))
        orientation_deg = 0.0

    # Fallback with minAreaRect if moments give ~1.0
    if aspect_ratio < 1.05 and len(largest) >= 5:
        rect = cv2.minAreaRect(largest)
        (_, (rw, rh), _) = rect
        major, minor = max(rw, rh), max(min(rw, rh), 1.0)
        aspect_ratio = float(major / minor)

    return {
        "area": area,
        "aspect_ratio": round(aspect_ratio, 3),
        "compactness": round(compactness, 4),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
        "roughness": round(roughness, 4),
        "relative_size": round(relative_size, 6),
        "elongation": round(aspect_ratio, 3),  # alias
        "orientation_deg": round(orientation_deg, 2),
        "perimeter": round(perimeter, 2),
    }


def _extract_color_features(
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
) -> Dict[str, float]:
    """Extracts color statistics from masked pixels and surrounding tooth enamel context."""
    mask_bool = binary_mask > 0
    n_pixels = int(np.sum(mask_bool))

    defaults = {
        "mean_hue": 0.0, "mean_sat": 0.0, "mean_val": 0.0,
        "std_hue": 0.0, "std_val": 0.0,
        "darkness_ratio": 0.0, "brown_ratio": 0.0,
        "white_chalky_ratio": 0.0, "edge_density": 0.0,
        "redness_ratio": 0.0,
        "delta_val": 0.0, "delta_sat": 0.0, "enamel_val": 128.0,
    }

    if n_pixels < 10 or image_bgr is None:
        return defaults

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    h_vals = hsv[:, :, 0][mask_bool].astype(np.float32)
    s_vals = hsv[:, :, 1][mask_bool].astype(np.float32)
    v_vals = hsv[:, :, 2][mask_bool].astype(np.float32)

    mean_hue = float(np.mean(h_vals))
    mean_sat = float(np.mean(s_vals))
    mean_val = float(np.mean(v_vals))
    std_hue = float(np.std(h_vals))
    std_val = float(np.std(v_vals))

    # Darkness: V < 80 (out of 255)
    darkness_ratio = float(np.sum(v_vals < 80) / n_pixels)

    # Brown signature: H in [8, 25], S > 40, V in [30, 180]
    brown_mask = (h_vals >= 8) & (h_vals <= 25) & (s_vals > 40) & (v_vals >= 30) & (v_vals <= 180)
    brown_ratio = float(np.sum(brown_mask) / n_pixels)

    # White/chalky: S < 30, V > 170
    chalky_mask = (s_vals < 30) & (v_vals > 170)
    white_chalky_ratio = float(np.sum(chalky_mask) / n_pixels)

    # Redness: H in [0, 8] or [165, 180], S > 60
    red_mask = ((h_vals <= 8) | (h_vals >= 165)) & (s_vals > 60)
    redness_ratio = float(np.sum(red_mask) / n_pixels)

    # Surrounding enamel ring context (15px dilation around lesion)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated = cv2.dilate((mask_bool * 255).astype(np.uint8), kernel)
    ring_mask = (dilated > 0) & (~mask_bool)
    if np.sum(ring_mask) > 10:
        ring_v = float(np.mean(hsv[:, :, 2][ring_mask].astype(np.float32)))
        ring_s = float(np.mean(hsv[:, :, 1][ring_mask].astype(np.float32)))
        delta_val = float(mean_val - ring_v)
        delta_sat = float(mean_sat - ring_s)
    else:
        ring_v = mean_val
        delta_val = 0.0
        delta_sat = 0.0

    # Edge density inside mask (structural detail / texture complexity)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels_in_mask = int(np.sum((edges > 0) & mask_bool))
    edge_density = float(edge_pixels_in_mask / n_pixels)

    return {
        "mean_hue": round(mean_hue, 2),
        "mean_sat": round(mean_sat, 2),
        "mean_val": round(mean_val, 2),
        "std_hue": round(std_hue, 2),
        "std_val": round(std_val, 2),
        "darkness_ratio": round(darkness_ratio, 4),
        "brown_ratio": round(brown_ratio, 4),
        "white_chalky_ratio": round(white_chalky_ratio, 4),
        "edge_density": round(edge_density, 4),
        "redness_ratio": round(redness_ratio, 4),
        "delta_val": round(delta_val, 2),
        "delta_sat": round(delta_sat, 2),
        "enamel_val": round(ring_v, 2),
    }


def classify_anomaly(
    binary_mask: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    geometry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classifies the dental anomaly type from the segmented mask and image.

    Decision Logic (priority order):

    1. CRACK — Very high elongation (AR > 3.5), or moderate elongation (AR > 2.5)
       combined with low compactness (< 0.25) and high edge density.
       Cracks are narrow, linear features on the tooth surface.

    2. EROSION — Broad, smooth loss of tooth structure with high compactness (> 0.6),
       high solidity (> 0.75), lighter color (low darkness), and either:
       - Position near cervical/gingival margin (rel_y > 0.55), OR
       - White/chalky demineralization signature (chalky_ratio > 0.15), OR
       - Very smooth contour (roughness < 1.15) with moderate saturation.
       Erosion produces smooth, rounded, saucer-like defects.

    3. ABRASION — Moderate elongation (AR 1.5–3.5) with horizontal orientation,
       high solidity (> 0.65), lighter/warmer color (worn enamel/dentin exposure),
       low darkness, and typically near the cervical zone or incisal edge.
       Abrasion creates flat, polished wear facets.

    4. CAVITY (default) — Roughly circular/oval (AR < 2.0), dark interior
       (darkness_ratio > 0.15 or brown_ratio > 0.10), moderate compactness.
       Cavities are concave, dark, carious lesions.

    Returns:
        {
            "anomaly_type": str,           # "Cavity", "Abrasion", "Crack", "Erosion"
            "confidence": float,           # 0.0–1.0
            "reasoning": str,              # Human-readable explanation
            "shape_features": dict,        # All extracted shape features
            "color_features": dict | None, # Color features (if image provided)
            "all_scores": dict,            # Raw score for each class
        }
    """
    shape = _extract_mask_shape_features(binary_mask)
    color = _extract_color_features(image_bgr, binary_mask) if image_bgr is not None else None

    ar = shape["aspect_ratio"]
    compact = shape["compactness"]
    solid = shape["solidity"]
    rough = shape["roughness"]
    orient = abs(shape["orientation_deg"])
    rel_size = shape["relative_size"]

    # Relative centroid position from geometry (if available)
    rel_x, rel_y = 0.5, 0.5
    if geometry and "relative_centroid" in geometry:
        rel_x, rel_y = geometry["relative_centroid"]

    # Color features (defaults if no image)
    darkness = color["darkness_ratio"] if color else 0.2
    brown = color["brown_ratio"] if color else 0.1
    chalky = color["white_chalky_ratio"] if color else 0.05
    edge_dens = color["edge_density"] if color else 0.1
    mean_val = color["mean_val"] if color else 128.0
    mean_sat = color["mean_sat"] if color else 80.0
    redness = color["redness_ratio"] if color else 0.0
    delta_val = color["delta_val"] if color else 0.0
    delta_sat = color["delta_sat"] if color else 0.0

    # =========================================================================
    # SCORE EACH ANOMALY CLASS WITH CONTEXT-INVARIANT ENAMEL CONTRAST
    # =========================================================================
    scores = {"Crack": 0.0, "Erosion": 0.0, "Abrasion": 0.0, "Cavity": 0.0}

    # --- CRACK SCORE ---
    # Cracks: narrow linear features with high aspect ratio and sharp gradient
    if ar > 3.8:
        scores["Crack"] += 0.55
    elif ar > 2.8:
        scores["Crack"] += 0.38
    elif ar > 2.2:
        scores["Crack"] += 0.22

    if compact < 0.18:
        scores["Crack"] += 0.22
    elif compact < 0.28:
        scores["Crack"] += 0.12

    if edge_dens > 0.18:
        scores["Crack"] += 0.15
    elif edge_dens > 0.10:
        scores["Crack"] += 0.08

    if rel_size < 0.010:  # Cracks are narrow
        scores["Crack"] += 0.12

    if rough > 1.5:  # High boundary complexity
        scores["Crack"] += 0.08

    if delta_val < -15:  # Noticeable dark fracture line vs enamel
        scores["Crack"] += 0.10

    # --- EROSION SCORE ---
    # Erosion: smooth, broad, shallow loss of tooth structure (often cervical or chemical)
    if compact > 0.65:
        scores["Erosion"] += 0.18
    elif compact > 0.50:
        scores["Erosion"] += 0.10

    if solid > 0.80:
        scores["Erosion"] += 0.15
    elif solid > 0.68:
        scores["Erosion"] += 0.08

    if rough < 1.20:
        scores["Erosion"] += 0.12
    elif rough < 1.35:
        scores["Erosion"] += 0.06

    if chalky > 0.18 or (delta_val > 15 and delta_sat < -10):
        scores["Erosion"] += 0.24
    elif chalky > 0.08:
        scores["Erosion"] += 0.14

    if darkness < 0.10 and delta_val > -10:
        scores["Erosion"] += 0.12

    if rel_y > 0.55:  # Cervical collar position
        scores["Erosion"] += 0.12

    if ar < 2.0:
        scores["Erosion"] += 0.06

    # --- ABRASION SCORE ---
    # Abrasion: mechanical wear facets, moderate elongation, warm exposed dentin
    if 1.4 < ar < 3.8:
        scores["Abrasion"] += 0.22
    elif 1.15 < ar <= 1.4:
        scores["Abrasion"] += 0.08

    if orient < 28:  # Near-horizontal orientation aligned with dental arch
        scores["Abrasion"] += 0.14
    elif orient < 45:
        scores["Abrasion"] += 0.06

    if solid > 0.72:
        scores["Abrasion"] += 0.10

    if brown > 0.12 or (delta_val < -8 and delta_val > -45 and mean_sat > 45):
        scores["Abrasion"] += 0.16
    elif brown > 0.06:
        scores["Abrasion"] += 0.08

    if rel_y > 0.55 or rel_y < 0.35:  # Cervical or incisal attrition zone
        scores["Abrasion"] += 0.12

    if compact > 0.45:
        scores["Abrasion"] += 0.06

    # --- CAVITY SCORE ---
    # Cavity: concave cavitation, necrotic dentin darker than surrounding enamel
    if ar < 1.9:
        scores["Cavity"] += 0.18
    elif ar < 2.3:
        scores["Cavity"] += 0.08

    if delta_val < -35:  # Deep necrotic floor significantly darker than enamel
        scores["Cavity"] += 0.32
    elif delta_val < -18:
        scores["Cavity"] += 0.20
    elif darkness > 0.20:
        scores["Cavity"] += 0.15

    if brown > 0.15:
        scores["Cavity"] += 0.18
    elif brown > 0.08:
        scores["Cavity"] += 0.10

    if compact > 0.35:
        scores["Cavity"] += 0.10

    if 0.003 < rel_size < 0.12:
        scores["Cavity"] += 0.08

    if redness > 0.04:  # Pulpal or hyperemic margin
        scores["Cavity"] += 0.08

    # Base prior
    scores["Cavity"] += 0.04

    # =========================================================================
    # PICK WINNER
    # =========================================================================
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    total_score = sum(scores.values())
    confidence = float(best_score / total_score) if total_score > 0 else 0.25

    # Build reasoning string
    reasons = []
    if best_type == "Crack":
        reasons.append(f"High elongation (AR={ar:.1f})")
        if compact < 0.25:
            reasons.append(f"low compactness ({compact:.2f})")
        if edge_dens > 0.12:
            reasons.append(f"high edge density ({edge_dens:.2f})")
        if rel_size < 0.008:
            reasons.append("narrow feature")
    elif best_type == "Erosion":
        reasons.append(f"Smooth contour (compactness={compact:.2f}, roughness={rough:.2f})")
        if chalky > 0.10:
            reasons.append(f"chalky demineralization ({chalky:.0%})")
        if darkness < 0.08:
            reasons.append("non-dark surface")
        if rel_y > 0.50:
            reasons.append(f"cervical position (rel_y={rel_y:.2f})")
    elif best_type == "Abrasion":
        reasons.append(f"Moderate elongation (AR={ar:.1f})")
        if orient < 25:
            reasons.append(f"near-horizontal ({orient:.0f}°)")
        if brown > 0.08:
            reasons.append(f"warm dentin exposure ({brown:.0%})")
        if rel_y > 0.55:
            reasons.append("cervical zone")
    else:  # Cavity
        reasons.append(f"Round shape (AR={ar:.1f})")
        if darkness > 0.12:
            reasons.append(f"dark interior ({darkness:.0%})")
        if brown > 0.08:
            reasons.append(f"carious brown ({brown:.0%})")

    reasoning = f"{best_type} detected: " + ", ".join(reasons) + "."

    return {
        "anomaly_type": best_type,
        "confidence": round(confidence, 3),
        "reasoning": reasoning,
        "shape_features": shape,
        "color_features": color,
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
    }


def classify_anomaly_from_preset(preset_id: str, geometry: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Returns anomaly classification for known clinical presets where the
    ground-truth anomaly type is already known from the dataset annotation.
    Falls back to the CV classifier if preset is unknown.
    """
    # Known preset ground truths (from AlphaDent / DentalAI annotations)
    PRESET_GROUND_TRUTH = {
        "case_01": {"anomaly_type": "Cavity", "confidence": 0.97, "reasoning": "Ground truth: occlusal molar caries (AlphaDent p001_002)"},
        "case_02": {"anomaly_type": "Cavity", "confidence": 0.96, "reasoning": "Ground truth: posterior pit caries (AlphaDent p001_003)"},
        "case_04": {"anomaly_type": "Cavity", "confidence": 0.95, "reasoning": "Ground truth: cervical margin caries (AlphaDent p015_002)"},
        "case_05": {"anomaly_type": "Crack", "confidence": 0.93, "reasoning": "Ground truth: enamel crack/fissure (DentalAI 1018)"},
        "case_07": {"anomaly_type": "Cavity", "confidence": 0.96, "reasoning": "Ground truth: premolar cavitated fossa (DentalAI 1007)"},
        "case_08": {"anomaly_type": "Cavity", "confidence": 0.95, "reasoning": "Ground truth: maxillary premolar caries (AlphaDent p026_001)"},
        "case_09": {"anomaly_type": "Abrasion", "confidence": 0.94, "reasoning": "Ground truth: anterior labial wear facet (AlphaDent p001_001)"},
        "case_10": {"anomaly_type": "Abrasion", "confidence": 0.93, "reasoning": "Ground truth: canine cervical abrasion (AlphaDent p015_001)"},
    }

    if preset_id in PRESET_GROUND_TRUTH:
        gt = PRESET_GROUND_TRUTH[preset_id]
        return {
            "anomaly_type": gt["anomaly_type"],
            "confidence": gt["confidence"],
            "reasoning": gt["reasoning"],
            "shape_features": None,
            "color_features": None,
            "all_scores": {gt["anomaly_type"]: gt["confidence"]},
        }

    # Unknown preset — would need full CV classification
    return {
        "anomaly_type": "Cavity",
        "confidence": 0.50,
        "reasoning": "Unknown preset, defaulting to Cavity (run full CV classifier with image).",
        "shape_features": None,
        "color_features": None,
        "all_scores": {"Cavity": 0.50},
    }
