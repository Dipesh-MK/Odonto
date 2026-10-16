"""
mask_utils.py
Utilities for:
- Converting polygon / bitmap annotations to binary masks
- Computing Intersection over Union (IoU)
- Extracting geometric shape parameters (Centroid, Bounding Ellipse, Aspect Ratio, Area, Effective Radius)
- Visualization overlays
"""

import zlib
import io
import base64
import numpy as np
import cv2
from PIL import Image

def yolo_poly_to_mask(coords, height, width):
    """
    Convert normalized polygon coordinates [x1, y1, x2, y2, ...] to a binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(coords) < 6:
        return mask
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = pts[:, 0] * width
    pts[:, 1] = pts[:, 1] * height
    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def decode_supervisely_bitmap(data_str, origin, height, width):
    """
    Decode a Supervisely compressed bitmap string into a full-size binary mask.
    origin is [col, row] (i.e. [x, y]).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        zlib_data = base64.b64decode(data_str)
        png_data = zlib.decompress(zlib_data)
        sub_img = Image.open(io.BytesIO(png_data))
        sub_mask = (np.array(sub_img) > 0).astype(np.uint8)
        
        col, row = origin[0], origin[1]
        h_sub, w_sub = sub_mask.shape[:2]
        
        # Clip to bounds
        r_start = max(0, row)
        c_start = max(0, col)
        r_end = min(height, row + h_sub)
        c_end = min(width, col + w_sub)
        
        sub_r_start = r_start - row
        sub_c_start = c_start - col
        sub_r_end = sub_r_start + (r_end - r_start)
        sub_c_end = sub_c_start + (c_end - c_start)
        
        if r_end > r_start and c_end > c_start:
            mask[r_start:r_end, c_start:c_end] = sub_mask[sub_r_start:sub_r_end, sub_c_start:sub_c_end]
    except Exception as e:
        print(f"Error decoding bitmap: {e}")
    return mask

def compute_mask_iou(mask_a, mask_b):
    """
    Computes Intersection over Union (IoU) between two binary boolean masks of the same shape.
    Returns float in [0.0, 1.0].
    """
    assert mask_a.shape == mask_b.shape, f"Mask shape mismatch: {mask_a.shape} vs {mask_b.shape}"
    a = mask_a > 0
    b = mask_b > 0
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

def extract_mask_shape_parameters(binary_mask):
    """
    Extracts shape parameters from a 2D binary mask:
    - centroid: (cx, cy) in pixels
    - bounding ellipse: center (cx, cy), axes (major, minor), angle
    - aspect_ratio: major / minor
    - effective_radius: sqrt(area / pi)
    - area_pixels: count of positive pixels
    - area_fraction: area_pixels / total_pixels
    """
    mask = (binary_mask > 0).astype(np.uint8)
    h, w = mask.shape
    total_pixels = h * w
    area_pixels = int(np.sum(mask))

    if area_pixels == 0:
        return {
            "cx": np.nan,
            "cy": np.nan,
            "area_pixels": 0,
            "area_fraction": 0.0,
            "major_axis": np.nan,
            "minor_axis": np.nan,
            "aspect_ratio": np.nan,
            "effective_radius": 0.0,
            "ellipse_angle": np.nan
        }

    # Centroid from moments or mean
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        y_indices, x_indices = np.where(mask)
        cx = float(np.mean(x_indices))
        cy = float(np.mean(y_indices))

    area_fraction = float(area_pixels / total_pixels)
    effective_radius = float(np.sqrt(area_pixels / np.pi))

    # Find contours for ellipse fitting
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    major_axis = np.nan
    minor_axis = np.nan
    aspect_ratio = 1.0
    ellipse_angle = 0.0

    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(largest_contour)
                (e_cx, e_cy), (d1, d2), angle = ellipse
                major = max(d1, d2)
                minor = min(d1, d2)
                if minor > 0:
                    major_axis = float(major)
                    minor_axis = float(minor)
                    aspect_ratio = float(major / minor)
                    ellipse_angle = float(angle)
            except Exception:
                pass
        
        if np.isnan(major_axis):
            # Fall back to minimum bounding box
            rect = cv2.minAreaRect(largest_contour)
            (r_cx, r_cy), (w_box, h_box), angle = rect
            major = max(w_box, h_box)
            minor = min(w_box, h_box)
            major_axis = float(major)
            minor_axis = float(minor) if minor > 0 else 1.0
            aspect_ratio = float(major_axis / minor_axis)
            ellipse_angle = float(angle)

    params = {
        "cx": cx,
        "cy": cy,
        "area_pixels": area_pixels,
        "area_fraction": area_fraction,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "aspect_ratio": aspect_ratio,
        "effective_radius": effective_radius,
        "ellipse_angle": ellipse_angle
    }

    # Strict sanity check
    for k, v in params.items():
        if k not in ["major_axis", "minor_axis", "aspect_ratio", "ellipse_angle"]:
            assert np.isfinite(v), f"Non-finite shape parameter {k}: {v}"

    return params

def overlay_mask_on_image(image_bgr, binary_mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlays a semi-transparent binary mask on a BGR image.
    """
    mask = binary_mask > 0
    overlay = image_bgr.copy()
    overlay[mask] = (
        overlay[mask] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha
    ).astype(np.uint8)
    return overlay
