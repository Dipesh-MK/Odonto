"""
geometry_extractor.py

Deterministic computer-vision geometry extraction for single anomaly masks.
Uses OpenCV and NumPy to compute spatial moments, bounding geometry,
aspect ratio, equivalent radius, and comparative validation metrics.
No LLM is involved in any geometric calculations.
"""

import math
import cv2
import numpy as np

def extract_anomaly_geometry(binary_mask, img_shape=None):
    """
    Extracts deterministic geometric parameters from a 2D binary anomaly mask.

    Args:
        binary_mask: 2D numpy array (uint8 or bool, non-zero indicates anomaly)
        img_shape: Optional tuple (height, width) of the original image.

    Returns:
        Dict of geometric parameters:
        {
            "centroid": [cx, cy],
            "area": area,
            "width": width,
            "height": height,
            "bounding_box": [x, y, w, h],
            "aspect_ratio": aspect_ratio,
            "orientation": angle_degrees,
            "radius": equivalent_radius,
            "relative_radius": relative_radius,
            "relative_centroid": [rel_cx, rel_cy],
            "major_axis": major_axis,
            "minor_axis": minor_axis,
            "contour": list of points
        }
    """
    mask = (binary_mask > 0).astype(np.uint8)
    h, w = mask.shape[:2]
    if img_shape is not None:
        img_h, img_w = img_shape[:2]
    else:
        img_h, img_w = h, w

    area = int(np.sum(mask))
    if area == 0:
        return {
            "centroid": [float(img_w / 2), float(img_h / 2)],
            "area": 0,
            "width": 0.0,
            "height": 0.0,
            "bounding_box": [0, 0, 0, 0],
            "aspect_ratio": 1.0,
            "orientation": 0.0,
            "radius": 0.0,
            "relative_radius": 0.0,
            "relative_centroid": [0.5, 0.5],
            "major_axis": 0.0,
            "minor_axis": 0.0,
            "contour": []
        }

    # 1. Centroid via spatial moments
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        y_indices, x_indices = np.where(mask)
        cx = float(np.mean(x_indices))
        cy = float(np.mean(y_indices))

    # 2. Equivalent radius: circle with identical area
    radius = float(math.sqrt(area / math.pi))
    min_dim = min(img_w, img_h)
    relative_radius = float(radius / min_dim)
    relative_centroid = [float(cx / img_w), float(cy / img_h)]

    # 3. Contour extraction
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x_box, y_box, w_box, h_box = cv2.boundingRect(largest_contour)
        bbox = [int(x_box), int(y_box), int(w_box), int(h_box)]
        bbox_w = float(w_box)
        bbox_h = float(h_box)
    else:
        bbox = [0, 0, 0, 0]
        bbox_w = 0.0
        bbox_h = 0.0
        largest_contour = None

    # 4. Aspect ratio & orientation via fitted ellipse or minAreaRect
    major_axis = bbox_w
    minor_axis = bbox_h
    aspect_ratio = 1.0
    angle = 0.0

    if largest_contour is not None and len(largest_contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(largest_contour)
            (e_cx, e_cy), (d1, d2), e_angle = ellipse
            major = max(d1, d2)
            minor = min(d1, d2)
            if minor > 0:
                major_axis = float(major)
                minor_axis = float(minor)
                aspect_ratio = float(major / minor)
                angle = float(e_angle)
        except Exception:
            pass

    if aspect_ratio == 1.0 and largest_contour is not None and len(largest_contour) >= 3:
        rect = cv2.minAreaRect(largest_contour)
        (r_cx, r_cy), (rw, rh), r_angle = rect
        major = max(rw, rh)
        minor = min(rw, rh)
        major_axis = float(major)
        minor_axis = float(minor) if minor > 0 else 1.0
        aspect_ratio = float(major_axis / minor_axis)
        angle = float(r_angle)

    # Simplified contour representation (JSON-serializable)
    contour_points = []
    if largest_contour is not None:
        # Approximate contour for compact storage
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        contour_points = approx.reshape(-1, 2).tolist()

    return {
        "centroid": [round(cx, 2), round(cy, 2)],
        "area": area,
        "width": round(bbox_w, 2),
        "height": round(bbox_h, 2),
        "bounding_box": bbox,
        "aspect_ratio": round(aspect_ratio, 2),
        "orientation": round(angle, 2),
        "radius": round(radius, 2),
        "relative_radius": round(relative_radius, 4),
        "relative_centroid": [round(relative_centroid[0], 4), round(relative_centroid[1], 4)],
        "major_axis": round(major_axis, 2),
        "minor_axis": round(minor_axis, 2),
        "contour": contour_points
    }

def compute_geometry_comparison_metrics(pred_mask, gt_mask, img_shape=None):
    """
    Computes comprehensive segmentation and geometric errors between
    predicted single anomaly mask and ground-truth mask.

    Returns:
        Dict of metrics:
        - IoU
        - Dice
        - centroid_error (pixels)
        - relative_centroid_error (normalized by GT bbox diagonal)
        - width_error (pixels)
        - height_error (pixels)
        - aspect_ratio_error
        - radius_error (relative |r_pred - r_gt| / r_gt)
    """
    p_mask = (pred_mask > 0).astype(np.uint8)
    g_mask = (gt_mask > 0).astype(np.uint8)

    # 1. IoU & Dice
    intersection = np.logical_and(p_mask, g_mask).sum()
    union = np.logical_or(p_mask, g_mask).sum()
    iou = float(intersection / union) if union > 0 else (1.0 if intersection == 0 else 0.0)

    p_sum = p_mask.sum()
    g_sum = g_mask.sum()
    dice = float((2.0 * intersection) / (p_sum + g_sum)) if (p_sum + g_sum) > 0 else 1.0

    # 2. Extract shape parameters for both
    pred_geom = extract_anomaly_geometry(p_mask, img_shape)
    gt_geom = extract_anomaly_geometry(g_mask, img_shape)

    # 3. Centroid error
    p_cx, p_cy = pred_geom["centroid"]
    g_cx, g_cy = gt_geom["centroid"]
    centroid_error = float(math.sqrt((p_cx - g_cx) ** 2 + (p_cy - g_cy) ** 2))

    # Normalize by GT bounding box diagonal
    gt_bw = gt_geom["width"]
    gt_bh = gt_geom["height"]
    gt_diag = math.sqrt(gt_bw ** 2 + gt_bh ** 2) if (gt_bw > 0 or gt_bh > 0) else 1.0
    relative_centroid_error = float(centroid_error / gt_diag)

    # 4. Dimension & aspect ratio errors
    width_error = float(abs(pred_geom["width"] - gt_geom["width"]))
    height_error = float(abs(pred_geom["height"] - gt_geom["height"]))
    aspect_ratio_error = float(abs(pred_geom["aspect_ratio"] - gt_geom["aspect_ratio"]))

    # 5. Radius relative error
    gt_r = gt_geom["radius"]
    if gt_r > 0:
        radius_error = float(abs(pred_geom["radius"] - gt_r) / gt_r)
    else:
        radius_error = 0.0 if pred_geom["radius"] == 0 else 1.0

    return {
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "pred_area": pred_geom["area"],
        "gt_area": gt_geom["area"],
        "pred_bbox": pred_geom["bounding_box"],
        "gt_bbox": gt_geom["bounding_box"],
        "centroid_error": round(centroid_error, 2),
        "relative_centroid_error": round(relative_centroid_error, 4),
        "width_error": round(width_error, 2),
        "height_error": round(height_error, 2),
        "aspect_ratio_error": round(aspect_ratio_error, 2),
        "radius_error": round(radius_error, 4),
        "pred_geometry": pred_geom,
        "gt_geometry": gt_geom
    }

def draw_geometry_overlay(image_bgr, mask, geometry, color=(0, 255, 0), alpha=0.45):
    """
    Renders mask overlay, centroid crosshair (yellow), lesion contour (green),
    and fitted ellipse / bounding box on the image.
    """
    overlay = image_bgr.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + np.array(color, dtype=np.uint8) * alpha
    ).astype(np.uint8)

    cx, cy = int(round(geometry["centroid"][0])), int(round(geometry["centroid"][1]))

    # Draw centroid crosshair
    cross_size = max(10, int(geometry["radius"] * 0.4))
    cv2.line(overlay, (cx - cross_size, cy), (cx + cross_size, cy), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(overlay, (cx, cy - cross_size), (cx, cy + cross_size), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1, cv2.LINE_AA)

    # Draw fitted ellipse if available
    major = geometry.get("major_axis", 0)
    minor = geometry.get("minor_axis", 0)
    angle = geometry.get("orientation", 0)
    if major > 2 and minor > 2:
        axes = (int(round(major / 2)), int(round(minor / 2)))
        cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, (0, 0, 255), 2, cv2.LINE_AA)

    return overlay
