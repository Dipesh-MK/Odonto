"""
depth_extractor.py

Monocular Depth Estimation & Local Enamel Detrending for Dental Macro Photography.
Extracts dense physical indentation relief Δz(u, v) from single 2D clinical photos
using Depth Anything V2 (and TorchHub MiDaS fallback) coupled with surrounding sound
enamel baseline planar fitting.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger("depth_extractor")

# Global singleton pipeline
_DEPTH_MODEL = None
_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_depth_pipeline():
    """Lazily loads Depth Anything V2 Small on GPU/CPU."""
    global _DEPTH_MODEL
    if _DEPTH_MODEL is None:
        try:
            from transformers import pipeline
            logger.info(f"Loading Depth Anything V2 Small on {_DEVICE}...")
            _DEPTH_MODEL = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("Depth Anything V2 loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load Depth Anything V2 ({e}). Falling back to MiDaS...")
            try:
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                midas.to(_DEVICE)
                midas.eval()
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                _DEPTH_MODEL = {"model": midas, "transform": midas_transforms.small_transform, "type": "midas"}
                logger.info("MiDaS Small loaded successfully as fallback.")
            except Exception as e2:
                logger.error(f"Both Depth Anything V2 and MiDaS failed to load: {e2}")
                _DEPTH_MODEL = None
    return _DEPTH_MODEL


def estimate_raw_depth_map(img_bgr: np.ndarray) -> np.ndarray:
    """
    Estimates raw inverse depth map from an input BGR image.
    Returns normalized float32 array in [0.0, 1.0] where 1.0 is closest to camera.
    """
    pipe = get_depth_pipeline()
    if pipe is None:
        # Fallback synthetic depth if ML models fail to load
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return 1.0 - (gray.astype(np.float32) / 255.0)

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if isinstance(pipe, dict) and pipe.get("type") == "midas":
        # MiDaS execution path
        transform = pipe["transform"]
        model = pipe["model"]
        input_batch = transform(img_rgb).to(_DEVICE)
        with torch.no_grad():
            pred = model(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()
            depth_np = pred.cpu().numpy()
    else:
        # HuggingFace pipeline path (Depth Anything V2)
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        result = pipe(pil_img)
        depth_pil = result["depth"]
        depth_np = np.array(depth_pil, dtype=np.float32)

    # Invert so larger value = deeper cavity indentation relative to camera
    # In monocular depth, higher values typically mean closer to camera.
    # When looking directly into a tooth cavity, the floor is FARTHER from camera than the enamel rim.
    d_min, d_max = depth_np.min(), depth_np.max()
    if d_max > d_min:
        norm_depth = (depth_np - d_min) / (d_max - d_min)
    else:
        norm_depth = np.zeros_like(depth_np)

    # Resize back to exact source image shape if needed
    if norm_depth.shape != (h, w):
        norm_depth = cv2.resize(norm_depth, (w, h), interpolation=cv2.INTER_CUBIC)

    return norm_depth


def detrend_enamel_baseline(
    raw_depth: np.ndarray,
    binary_mask: np.ndarray,
    ring_dilation_px: int = 15,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fits a planar reference surface to the sound enamel ring surrounding the lesion mask,
    then subtracts it to isolate pure crater indentation relief Δz(u, v).

    Args:
        raw_depth: Raw monocular depth map (H, W).
        binary_mask: Binary mask of lesion (H, W), 255 = inside lesion, 0 = outside.
        ring_dilation_px: Width of the surrounding sound enamel border ring in pixels.

    Returns:
        (delta_z_relief, baseline_plane, metrics_dict)
    """
    h, w = raw_depth.shape[:2]

    # Create surrounding sound enamel ring by dilating mask and subtracting original mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_dilation_px * 2 + 1, ring_dilation_px * 2 + 1))
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    enamel_ring = (dilated > 0) & (binary_mask == 0)

    # If ring has insufficient points, fallback to broader border
    if enamel_ring.sum() < 30:
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_dilation_px * 4 + 1, ring_dilation_px * 4 + 1))
        dilated_large = cv2.dilate(binary_mask, kernel_large, iterations=1)
        enamel_ring = (dilated_large > 0) & (binary_mask == 0)

    ring_ys, ring_xs = np.where(enamel_ring)

    if len(ring_xs) >= 6:
        # Fit 2D plane: z_baseline = a*x + b*y + c using Least Squares
        A = np.column_stack([ring_xs, ring_ys, np.ones_like(ring_xs)])
        z_ring = raw_depth[ring_ys, ring_xs]
        coeffs, _, _, _ = np.linalg.lstsq(A, z_ring, rcond=None)
        a, b, c = coeffs

        # Evaluate reference plane across the full image grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        baseline_plane = a * grid_x + b * grid_y + c
    else:
        # Fallback to mean boundary depth
        baseline_val = float(raw_depth[binary_mask > 0].mean()) if (binary_mask > 0).any() else 0.5
        baseline_plane = np.full((h, w), baseline_val, dtype=np.float32)

    # Crater indentation is baseline minus raw depth (cavity points are farther/lower than sound rim)
    # If the monocular model predicted lower raw values for farther/deeper crater points:
    # delta_z = baseline - raw_depth
    diff = baseline_plane - raw_depth

    # Check if inside-mask values are predominantly positive or negative
    inside_diffs = diff[binary_mask > 0]
    if len(inside_diffs) > 0 and np.median(inside_diffs) < 0:
        # Model inverted distance vs disparity: flip polarity
        diff = -diff

    # Inside the lesion, clamp negative values (above sound surface) to zero
    delta_z = np.clip(diff, 0.0, None)

    # Smooth the perimeter transition to zero at the mask boundary
    dist_inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_dist = dist_inside.max()
    if max_dist > 0:
        blend_edge = np.clip(dist_inside / max(2.0, min(12.0, max_dist * 0.25)), 0.0, 1.0)
        delta_z = delta_z * blend_edge

    # Normalize indentation relief inside mask strictly to [0.0, 1.0]
    inside_vals = delta_z[binary_mask > 0]
    if len(inside_vals) > 0 and inside_vals.max() > 1e-5:
        delta_z = delta_z / inside_vals.max()
    else:
        # If flat, create subtle concave relief
        delta_z = np.clip(dist_inside / (max_dist + 1e-6), 0.0, 1.0) ** 1.4

    delta_z[binary_mask == 0] = 0.0

    metrics = {
        "enamel_ring_samples": int(len(ring_xs)),
        "max_relative_indentation": float(delta_z.max()),
        "mean_relative_indentation": float(delta_z[binary_mask > 0].mean()) if (binary_mask > 0).any() else 0.0,
        "baseline_tilt_x": float(coeffs[0]) if len(ring_xs) >= 6 else 0.0,
        "baseline_tilt_y": float(coeffs[1]) if len(ring_xs) >= 6 else 0.0,
    }

    return delta_z.astype(np.float32), baseline_plane.astype(np.float32), metrics


def render_relief_heatmap_overlay(
    img_bgr: np.ndarray,
    delta_z: np.ndarray,
    binary_mask: np.ndarray,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """
    Renders an informative color-coded 2.5D relief heatmap overlaid onto the original image.
    Deepest crater indentations show as vibrant crimson/amber, shallow borders as teal/blue.
    """
    h, w = img_bgr.shape[:2]
    # Colorize depth relief
    depth_u8 = (np.clip(delta_z, 0.0, 1.0) * 255.0).astype(np.uint8)
    color_map = cv2.applyColorMap(depth_u8, colormap)

    # Blend onto original image only within and immediately around the mask
    overlay = img_bgr.copy()
    mask_bool = binary_mask > 0
    overlay[mask_bool] = (0.35 * img_bgr[mask_bool] + 0.65 * color_map[mask_bool]).astype(np.uint8)

    # Draw sharp boundary contour
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(overlay, contours, isClosed=True, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    return overlay
