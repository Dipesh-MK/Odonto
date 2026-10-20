"""
parameter_converter.py

Explicit 2D-to-3D parameter conversion layer.
Transforms 2D pixel geometry (centroid, area, aspect ratio, radius)
into 3D tooth-relative carving parameters.

Explicit Design Rules:
1. No physical scale calibration: does NOT assume pixels equal millimeters.
2. No depth estimation: uses documented DEFAULT_DEPTH.
3. Labeled as: "geometry-informed approximation rather than physically measured depth."
"""

import numpy as np

# Documented default depth ratio relative to tooth bounding box diagonal
# Matches the synthetic anomaly generator's range (0.05 to 0.15 * tooth_size)
DEFAULT_DEPTH_RATIO = 0.08  # ~0.8 - 1.2 mm on a typical 10-15mm human tooth
MIN_CARVE_RADIUS_MM = 0.6
MAX_CARVE_RADIUS_MM = 4.0
MIN_CARVE_DEPTH_MM = 0.4
MAX_CARVE_DEPTH_MM = 1.8

def convert_2d_geometry_to_3d_params(geometry, tooth_mesh, default_depth_ratio=DEFAULT_DEPTH_RATIO, anomaly_type="Cavity", surface="occlusal"):
    """
    Converts 2D shape geometry into 3D carving arguments for carve_cavity().
    Maps the 2D segmentation centroid position onto the correct anatomical surface
    of the 3D tooth mesh, with surface-aware coordinate remapping.
    """
    from tooth_carver import find_surface_vertex

    verts = tooth_mesh.vertices
    bounds = tooth_mesh.bounds
    tooth_size = float(np.linalg.norm(bounds[1] - bounds[0]))

    # 1. Radius conversion:
    rel_r = geometry.get("relative_radius", 0.05)
    estimated_radius_3d = rel_r * tooth_size * 2.2
    carve_radius = float(np.clip(estimated_radius_3d, MIN_CARVE_RADIUS_MM, MAX_CARVE_RADIUS_MM))

    # 2. Depth calculation:
    depth_multiplier = 1.0
    anomaly_lower = str(anomaly_type).lower()
    if "crack" in anomaly_lower:
        depth_multiplier = 1.3
    elif "abrasion" in anomaly_lower:
        depth_multiplier = 0.7
    elif "deep" in anomaly_lower:
        depth_multiplier = 1.5

    carve_depth = float(np.clip(tooth_size * default_depth_ratio * depth_multiplier, MIN_CARVE_DEPTH_MM, MAX_CARVE_DEPTH_MM * 1.5))

    # 3. Surface determination:
    if "cervical" in anomaly_lower or "caries 5" in anomaly_lower:
        actual_surface = "cervical"
    elif surface in ["buccal", "lingual", "cervical", "interproximal"]:
        actual_surface = surface
    else:
        actual_surface = "occlusal"

    # 4. Surface-aware relative coordinate mapping:
    # The 2D centroid (rel_x, rel_y) from the photo maps differently depending
    # on which surface we're projecting onto the 3D mesh.
    raw_rel = geometry.get("relative_centroid", [0.5, 0.5])
    rel_x, rel_y = raw_rel

    if actual_surface == "occlusal":
        # Top-down view: photo X -> mesh X, photo Y -> mesh Y
        rel_coords = [rel_x, rel_y]
    elif actual_surface == "buccal":
        # Front-facing view: photo X -> mesh X (mesial-distal), photo Y -> inverted mesh Z (height)
        rel_coords = [rel_x, 1.0 - rel_y]
    elif actual_surface == "lingual":
        # Back-facing view: photo X mirrored -> mesh X, photo Y -> inverted mesh Z
        rel_coords = [1.0 - rel_x, 1.0 - rel_y]
    elif actual_surface == "cervical":
        # Cervical margin: photo X -> mesh X, photo Y -> mesh Y (buccal-lingual)
        rel_coords = [rel_x, rel_y]
    elif actual_surface == "interproximal":
        # Side view: photo X hints mesial/distal, photo Y -> inverted mesh Z
        rel_coords = [rel_x, 1.0 - rel_y]
    else:
        rel_coords = [rel_x, rel_y]

    center_idx = find_surface_vertex(tooth_mesh, surface=actual_surface, rel_coords=rel_coords)

    return {
        "center_idx": int(center_idx),
        "radius": round(carve_radius, 3),
        "depth": round(carve_depth, 3),
        "aspect_ratio": geometry.get("aspect_ratio", 1.0),
        "orientation": geometry.get("orientation", 0.0),
        "tooth_size": round(tooth_size, 3),
        "surface": actual_surface,
        "notes": f"Anatomic {actual_surface} projection, {anomaly_type}"
    }
