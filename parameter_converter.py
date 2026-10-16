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

def convert_2d_geometry_to_3d_params(geometry, tooth_mesh, default_depth_ratio=DEFAULT_DEPTH_RATIO):
    """
    Converts 2D shape geometry into 3D carving arguments for carve_cavity().

    Args:
        geometry: Dict from extract_anomaly_geometry() containing:
                  'relative_radius', 'aspect_ratio', 'relative_centroid', 'orientation'
        tooth_mesh: trimesh.Trimesh object of the target isolated tooth
        default_depth_ratio: float, ratio of tooth bounding diagonal for default cavity depth

    Returns:
        Dict:
        {
            "center_idx": int, vertex index on tooth crown,
            "radius": float, radius in 3D tooth units (mm),
            "depth": float, depth in 3D tooth units (mm),
            "aspect_ratio": float,
            "orientation": float,
            "tooth_size": float,
            "notes": "geometry-informed approximation (default depth)"
        }
    """
    verts = tooth_mesh.vertices
    bounds = tooth_mesh.bounds
    # Physical/mesh scale: diagonal of tooth bounding box (typically ~10-15mm)
    tooth_size = float(np.linalg.norm(bounds[1] - bounds[0]))

    # 1. Radius conversion:
    # 2D relative_radius is (radius / min_dim_of_image).
    # In dental macro photos, a single tooth typically spans 30% to 70% of the image field.
    # We calibrate relative_radius against the tooth mesh size with an empirical factor:
    rel_r = geometry.get("relative_radius", 0.05)
    # Clamp scale factor to ensure realistic anatomical dimensions
    estimated_radius_3d = rel_r * tooth_size * 2.2
    carve_radius = float(np.clip(estimated_radius_3d, MIN_CARVE_RADIUS_MM, MAX_CARVE_RADIUS_MM))

    # 2. Depth: Documented default approximation (NO depth estimation)
    carve_depth = float(np.clip(tooth_size * default_depth_ratio, MIN_CARVE_DEPTH_MM, MAX_CARVE_DEPTH_MM))

    # 3. Center vertex index:
    # In standard intraoral scans (Teeth3DS+), the occlusal (chewing) surface is topmost (highest Z).
    # If the relative centroid is in the central region of the image, we map to the occlusal crown.
    # Find crown vertices (top 25% of Z-extent)
    z_min, z_max = bounds[0][2], bounds[1][2]
    z_threshold = z_min + 0.75 * (z_max - z_min)
    crown_mask = verts[:, 2] >= z_threshold

    if np.any(crown_mask):
        crown_indices = np.where(crown_mask)[0]
        # Map 2D relative centroid (X, Y) to the XY plane of the crown
        rel_x, rel_y = geometry.get("relative_centroid", [0.5, 0.5])
        # Normalized XY on the crown surface:
        # Dental photos typically view the occlusal or buccal face.
        # Project relative coordinates to the bounding box of crown vertices:
        crown_xy_min = verts[crown_indices, :2].min(axis=0)
        crown_xy_max = verts[crown_indices, :2].max(axis=0)
        target_xy = crown_xy_min + np.array([rel_x, rel_y]) * (crown_xy_max - crown_xy_min)

        # Find closest crown vertex to target_xy
        dists_2d = np.linalg.norm(verts[crown_indices, :2] - target_xy, axis=1)
        center_idx = int(crown_indices[np.argmin(dists_2d)])
    else:
        # Fallback to topmost occlusal candidate
        center_idx = int(np.argmax(verts[:, 2]))

    return {
        "center_idx": center_idx,
        "radius": round(carve_radius, 3),
        "depth": round(carve_depth, 3),
        "aspect_ratio": geometry.get("aspect_ratio", 1.0),
        "orientation": geometry.get("orientation", 0.0),
        "tooth_size": round(tooth_size, 3),
        "notes": "geometry-informed approximation (default depth, no physical scale calibration)"
    }
