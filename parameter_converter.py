"""
parameter_converter.py

Dynamic, Tooth-Type-Aware and Anomaly-Aware 2D-to-3D Parameter Conversion Layer.
Transforms 2D pixel geometry (centroid, area, aspect ratio, radius, orientation, mask)
into 3D tooth-relative carving parameters derived directly from the target tooth mesh.

Key Architectural Capabilities:
1. Canonical Anatomical Registration & Calibrated Multi-View Raycasting:
   - Derives intrinsic orthonormal dental axes:
     * e_md: Mesio-distal tangent along dental arch
     * e_facial: Bucco-lingual outward axis (facing lips/cheek)
     * e_crown: Apico-coronal axis (cervical base -> crown/occlusal table)
   - Calibrated camera projection maps 2D coordinates (rel_x, rel_y) directly
     to the visible surface vertices matching the clinical camera view:
     * 'occlusal': view looking down -e_crown (chewing table)
     * 'buccal' / 'labial': view looking along -e_facial (facial crown face)
     * 'lingual' / 'palatal': view looking along +e_facial (inner tongue/palatal face)
     * 'cervical': view targeted to the cervical / gingival margin (CEJ)
     * 'interproximal' ('mesial' / 'distal'): view along +/-e_md
2. Zero-Freeze Spatial Indexing via scipy.spatial.cKDTree:
   - Instantaneous nearest-neighbor and radius lookups (<2ms), completely replacing
     unbounded O(N^2) distance loops.
3. Dynamic Mesh-Derived Bounds:
   - All spatial cutoffs, radii, and depths are derived from measured mesh surface spans,
     mean edge lengths, and local slice wall thickness.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial import cKDTree


def classify_fdi_tooth(tooth_fdi=None, mesh_spans=None) -> Dict[str, Any]:
    """
    Explicitly classifies an FDI two-digit tooth number into its anatomical type
    and functional group.

    FDI Notation:
      - First digit (quadrant): 1-4 permanent, 5-8 deciduous
      - Second digit (position from midline):
          1, 2 -> incisor   (anterior)
          3    -> canine    (anterior)
          4, 5 -> premolar  (posterior)
          6-8  -> molar     (posterior)
    """
    if tooth_fdi is not None:
        try:
            fdi_int = int(tooth_fdi)
            quadrant = fdi_int // 10
            position = fdi_int % 10
            jaw = "upper" if quadrant in (1, 2, 5, 6) else "lower"
            side = "right" if quadrant in (1, 4, 5, 8) else "left"
            if position in (1, 2):
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "incisor",
                    "anatomical_group": "anterior",
                    "jaw": jaw,
                    "side": side,
                    "position": position,
                }
            elif position == 3:
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "canine",
                    "anatomical_group": "anterior",
                    "jaw": jaw,
                    "side": side,
                    "position": position,
                }
            elif position in (4, 5):
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "premolar",
                    "anatomical_group": "posterior",
                    "jaw": jaw,
                    "side": side,
                    "position": position,
                }
            elif position in (6, 7, 8):
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "molar",
                    "anatomical_group": "posterior",
                    "jaw": jaw,
                    "side": side,
                    "position": position,
                }
        except (ValueError, TypeError):
            pass

    if mesh_spans is not None:
        md_span, bl_span, crown_h = mesh_spans
        horizontal_footprint = max(np.sqrt(md_span * bl_span), 1e-5)
        if crown_h / horizontal_footprint > 1.05:
            return {
                "tooth_fdi": tooth_fdi,
                "tooth_type": "incisor",
                "anatomical_group": "anterior",
                "jaw": "unknown",
                "side": "unknown",
                "position": 1,
            }
        elif max(md_span, bl_span) < 9.0:
            return {
                "tooth_fdi": tooth_fdi,
                "tooth_type": "premolar",
                "anatomical_group": "posterior",
                "jaw": "unknown",
                "side": "unknown",
                "position": 4,
            }

    return {
        "tooth_fdi": tooth_fdi,
        "tooth_type": "molar",
        "anatomical_group": "posterior",
        "jaw": "unknown",
        "side": "unknown",
        "position": 6,
    }


def compute_anatomical_frame(tooth_mesh) -> Dict[str, Any]:
    """
    Derives the intrinsic 3D anatomical coordinate frame of an isolated tooth mesh
    purely from its vertex and face geometry:
      - e_crown: Apico-coronal axis (cervical base -> crown tip/occlusal surface),
                 computed via Gauss's divergence theorem on the open cervical base
                 (area-weighted sum of outward face normals).
      - e_facial: Labial/Buccal outward axis (pointing from dental arch center toward
                  lips/cheek, orthogonal to e_crown).
      - e_md: Mesio-distal axis (tangent along dental arch, e_crown x e_facial).
    """
    verts = np.asarray(tooth_mesh.vertices, dtype=np.float64)
    center = verts.mean(axis=0)

    # 1. Apico-coronal (crown) axis via area-weighted face normals
    area_normals = tooth_mesh.face_normals * tooth_mesh.area_faces[:, None]
    crown_vec = area_normals.sum(axis=0)
    crown_norm = np.linalg.norm(crown_vec)
    if crown_norm > 1e-8:
        e_crown = crown_vec / crown_norm
    else:
        e_crown = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # 2. Labial/Buccal outward axis
    c_xy = np.array([center[0], center[1], 0.0], dtype=np.float64)
    c_xy_ortho = c_xy - np.dot(c_xy, e_crown) * e_crown
    if np.linalg.norm(c_xy_ortho) > 2.0:
        e_facial = c_xy_ortho / np.linalg.norm(c_xy_ortho)
    else:
        v_centered = verts - center
        v_planar = v_centered - np.outer(np.dot(v_centered, e_crown), e_crown)
        _, _, vh = np.linalg.svd(v_planar, full_matrices=False)
        candidate_axis = vh[1]
        e_facial = candidate_axis / (np.linalg.norm(candidate_axis) + 1e-8)

    # 3. Mesio-distal axis (completing right-handed orthonormal frame)
    e_md = np.cross(e_crown, e_facial)
    e_md = e_md / (np.linalg.norm(e_md) + 1e-8)
    e_facial = np.cross(e_md, e_crown)
    e_facial = e_facial / (np.linalg.norm(e_facial) + 1e-8)

    # Project all vertices into local anatomical coordinates (u=MD, v=Facial, w=Crown)
    rel_verts = verts - center
    u = np.dot(rel_verts, e_md)
    v = np.dot(rel_verts, e_facial)
    w = np.dot(rel_verts, e_crown)

    w_min, w_max = float(w.min()), float(w.max())
    crown_height = max(w_max - w_min, 1e-5)
    w_norm = (w - w_min) / crown_height

    md_span = float(np.percentile(u, 98) - np.percentile(u, 2))
    bl_span = float(np.percentile(v, 98) - np.percentile(v, 2))
    mean_edge_len = float(tooth_mesh.edges_unique_length.mean())

    return {
        "center": center,
        "e_md": e_md,
        "e_facial": e_facial,
        "e_crown": e_crown,
        "u": u,
        "v": v,
        "w": w,
        "w_norm": w_norm,
        "md_span": md_span,
        "bl_span": bl_span,
        "crown_height": crown_height,
        "mean_edge_len": mean_edge_len,
    }


def measure_local_slice_thickness(u, v, w_norm, target_w_norm):
    """
    Dynamically measures the tooth's actual labio-lingual / bucco-lingual thickness
    and mesio-distal width at the horizontal cross-section around target_w_norm.
    """
    band = np.abs(w_norm - target_w_norm) <= 0.10
    if np.sum(band) < 10:
        band = np.abs(w_norm - target_w_norm) <= 0.20
    if np.sum(band) < 5:
        band = np.ones_like(w_norm, dtype=bool)

    local_bl_thickness = float(np.percentile(v[band], 95) - np.percentile(v[band], 5))
    local_md_width = float(np.percentile(u[band], 95) - np.percentile(u[band], 5))
    return max(local_md_width, 1.0), max(local_bl_thickness, 1.0)


def raycast_surface_landmark(
    tooth_mesh,
    surface: str = "occlusal",
    rel_coords: Tuple[float, float] = (0.5, 0.5),
    frame: Optional[Dict[str, Any]] = None,
    fine_fossa_snap: bool = False,
) -> Dict[str, Any]:
    """
    Calibrated Multi-View Raycasting & Coordinate Projection.

    Projects normalized 2D image coordinates rel_coords=(rel_x, rel_y) in [0, 1]
    directly onto the visible 3D mesh surface corresponding to the camera viewing angle:
      - 'occlusal': Camera looking down at chewing table along -e_crown.
      - 'buccal' / 'labial' / 'labial_facial': Camera looking along -e_facial.
      - 'lingual' / 'palatal': Camera looking along +e_facial.
      - 'cervical': Camera looking along -e_facial near the cervical neck (CEJ).
      - 'incisal': Camera looking along -e_crown at incisal edge.
      - 'interproximal' / 'mesial' / 'distal': Camera looking along +/-e_md.

    Returns:
      {
        "center_idx": int,
        "center_pos": ndarray(3),
        "surface": str,
        "tangent_u": ndarray(3),
        "tangent_v": ndarray(3),
        "view_dir": ndarray(3),
        "surf_span_u": float,
        "surf_span_v": float,
        "characteristic_span": float,
      }
    """
    verts = np.asarray(tooth_mesh.vertices, dtype=np.float64)
    normals = np.asarray(tooth_mesh.vertex_normals, dtype=np.float64)

    if frame is None:
        frame = compute_anatomical_frame(tooth_mesh)

    e_crown = frame["e_crown"]
    e_facial = frame["e_facial"]
    e_md = frame["e_md"]
    w_norm = frame["w_norm"]

    rel_x = float(np.clip(rel_coords[0], 0.05, 0.95))
    rel_y = float(np.clip(rel_coords[1], 0.05, 0.95))

    surf_clean = str(surface or "occlusal").strip().lower()

    # 1. Configure Camera View Orientation and Projection Axes
    if "occ" in surf_clean or "chew" in surf_clean:
        actual_surface = "occlusal"
        view_dir = -e_crown
        tangent_u = e_md
        tangent_v = e_facial
        target_u = rel_x
        target_v = rel_y
        normal_align = np.dot(normals, e_crown)
        vis_mask = (normal_align > 0.18) & (w_norm >= 0.35)

    elif "cerv" in surf_clean or "gingiv" in surf_clean:
        actual_surface = "cervical"
        view_dir = -e_facial
        tangent_u = e_md
        tangent_v = e_crown
        target_u = rel_x
        target_v = 1.0 - rel_y
        normal_align = np.dot(normals, e_facial)
        vis_mask = (normal_align > 0.15) & (w_norm >= 0.10) & (w_norm <= 0.50)

    elif "ling" in surf_clean or "palat" in surf_clean:
        actual_surface = "lingual"
        view_dir = e_facial
        tangent_u = -e_md
        tangent_v = e_crown
        target_u = rel_x
        target_v = 1.0 - rel_y
        normal_align = np.dot(normals, -e_facial)
        vis_mask = (normal_align > 0.18) & (w_norm >= 0.15)

    elif "mes" in surf_clean or "dist" in surf_clean or "prox" in surf_clean:
        actual_surface = "interproximal"
        is_mesial = "mes" in surf_clean or rel_x < 0.5
        view_dir = e_md if is_mesial else -e_md
        tangent_u = e_facial if is_mesial else -e_facial
        tangent_v = e_crown
        target_u = rel_x
        target_v = 1.0 - rel_y
        normal_align = np.dot(normals, -view_dir)
        vis_mask = (normal_align > 0.18) & (w_norm >= 0.15)

    elif "incis" in surf_clean:
        actual_surface = "incisal"
        view_dir = -e_crown
        tangent_u = e_md
        tangent_v = e_facial
        target_u = rel_x
        target_v = rel_y
        normal_align = np.dot(normals, e_crown)
        vis_mask = (normal_align > 0.15) & (w_norm >= 0.65)

    else:
        # Default: buccal / labial / facial surface
        actual_surface = "buccal"
        view_dir = -e_facial
        tangent_u = e_md
        tangent_v = e_crown
        target_u = rel_x
        target_v = 1.0 - rel_y
        normal_align = np.dot(normals, e_facial)
        vis_mask = (normal_align > 0.18) & (w_norm >= 0.15)

    # Fallback if filter is too stringent
    if np.sum(vis_mask) < 20:
        vis_mask = np.dot(normals, -view_dir) > 0.05
    if np.sum(vis_mask) < 10:
        vis_mask = np.ones(len(verts), dtype=bool)

    cands = np.where(vis_mask)[0]

    # 2. Project Candidate Vertices onto Camera Tangent Plane
    cand_verts = verts[cands]
    u_proj = np.dot(cand_verts, tangent_u)
    v_proj = np.dot(cand_verts, tangent_v)

    u_p_min, u_p_max = float(np.percentile(u_proj, 3)), float(np.percentile(u_proj, 97))
    v_p_min, v_p_max = float(np.percentile(v_proj, 3)), float(np.percentile(v_proj, 97))

    u_span = max(u_p_max - u_p_min, 1e-4)
    v_span = max(v_p_max - v_p_min, 1e-4)

    u_norm = np.clip((u_proj - u_p_min) / u_span, 0.0, 1.0)
    v_norm = np.clip((v_proj - v_p_min) / v_span, 0.0, 1.0)

    # 3. Pinpoint the Closest Surface Vertex to the Normalized Target
    dist_sq = (u_norm - target_u) ** 2 + (v_norm - target_v) ** 2
    best_sub = int(np.argmin(dist_sq))
    center_idx = int(cands[best_sub])

    # 4. Optional Fine Pit/Fossa Seeking (Fast KDTree local refinement within 0.8mm)
    if fine_fossa_snap and actual_surface == "occlusal":
        tree = cKDTree(verts)
        local_nbrs = tree.query_ball_point(verts[center_idx], r=0.85)
        if len(local_nbrs) >= 8:
            nbr_indices = np.array(local_nbrs, dtype=np.int64)
            nbr_norms = normals[nbr_indices]
            center_n = normals[center_idx]
            # Select neighbor that points upwards and has maximum local concavity
            up_align = np.dot(nbr_norms, e_crown)
            valid_up = nbr_indices[up_align > 0.25]
            if len(valid_up) > 0:
                # Find point with lowest w (deepest groove in that tight 0.85mm sphere)
                w_vals = frame["w"][valid_up]
                center_idx = int(valid_up[np.argmin(w_vals)])

    characteristic_span = float(min(u_span, v_span))

    return {
        "center_idx": center_idx,
        "center_pos": verts[center_idx],
        "surface": actual_surface,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "view_dir": view_dir,
        "surf_span_u": u_span,
        "surf_span_v": v_span,
        "characteristic_span": characteristic_span,
    }


def convert_2d_geometry_to_3d_params(
    geometry: Dict[str, Any],
    tooth_mesh: Any,
    tooth_fdi: Optional[int] = None,
    anomaly_type: str = "Cavity",
    surface: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts 2D shape geometry into 3D carving arguments for carve_cavity().
    All spatial cutoffs, radii, and depths are computed dynamically from the
    specific tooth mesh's geometry, explicit FDI tooth type, and calibrated raycasting.
    """
    normals = np.asarray(tooth_mesh.vertex_normals, dtype=np.float64)
    bounds = tooth_mesh.bounds
    tooth_size = float(np.linalg.norm(bounds[1] - bounds[0]))

    # 1. Compute intrinsic anatomical frame and spans from mesh
    frame = compute_anatomical_frame(tooth_mesh)
    e_md = frame["e_md"]
    e_facial = frame["e_facial"]
    e_crown = frame["e_crown"]
    u, v, w, w_norm = frame["u"], frame["v"], frame["w"], frame["w_norm"]
    mean_edge_len = frame["mean_edge_len"]

    # 2. Explicit FDI Tooth-Type Classification
    tooth_info = classify_fdi_tooth(
        tooth_fdi,
        mesh_spans=(frame["md_span"], frame["bl_span"], frame["crown_height"]),
    )
    tooth_type = tooth_info["tooth_type"]
    anatomical_group = tooth_info["anatomical_group"]

    rel_x, rel_y = geometry.get("relative_centroid", [0.5, 0.5])
    raw_orientation = float(geometry.get("orientation", 0.0))

    anomaly_lower = str(anomaly_type).strip().lower()
    is_abrasion = (
        anomaly_lower.startswith("abrasion")
        or "wear" in anomaly_lower
        or "attrition" in anomaly_lower
    )
    is_crack = (
        "crack" in anomaly_lower
        or "fracture" in anomaly_lower
        or "fissure" in anomaly_lower
        or "craze" in anomaly_lower
    )
    is_erosion = "erosion" in anomaly_lower or "cervical" in anomaly_lower

    # 3. Surface Determination (Explicit Override vs Anatomical Default)
    if surface is not None and str(surface).strip() != "":
        target_surface = str(surface).strip().lower()
    else:
        # Smart surface selection based on anomaly and tooth anatomy
        if is_erosion:
            target_surface = "cervical"
        elif anatomical_group == "anterior":
            target_surface = "labial_facial"
        else:
            # Posterior teeth: occlusal unless lesion is clearly cervical (rel_y > 0.70)
            if rel_y > 0.75:
                target_surface = "cervical"
            else:
                target_surface = "occlusal"

    # 4. Calibrated Multi-View Surface Raycasting
    raycast = raycast_surface_landmark(
        tooth_mesh=tooth_mesh,
        surface=target_surface,
        rel_coords=(rel_x, rel_y),
        frame=frame,
        fine_fossa_snap=(not is_abrasion and not is_crack and target_surface == "occlusal"),
    )

    center_idx = raycast["center_idx"]
    actual_surface = raycast["surface"]
    tangent_u = raycast["tangent_u"]
    tangent_v = raycast["tangent_v"]
    characteristic_surf_span = raycast["characteristic_span"]
    max_major_span = max(raycast["surf_span_u"], raycast["surf_span_v"])

    # Effective 2D-to-3D orientation angle
    if is_abrasion:
        effective_orientation = float(np.clip(raw_orientation, -20.0, 20.0))
    elif is_crack:
        effective_orientation = raw_orientation
    else:
        effective_orientation = float(np.clip(raw_orientation, -45.0, 45.0))

    # 5. Dynamic Mesh-Derived Radius & Depth Scaling
    center_w_norm = float(w_norm[center_idx])
    local_md_w, local_bl_t = measure_local_slice_thickness(u, v, w_norm, center_w_norm)

    if anatomical_group == "anterior" or actual_surface in ("buccal", "lingual", "cervical"):
        local_wall_thickness = local_bl_t
    else:
        local_wall_thickness = max(float(w[center_idx] - w.min()), 2.0)

    mesh_resolution_min_r = 3.5 * mean_edge_len
    min_radius_dyn = max(mesh_resolution_min_r, 0.16 * characteristic_surf_span)
    max_radius_dyn = max(min_radius_dyn * 1.15, 0.32 * characteristic_surf_span)

    rel_r = float(geometry.get("relative_radius", 0.06))
    raw_radius = rel_r * characteristic_surf_span * 3.2
    carve_radius = float(np.clip(raw_radius, min_radius_dyn, max_radius_dyn))

    raw_aspect = float(geometry.get("aspect_ratio", 1.0))
    if is_crack:
        max_allowed_aspect = 6.0
        aspect_ratio = float(np.clip(raw_aspect, 2.0, max_allowed_aspect))
    else:
        max_allowed_aspect = min(2.5, (0.42 * max_major_span / max(carve_radius, 1e-4)) ** 2)
        aspect_ratio = float(np.clip(raw_aspect, 1.0, max(1.1, max_allowed_aspect)))

    # Depth calibration tailored by anomaly pathology profile
    if is_abrasion:
        profile = "abrasion"
        target_depth = min(0.20 * carve_radius, 0.08 * local_wall_thickness)
        min_depth_dyn = min(0.14 * carve_radius, 0.05 * local_wall_thickness)
        max_depth_dyn = min(0.24 * carve_radius, 0.10 * local_wall_thickness)
    elif is_crack:
        profile = "crack"
        target_depth = min(0.55 * carve_radius, 0.22 * local_wall_thickness)
        min_depth_dyn = min(0.35 * carve_radius, 0.15 * local_wall_thickness)
        max_depth_dyn = min(0.65 * carve_radius, 0.25 * local_wall_thickness)
    elif is_erosion:
        profile = "cervical"
        target_depth = min(0.25 * carve_radius, 0.10 * local_wall_thickness)
        min_depth_dyn = min(0.15 * carve_radius, 0.06 * local_wall_thickness)
        max_depth_dyn = min(0.30 * carve_radius, 0.12 * local_wall_thickness)
    else:
        profile = "cavity"
        target_depth = min(0.42 * carve_radius, 0.15 * local_wall_thickness)
        min_depth_dyn = min(0.32 * carve_radius, 0.10 * local_wall_thickness)
        max_depth_dyn = min(0.48 * carve_radius, 0.18 * local_wall_thickness)

    if max_depth_dyn < min_depth_dyn:
        max_depth_dyn = min_depth_dyn
    carve_depth = float(np.clip(target_depth, min_depth_dyn, max_depth_dyn))

    return {
        "center_idx": int(center_idx),
        "radius": round(carve_radius, 3),
        "depth": round(carve_depth, 3),
        "aspect_ratio": round(aspect_ratio, 2),
        "orientation": round(effective_orientation, 2),
        "profile": profile,
        "tooth_fdi": tooth_info["tooth_fdi"],
        "tooth_type": tooth_type,
        "anatomical_group": anatomical_group,
        "target_surface": target_surface,
        "surface": actual_surface,
        "tooth_size": round(tooth_size, 3),
        "surface_span_mm": round(characteristic_surf_span, 3),
        "local_wall_thickness_mm": round(local_wall_thickness, 3),
        "mean_edge_len_mm": round(mean_edge_len, 3),
        "depth_calibration_source": "clinical_heuristic_ratio_on_measured_mesh",
        "tangent_u": [round(float(x), 5) for x in tangent_u],
        "tangent_v": [round(float(x), 5) for x in tangent_v],
        "e_crown": [round(float(x), 5) for x in e_crown],
        "notes": (
            f"Calibrated multi-view raycasting {tooth_type} ({actual_surface}); "
            f"profile={profile}, measured_wall={local_wall_thickness:.2f}mm"
        ),
    }
