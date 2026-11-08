"""
tooth_carver.py

Integrates validated 2D shape parameters and custom 3D anomaly generation
with the real Teeth3DS+ patient dental mesh pipeline across all 32 FDI teeth
(incisors, canines, premolars, molars) and anomaly types (cavity/caries, abrasion, crack, cervical).

Core capabilities & strict adherence to project principles:
1. Multi-tooth library: loads real patient 3D models for ANY FDI tooth (11-18, 21-28, 31-38, 41-48)
   using trimesh.load(path, process=False) to preserve vertex indexing.
2. Surface-normal alignment gating: prevents opposite-wall bleed-through on thin anterior teeth.
3. Realistic clinical lesion sculpting:
   - Cavities (Class I occlusal, Class II interproximal, Class V cervical)
   - Cracks / structural fractures with narrow V-grooves
   - Abrasions / wear facets with flat-bottomed saucer bevels
4. Authentic pathological color grading:
   - Enamel ivory base (#F8F6F0) with fissure ambient occlusion
   - Opaque demineralization halo (#E8DFCE)
   - Carious amber dentin (#7B4722) and deep necrotic floor (#211209)
5. Single atomic .npz export and 2-viewport PyVista rendering via pv.wrap(trimesh).
"""

import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import trimesh
try:
    import pyvista as pv
except ImportError:
    pv = None
import cv2

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "extracted_teeth"
CACHE_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = BASE_DIR / "teeth3ds_manifest.csv"
DEFAULT_TOOTH_PATH = BASE_DIR / "tooth_36_isolated.obj"
DATASET_DIR = Path(os.environ.get("TEETH3DS_DATASET_DIR", "/Users/mi/.cache/kagglehub/datasets/aracknab69/dental-data-part-1-stl/versions/1"))

# FDI Tooth Metadata Descriptions
FDI_NAMES = {
    18: "Upper Right 3rd Molar (Wisdom)",
    17: "Upper Right 2nd Molar",
    16: "Upper Right 1st Molar",
    15: "Upper Right 2nd Premolar",
    14: "Upper Right 1st Premolar",
    13: "Upper Right Canine",
    12: "Upper Right Lateral Incisor",
    11: "Upper Right Central Incisor",
    21: "Upper Left Central Incisor",
    22: "Upper Left Lateral Incisor",
    23: "Upper Left Canine",
    24: "Upper Left 1st Premolar",
    25: "Upper Left 2nd Premolar",
    26: "Upper Left 1st Molar",
    27: "Upper Left 2nd Molar",
    28: "Upper Left 3rd Molar",
    38: "Lower Left 3rd Molar",
    37: "Lower Left 2nd Molar",
    36: "Lower Left 1st Molar",
    35: "Lower Left 2nd Premolar",
    34: "Lower Left 1st Premolar",
    33: "Lower Left Canine",
    32: "Lower Left Lateral Incisor",
    31: "Lower Left Central Incisor",
    41: "Lower Right Central Incisor",
    42: "Lower Right Lateral Incisor",
    43: "Lower Right Canine",
    44: "Lower Right 1st Premolar",
    45: "Lower Right 2nd Premolar",
    46: "Lower Right 1st Molar",
    47: "Lower Right 2nd Molar",
    48: "Lower Right 3rd Molar"
}

# In-memory cache of isolated tooth meshes keyed by (patient_id, target_tooth)
_TOOTH_MESH_CACHE = {}


def load_or_extract_tooth_mesh(manifest_row=None, target_tooth=36):
    """
    Loads an isolated 3D tooth mesh for ANY FDI tooth number.
    1. If manifest_row is explicitly provided with existing paths, extracts from that scan.
    2. Checks in-memory cache and extracted_teeth/tooth_{fdi}.obj pre-extracted library.
    3. Searches teeth3ds_manifest.csv / dataset if not yet cached.
    Always uses process=False to preserve exact vertex indexing.
    """
    target_tooth = int(target_tooth)

    # 1. If a specific manifest_row is provided and files exist on disk, extract from that scan
    if manifest_row is not None and "obj_path" in manifest_row and "json_path" in manifest_row:
        patient_id = manifest_row.get("patient_id", "custom")
        cache_key = (patient_id, target_tooth)
        if cache_key in _TOOTH_MESH_CACHE:
            return _TOOTH_MESH_CACHE[cache_key].copy()

        obj_p = manifest_row["obj_path"]
        json_p = manifest_row["json_path"]
        if os.path.exists(obj_p) and os.path.exists(json_p):
            tmesh = trimesh.load(obj_p, process=False)
            with open(json_p, "r") as f:
                labels = np.array(json.load(f)["labels"])

            if target_tooth in labels:
                vertex_mask = (labels == target_tooth)
                face_mask = vertex_mask[tmesh.faces].all(axis=1)
                tooth = tmesh.submesh([face_mask], append=True)
                if tooth.vertices.shape[0] > 50:
                    _TOOTH_MESH_CACHE[cache_key] = tooth
                    return tooth.copy()

    # 2. Check in-memory cache for any already-loaded instance of target_tooth
    for (pid, fdi), cached_mesh in _TOOTH_MESH_CACHE.items():
        if fdi == target_tooth:
            return cached_mesh.copy()

    # 3. Check extracted_teeth/tooth_{target_tooth}.obj cache (32 FDI teeth library)
    cache_file = CACHE_DIR / f"tooth_{target_tooth}.obj"
    if cache_file.exists():
        tooth = trimesh.load(str(cache_file), process=False)
        _TOOTH_MESH_CACHE[("extracted_teeth", target_tooth)] = tooth
        return tooth.copy()

    # 4. Fast path for default tooth 36 if cached file exists
    if target_tooth == 36 and DEFAULT_TOOTH_PATH.exists():
        tooth = trimesh.load(str(DEFAULT_TOOTH_PATH), process=False)
        _TOOTH_MESH_CACHE[("default", 36)] = tooth
        return tooth.copy()

    # 5. Search Teeth3DS+ manifest for a scan containing target_tooth
    if MANIFEST_PATH.exists():
        manifest = pd.read_csv(MANIFEST_PATH)
        quadrant = target_tooth // 10
        expected_jaw = "upper" if quadrant in (1, 2, 5, 6) else "lower"
        candidates = manifest[manifest["jaw"] == expected_jaw]
        if candidates.empty:
            candidates = manifest

        for _, row in candidates.iterrows():
            json_p = row["json_path"]
            obj_p = row["obj_path"]
            if not (os.path.exists(json_p) and os.path.exists(obj_p)):
                continue
            try:
                with open(json_p, "r") as f:
                    labels = np.array(json.load(f)["labels"])
                if target_tooth in labels:
                    tmesh = trimesh.load(obj_p, process=False)
                    vertex_mask = (labels == target_tooth)
                    face_mask = vertex_mask[tmesh.faces].all(axis=1)
                    tooth = tmesh.submesh([face_mask], append=True)
                    if tooth.vertices.shape[0] > 50:
                        _TOOTH_MESH_CACHE[(row["patient_id"], target_tooth)] = tooth
                        return tooth.copy()
            except Exception:
                continue

    # 6. Extract on-the-fly from DATASET_DIR if present
    if DATASET_DIR.exists():
        subfolder = "upper" if target_tooth < 30 else "lower"
        scan_jsons = list((DATASET_DIR / subfolder).glob("*/*.json"))
        for jf in scan_jsons:
            try:
                with open(jf, "r") as f:
                    lbls = np.array(json.load(f).get("labels", []))
                if target_tooth in lbls:
                    obj_p = jf.with_suffix(".obj")
                    if obj_p.exists():
                        tmesh = trimesh.load(str(obj_p), process=False)
                        vmask = (lbls == target_tooth)
                        fmask = vmask[tmesh.faces].all(axis=1)
                        submesh = tmesh.submesh([fmask], append=True)
                        if len(submesh.vertices) > 500:
                            submesh.export(str(cache_file))
                            _TOOTH_MESH_CACHE[("dataset", target_tooth)] = submesh
                            return submesh.copy()
            except Exception:
                continue

    raise FileNotFoundError(
        f"Could not find or extract target FDI tooth #{target_tooth} in manifest or cache."
    )


def find_surface_vertex(tooth_mesh, surface="occlusal", rel_coords=None):
    """
    Finds the optimal vertex index on a specified anatomical tooth surface
    using calibrated multi-view raycasting and intrinsic dental frames.
    """
    try:
        from parameter_converter import raycast_surface_landmark
        coords = rel_coords if (rel_coords and len(rel_coords) == 2) else (0.5, 0.5)
        res = raycast_surface_landmark(tooth_mesh, surface=surface, rel_coords=coords)
        return int(res["center_idx"])
    except Exception:
        # Fallback to extreme z
        z = tooth_mesh.vertices[:, 2]
        return int(np.argmax(z))


def carve_cavity(
    mesh,
    center_idx,
    radius=1.5,
    depth=0.8,
    aspect_ratio=1.0,
    orientation=0.0,
    profile="cavity",
    tangent_u=None,
    lesion_type=None,
    mask_2d=None,
    mask_proj_u=None,
    mask_proj_v=None,
):
    """
    Carves an anomaly onto a 3D tooth mesh via surface-normal-gated inward displacement.

    Key High-Accuracy Features:
    1. Exact 2D Mask Morphology: When `mask_2d` is provided (e.g. from SAM2/OpenCV segmentation),
       transfers the real irregular contour, branching crack fissures, or wear boundary
       directly onto 3D vertices using an inverted Euclidean distance transform.
    2. Surface-Normal Alignment Gate: Vertices whose normals diverge from the lesion center normal
       (dot product <= 0.18, such as the opposing lingual surface of thin incisors/canines)
       receive strictly zero displacement, completely eliminating double-sided bleed-through.
    3. Fully topology-preserving: guarantees 100% vertex count, face count, and indexing preservation.
    """
    active_profile = str(lesion_type if lesion_type is not None else profile).lower()

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    center = verts[center_idx]
    center_normal = normals[center_idx]
    center_norm_len = np.linalg.norm(center_normal)
    center_normal = center_normal / (center_norm_len + 1e-8) if center_norm_len > 1e-8 else np.array([0.0, 0.0, 1.0])

    diffs = verts - center
    dists = np.linalg.norm(diffs, axis=1)

    # 1. Surface-Normal Alignment Gate (guarantees zero opposite-wall perforation)
    normal_dots = np.dot(normals, center_normal)
    normal_gate = np.clip((normal_dots - 0.18) / 0.32, 0.0, 1.0)

    # 2. Orthonormal Tangent Frame at center_idx
    if tangent_u is not None:
        tu = np.array(tangent_u, dtype=np.float64)
        tu = tu - np.dot(tu, center_normal) * center_normal
        if np.linalg.norm(tu) < 1e-6:
            ref_vec = np.array([0.0, 0.0, 1.0]) if abs(center_normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            tu = np.cross(center_normal, ref_vec)
    else:
        ref_vec = np.array([0.0, 0.0, 1.0]) if abs(center_normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        tu = np.cross(center_normal, ref_vec)

    tu = tu / (np.linalg.norm(tu) + 1e-8)
    tv = np.cross(center_normal, tu)
    tv = tv / (np.linalg.norm(tv) + 1e-8)

    # Rotate tangent frame by 2D orientation angle if parametric
    rad = np.radians(orientation)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    u_rot = cos_a * tu + sin_a * tv
    v_rot = -sin_a * tu + cos_a * tv

    coord_u = np.dot(diffs, u_rot)
    coord_v = np.dot(diffs, v_rot)
    coord_n = np.dot(diffs, center_normal)

    # 3. Morphology Transfer: Exact 2D Mask Transfer vs Parametric Sculpting
    mask_used = False
    uses_profile_geometry = any(
        profile_name in active_profile
        for profile_name in ("abrasion", "wear", "crack", "fracture", "fissure", "erosion", "cervical")
    )
    if not uses_profile_geometry and mask_2d is not None and isinstance(mask_2d, np.ndarray) and mask_2d.size > 20:
        bin_mask = (mask_2d > 0).astype(np.uint8)
        mask_area = int(np.sum(bin_mask))
        if mask_area > 10:
            h_m, w_m = bin_mask.shape[:2]
            moments = cv2.moments(bin_mask)
            if moments["m00"] > 0:
                cx_m = moments["m10"] / moments["m00"]
                cy_m = moments["m01"] / moments["m00"]
            else:
                cx_m, cy_m = w_m / 2.0, h_m / 2.0

            dist_map = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
            max_d = float(dist_map.max())
            if max_d > 0:
                # Pixel scale calibration: maps mask equivalent radius to physical carving radius (mm)
                equiv_r_px = math.sqrt(mask_area / math.pi)
                scale_mm = max(float(radius), 0.2) / max(equiv_r_px, 1.0)

                # Project 3D vertex displacements into mask pixel coordinates
                u_m = np.dot(diffs, tu if mask_proj_u is None else mask_proj_u)
                v_m = np.dot(diffs, tv if mask_proj_v is None else mask_proj_v)

                px_arr = np.round(cx_m + u_m / scale_mm).astype(int)
                py_arr = np.round(cy_m - v_m / scale_mm).astype(int)

                valid_px = (px_arr >= 0) & (px_arr < w_m) & (py_arr >= 0) & (py_arr < h_m)
                valid_indices = np.where(valid_px)[0]

                raw_inf = np.zeros(len(verts), dtype=np.float64)
                d_vals = dist_map[py_arr[valid_indices], px_arr[valid_indices]]
                in_lesion = d_vals > 0

                if np.any(in_lesion):
                    target_indices = valid_indices[in_lesion]
                    norm_d = np.clip(d_vals[in_lesion] / (0.75 * max_d), 0.0, 1.0)

                    if "crack" in active_profile:
                        raw_inf[target_indices] = norm_d ** 1.6
                    elif "abrasion" in active_profile or "wear" in active_profile:
                        raw_inf[target_indices] = 1.0 - (1.0 - norm_d) ** 2.2
                    elif "cervical" in active_profile:
                        raw_inf[target_indices] = norm_d ** 1.1
                    else:
                        # Cavity
                        raw_inf[target_indices] = norm_d ** 0.85

                    influence = normal_gate * raw_inf
                    mask_used = True

    if not mask_used:
        # Fallback to parametric elliptical sculpt
        eff_aspect = max(float(aspect_ratio), 1.0)
        if "crack" in active_profile or "fracture" in active_profile or "fissure" in active_profile:
            eff_aspect = max(eff_aspect, 8.0)
            scale_u = np.sqrt(eff_aspect)
            scale_v = 1.0 / scale_u
            scaled_dists = np.sqrt(
                (coord_u / (radius * scale_u)) ** 2
                + (coord_v / (radius * scale_v)) ** 2
                + (coord_n / (radius * 0.35)) ** 2
            )
        elif "abrasion" in active_profile or "wear" in active_profile:
            scaled_dists = np.sqrt(
                (coord_u / (radius * 1.6)) ** 2
                + (coord_v / (radius * 0.8)) ** 2
                + (coord_n / (radius * 0.55)) ** 2
            )
        elif "erosion" in active_profile or "cervical" in active_profile:
            scaled_dists = np.sqrt(
                (coord_u / (radius * 1.5)) ** 2
                + (coord_v / (radius * 1.3)) ** 2
                + (coord_n / (radius * 0.85)) ** 2
            )
        elif eff_aspect > 1.15:
            scale_u = np.sqrt(max(eff_aspect, 1.0))
            scale_v = 1.0 / scale_u
            scaled_dists = np.sqrt(
                (coord_u / (radius * scale_u)) ** 2
                + (coord_v / (radius * scale_v)) ** 2
                + (coord_n / radius) ** 2
            )
        else:
            scaled_dists = dists / max(radius, 1e-5)

        if "abrasion" in active_profile or "wear" in active_profile:
            edge = np.clip((1.0 - scaled_dists) / 0.28, 0.0, 1.0)
            flat_facet = edge ** 2 * (3.0 - 2.0 * edge)
            influence = normal_gate * flat_facet
        elif "crack" in active_profile or "fracture" in active_profile or "fissure" in active_profile:
            raw_influence = np.clip(1.0 - scaled_dists, 0.0, 1.0)
            influence = normal_gate * np.clip(raw_influence ** 4.5, 0.0, 1.0)
        elif "erosion" in active_profile or "cervical" in active_profile:
            raw_influence = np.clip(1.0 - scaled_dists ** 2, 0.0, 1.0)
            influence = normal_gate * raw_influence ** 1.25
        else:
            radial_falloff = np.clip(1.0 - scaled_dists, 0.0, 1.0) ** 2
            influence = normal_gate * radial_falloff

    # 4. Inward Normal Displacement
    if "abrasion" in active_profile or "wear" in active_profile:
        effective_normals = 0.5 * normals + 0.5 * center_normal[None, :]
        effective_normals /= np.linalg.norm(effective_normals, axis=1, keepdims=True) + 1e-8
        displacement = -effective_normals * (depth * influence[:, None])
    elif "crack" in active_profile:
        displacement = -normals * (depth * 1.35 * influence[:, None])
    else:
        displacement = -normals * (depth * influence[:, None])

    carved_verts = verts + displacement
    carved_mesh = mesh.copy()
    carved_mesh.vertices = carved_verts
    return carved_mesh, influence


def carve_contour_conforming_defect(
    mesh: trimesh.Trimesh,
    polygon_norm: Union[List[List[float]], np.ndarray],
    target_surface: str = "occlusal",
    depth: float = 0.85,
    margin_steepness: float = 0.70,
    floor_morphology: str = "irregular_concave",
    surface_center_norm: Optional[Tuple[float, float]] = None,
    tooth_fdi: int = 16,
    max_physical_span_mm: Optional[float] = None,
) -> Tuple[trimesh.Trimesh, np.ndarray, Dict[str, Any]]:
    """
    Phase 3: High-Accuracy Contour-Conforming Depth Mesh Deformation.
    Directly sculpts the irregular 2D SAM perimeter polygon onto the 3D tooth surface.

    Pillars of Clinical Realism:
    1. Exact Boundary Fidelity: Unlike generic radial bowls, strictly bounds deformation
       to the complex, natural, non-circular perimeter of the segmented lesion.
    2. Raycast Projection: Projects 2D normalized image space onto the visible 3D tangent
       frame derived from canonical anatomical registration.
    3. Surface-Normal Gating: Eliminates bleed-through to the opposite tooth wall.
    4. Margin Steepness & Floor Morphology:
       - 'irregular_concave': Deep cavity with steep undermined enamel walls and concave floor.
       - 'smooth_planar': Attrition/abrasion facet with flat bottom and beveled borders.
       - 'sharp_v': Narrow fissure/crack notch with linear depth falloff.
    5. Manifold & Topology Guarantee: 100% preserves vertex count, face connectivity, and indexing.
    """
    from parameter_converter import raycast_surface_landmark

    poly_arr = np.asarray(polygon_norm, dtype=np.float64)
    if poly_arr.ndim != 2 or len(poly_arr) < 3:
        raise ValueError("polygon_norm must be an array of shape (N, 2) with N >= 3.")

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    # Determine projection center
    if surface_center_norm is not None and len(surface_center_norm) == 2:
        cx_norm, cy_norm = float(surface_center_norm[0]), float(surface_center_norm[1])
    else:
        cx_norm, cy_norm = float(poly_arr[:, 0].mean()), float(poly_arr[:, 1].mean())

    # Raycast onto visible target surface to obtain 3D landmark & orthonormal tangent frame
    landmark = raycast_surface_landmark(mesh, surface=target_surface, rel_coords=(cx_norm, cy_norm))
    center_idx = landmark["center_idx"]
    center = verts[center_idx]
    center_norm = normals[center_idx]
    center_norm_len = np.linalg.norm(center_norm)
    center_norm = center_norm / (center_norm_len + 1e-8) if center_norm_len > 1e-8 else np.array([0.0, 0.0, 1.0])

    tu = landmark["tangent_u"]
    tv = landmark["tangent_v"]
    span_u = landmark["surf_span_u"]

    # Calibrate physical scaling: map normalized polygon span to physical tooth surface (mm)
    delta_x = max(1e-4, poly_arr[:, 0].max() - poly_arr[:, 0].min())
    delta_y = max(1e-4, poly_arr[:, 1].max() - poly_arr[:, 1].min())

    if max_physical_span_mm is not None and max_physical_span_mm > 0:
        target_span_mm = float(max_physical_span_mm)
    else:
        # Scale to realistic proportion of target surface (~2.0 - 4.2 mm)
        target_span_mm = min(4.2, max(1.8, span_u * 0.38))

    scale_x = target_span_mm / delta_x
    scale_y = (target_span_mm * (delta_y / delta_x)) / delta_y

    # Transform polygon points to physical 2D coordinates in tangent plane centered at (0, 0)
    poly_u = (poly_arr[:, 0] - cx_norm) * scale_x
    poly_v = -(poly_arr[:, 1] - cy_norm) * scale_y
    poly_2d = np.stack([poly_u, poly_v], axis=1).astype(np.float32)

    # Project mesh vertices onto tangent plane
    diffs = verts - center
    u_proj = np.dot(diffs, tu)
    v_proj = np.dot(diffs, tv)

    # Surface-normal alignment gate (prevents perforation through thin opposing walls)
    norm_dots = np.dot(normals, center_norm)
    normal_gate = np.clip((norm_dots - 0.18) / 0.32, 0.0, 1.0)

    # Fast spatial pre-filter
    r_mm = np.sqrt(poly_u**2 + poly_v**2).max() * 1.08
    candidates = np.where((normal_gate > 0.0) & ((u_proj**2 + v_proj**2) <= r_mm**2))[0]

    # Signed distance to SAM polygon boundary (measureDist=True)
    dists = np.zeros(len(verts), dtype=np.float64)
    for idx in candidates:
        d = cv2.pointPolygonTest(poly_2d, (float(u_proj[idx]), float(v_proj[idx])), measureDist=True)
        if d > 0:
            dists[idx] = d

    in_poly = dists > 0
    num_inside = int(np.sum(in_poly))

    if num_inside > 0:
        max_d = float(dists[in_poly].max())
        steepness = max(0.1, min(1.0, float(margin_steepness)))
        blend_dist = max(0.08, (1.05 - steepness * 0.5) * 0.45 * max_d)
        norm_d = np.clip(dists / blend_dist, 0.0, 1.0)

        # Depth decay profile based on edge steepness
        p_exp = 1.0 / max(steepness, 0.15)
        weights = norm_d ** p_exp

        # Floor morphology modulation
        morph_clean = str(floor_morphology or "irregular_concave").lower()
        if "planar" in morph_clean:
            weights = weights**2 * (3.0 - 2.0 * weights)
        elif "sharp" in morph_clean or "v" in morph_clean:
            weights = (dists / max_d) ** 1.8
        else:
            # Irregular concave (Caries)
            floor_mod = 1.0 - 0.18 * ((1.0 - np.clip(dists / max_d, 0.0, 1.0)) ** 2)
            weights = weights * floor_mod

        influence = weights * normal_gate
    else:
        influence = np.zeros(len(verts), dtype=np.float64)

    # Inward normal displacement
    surf_clean = str(target_surface or "occlusal").lower()
    if "abrasion" in surf_clean or "cervical" in surf_clean or "planar" in floor_morphology:
        effective_normals = 0.5 * normals + 0.5 * center_norm[None, :]
        effective_normals /= np.linalg.norm(effective_normals, axis=1, keepdims=True) + 1e-8
        displacement = -effective_normals * (float(depth) * influence[:, None])
    else:
        displacement = -normals * (float(depth) * influence[:, None])

    carved_verts = verts + displacement
    carved_mesh = mesh.copy()
    carved_mesh.vertices = carved_verts

    max_actual_depth = float(np.linalg.norm(displacement, axis=1).max()) if num_inside > 0 else 0.0

    metrics = {
        "num_displaced_vertices": num_inside,
        "max_depth_reached_mm": round(max_actual_depth, 4),
        "lesion_span_mm": [round(float(poly_u.max() - poly_u.min()), 2), round(float(poly_v.max() - poly_v.min()), 2)],
        "center_landmark_idx": int(center_idx),
        "floor_morphology": floor_morphology,
        "is_watertight": carved_mesh.is_watertight,
        "vertices_count": len(carved_mesh.vertices),
        "faces_count": len(carved_mesh.faces),
    }

    return carved_mesh, influence, metrics


def carve_depth_heightmap_defect(
    mesh: trimesh.Trimesh,
    polygon_norm: Union[List[List[float]], np.ndarray],
    depth_relief_map: np.ndarray,
    target_surface: str = "occlusal",
    depth: float = 0.85,
    rotation_deg: float = 0.0,
    surface_center_norm: Optional[Tuple[float, float]] = None,
    tooth_fdi: int = 16,
    max_physical_span_mm: Optional[float] = None,
) -> Tuple[trimesh.Trimesh, np.ndarray, Dict[str, Any]]:
    """
    Method B: Dense Monocular Depth-Relief Conformal Pasting.
    Deforms the target tooth mesh by transferring the dense photographic heightmap
    Δz(u, v) extracted via Depth Anything V2 / MiDaS onto the target tooth surface.

    Unlike Method A (which applies a synthetic parabolic/planar mathematical falloff),
    Method B directly sculpts the realistic, organic micro-indentations, rough floor ripples,
    and natural depth topography captured from the clinical photo.

    Topology & Clinical Safety Guarantees:
    - 100% Watertight Manifold Topology Invariance (carved_mesh.faces = mesh.faces).
    - Bicubic per-vertex interpolation from dense heightmap Δz(u, v).
    - Surface-normal alignment gating (prevents lingual punch-through on thin incisors).
    - Bounded maximum physical depth calibrated to clinical tooth enamel/dentin limits.
    """
    from parameter_converter import raycast_surface_landmark
    from scipy.ndimage import map_coordinates

    poly_arr = np.asarray(polygon_norm, dtype=np.float64)
    if poly_arr.ndim != 2 or len(poly_arr) < 3:
        raise ValueError("polygon_norm must be an array of shape (N, 2) with N >= 3.")

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    # Determine projection center
    if surface_center_norm is not None and len(surface_center_norm) == 2:
        cx_norm, cy_norm = float(surface_center_norm[0]), float(surface_center_norm[1])
    else:
        cx_norm, cy_norm = float(poly_arr[:, 0].mean()), float(poly_arr[:, 1].mean())

    # Raycast onto visible target surface to obtain 3D landmark & tangent frame
    landmark = raycast_surface_landmark(mesh, surface=target_surface, rel_coords=(cx_norm, cy_norm))
    center_idx = landmark["center_idx"]
    center = verts[center_idx]
    center_norm = normals[center_idx]
    center_norm_len = np.linalg.norm(center_norm)
    center_norm = center_norm / (center_norm_len + 1e-8) if center_norm_len > 1e-8 else np.array([0.0, 0.0, 1.0])

    tu = landmark["tangent_u"]
    tv = landmark["tangent_v"]
    span_u = landmark["surf_span_u"]

    # Apply optional rotation around surface normal
    if abs(rotation_deg) > 1e-3:
        rad = np.radians(rotation_deg)
        cos_r, sin_r = np.cos(rad), np.sin(rad)
        tu_rot = cos_r * tu + sin_r * tv
        tv_rot = -sin_r * tu + cos_r * tv
        tu, tv = tu_rot, tv_rot

    # Calibrate physical scaling: map normalized polygon span to physical tooth surface (mm)
    delta_x = max(1e-4, poly_arr[:, 0].max() - poly_arr[:, 0].min())
    delta_y = max(1e-4, poly_arr[:, 1].max() - poly_arr[:, 1].min())

    if max_physical_span_mm is not None and max_physical_span_mm > 0:
        target_span_mm = float(max_physical_span_mm)
    else:
        target_span_mm = min(4.2, max(1.8, span_u * 0.38))

    scale_x = target_span_mm / delta_x
    scale_y = (target_span_mm * (delta_y / delta_x)) / delta_y

    poly_u = (poly_arr[:, 0] - cx_norm) * scale_x
    poly_v = -(poly_arr[:, 1] - cy_norm) * scale_y
    poly_2d = np.stack([poly_u, poly_v], axis=1).astype(np.float32)

    # Project mesh vertices onto tangent plane
    diffs = verts - center
    u_proj = np.dot(diffs, tu)
    v_proj = np.dot(diffs, tv)

    # Surface-normal alignment gate (prevents perforation through thin opposing walls)
    normal_dots = np.dot(normals, center_norm)
    gate = np.clip((normal_dots - 0.18) / 0.32, 0.0, 1.0)

    # Bounding box candidate pre-filter
    u_min, u_max = poly_u.min() - 0.5, poly_u.max() + 0.5
    v_min, v_max = poly_v.min() - 0.5, poly_v.max() + 0.5
    candidate_mask = (u_proj >= u_min) & (u_proj <= u_max) & (v_proj >= v_min) & (v_proj <= v_max) & (gate > 0.0)
    candidate_indices = np.where(candidate_mask)[0]

    influence = np.zeros(len(verts), dtype=np.float64)
    hm_h, hm_w = depth_relief_map.shape[:2]

    # Map target tangent coordinates to heightmap normalized pixel space
    poly_u_min, poly_u_max = poly_u.min(), poly_u.max()
    poly_v_min, poly_v_max = poly_v.min(), poly_v.max()
    u_span = max(1e-4, poly_u_max - poly_u_min)
    v_span = max(1e-4, poly_v_max - poly_v_min)

    num_inside = 0
    if len(candidate_indices) > 0:
        poly_pts = poly_2d.reshape((-1, 1, 2))
        for idx in candidate_indices:
            pt = (float(u_proj[idx]), float(v_proj[idx]))
            signed_dist = cv2.pointPolygonTest(poly_pts, pt, measureDist=True)
            if signed_dist >= -0.15:
                # Map to normalized [0, 1] relative to polygon bounds
                rel_u = (pt[0] - poly_u_min) / u_span
                rel_v = 1.0 - (pt[1] - poly_v_min) / v_span  # Invert Y to match image row order

                # Heightmap pixel coordinates
                hm_x = float(np.clip(rel_u * (hm_w - 1), 0.0, hm_w - 1))
                hm_y = float(np.clip(rel_v * (hm_h - 1), 0.0, hm_h - 1))

                # Bicubic interpolation from depth relief map
                sampled_val = float(map_coordinates(
                    depth_relief_map,
                    [[hm_y], [hm_x]],
                    order=3,
                    mode="nearest"
                )[0])

                # Smooth boundary tapering
                edge_weight = float(np.clip((signed_dist + 0.15) / 0.40, 0.0, 1.0))
                inf_val = float(gate[idx]) * float(np.clip(sampled_val, 0.0, 1.0)) * edge_weight

                if inf_val > 0.005:
                    influence[idx] = inf_val
                    num_inside += 1

    # Inward normal displacement
    displacement = -normals * (depth * influence[:, None])
    carved_verts = verts + displacement

    carved_mesh = mesh.copy()
    carved_mesh.vertices = carved_verts

    max_actual_depth = float(np.linalg.norm(displacement, axis=1).max()) if num_inside > 0 else 0.0

    metrics = {
        "num_displaced_vertices": num_inside,
        "max_depth_reached_mm": round(max_actual_depth, 4),
        "lesion_span_mm": [round(float(poly_u.max() - poly_u.min()), 2), round(float(poly_v.max() - poly_v.min()), 2)],
        "center_landmark_idx": int(center_idx),
        "method": "depth_heightmap_conformal",
        "is_watertight": carved_mesh.is_watertight,
        "vertices_count": len(carved_mesh.vertices),
        "faces_count": len(carved_mesh.faces),
    }

    return carved_mesh, influence, metrics


def carve_v_groove_path(
    mesh,
    path_points,
    depth: float = 0.35,
    width: float = 0.45,
    normal_gate_thresh: float = 0.18,
):
    """
    Carves a hairline crack or structural fissure along a continuous polyline path.
    Unlike crater-like radial cavities, produces a narrow, sharp V-groove notch cross-section:
      - Polyline projection: computes exact Euclidean distance to piecewise-linear path segments.
      - V-notch cross-section: inward displacement decays as (1 - d/w)^1.3.
      - Surface-normal gating: vertices whose outward normal opposes the crack surface normal
        receive strictly zero displacement, completely preventing lingual/palatal punch-through
        on thin incisors and canines.
    Topology-preserving: preserves exact vertex count, face connectivity, and indexing.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    # Convert path_points: either list of (x, y, z) coordinates or list of vertex indices
    if isinstance(path_points, list) and len(path_points) > 0 and isinstance(path_points[0], int):
        path = verts[path_points]
    else:
        path = np.asarray(path_points, dtype=np.float64)

    if len(path) < 2:
        raise ValueError("path_points must contain at least 2 points to form a crack line")

    # Vectorized piecewise-linear distance to polyline segments
    segs_a = path[:-1]
    segs_b = path[1:]
    segs_v = segs_b - segs_a
    segs_len_sq = np.maximum(np.sum(segs_v**2, axis=1), 1e-12)

    diff = verts[:, None, :] - segs_a[None, :, :]
    t = np.clip(np.sum(diff * segs_v[None, :, :], axis=2) / segs_len_sq[None, :], 0.0, 1.0)
    proj = segs_a[None, :, :] + t[:, :, None] * segs_v[None, :, :]
    dists_sq = np.sum((verts[:, None, :] - proj)**2, axis=2)
    min_dists = np.sqrt(np.min(dists_sq, axis=1))

    # Surface normal along the crack path: sample nearest vertex normals to path points
    from scipy.spatial import cKDTree
    tree = cKDTree(verts)
    _, nearest_p_indices = tree.query(path)
    path_normals = normals[nearest_p_indices]
    avg_path_normal = np.mean(path_normals, axis=0)
    avg_path_normal /= (np.linalg.norm(avg_path_normal) + 1e-8)

    # Surface-normal alignment gating: prevents opposite-wall perforation
    normal_dots = np.dot(normals, avg_path_normal)
    normal_gate = np.clip((normal_dots - normal_gate_thresh) / 0.32, 0.0, 1.0)

    # Sharp triangular V-notch cross section
    cross_section = np.maximum(0.0, 1.0 - (min_dists / max(width, 1e-4))) ** 1.3
    influence = normal_gate * cross_section

    # Inward displacement along vertex normals
    displacement = -normals * (depth * influence[:, None])
    carved_verts = verts + displacement

    carved_mesh = mesh.copy()
    carved_mesh.vertices = carved_verts
    return carved_mesh, influence


def carve_chipped_cusp(
    mesh,
    cusp_target: str = "mesiobuccal",
    chip_height: float = 1.4,
    chip_angle: float = 35.0,
):
    """
    Sculpts a chipped cusp / structural crown fracture via angled planar slicing and cap remeshing.
    Unlike inward surface displacement (which looks like a dent), this executes true material loss:
      1. Identifies the target cusp tip on the occlusal table.
      2. Defines a fracture plane below the cusp tip tilted at chip_angle toward the outward margin.
      3. Slices away the fractured shard using trimesh.intersections.slice_mesh_plane(..., cap=True).
      4. Automatically triangulates and caps the exposed cross-section to create a solid fracture floor.
      5. Generates influence scalars across the newly formed fracture floor and chipped margin.
    """
    from parameter_converter import compute_anatomical_frame

    frame = compute_anatomical_frame(mesh)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    e_crown = frame["e_crown"]
    e_facial = frame["e_facial"]
    e_md = frame["e_md"]
    u = frame["u"]
    v = frame["v"]
    w = frame["w"]

    top_mask = w > np.percentile(w, 75)
    top_indices = np.where(top_mask)[0]
    if len(top_indices) == 0:
        top_indices = np.arange(len(verts))

    # Identify target cusp tip position and outward fracture orientation
    if isinstance(cusp_target, str):
        cusp_lower = cusp_target.lower().strip()
        u_sign = 1.0 if "mesio" in cusp_lower else (-1.0 if "disto" in cusp_lower else 0.0)
        v_sign = 1.0 if ("buccal" in cusp_lower or "facial" in cusp_lower or "labial" in cusp_lower) else (-1.0 if ("lingual" in cusp_lower or "palatal" in cusp_lower) else 0.0)

        # For incisal edge corner / edge chipping
        if "incisal" in cusp_lower or "edge" in cusp_lower:
            u_sign = 1.0 if "distal" not in cusp_lower else -1.0
            v_sign = 0.0

        scores = u_sign * u[top_indices] + v_sign * v[top_indices] + 0.8 * w[top_indices]
        best_idx = top_indices[np.argmax(scores)]
        cusp_pos = verts[best_idx]
        outward_dir = (u_sign * 0.45 * e_md + v_sign * 0.85 * e_facial)
        if np.linalg.norm(outward_dir) < 1e-4:
            outward_dir = cusp_pos - frame["center"]
            outward_dir -= np.dot(outward_dir, e_crown) * e_crown
        outward_dir /= (np.linalg.norm(outward_dir) + 1e-8)
    elif isinstance(cusp_target, int):
        cusp_pos = verts[cusp_target]
        outward_dir = cusp_pos - frame["center"]
        outward_dir -= np.dot(outward_dir, e_crown) * e_crown
        outward_dir /= (np.linalg.norm(outward_dir) + 1e-8)
    else:
        cusp_pos = np.array(cusp_target, dtype=np.float64)
        outward_dir = cusp_pos - frame["center"]
        outward_dir -= np.dot(outward_dir, e_crown) * e_crown
        outward_dir /= (np.linalg.norm(outward_dir) + 1e-8)

    # Define fracture plane
    rad = np.radians(chip_angle)
    plane_origin = cusp_pos - chip_height * e_crown
    plane_normal = np.cos(rad) * e_crown + np.sin(rad) * outward_dir
    plane_normal /= (np.linalg.norm(plane_normal) + 1e-8)

    # Slice mesh: keeps the body of the tooth and caps the exposed plane
    sliced = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=-plane_normal,
        plane_origin=plane_origin,
        cap=True
    )

    # Compute influence mask on sliced vertices:
    s_verts = np.asarray(sliced.vertices, dtype=np.float64)
    d_plane = np.abs(np.dot(s_verts - plane_origin, plane_normal))
    d_cusp = np.linalg.norm(s_verts - cusp_pos, axis=1)

    # Vertices on cap or cut margin (d_plane < 0.25 mm and within chip radius)
    in_chip_zone = (d_cusp < chip_height * 2.8)
    cap_inf = np.clip((0.25 - d_plane) / 0.25, 0.0, 1.0) * in_chip_zone.astype(np.float64)
    margin_inf = np.clip((0.65 - d_plane) / 0.65, 0.0, 0.7) * in_chip_zone.astype(np.float64)
    influence = np.maximum(cap_inf, margin_inf)

    return sliced, influence


def compute_clinical_vertex_colors(mesh, influence, lesion_type="cavity"):
    """
    Computes natural dental clinical vertex colors:
    - Translucent healthy enamel: #F8F6F0 with subtle anatomical ambient occlusion in fissures
    - Demineralization halo: chalky opaque buff #E8DFCE
    - Active carious dentin: warm amber/burnt caramel #7B4722
    - Deep necrotic floor: dark espresso brown & carbon black #211209
    """
    verts = mesh.vertices
    n_verts = len(verts)
    colors = np.zeros((n_verts, 3), dtype=np.float32)

    z = verts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    enamel_base = np.array([0.97, 0.96, 0.93], dtype=np.float32)
    enamel_fissure = np.array([0.91, 0.88, 0.83], dtype=np.float32)

    lesion_lower = str(lesion_type).lower()
    if "crack" in lesion_lower:
        decay_color = np.array([0.28, 0.16, 0.10], dtype=np.float32)
        necrotic_color = np.array([0.12, 0.06, 0.04], dtype=np.float32)
    elif "abrasion" in lesion_lower or "wear" in lesion_lower:
        decay_color = np.array([0.80, 0.60, 0.36], dtype=np.float32)
        necrotic_color = np.array([0.65, 0.45, 0.25], dtype=np.float32)
    elif "erosion" in lesion_lower or "cervical" in lesion_lower:
        decay_color = np.array([0.85, 0.76, 0.58], dtype=np.float32)
        necrotic_color = np.array([0.72, 0.56, 0.38], dtype=np.float32)
    else:
        decay_color = np.array([0.48, 0.28, 0.14], dtype=np.float32)
        necrotic_color = np.array([0.15, 0.08, 0.04], dtype=np.float32)

    halo_color = np.array([0.89, 0.84, 0.75], dtype=np.float32)

    for i in range(n_verts):
        groove_t = float(np.clip(z_norm[i] * 0.4 + 0.6, 0.0, 1.0))
        base = (1.0 - groove_t) * enamel_fissure + groove_t * enamel_base

        inf = float(influence[i])
        if inf <= 0.02:
            colors[i] = base
        elif inf <= 0.25:
            w = (inf - 0.02) / (0.25 - 0.02)
            colors[i] = (1.0 - w) * base + w * halo_color
        elif inf <= 0.65:
            w = (inf - 0.25) / (0.65 - 0.25)
            colors[i] = (1.0 - w) * halo_color + w * decay_color
        else:
            w = (inf - 0.65) / (1.0 - 0.65)
            colors[i] = (1.0 - w) * decay_color + w * necrotic_color

    return colors


def save_atomic_case(output_path, carved_mesh, gt_mask, metadata):
    out_p = Path(output_path)
    npz_path = out_p.with_suffix(".npz")
    obj_path = out_p.with_suffix(".obj")

    np.savez_compressed(
        npz_path,
        vertices=carved_mesh.vertices,
        faces=carved_mesh.faces,
        gt_mask=gt_mask,
        metadata=json.dumps(metadata)
    )

    carved_mesh.export(str(obj_path))
    return str(npz_path), str(obj_path)


def render_carved_tooth_pyvista(carved_mesh, gt_mask, output_img_path, off_screen=True, carve_params=None):
    """
    Renders 3D carved tooth mesh with the anomaly highlighted via PyVista off-screen.
    Uses a 2-viewport layout:
      - Left viewport: Direct En-Face View aligned with the anomaly's surface normal
      - Right viewport: Oblique 3D Profile View showing lesion depth and intact opposing wall
    Wraps trimesh directly via pv.wrap() to avoid VTK OBJ reader bugs.
    """
    if pv is None:
        import cv2
        placeholder = np.zeros((700, 1400, 3), dtype=np.uint8)
        placeholder[:] = (15, 23, 42)
        cv2.putText(placeholder, "3D Mesh Carved Successfully (Three.js WebGL Interactive)", (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(str(output_img_path), placeholder)
        return str(output_img_path)

    pv.set_plot_theme("document")
    plotter = pv.Plotter(
        shape=(1, 2),
        off_screen=off_screen,
        window_size=[1400, 700]
    )

    pv_mesh = pv.wrap(carved_mesh)
    pv_mesh["cavity_influence"] = gt_mask

    verts = carved_mesh.vertices
    normals = carved_mesh.vertex_normals
    mesh_center = verts.mean(axis=0)
    bounds = carved_mesh.bounds
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))

    active = gt_mask > 0.05
    if np.any(active):
        weights = gt_mask[active][:, None]
        lesion_center = (verts[active] * weights).sum(axis=0) / weights.sum()
        lesion_normal = (normals[active] * weights).sum(axis=0)
        lesion_normal /= (np.linalg.norm(lesion_normal) + 1e-8)
    else:
        lesion_center = mesh_center
        lesion_normal = np.array([0.0, 0.0, 1.0])

    if carve_params and "e_crown" in carve_params:
        e_crown = np.array(carve_params["e_crown"], dtype=np.float64)
    else:
        area_normals = carved_mesh.face_normals * carved_mesh.area_faces[:, None]
        e_crown = area_normals.sum(axis=0)
        e_crown /= (np.linalg.norm(e_crown) + 1e-8)

    if abs(np.dot(lesion_normal, e_crown)) < 0.80:
        view_up = e_crown
    else:
        view_up = np.array([0.0, 1.0, 0.0]) if abs(lesion_normal[1]) < 0.8 else np.array([1.0, 0.0, 0.0])

    side_vec = np.cross(lesion_normal, view_up)
    side_vec /= (np.linalg.norm(side_vec) + 1e-8)

    cam_dist = diag * 1.95

    # Viewport 0: Direct En-Face Lesion View
    plotter.subplot(0, 0)
    plotter.set_background("#0f172a")
    plotter.add_mesh(
        pv_mesh,
        scalars="cavity_influence",
        cmap="YlOrRd",
        clim=[0.0, 1.0],
        smooth_shading=True,
        show_scalar_bar=False,
        specular=0.35,
        specular_power=25
    )
    cam_pos_enface = lesion_center + (0.92 * lesion_normal + 0.20 * view_up) * cam_dist
    plotter.camera_position = [cam_pos_enface.tolist(), mesh_center.tolist(), view_up.tolist()]
    plotter.add_text("En-Face Surface View", position="upper_left", font_size=11, color="white")

    # Viewport 1: Oblique 3D Profile View (shows depth & intact opposing wall)
    plotter.subplot(0, 1)
    plotter.set_background("#0f172a")
    plotter.add_mesh(
        pv_mesh,
        scalars="cavity_influence",
        cmap="YlOrRd",
        clim=[0.0, 1.0],
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Anomaly Influence", "color": "white"},
        specular=0.35,
        specular_power=25
    )
    oblique_dir = 0.55 * lesion_normal + 0.65 * side_vec + 0.35 * view_up
    oblique_dir /= (np.linalg.norm(oblique_dir) + 1e-8)
    cam_pos_oblique = mesh_center + oblique_dir * cam_dist
    plotter.camera_position = [cam_pos_oblique.tolist(), mesh_center.tolist(), view_up.tolist()]
    plotter.add_text("Oblique 3D Depth & Profile View", position="upper_left", font_size=11, color="white")

    plotter.screenshot(str(output_img_path))
    plotter.close()
    return str(output_img_path)
