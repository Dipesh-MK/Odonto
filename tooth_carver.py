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
DATASET_DIR = Path("/Users/mi/.cache/kagglehub/datasets/aracknab69/dental-data-part-1-stl/versions/1")

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
    if mask_2d is not None and isinstance(mask_2d, np.ndarray) and mask_2d.size > 20:
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
        if eff_aspect > 1.15 or "crack" in active_profile:
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
            clamped_d = np.clip(scaled_dists, 0.0, 1.0)
            radial_falloff = (1.0 - clamped_d ** 3) ** 1.5
            influence = normal_gate * radial_falloff
        elif "crack" in active_profile:
            raw_influence = np.clip(1.0 - scaled_dists, 0.0, 1.0)
            influence = normal_gate * np.clip(raw_influence ** 3.5, 0.0, 1.0)
        elif "cervical" in active_profile:
            np.random.seed(42)
            noise = 1.0 + 0.10 * np.sin(diffs[:, 0] * 3.8) * np.cos(diffs[:, 1] * 3.8)
            raw_influence = np.clip(1.0 - scaled_dists, 0.0, 1.0)
            influence = normal_gate * np.clip((raw_influence ** 1.7) * noise, 0.0, 1.0)
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
        cmap="copper",
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
