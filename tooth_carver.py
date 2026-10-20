"""
tooth_carver.py

Integrates validated 2D shape parameters with the 3D dental mesh carving pipeline
across all tooth types (incisors, canines, premolars, molars) and anomaly types
(cavity/caries, abrasion, crack).

Strict adherence to project core principles:
1. Always load meshes with trimesh.load(path, process=False)
2. Never rely on OBJ round-trips for label arrays; save single atomic .npz
3. Use vertex-normal inward displacement with surface-normal alignment gating
   to prevent opposite-face bleed-through on thin anterior teeth
4. Render 3D tooth using PyVista via pv.wrap(trimesh) with dynamic lesion-aligned camera
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import pyvista as pv

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "dental_manifest_part1.csv"
DEFAULT_TOOTH_PATH = BASE_DIR / "tooth_36_isolated.obj"

# In-memory cache of isolated tooth meshes keyed by (patient_id, target_tooth)
_TOOTH_MESH_CACHE = {}


def load_or_extract_tooth_mesh(manifest_row=None, target_tooth=36):
    """
    Loads an isolated tooth mesh for the requested FDI tooth number from the
    Teeth3DS+ manifest. Always uses process=False to preserve vertex indexing.
    Never silently substitutes tooth 36 when a different target_tooth is requested.
    """
    target_tooth = int(target_tooth)

    # 1. If a specific manifest_row is provided, extract from that scan
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

    # 2. Check cache for any already-loaded instance of target_tooth
    for (pid, fdi), cached_mesh in _TOOTH_MESH_CACHE.items():
        if fdi == target_tooth:
            return cached_mesh.copy()

    # 3. Fast path for default tooth 36 if cached file exists
    if target_tooth == 36 and DEFAULT_TOOTH_PATH.exists():
        tooth = trimesh.load(str(DEFAULT_TOOTH_PATH), process=False)
        _TOOTH_MESH_CACHE[("default", 36)] = tooth
        return tooth.copy()

    # 4. Search Teeth3DS+ manifest for a scan containing target_tooth
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

    raise FileNotFoundError(
        f"Could not find or extract target FDI tooth #{target_tooth} in manifest."
    )


def carve_cavity(
    mesh,
    center_idx,
    radius=1.5,
    depth=0.8,
    aspect_ratio=1.0,
    orientation=0.0,
    profile="cavity",
    tangent_u=None
):
    """
    Carves an anomaly onto a 3D tooth mesh via surface-normal-gated inward displacement.

    Key Generalization Features:
    - Surface-Normal Alignment Gate: Vertices whose normals diverge strongly from
      the lesion center normal (dot product <= 0.15, such as the opposing lingual
      surface of thin incisors/canines) receive zero displacement, eliminating
      double-sided bleed-through.
    - Anomaly Profile Support:
        * "cavity": Smooth quadratic concave pit (1 - d)^2
        * "abrasion": Flat-bottomed superficial wear facet (1 - d^3)^1.5 with
          blended planar normal displacement.

    Args:
        mesh: trimesh.Trimesh
        center_idx: int, vertex index of the anomaly center
        radius: float, effective radius in mm
        depth: float, inward displacement depth in mm
        aspect_ratio: float, ratio of major to minor axis (default 1.0)
        orientation: float, orientation angle in degrees
        profile: str, "cavity" or "abrasion"
        tangent_u: Optional 3D unit vector for primary tangent axis

    Returns:
        carved_mesh: trimesh.Trimesh with displaced vertices
        influence: 1D np.ndarray of ground-truth anomaly influence (0.0 to 1.0)
    """
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals
    center = verts[center_idx]
    center_normal = normals[center_idx]
    center_normal = center_normal / (np.linalg.norm(center_normal) + 1e-8)

    diffs = verts - center

    # 1. Surface-Normal Alignment Gate (prevents opposite-wall perforation on thin teeth)
    normal_dots = np.dot(normals, center_normal)
    # Strict zero for vertices facing >81 deg away (dot <= 0.15); smooth ramp up to dot = 0.50
    normal_gate = np.clip((normal_dots - 0.15) / 0.35, 0.0, 1.0)

    # 2. Construct orthonormal tangent plane at center_idx
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

    # Rotate tangent frame by 2D orientation angle
    rad = np.radians(orientation)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    u_rot = cos_a * tu + sin_a * tv
    v_rot = -sin_a * tu + cos_a * tv

    coord_u = np.dot(diffs, u_rot)
    coord_v = np.dot(diffs, v_rot)
    coord_n = np.dot(diffs, center_normal)

    # 3. Compute normalized distance (isotropic or elliptical)
    eff_aspect = max(float(aspect_ratio), 1.0)
    if eff_aspect > 1.15:
        scale_u = np.sqrt(eff_aspect)
        scale_v = 1.0 / scale_u
        scaled_dists = np.sqrt(
            (coord_u / (radius * scale_u)) ** 2 +
            (coord_v / (radius * scale_v)) ** 2 +
            (coord_n / radius) ** 2
        )
    else:
        dists = np.linalg.norm(diffs, axis=1)
        scaled_dists = dists / max(radius, 1e-5)

    # 4. Anomaly-specific falloff and inward displacement
    if str(profile).lower() == "abrasion":
        # Flat-bottomed saucer wear facet with smooth rim bevel
        clamped_d = np.clip(scaled_dists, 0.0, 1.0)
        radial_falloff = (1.0 - clamped_d ** 3) ** 1.5
        influence = normal_gate * radial_falloff
        # Blend vertex normal with center facet normal for a flatter worn facet plane
        effective_normals = 0.5 * normals + 0.5 * center_normal[None, :]
        effective_normals /= (np.linalg.norm(effective_normals, axis=1, keepdims=True) + 1e-8)
        displacement = -effective_normals * (depth * influence[:, None])
    else:
        # Standard parabolic carious cavitation falloff
        radial_falloff = np.clip(1.0 - scaled_dists, 0.0, 1.0) ** 2
        influence = normal_gate * radial_falloff
        displacement = -normals * (depth * influence[:, None])

    carved_verts = verts + displacement

    carved = mesh.copy()
    carved.vertices = carved_verts
    return carved, influence


def save_atomic_case(output_path, carved_mesh, gt_mask, metadata):
    """
    Saves the 3D anomaly case as an atomic .npz file and companion .obj.
    Guarantees vertices, faces, and cavity ground truth remain in lockstep.
    """
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

    # Determine lesion center and outward normal dynamically from gt_mask
    active = gt_mask > 0.05
    if np.any(active):
        weights = gt_mask[active][:, None]
        lesion_center = (verts[active] * weights).sum(axis=0) / weights.sum()
        lesion_normal = (normals[active] * weights).sum(axis=0)
        lesion_normal /= (np.linalg.norm(lesion_normal) + 1e-8)
    else:
        lesion_center = mesh_center
        lesion_normal = np.array([0.0, 0.0, 1.0])

    # Determine crown up vector
    if carve_params and "e_crown" in carve_params:
        e_crown = np.array(carve_params["e_crown"], dtype=np.float64)
    else:
        area_normals = carved_mesh.face_normals * carved_mesh.area_faces[:, None]
        e_crown = area_normals.sum(axis=0)
        e_crown /= (np.linalg.norm(e_crown) + 1e-8)

    # Choose stable camera view_up vector orthogonal to lesion_normal
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
