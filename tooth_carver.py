"""
tooth_carver.py

Integrates validated 2D shape parameters with the existing 3D dental mesh carving pipeline.
Strict adherence to project core principles:
1. Always load meshes with trimesh.load(path, process=False)
2. Never rely on OBJ round-trips for label arrays; save single atomic .npz
3. Use vertex-normal inward displacement with smooth radial falloff
4. Render 3D tooth using PyVista via pv.wrap(trimesh) with zero VTK OBJ parser warnings
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

def load_or_extract_tooth_mesh(manifest_row=None, target_tooth=36):
    """
    Loads an isolated tooth mesh from Teeth3DS+ manifest or cached default.
    Always uses process=False to preserve vertex indexing.
    """
    if manifest_row is not None and "obj_path" in manifest_row and "json_path" in manifest_row:
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
                    return tooth

    # Fallback to default isolated tooth
    if DEFAULT_TOOTH_PATH.exists():
        return trimesh.load(str(DEFAULT_TOOTH_PATH), process=False)
    
    raise FileNotFoundError("Could not load or extract target tooth mesh.")

def carve_cavity(mesh, center_idx, radius=1.5, depth=0.8, aspect_ratio=1.0, orientation=0.0):
    """
    Carves an anomaly onto a 3D tooth mesh via vertex-normal inward displacement.
    Preserves exact behavior of the existing carve_cavity() pipeline,
    with directional elliptical falloff support for elongated anomalies (cracks/fissures).

    Args:
        mesh: trimesh.Trimesh
        center_idx: int, vertex index of the anomaly center
        radius: float, effective radius in mm
        depth: float, inward displacement depth in mm
        aspect_ratio: float, ratio of major to minor axis (default 1.0 for circular)
        orientation: float, orientation angle in degrees

    Returns:
        carved_mesh: trimesh.Trimesh with displaced vertices
        influence: 1D np.ndarray of ground-truth cavity influence (0.0 to 1.0)
    """
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals
    center = verts[center_idx]
    normal = normals[center_idx]

    diffs = verts - center

    if aspect_ratio > 1.25:
        # Construct an orthonormal tangent plane at the anomaly center
        # to apply anisotropic elliptical falloff matching 2D aspect ratio
        ref_vec = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
        tangent_u = np.cross(normal, ref_vec)
        tangent_u = tangent_u / (np.linalg.norm(tangent_u) + 1e-8)
        tangent_v = np.cross(normal, tangent_u)

        # Rotate tangent frame by orientation angle
        rad = np.radians(orientation)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        u_rot = cos_a * tangent_u + sin_a * tangent_v
        v_rot = -sin_a * tangent_u + cos_a * tangent_v

        coord_u = np.dot(diffs, u_rot)
        coord_v = np.dot(diffs, v_rot)
        coord_n = np.dot(diffs, normal)

        # Scaled elliptical distance
        # Major axis = radius * sqrt(aspect_ratio), Minor axis = radius / sqrt(aspect_ratio)
        scale_u = math_sqrt = np.sqrt(aspect_ratio)
        scale_v = 1.0 / scale_u
        scaled_dists = np.sqrt((coord_u / (radius * scale_u)) ** 2 + 
                               (coord_v / (radius * scale_v)) ** 2 + 
                               (coord_n / radius) ** 2)
        influence = np.clip(1.0 - scaled_dists, 0.0, 1.0) ** 2
    else:
        # Standard isotropic radial falloff (exact original formulation)
        dists = np.linalg.norm(diffs, axis=1)
        influence = np.clip(1.0 - (dists / radius), 0.0, 1.0) ** 2

    # Inward displacement along vertex normals
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

    # Atomic .npz package
    np.savez_compressed(
        npz_path,
        vertices=carved_mesh.vertices,
        faces=carved_mesh.faces,
        gt_mask=gt_mask,
        metadata=json.dumps(metadata)
    )

    # Export .obj for 3D viewers
    carved_mesh.export(str(obj_path))
    return str(npz_path), str(obj_path)

def render_carved_tooth_pyvista(carved_mesh, gt_mask, output_img_path, off_screen=True):
    """
    Renders 3D carved tooth mesh with cavity highlighted via PyVista off-screen.
    Wraps trimesh directly to avoid VTK OBJ reader warnings.
    """
    pv.set_plot_theme('document')
    plotter = pv.Plotter(off_screen=off_screen, window_size=[1024, 768])
    
    # Wrap trimesh safely into PyVista PolyData
    pv_mesh = pv.wrap(carved_mesh)
    pv_mesh["cavity_influence"] = gt_mask

    # Color enamel base tooth white, cavity influence deep crimson/red
    plotter.add_mesh(
        pv_mesh,
        scalars="cavity_influence",
        cmap="YlOrRd",
        clim=[0.0, 1.0],
        smooth_shading=True,
        show_scalar_bar=False
    )
    
    # Position camera above occlusal surface looking down and slightly tilted
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    plotter.screenshot(str(output_img_path))
    plotter.close()
    return str(output_img_path)
