"""
tooth_carver.py

Integrates validated 2D shape parameters and custom 3D anomaly generation
with the real Teeth3DS+ patient dental mesh pipeline.

Core capabilities:
1. Multi-tooth library: loads real patient 3D models for ANY FDI tooth (11-18, 21-28, 31-38, 41-48).
2. Realistic clinical lesion sculpting:
   - Cavities (Class I occlusal, Class II interproximal, Class V cervical) with organic micro-texture,
     steep demineralized margins, and excavated necrotic dentin floor.
   - Cracks / structural fractures with narrow V-grooves.
   - Abrasions / wear facets with dish-like bevels.
3. Authentic pathological color grading:
   - Enamel ivory base with fissure ambient occlusion
   - Opaque demineralization halo
   - Carious amber dentin and deep carbonized necrotic decay (NO artificial bright red blobs).
4. Topology-preserving vertex displacement and atomic .npz export.
"""

import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import trimesh
import pyvista as pv

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "extracted_teeth"
CACHE_DIR.mkdir(exist_ok=True)
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

def load_or_extract_tooth_mesh(manifest_row=None, target_tooth=36):
    """
    Loads an isolated 3D tooth mesh for ANY FDI tooth number.
    Checks extracted_teeth cache first, then scans dataset, with fallback.
    Always uses process=False to preserve exact vertex indexing.
    """
    target_tooth = int(target_tooth)
    cache_file = CACHE_DIR / f"tooth_{target_tooth}.obj"

    # 1. Check extracted_teeth cache
    if cache_file.exists():
        return trimesh.load(str(cache_file), process=False)

    # 2. Extract on-the-fly from Teeth3DS+ dataset
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
                            return submesh
            except Exception:
                continue

    # 3. Fallback to default tooth 36 if available
    if DEFAULT_TOOTH_PATH.exists():
        return trimesh.load(str(DEFAULT_TOOTH_PATH), process=False)

    raise FileNotFoundError(f"Could not load or extract tooth mesh for FDI #{target_tooth}")

def find_surface_vertex(tooth_mesh, surface="occlusal", rel_coords=None):
    """
    Finds the optimal vertex index on a specified anatomical tooth surface.
    Uses rel_coords (2D segmentation centroid normalized to [0,1]) to map
    the anomaly position onto the correct 3D surface region.

    Surfaces:
      - 'occlusal': Chewing table (central groove / pit)
      - 'buccal': Facial / cheek-facing outer surface
      - 'lingual': Palatal / tongue-facing inner surface
      - 'cervical': Near the neck / gumline margin
      - 'interproximal': Mesial / distal contact margin
    """
    verts = tooth_mesh.vertices
    bounds = tooth_mesh.bounds
    center_pos = (bounds[0] + bounds[1]) / 2.0
    z = verts[:, 2]
    x = verts[:, 0]
    y = verts[:, 1]

    def _pick_from_candidates(cands, primary_axes, fallback_pos):
        """
        Given candidate vertex indices and axes to search within,
        use rel_coords to find the best vertex matching the 2D anomaly position.
        primary_axes: list of axis indices to use for spatial matching (e.g. [0,2] = X,Z)
        fallback_pos: position to use if no rel_coords provided
        """
        if len(cands) == 0:
            return int(np.argmax(z))  # absolute fallback
        if rel_coords and len(rel_coords) == 2:
            c_min = verts[cands][:, primary_axes].min(axis=0)
            c_max = verts[cands][:, primary_axes].max(axis=0)
            c_range = c_max - c_min
            c_range[c_range < 1e-6] = 1.0  # prevent division by zero
            target = c_min + np.array(rel_coords) * c_range
            dists = np.linalg.norm(verts[cands][:, primary_axes] - target, axis=1)
            return int(cands[np.argmin(dists)])
        else:
            dists = np.linalg.norm(verts[cands][:, primary_axes] - fallback_pos[primary_axes], axis=1)
            return int(cands[np.argmin(dists)])

    if surface == "occlusal":
        # Top 18% Z of the crown — the chewing table
        z_thresh = np.percentile(z, 82)
        cands = np.where(z >= z_thresh)[0]
        if len(cands) == 0:
            cands = np.array([int(np.argmax(z))])
        return _pick_from_candidates(cands, [0, 1], center_pos)

    elif surface == "buccal":
        # Outer surface: top 20% of Y axis (cheek-facing)
        y_thresh = np.percentile(y, 80)
        cands = np.where(y >= y_thresh)[0]
        if len(cands) == 0:
            cands = np.where(z >= np.median(z))[0]
        # Map rel_coords onto X (mesial-distal) and Z (height) within buccal face
        return _pick_from_candidates(cands, [0, 2], center_pos)

    elif surface == "lingual":
        # Inner surface: bottom 20% of Y axis (tongue-facing)
        y_thresh = np.percentile(y, 20)
        cands = np.where(y <= y_thresh)[0]
        if len(cands) == 0:
            cands = np.where(z >= np.median(z))[0]
        # Map rel_coords onto X (mesial-distal) and Z (height) within lingual face
        return _pick_from_candidates(cands, [0, 2], center_pos)

    elif surface == "cervical":
        # Lower 15-40% Z near the cervical neck / gumline
        cands = np.where((z >= np.percentile(z, 15)) & (z <= np.percentile(z, 40)))[0]
        if len(cands) == 0:
            cands = np.where(z <= np.percentile(z, 30))[0]
        # Map rel_coords onto X (mesial-distal) and Y (buccal-lingual) at cervical height
        return _pick_from_candidates(cands, [0, 1], center_pos)

    elif surface == "interproximal":
        # Side margin: high or low X (mesial or distal contact)
        # Use rel_coords X to decide mesial vs distal
        if rel_coords and len(rel_coords) == 2 and rel_coords[0] < 0.5:
            x_thresh = np.percentile(x, 12)
            cands = np.where(x <= x_thresh)[0]
        else:
            x_thresh = np.percentile(x, 88)
            cands = np.where(x >= x_thresh)[0]
        if len(cands) == 0:
            cands = np.where(z >= np.median(z))[0]
        # Map rel_coords onto Y (buccal-lingual) and Z (height) on the side face
        return _pick_from_candidates(cands, [1, 2], center_pos)

    # Fallback: topmost occlusal point
    return int(np.argmax(z))

def carve_cavity(mesh, center_idx, radius=1.8, depth=1.2, aspect_ratio=1.0, orientation=0.0, lesion_type="cavity"):
    """
    Sculpts an anatomically authentic dental pathology onto the 3D tooth mesh.
    Supports:
      - 'cavity' / 'caries': steep demineralized crater with natural textured floor
      - 'crack': narrow V-notch fissure along the crack angle
      - 'abrasion': shallow, saucer-shaped wear facet
      - 'cervical': wedge-shaped abfraction defect
    """
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals
    center = verts[center_idx]
    normal = normals[center_idx]

    diffs = verts - center
    dists = np.linalg.norm(diffs, axis=1)

    # Anisotropic elliptical coordinates for elongated lesions or cracks
    if aspect_ratio > 1.2 or lesion_type == "crack":
        ref_vec = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
        tangent_u = np.cross(normal, ref_vec)
        tangent_u = tangent_u / (np.linalg.norm(tangent_u) + 1e-8)
        tangent_v = np.cross(normal, tangent_u)

        rad = np.radians(orientation)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        u_rot = cos_a * tangent_u + sin_a * tangent_v
        v_rot = -sin_a * tangent_u + cos_a * tangent_v

        coord_u = np.dot(diffs, u_rot)
        coord_v = np.dot(diffs, v_rot)
        coord_n = np.dot(diffs, normal)

        scale_u = np.sqrt(max(1.0, aspect_ratio))
        scale_v = 1.0 / scale_u

        scaled_dists = np.sqrt(
            (coord_u / (radius * scale_u)) ** 2 +
            (coord_v / (radius * scale_v)) ** 2 +
            (coord_n / radius) ** 2
        )
        raw_influence = np.clip(1.0 - scaled_dists, 0.0, 1.0)
    else:
        raw_influence = np.clip(1.0 - (dists / radius), 0.0, 1.0)

    # Subtle organic micro-texture for realistic carious cavity floor
    np.random.seed(42)
    noise = 1.0 + 0.10 * np.sin(diffs[:, 0] * 3.8) * np.cos(diffs[:, 1] * 3.8)

    lesion_lower = lesion_type.lower()
    if "crack" in lesion_lower:
        # Sharp V-groove structural fissure
        influence = np.clip(raw_influence ** 3.5, 0.0, 1.0)
        displacement = -normals * (depth * 1.35 * influence[:, None])
    elif "abrasion" in lesion_lower or "wear" in lesion_lower:
        # Shallow dish-like bevel
        influence = np.clip(raw_influence ** 1.3, 0.0, 1.0)
        displacement = -normals * (depth * 0.7 * influence[:, None])
    elif "cervical" in lesion_lower or "caries 5" in lesion_lower:
        # Wedge-shaped notch
        influence = np.clip((raw_influence ** 1.7) * noise, 0.0, 1.0)
        displacement = -normals * (depth * influence[:, None])
    else:
        # Authentic carious cavity: steep outer rim with excavated dentin core
        steep_falloff = 1.0 / (1.0 + np.exp(-10.0 * (raw_influence - 0.45)))
        influence = np.clip(raw_influence * 0.5 + steep_falloff * 0.5, 0.0, 1.0) * noise
        influence = np.clip(influence, 0.0, 1.0)
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

    # Base enamel ivory with subtle groove shading
    z = verts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    enamel_base = np.array([0.97, 0.96, 0.93], dtype=np.float32)
    enamel_fissure = np.array([0.91, 0.88, 0.83], dtype=np.float32)

    lesion_lower = str(lesion_type).lower()
    if "crack" in lesion_lower:
        decay_color = np.array([0.28, 0.16, 0.10], dtype=np.float32)
        necrotic_color = np.array([0.12, 0.06, 0.04], dtype=np.float32)
    elif "abrasion" in lesion_lower or "wear" in lesion_lower:
        # Exposed amber secondary dentin
        decay_color = np.array([0.80, 0.60, 0.36], dtype=np.float32)
        necrotic_color = np.array([0.65, 0.45, 0.25], dtype=np.float32)
    else:
        # Classic carious cavitation
        decay_color = np.array([0.48, 0.28, 0.14], dtype=np.float32)
        necrotic_color = np.array([0.15, 0.08, 0.04], dtype=np.float32)

    halo_color = np.array([0.89, 0.84, 0.75], dtype=np.float32)

    for i in range(n_verts):
        # Anatomical groove ambient occlusion
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

def render_carved_tooth_pyvista(carved_mesh, gt_mask, output_img_path, off_screen=True):
    pv.set_plot_theme('document')
    plotter = pv.Plotter(off_screen=off_screen, window_size=[1024, 768])
    pv_mesh = pv.wrap(carved_mesh)
    pv_mesh["cavity_influence"] = gt_mask

    plotter.add_mesh(
        pv_mesh,
        scalars="cavity_influence",
        cmap="copper",
        clim=[0.0, 1.0],
        smooth_shading=True,
        show_scalar_bar=False
    )
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    plotter.screenshot(str(output_img_path))
    plotter.close()
    return str(output_img_path)
