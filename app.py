"""
app.py

Interactive Human-in-the-Loop Dental Anomaly Segmentation & 3D Reconstruction Web Application.
Supports:
1. Uploading real dental photo or choosing preset clinical samples.
2. Clicking anomaly or drawing a bounding box.
3. Prompting SAM2 for single-anomaly segmentation.
4. Human validation: [ ACCEPT SEGMENTATION ] or [ RE-SEGMENT ].
5. Deterministic geometry extraction.
6. Multi-Tooth Selection (FDI 11-48): Loads real patient 3D models for the specific tooth.
7. Realistic 3D cavity & lesion sculpting (Cavities, Cracks, Abrasions, Cervical defects).
8. Authentic clinical shader colors (Enamel ivory, demineralized halo, necrotic decay).
9. Interactive WebGL Three.js viewport with auto-framing and orbit controls.
"""

import os
import shutil
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np

from sam2_segmenter import SAM2AnomalySegmenter
from geometry_extractor import extract_anomaly_geometry
from parameter_converter import convert_2d_geometry_to_3d_params
from tooth_carver import (
    load_or_extract_tooth_mesh,
    find_surface_vertex,
    carve_cavity,
    compute_clinical_vertex_colors,
    save_atomic_case,
    render_carved_tooth_pyvista,
    FDI_NAMES
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RENDER_DIR = STATIC_DIR / "renders"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize SAM2 predictor singleton
segmenter = SAM2AnomalySegmenter.get_instance(device="cpu")

# Tooth mesh memory cache for instant responsiveness
tooth_cache = {}

def get_cached_tooth(fdi):
    fdi = int(fdi)
    if fdi not in tooth_cache:
        tooth_cache[fdi] = load_or_extract_tooth_mesh(target_tooth=fdi)
    return tooth_cache[fdi]

# Preset mapping with accurate clinical tooth FDI numbers
PRESETS = {
    "case_01": {
        "name": "Caries 1 (Maxillary Molar - FDI #26)",
        "path": "data/dentalai/train/1001_jpg.rf.c00322f19b0c2d53472ad0181ae21683.jpg",
        "default_fdi": 26,
        "default_type": "Cavity",
        "default_surface": "occlusal"
    },
    "case_04": {
        "name": "Caries 5 (Cervical Margin - FDI #14)",
        "path": "data/dentalai/train/1008_jpg.rf.e756a25bd9b1eca086c9e49c5fbdf9f1.jpg",
        "default_fdi": 14,
        "default_type": "Cervical",
        "default_surface": "cervical"
    },
    "case_05": {
        "name": "Crack (Enamel Fracture - FDI #11)",
        "path": "data/dentalai/train/1025_jpg.rf.022c471c8278de40a975ab595fb64270.jpg",
        "default_fdi": 11,
        "default_type": "Crack",
        "default_surface": "buccal"
    },
    "case_07": {
        "name": "Cavity (Mandibular Molar - FDI #36)",
        "path": "data/dentalai/train/1007_jpg.rf.d08c6d48aad78ebc1ae9f48075dc628a.jpg",
        "default_fdi": 36,
        "default_type": "Cavity",
        "default_surface": "occlusal"
    },
    "case_09": {
        "name": "Wear Facet / Abrasion - FDI #24",
        "path": "data/dentalai/train/100_jpg.rf.b82625b276769d80f0c12c1f0b318d6f.jpg",
        "default_fdi": 24,
        "default_type": "Abrasion",
        "default_surface": "occlusal"
    }
}

loaded_images = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/teeth_list")
def get_teeth_list():
    """Returns available FDI teeth with anatomical descriptions."""
    items = []
    for fdi, name in FDI_NAMES.items():
        items.append({"fdi": fdi, "name": f"{fdi} — {name}"})
    return jsonify({"teeth": items})

@app.route("/api/tooth/<int:tooth_fdi>")
def get_base_tooth(tooth_fdi):
    """
    Returns the real 3D mesh for the requested tooth FDI with healthy enamel color grading.
    """
    try:
        tooth = get_cached_tooth(tooth_fdi)
    except Exception as e:
        return jsonify({"error": f"Tooth FDI #{tooth_fdi} not found: {str(e)}"}), 404

    center = (tooth.bounds[0] + tooth.bounds[1]) / 2.0
    centered_verts = (tooth.vertices - center).astype(np.float32)

    # Clean healthy enamel colors
    zero_inf = np.zeros(len(tooth.vertices), dtype=np.float32)
    colors = compute_clinical_vertex_colors(tooth, zero_inf)

    bounds = tooth.bounds
    tooth_size = float(np.linalg.norm(bounds[1] - bounds[0]))

    return jsonify({
        "tooth_fdi": tooth_fdi,
        "tooth_name": FDI_NAMES.get(tooth_fdi, f"Tooth #{tooth_fdi}"),
        "mesh_data": {
            "vertices": centered_verts.flatten().tolist(),
            "faces": tooth.faces.flatten().tolist(),
            "colors": colors.flatten().tolist(),
            "tooth_size": round(tooth_size, 3),
            "num_vertices": int(tooth.vertices.shape[0]),
            "num_faces": int(tooth.faces.shape[0])
        }
    })

@app.route("/api/preset/<preset_id>")
def get_preset(preset_id):
    if preset_id not in PRESETS:
        return jsonify({"error": "Unknown preset"}), 404
    src_rel = PRESETS[preset_id]["path"]
    src_p = BASE_DIR / src_rel
    if not src_p.exists():
        return jsonify({
            "error": f"Preset image not found at {src_rel}. You can use the 'Upload Image' button to test any photo."
        }), 404
    dest_name = f"{preset_id}_{src_p.name}"
    dest_p = UPLOAD_DIR / dest_name
    shutil.copy(str(src_p), str(dest_p))
    loaded_images[preset_id] = str(dest_p)
    return jsonify({
        "image_id": preset_id,
        "image_url": f"/static/uploads/{dest_name}",
        "name": PRESETS[preset_id]["name"],
        "default_fdi": PRESETS[preset_id].get("default_fdi", 36),
        "default_type": PRESETS[preset_id].get("default_type", "Cavity"),
        "default_surface": PRESETS[preset_id].get("default_surface", "occlusal")
    })

@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    image_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".jpg"
    dest_name = f"upload_{image_id}{ext}"
    dest_p = UPLOAD_DIR / dest_name
    file.save(str(dest_p))
    loaded_images[image_id] = str(dest_p)
    return jsonify({
        "image_id": image_id,
        "image_url": f"/static/uploads/{dest_name}"
    })

@app.route("/api/segment", methods=["POST"])
def segment():
    from tooth_classifier import classify_tooth

    data = request.get_json() or {}
    image_id = data.get("image_id")
    if image_id not in loaded_images:
        return jsonify({"error": "Image not found"}), 404

    img_path = loaded_images[image_id]
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 500
    h, w = img.shape[:2]

    points = data.get("points")
    box = data.get("box")

    segmenter.set_image(img)
    mask, conf = segmenter.segment_anomaly(point_coords=points, box=box)
    geometry = extract_anomaly_geometry(mask, (h, w))

    # Automatic FDI tooth classification based on spatial position and anatomy
    identified_tooth = classify_tooth(image_id, geometry, (h, w))

    # Transparent colored RGBA overlay
    rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_mask[mask > 0] = [16, 185, 129, 180]  # Emerald green

    mask_fname = f"mask_{image_id}_{uuid.uuid4().hex[:6]}.png"
    mask_p = RENDER_DIR / mask_fname
    cv2.imwrite(str(mask_p), rgba_mask)

    return jsonify({
        "mask_url": f"/static/renders/{mask_fname}",
        "confidence": round(conf, 4),
        "geometry": geometry,
        "identified_tooth": identified_tooth
    })

@app.route("/api/carve", methods=["POST"])
def carve():
    """
    Carves 2D photo-segmented anomaly onto the selected tooth FDI.
    """
    data = request.get_json() or {}
    geometry = data.get("geometry")
    tooth_fdi = int(data.get("tooth_fdi", 36))
    anomaly_type = data.get("anomaly_type", "Cavity")
    surface = data.get("surface", "occlusal")

    if not geometry:
        return jsonify({"error": "Missing geometry"}), 400

    try:
        tooth = get_cached_tooth(tooth_fdi)
    except Exception as e:
        return jsonify({"error": f"Tooth FDI #{tooth_fdi} mesh not available: {str(e)}"}), 404

    # 1. Parameter conversion on the actual selected tooth
    carve_params = convert_2d_geometry_to_3d_params(
        geometry,
        tooth,
        anomaly_type=anomaly_type,
        surface=surface
    )

    # 2. Carve realistic cavity
    carved_mesh, gt_cavity = carve_cavity(
        tooth,
        center_idx=carve_params["center_idx"],
        radius=carve_params["radius"],
        depth=carve_params["depth"],
        aspect_ratio=carve_params["aspect_ratio"],
        orientation=carve_params["orientation"],
        lesion_type=anomaly_type
    )

    # 3. Save atomic .npz and .obj
    case_uid = f"case_{uuid.uuid4().hex[:8]}"
    npz_path, obj_path = save_atomic_case(
        RENDER_DIR / case_uid,
        carved_mesh,
        gt_cavity,
        {
            "tooth_fdi": tooth_fdi,
            "anomaly_type": anomaly_type,
            "carve_params": carve_params,
            "geometry_2d": geometry
        }
    )

    # 4. Render 3D PyVista screenshot
    render_fname = f"{case_uid}_render.png"
    render_p = RENDER_DIR / render_fname
    render_carved_tooth_pyvista(carved_mesh, gt_cavity, render_p)

    # Center mesh at origin for smooth rotation
    center = (carved_mesh.bounds[0] + carved_mesh.bounds[1]) / 2.0
    centered_verts = (carved_mesh.vertices - center).astype(np.float32)

    # Compute authentic clinical colors
    colors = compute_clinical_vertex_colors(carved_mesh, gt_cavity, lesion_type=anomaly_type)

    mesh_data = {
        "vertices": centered_verts.flatten().tolist(),
        "faces": carved_mesh.faces.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "tooth_size": carve_params["tooth_size"],
        "num_vertices": int(carved_mesh.vertices.shape[0]),
        "num_faces": int(carved_mesh.faces.shape[0])
    }

    return jsonify({
        "render_url": f"/static/renders/{render_fname}",
        "npz_url": f"/static/renders/{case_uid}.npz",
        "obj_url": f"/static/renders/{case_uid}.obj",
        "carve_params": carve_params,
        "mesh_data": mesh_data
    })

@app.route("/api/carve_custom", methods=["POST"])
def carve_custom():
    """
    Direct 3D Cavity Generator: Carves any custom pathology requested by the user
    (Tooth FDI, Cavity Type, Surface, Radius, Depth).
    """
    data = request.get_json() or {}
    tooth_fdi = int(data.get("tooth_fdi", 36))
    lesion_type = data.get("lesion_type", "Cavity")
    surface = data.get("surface", "occlusal")
    radius = float(data.get("radius", 2.2))
    depth = float(data.get("depth", 1.5))
    aspect_ratio = float(data.get("aspect_ratio", 1.0))
    orientation = float(data.get("orientation", 0.0))

    try:
        tooth = get_cached_tooth(tooth_fdi)
    except Exception as e:
        return jsonify({"error": f"Tooth FDI #{tooth_fdi} mesh not available: {str(e)}"}), 404

    center_idx = find_surface_vertex(tooth, surface=surface)

    carved_mesh, gt_cavity = carve_cavity(
        tooth,
        center_idx=center_idx,
        radius=radius,
        depth=depth,
        aspect_ratio=aspect_ratio,
        orientation=orientation,
        lesion_type=lesion_type
    )

    case_uid = f"custom_{uuid.uuid4().hex[:8]}"
    save_atomic_case(
        RENDER_DIR / case_uid,
        carved_mesh,
        gt_cavity,
        {
            "tooth_fdi": tooth_fdi,
            "lesion_type": lesion_type,
            "surface": surface,
            "radius": radius,
            "depth": depth
        }
    )

    render_fname = f"{case_uid}_render.png"
    render_p = RENDER_DIR / render_fname
    render_carved_tooth_pyvista(carved_mesh, gt_cavity, render_p)

    center = (carved_mesh.bounds[0] + carved_mesh.bounds[1]) / 2.0
    centered_verts = (carved_mesh.vertices - center).astype(np.float32)
    colors = compute_clinical_vertex_colors(carved_mesh, gt_cavity, lesion_type=lesion_type)

    bounds = tooth.bounds
    tooth_size = float(np.linalg.norm(bounds[1] - bounds[0]))

    mesh_data = {
        "vertices": centered_verts.flatten().tolist(),
        "faces": carved_mesh.faces.flatten().tolist(),
        "colors": colors.flatten().tolist(),
        "tooth_size": round(tooth_size, 3),
        "num_vertices": int(carved_mesh.vertices.shape[0]),
        "num_faces": int(carved_mesh.faces.shape[0])
    }

    return jsonify({
        "render_url": f"/static/renders/{render_fname}",
        "obj_url": f"/static/renders/{case_uid}.obj",
        "carve_params": {
            "center_idx": center_idx,
            "radius": radius,
            "depth": depth,
            "aspect_ratio": aspect_ratio,
            "surface": surface,
            "tooth_size": round(tooth_size, 3)
        },
        "mesh_data": mesh_data
    })

if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 0))
    if port == 0:
        for candidate in [5000, 5001, 5050, 8080]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', candidate))
                    port = candidate
                    break
                except OSError:
                    continue
        if port == 0:
            port = 5001

    print(f"Starting Odonto Human-in-the-Loop MVP at http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=False)
