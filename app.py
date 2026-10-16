"""
app.py

Interactive Human-in-the-Loop Dental Anomaly Segmentation & 3D Reconstruction Web Application.
Supports:
1. Uploading real dental photo or choosing preset clinical samples.
2. Clicking anomaly or drawing a bounding box.
3. Prompting SAM2 for single-anomaly segmentation.
4. Human validation: [ ACCEPT SEGMENTATION ] or [ RE-SEGMENT ].
5. Deterministic geometry extraction.
6. Manual tooth selection & 2D-to-3D parameter conversion.
7. Real 3D tooth mesh cavity carving via carve_cavity() and PyVista rendering.
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
    carve_cavity,
    save_atomic_case,
    render_carved_tooth_pyvista
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
# Pre-load base tooth mesh
base_tooth = load_or_extract_tooth_mesh(target_tooth=36)

# Preset mapping
PRESETS = {
    "case_01": {
        "name": "Caries 1 (Occlusal Molar)",
        "path": "data/alphadent/images/valid/p001_F_51_002.jpg"
    },
    "case_04": {
        "name": "Caries 5 (Cervical Margin)",
        "path": "data/alphadent/images/valid/p015_M_53_002.jpg"
    },
    "case_05": {
        "name": "Crack (Enamel Fracture)",
        "path": "data/dentalai/train/1025_jpg.rf.022c471c8278de40a975ab595fb64270.jpg"
    },
    "case_07": {
        "name": "Cavity (Occlusal Breakdown)",
        "path": "data/dentalai/train/1007_jpg.rf.d08c6d48aad78ebc1ae9f48075dc628a.jpg"
    },
    "case_09": {
        "name": "Abrasion (Wear Facet)",
        "path": "data/alphadent/images/valid/p001_F_51_001.jpg"
    }
}

# Image cache
loaded_images = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/preset/<preset_id>")
def get_preset(preset_id):
    if preset_id not in PRESETS:
        return jsonify({"error": "Unknown preset"}), 404
    src_rel = PRESETS[preset_id]["path"]
    src_p = BASE_DIR / src_rel
    dest_name = f"{preset_id}_{src_p.name}"
    dest_p = UPLOAD_DIR / dest_name
    shutil.copy(str(src_p), str(dest_p))
    loaded_images[preset_id] = str(dest_p)
    return jsonify({
        "image_id": preset_id,
        "image_url": f"/static/uploads/{dest_name}",
        "name": PRESETS[preset_id]["name"]
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

    # Generate transparent colored RGBA overlay for frontend canvas
    rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_mask[mask > 0] = [0, 240, 0, 180]  # Green with 70% opacity
    
    mask_fname = f"mask_{image_id}_{uuid.uuid4().hex[:6]}.png"
    mask_p = RENDER_DIR / mask_fname
    cv2.imwrite(str(mask_p), rgba_mask)

    return jsonify({
        "mask_url": f"/static/renders/{mask_fname}",
        "confidence": round(conf, 4),
        "geometry": geometry
    })

@app.route("/api/carve", methods=["POST"])
def carve():
    data = request.get_json() or {}
    geometry = data.get("geometry")
    tooth_fdi = int(data.get("tooth_fdi", 36))
    anomaly_type = data.get("anomaly_type", "Cavity")

    if not geometry:
        return jsonify({"error": "Missing geometry"}), 400

    # 1. Parameter conversion
    carve_params = convert_2d_geometry_to_3d_params(geometry, base_tooth)

    # 2. Carve cavity
    carved_mesh, gt_cavity = carve_cavity(
        base_tooth,
        center_idx=carve_params["center_idx"],
        radius=carve_params["radius"],
        depth=carve_params["depth"],
        aspect_ratio=carve_params["aspect_ratio"],
        orientation=carve_params["orientation"]
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

    # 4. Render 3D PyVista screenshot for download/fallback
    render_fname = f"{case_uid}_render.png"
    render_p = RENDER_DIR / render_fname
    render_carved_tooth_pyvista(carved_mesh, gt_cavity, render_p)

    # Center mesh at origin for smooth interactive 3D rotation in browser
    center = (carved_mesh.bounds[0] + carved_mesh.bounds[1]) / 2.0
    centered_verts = (carved_mesh.vertices - center).astype(np.float32)

    # Compute RGB vertex colors: enamel ivory base with crimson cavity influence
    t = gt_cavity[:, None].astype(np.float32)
    enamel_color = np.array([0.96, 0.95, 0.92], dtype=np.float32)
    cavity_color = np.array([0.88, 0.15, 0.15], dtype=np.float32)
    colors = enamel_color * (1.0 - t) + cavity_color * t

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

if __name__ == "__main__":
    print("Starting Odonto Human-in-the-Loop MVP at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
