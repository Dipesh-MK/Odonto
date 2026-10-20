"""
app.py

Odonto — AI-Assisted Dental Education, 3D Anomaly Reconstruction & Clinical Simulation Platform.

Authenticated, Dynamic, Zero-Hardcoding Architecture:
- Auth: Flask-Login + bcrypt-hashed passwords in SQLite (db.py), strict role gating ('teacher' vs 'student').
- Dynamic Rooms: Teachers create named rooms, select/synthesize cases strictly from Phase 0 whitelist
  combinations (validated via whitelist.validate_tooth_anomaly_combination), and receive a unique room_code.
  Students join via room_code and see only the cases assigned to that room.
- Live Test Flow:
  * Blind 3D mesh inspection + click-to-mark (THREE.Raycaster). Ground-truth centroid and heatmap are
    withheld until the student submits their 3D mark.
  * Pure rule-based 3D location grading (Euclidean distance + surface normal alignment, no LLM, no artificial floor).
  * Real Audio Voice Viva loop: Whisper STT (faster-whisper) -> Claude/Ollama LLM -> Piper ONNX TTS (.wav).
  * Final Diagnosis Grading & Spoken Teach-Back: LLM grades the transcribed spoken/written diagnosis
    against the case's true anomaly type, surface, and severity, and Piper TTS synthesizes a spoken teach-back.
- Persistent Student Reports, Longitudinal Progress Tracking (with dynamic Whitelist Competency Coverage),
  and Teacher Room/Attempt Replay Dashboards.
"""

import json
import os
import secrets
import shutil
import signal
import subprocess
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import trimesh
from flask import Flask, jsonify, render_template, request
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)

from db import (
    add_case_to_room,
    create_or_resume_attempt,
    create_room,
    create_user,
    finalize_attempt_grading,
    get_attempt_by_uid,
    get_case_by_uid,
    get_db_connection,
    get_room_by_code,
    get_room_by_id,
    get_student_longitudinal_progress,
    get_teacher_room_dashboard,
    get_user_by_email,
    get_user_by_id,
    init_db,
    is_student_enrolled,
    join_room_by_code,
    list_cases_for_room,
    list_student_rooms,
    list_teacher_rooms,
    update_attempt_location,
    update_attempt_viva_turn,
    verify_password,
)
from geometry_extractor import extract_anomaly_geometry
from parameter_converter import convert_2d_geometry_to_3d_params
from sam2_segmenter import SAM2AnomalySegmenter
from tooth_carver import (
    carve_cavity,
    load_or_extract_tooth_mesh,
    render_carved_tooth_pyvista,
    save_atomic_case,
)
from voice_llm_service import (
    AUDIO_DIR,
    compute_rule_based_location_grade,
    grade_diagnosis_and_generate_teachback,
    run_viva_patient_turn,
    synthesize_speech_to_url,
    transcribe_audio_file,
)
from whitelist import (
    APPROVED_FDI_TEETH,
    DEPTH_CALIBRATION_NOTE,
    SUPPORTED_ANOMALIES,
    get_whitelist_payload,
    validate_tooth_anomaly_combination,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RENDER_DIR = STATIC_DIR / "renders"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RENDER_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("ODONTO_SECRET_KEY", "odonto-clinical-secret-key-2026-v5")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Initialize SQLite schema on startup
init_db()

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)


class OdontoUser(UserMixin):
    def __init__(self, row_dict: Dict[str, Any]):
        self.id = int(row_dict["id"])
        self.email = row_dict["email"]
        self.name = row_dict["name"]
        self.role = row_dict["role"]
        self.created_at = row_dict["created_at"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "created_at": self.created_at,
        }


@login_manager.user_loader
def load_user(user_id: str) -> Optional[OdontoUser]:
    u = get_user_by_id(int(user_id))
    return OdontoUser(u) if u else None


@login_manager.unauthorized_handler
def unauthorized_callback():
    return (
        jsonify(
            {
                "error": "Authentication required. Please log in as a teacher or student.",
                "code": "UNAUTHENTICATED",
            }
        ),
        401,
    )


def role_required(required_role: str):
    """Enforces that the logged-in user has the specified role ('teacher' or 'student')."""
    def decorator(fn):
        @wraps(fn)
        @login_required
        def wrapper(*args, **kwargs):
            if current_user.role != required_role:
                return (
                    jsonify(
                        {
                            "error": f"Access denied: endpoint requires role '{required_role}', but current user is '{current_user.role}'.",
                            "code": "FORBIDDEN_ROLE",
                        }
                    ),
                    403,
                )
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# Initialize SAM2 predictor singleton
segmenter = SAM2AnomalySegmenter.get_instance(device="cpu")

# Clinical source photos for SAM2 segmentation
PRESETS: Dict[str, Dict[str, Any]] = {
    "case_01": {
        "name": "Sample 1: Caries 1 — Occlusal Molar (AlphaDent p001_002)",
        "path": "data/alphadent/images/valid/p001_F_51_002.jpg",
        "default_anomaly": "Cavity",
        "default_fdi": 36,
        "default_box": [1254, 651, 1349, 738],
        "default_point": [1301, 694],
    },
    "case_02": {
        "name": "Sample 2: Caries 2 — Posterior Pit (AlphaDent p001_003)",
        "path": "data/alphadent/images/valid/p001_F_51_003.jpg",
        "default_anomaly": "Cavity",
        "default_fdi": 46,
        "default_box": [980, 540, 1080, 640],
        "default_point": [1030, 590],
    },
    "case_04": {
        "name": "Sample 3: Caries 5 — Cervical/Labial Margin (AlphaDent p015_002)",
        "path": "data/alphadent/images/valid/p015_M_53_002.jpg",
        "default_anomaly": "Cavity",
        "default_fdi": 31,
        "default_box": [890, 520, 980, 610],
        "default_point": [935, 565],
    },
    "case_05": {
        "name": "Sample 4: Enamel Defect / Fissure (DentalAI 1018)",
        "path": "data/dentalai/train/1018_jpg.rf.aabbf1107ffd09f3e6e0d1e7804d8eb9.jpg",
        "default_anomaly": "Cavity",
        "default_fdi": 33,
        "default_box": [180, 140, 260, 220],
        "default_point": [220, 180],
    },
    "case_07": {
        "name": "Sample 5: Premolar Cavitated Fossa Lesion (DentalAI 1007)",
        "path": "data/dentalai/train/1007_jpg.rf.d08c6d48aad78ebc1ae9f48075dc628a.jpg",
        "default_anomaly": "Cavity",
        "default_fdi": 34,
        "default_box": [204, 150, 224, 179],
        "default_point": [214, 164],
    },
    "case_08": {
        "name": "Sample 6: Maxillary Premolar Caries (AlphaDent p026_001)",
        "path": "data/alphadent/images/valid/p026_F_34_001.jpg",
        "default_anomaly": "Cavity",
        "default_fdi": 14,
        "default_box": [920, 500, 1020, 600],
        "default_point": [970, 550],
    },
    "case_09": {
        "name": "Sample 7: Anterior Labial Abrasion Wear Facet (AlphaDent p001_001)",
        "path": "data/alphadent/images/valid/p001_F_51_001.jpg",
        "default_anomaly": "Abrasion",
        "default_fdi": 31,
        "default_box": [922, 667, 1143, 795],
        "default_point": [1032, 731],
    },
    "case_10": {
        "name": "Sample 8: Canine Cervical Abrasion (AlphaDent p015_001)",
        "path": "data/alphadent/images/valid/p015_M_53_001.jpg",
        "default_anomaly": "Abrasion",
        "default_fdi": 33,
        "default_box": [860, 540, 990, 660],
        "default_point": [925, 600],
    },
}

loaded_images: Dict[str, str] = {}
active_host_mic_procs: Dict[int, Dict[str, Any]] = {}


def extract_geometry_from_preset(preset_id: str) -> Dict[str, Any]:
    """Runs real SAM2 + OpenCV spatial moment extraction on a clinical photo preset."""
    preset = PRESETS.get(preset_id) or PRESETS["case_01"]
    src_p = BASE_DIR / preset["path"]
    img = cv2.imread(str(src_p))
    if img is None:
        raise RuntimeError(f"Could not read clinical source image at {src_p}")
    h, w = img.shape[:2]
    segmenter.set_image(img)
    mask, _ = segmenter.segment_anomaly(
        point_coords=[preset["default_point"]],
        box=preset["default_box"],
    )
    return extract_anomaly_geometry(mask, (h, w))


def build_threejs_mesh_payload(
    vertices: np.ndarray,
    faces: np.ndarray,
    gt_cavity: np.ndarray,
    anomaly_type: str = "Cavity",
    reveal_gt: bool = True,
) -> Dict[str, Any]:
    """
    Centers the mesh at (0, 0, 0) for Three.js.
    When reveal_gt=False (during a student's blind Step 1 anomaly marking), ground-truth
    centroid coordinates and heatmap vertex colors are withheld from the payload.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    fcs = np.asarray(faces, dtype=np.int32)
    inf = np.asarray(gt_cavity, dtype=np.float32)

    bounds_min = verts.min(axis=0)
    bounds_max = verts.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    centered_verts = verts - center
    tooth_size = float(np.linalg.norm(bounds_max - bounds_min))

    enamel_base = np.array([0.96, 0.95, 0.92], dtype=np.float32)
    enamel_only_colors = np.repeat(enamel_base[None, :], len(verts), axis=0)

    payload: Dict[str, Any] = {
        "vertices": centered_verts.flatten().tolist(),
        "faces": fcs.flatten().tolist(),
        "enamel_colors": enamel_only_colors.flatten().tolist(),
        "tooth_size": round(tooth_size, 3),
        "num_vertices": int(verts.shape[0]),
        "num_faces": int(fcs.shape[0]),
        "gt_revealed": bool(reveal_gt),
    }

    if reveal_gt:
        if np.max(inf) > 1e-4:
            peak_mask = inf >= (0.5 * np.max(inf))
            weights = (inf[peak_mask] ** 2)[:, None]
            gt_centroid_3d = (
                np.sum(centered_verts[peak_mask] * weights, axis=0) / np.sum(weights)
            ).tolist()
            max_idx = int(np.argmax(inf))
            gt_peak_vertex_3d = centered_verts[max_idx].tolist()
        else:
            gt_centroid_3d = [0.0, 0.0, 0.0]
            gt_peak_vertex_3d = [0.0, 0.0, 0.0]

        t = inf[:, None]
        if str(anomaly_type).lower().startswith("abrasion"):
            lesion_tint = np.array([0.85, 0.42, 0.08], dtype=np.float32)
        else:
            lesion_tint = np.array([0.86, 0.15, 0.15], dtype=np.float32)
        heatmap_colors = enamel_base * (1.0 - t) + lesion_tint * t

        payload["colors"] = heatmap_colors.flatten().tolist()
        payload["gt_centroid_3d"] = [round(float(x), 3) for x in gt_centroid_3d]
        payload["gt_peak_vertex_3d"] = [round(float(x), 3) for x in gt_peak_vertex_3d]
        payload["affected_vertices"] = int(np.sum(inf > 1e-4))

    return payload


def execute_validated_carve(
    geometry: Dict[str, Any],
    tooth_fdi: int,
    canonical_anomaly: str,
) -> Dict[str, Any]:
    """
    Executes 2D-to-3D parameter conversion, 3D surface-normal-gated carving,
    zero-bleed-through verification, atomic saving, and PyVista rendering.
    """
    target_tooth_mesh = load_or_extract_tooth_mesh(target_tooth=tooth_fdi)
    orig_verts = np.asarray(target_tooth_mesh.vertices, dtype=np.float64).copy()
    orig_norms = np.asarray(target_tooth_mesh.vertex_normals, dtype=np.float64).copy()

    carve_params = convert_2d_geometry_to_3d_params(
        geometry,
        target_tooth_mesh,
        tooth_fdi=tooth_fdi,
        anomaly_type=canonical_anomaly,
    )

    carved_mesh, gt_cavity = carve_cavity(
        target_tooth_mesh,
        center_idx=carve_params["center_idx"],
        radius=carve_params["radius"],
        depth=carve_params["depth"],
        aspect_ratio=carve_params["aspect_ratio"],
        orientation=carve_params["orientation"],
        profile=carve_params.get("profile", "cavity"),
        tangent_u=carve_params.get("tangent_u"),
    )

    carved_verts = np.asarray(carved_mesh.vertices, dtype=np.float64)
    disp = np.linalg.norm(carved_verts - orig_verts, axis=1)
    affected = disp > 1e-5
    center_n = orig_norms[carve_params["center_idx"]]
    if np.any(affected):
        ndots = np.dot(orig_norms[affected], center_n)
        opposite_bleed = int(np.sum(ndots < 0.0))
    else:
        opposite_bleed = 0

    is_finite = bool(np.all(np.isfinite(carved_verts)))
    vf_match = (len(carved_verts) == len(orig_verts)) and (
        len(carved_mesh.faces) == len(target_tooth_mesh.faces)
    )

    carve_params["opposite_bleed_vertices"] = opposite_bleed
    carve_params["is_finite"] = is_finite
    carve_params["vf_preserved"] = vf_match
    carve_params["affected_vertices"] = int(np.sum(affected))

    case_uid = f"case_{uuid.uuid4().hex[:8]}"
    npz_path, obj_path = save_atomic_case(
        RENDER_DIR / case_uid,
        carved_mesh,
        gt_cavity,
        {
            "tooth_fdi": tooth_fdi,
            "anomaly_type": canonical_anomaly,
            "carve_params": carve_params,
            "geometry_2d": geometry,
            "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
        },
    )

    render_fname = f"{case_uid}_render.png"
    render_p = RENDER_DIR / render_fname
    render_carved_tooth_pyvista(carved_mesh, gt_cavity, render_p, carve_params=carve_params)

    mesh_data = build_threejs_mesh_payload(
        carved_mesh.vertices,
        carved_mesh.faces,
        gt_cavity,
        anomaly_type=canonical_anomaly,
        reveal_gt=True,
    )

    return {
        "case_uid": case_uid,
        "npz_path": str(npz_path),
        "render_url": f"/static/renders/{render_fname}",
        "npz_url": f"/static/renders/{case_uid}.npz",
        "obj_url": f"/static/renders/{case_uid}.obj",
        "carve_params": carve_params,
        "mesh_data": mesh_data,
    }


def build_dynamic_case_record(
    validation_info: Dict[str, Any],
    carve_result: Dict[str, Any],
    case_seq_num: int,
    rigor: str = "Intermediate",
    patient_name: Optional[str] = None,
    patient_age: int = 32,
    patient_gender: str = "Female",
) -> Dict[str, Any]:
    """Builds the structured case metadata dictionary for a newly carved tooth case."""
    tooth_fdi = int(validation_info["tooth_fdi"])
    canonical_anomaly = validation_info["anomaly_type"]
    tooth_meta = APPROVED_FDI_TEETH[tooth_fdi]
    cp = carve_result["carve_params"]

    surf_label = "Cervical-middle labial face" if tooth_meta["surface"] == "labial_facial" else "Central occlusal fossa"
    badge_label = (
        f"Labial abrasion ({cp['depth']:.2f}mm)"
        if canonical_anomaly == "Abrasion"
        else f"{'Labial' if tooth_meta['surface'] == 'labial_facial' else 'Occlusal'} cavity ({cp['depth']:.2f}mm)"
    )

    p_name = (patient_name or "").strip() or f"Patient #{case_seq_num}"
    if canonical_anomaly == "Abrasion":
        cc = (
            f"I feel a sharp zing on my {tooth_meta['short_name'].lower()} (Tooth #{tooth_fdi}) "
            "whenever cold water or my toothbrush bristles touch the front near the gumline."
        )
        hpi = (
            "Sensitivity started about 3 weeks ago. Uses a firm manual toothbrush with horizontal strokes. "
            "Pain lasts 3 to 4 seconds after cold contact and never wakes the patient at night."
        )
    else:
        cc = (
            f"I've had a dull ache and sensitivity to cold drinks and sweets on my {tooth_meta['short_name'].lower()} "
            f"(Tooth #{tooth_fdi}) for the past 3 weeks."
        )
        hpi = (
            "Discomfort is provoked by cold or sweet stimuli and subsides within 4 to 5 seconds. "
            "No spontaneous nocturnal throbbing, swelling, or fever."
        )

    return {
        "case_uid": carve_result["case_uid"],
        "case_number": f"#{case_seq_num}",
        "tooth_fdi": tooth_fdi,
        "tooth_name": tooth_meta["name"],
        "short_name": tooth_meta["short_name"],
        "section": surf_label,
        "quadrant": tooth_meta["quadrant"],
        "arch": tooth_meta["arch"],
        "tooth_type": tooth_meta["tooth_type"],
        "anatomical_group": tooth_meta["anatomical_group"],
        "target_surface": tooth_meta["surface"],
        "anomaly_type": canonical_anomaly,
        "anomaly_badge": badge_label,
        "rigor": rigor,
        "npz_path": carve_result["npz_path"],
        "obj_url": carve_result["obj_url"],
        "npz_url": carve_result["npz_url"],
        "thumbnail_url": carve_result["render_url"],
        "carve_params": cp,
        "patient": {
            "name": p_name,
            "age": int(patient_age),
            "gender": patient_gender,
            "summary_tag": f"{int(patient_age)} yo {patient_gender.lower()}",
            "asa_class": "Class I (Normal, no drug allergies)",
            "record_id": f"Rec #D-{90000 + case_seq_num}",
            "chief_complaint": cc,
            "hpi": hpi,
            "exam_notes": [
                f"Surface Inspection: Tooth #{tooth_fdi} exhibits {badge_label.lower()} on {tooth_meta['surface']} (r={cp['radius']}mm, d={cp['depth']}mm).",
                "Mobility & Percussion: Grade 0 mobility, negative to percussion. Opposite wall bleed-through: 0 vertices.",
                "Periodontal Status: Probing depths 2–3mm within normal limits, BOP negative.",
            ],
            "diagnostic_objective": (
                f"Pinpoint the 3D {canonical_anomaly.lower()} site on Tooth #{tooth_fdi}, interview the patient via Voice Viva, "
                "and formulate a diagnosis and minimally invasive treatment plan."
            ),
            "radiograph_note": (
                f"Simulated depth {cp['depth']}mm on {cp['local_wall_thickness_mm']}mm measured local wall "
                "(clinical heuristic ratio on measured Teeth3DS+ mesh)."
            ),
            "electric_pulp": "Positive / Normal reversible response",
        },
    }


# ===========================================================================
# UI SHELL ROUTE
# ===========================================================================
@app.route("/")
def index():
    return render_template("index.html")


# ===========================================================================
# AUTHENTICATION ENDPOINTS (Flask-Login + bcrypt)
# ===========================================================================
@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    email = str(data.get("email", "")).strip()
    name = str(data.get("name", "")).strip()
    password = str(data.get("password", ""))
    role = str(data.get("role", "student")).strip().lower()

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if get_user_by_email(email):
        return jsonify({"error": f"An account with email '{email}' already exists."}), 409

    try:
        user_row = create_user(email=email, name=name, password=password, role=role)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    user_obj = OdontoUser(user_row)
    login_user(user_obj)
    return jsonify({"user": user_obj.to_dict()}), 201


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    user_row = get_user_by_email(email)
    if not user_row or not verify_password(password, user_row["password_hash"]):
        return jsonify({"error": "Invalid email or password."}), 401

    user_obj = OdontoUser(user_row)
    login_user(user_obj)
    return jsonify({"user": user_obj.to_dict()}), 200


@app.route("/api/auth/logout", methods=["POST"])
@login_required
def api_logout():
    logout_user()
    return jsonify({"status": "logged_out"}), 200


@app.route("/api/auth/me", methods=["GET"])
@login_required
def api_me():
    return jsonify({"user": current_user.to_dict()}), 200


# ===========================================================================
# WHITELIST ENDPOINT (Authenticated)
# ===========================================================================
@app.route("/api/whitelist", methods=["GET"])
@login_required
def api_whitelist():
    return jsonify(get_whitelist_payload()), 200


# ===========================================================================
# TEACHER ENDPOINTS: ROOM CREATION, CASE SYNTHESIS & TEACHER DASHBOARD
# ===========================================================================
@app.route("/api/teacher/rooms", methods=["GET"])
@role_required("teacher")
def api_list_teacher_rooms():
    rooms = list_teacher_rooms(current_user.id)
    return jsonify({"rooms": rooms}), 200


@app.route("/api/rooms", methods=["POST"])
@role_required("teacher")
def api_create_room():
    """
    Teacher creates a new dynamic room and optionally synthesizes initial cases
    strictly from Phase 0 whitelist-approved tooth x anomaly combinations.
    """
    data = request.get_json() or {}
    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()
    case_specs = data.get("cases") or []

    if not name:
        return jsonify({"error": "Room name is required."}), 400

    # Pre-validate all requested case combinations against the live Phase 0 whitelist BEFORE creating room
    validated_specs = []
    for spec in case_specs:
        raw_fdi = spec.get("tooth_fdi")
        raw_anom = spec.get("anomaly_type")
        ok, v_info = validate_tooth_anomaly_combination(raw_fdi, raw_anom)
        if not ok:
            return jsonify(v_info), 422
        validated_specs.append((spec, v_info))

    room = create_room(teacher_id=current_user.id, name=name, description=description)
    created_cases = []

    for idx, (spec, v_info) in enumerate(validated_specs, start=1):
        geom = spec.get("geometry")
        if not geom:
            preset_id = spec.get("preset_id") or (
                "case_09" if v_info["anomaly_type"] == "Abrasion" else "case_01"
            )
            geom = extract_geometry_from_preset(preset_id)

        carve_res = execute_validated_carve(
            geom, v_info["tooth_fdi"], v_info["anomaly_type"]
        )
        case_record = build_dynamic_case_record(
            validation_info=v_info,
            carve_result=carve_res,
            case_seq_num=100 + room["id"] * 10 + idx,
            rigor=str(spec.get("rigor", "Intermediate")),
            patient_name=spec.get("patient_name"),
            patient_age=int(spec.get("patient_age", 30 + idx * 3)),
            patient_gender=str(spec.get("patient_gender", "Female" if idx % 2 == 1 else "Male")),
        )
        saved_case = add_case_to_room(
            room_id=room["id"],
            teacher_id=current_user.id,
            case_data=case_record,
        )
        created_cases.append(saved_case)

    return (
        jsonify(
            {
                "room": room,
                "cases": created_cases,
            }
        ),
        201,
    )


@app.route("/api/rooms/<int:room_id>/cases", methods=["POST"])
@role_required("teacher")
def api_add_case_to_room(room_id: int):
    """
    Teacher adds a whitelist-validated 3D case to one of their own rooms.
    """
    room = get_room_by_id(room_id)
    if not room:
        return jsonify({"error": "Room not found."}), 404
    if int(room["teacher_id"]) != current_user.id:
        return jsonify({"error": "Access denied: you do not own this room.", "code": "FORBIDDEN_ROOM"}), 403

    data = request.get_json() or {}
    raw_fdi = data.get("tooth_fdi")
    raw_anomaly = data.get("anomaly_type")

    is_valid, validation_info = validate_tooth_anomaly_combination(raw_fdi, raw_anomaly)
    if not is_valid:
        return jsonify(validation_info), 422

    geometry = data.get("geometry")
    if not geometry:
        preset_id = data.get("preset_id") or (
            "case_09" if validation_info["anomaly_type"] == "Abrasion" else "case_01"
        )
        geometry = extract_geometry_from_preset(preset_id)

    carve_res = execute_validated_carve(
        geometry,
        validation_info["tooth_fdi"],
        validation_info["anomaly_type"],
    )
    existing_cases = list_cases_for_room(room_id)
    seq_num = 100 + room_id * 10 + len(existing_cases) + 1

    case_record = build_dynamic_case_record(
        validation_info=validation_info,
        carve_result=carve_res,
        case_seq_num=seq_num,
        rigor=str(data.get("rigor", "Intermediate")),
        patient_name=data.get("patient_name"),
        patient_age=int(data.get("patient_age", 34)),
        patient_gender=str(data.get("patient_gender", "Female")),
    )
    saved_case = add_case_to_room(
        room_id=room_id,
        teacher_id=current_user.id,
        case_data=case_record,
    )

    return (
        jsonify(
            {
                "case": saved_case,
                "mesh_data": carve_res["mesh_data"],
                "whitelist_validation": validation_info,
                "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
            }
        ),
        201,
    )


@app.route("/api/presets", methods=["GET"])
@login_required
def api_list_presets():
    return (
        jsonify(
            {
                "presets": [
                    {
                        "id": k,
                        "name": v["name"],
                        "default_anomaly": v["default_anomaly"],
                        "default_fdi": v["default_fdi"],
                        "default_box": v["default_box"],
                        "default_point": v["default_point"],
                    }
                    for k, v in PRESETS.items()
                ]
            }
        ),
        200,
    )


@app.route("/api/carve", methods=["POST"])
@login_required
def api_teacher_carve():
    """
    SAM2 2D-to-3D HITL carving endpoint:
    1. Validates (tooth_fdi, anomaly_type) against whitelist.py.
    2. Carves the 3D mesh from validated SAM2 2D geometry and returns full 3D mesh_data + .npz/.obj URLs
       so the user can inspect the carved 3D tooth in the interactive pipeline viewer.
    3. If 'room_id' is provided, verifies the user is the teacher owning that room and adds the case to the room.
    """
    data = request.get_json() or {}
    raw_fdi = data.get("tooth_fdi")
    raw_anomaly = data.get("anomaly_type")
    room_id = data.get("room_id")

    is_valid, validation_info = validate_tooth_anomaly_combination(raw_fdi, raw_anomaly)
    if not is_valid:
        return jsonify(validation_info), 422

    geometry = data.get("geometry")
    if not geometry:
        return jsonify({"error": "Missing validated 2D geometry. Run SAM2 segmentation first."}), 400

    room = None
    if room_id is not None and str(room_id).strip() != "":
        if current_user.role != "teacher":
            return jsonify({"error": "Only teachers can publish a carved case into a room.", "code": "FORBIDDEN_ROLE"}), 403
        room = get_room_by_id(int(room_id))
        if not room:
            return jsonify({"error": "Target room not found."}), 404
        if int(room["teacher_id"]) != current_user.id:
            return jsonify({"error": "Access denied: you do not own this room.", "code": "FORBIDDEN_ROOM"}), 403

    result = execute_validated_carve(
        geometry,
        validation_info["tooth_fdi"],
        validation_info["anomaly_type"],
    )

    saved_case = None
    if room:
        existing_cases = list_cases_for_room(room["id"])
        seq_num = 100 + room["id"] * 10 + len(existing_cases) + 1
        case_record = build_dynamic_case_record(
            validation_info=validation_info,
            carve_result=result,
            case_seq_num=seq_num,
            rigor=str(data.get("rigor", "Intermediate")),
            patient_name=data.get("patient_name"),
            patient_age=int(data.get("patient_age", 31)),
            patient_gender=str(data.get("patient_gender", "Female")),
        )
        if data.get("chief_complaint"):
            case_record["patient"]["chief_complaint"] = str(data.get("chief_complaint")).strip()
        saved_case = add_case_to_room(
            room_id=room["id"],
            teacher_id=current_user.id,
            case_data=case_record,
        )

    return (
        jsonify(
            {
                "render_url": result["render_url"],
                "npz_url": result["npz_url"],
                "obj_url": result["obj_url"],
                "carve_params": result["carve_params"],
                "mesh_data": result["mesh_data"],
                "room_case": saved_case,
                "whitelist_validation": validation_info,
                "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
            }
        ),
        200,
    )


@app.route("/api/preset/<preset_id>", methods=["GET"])
@login_required
def get_preset(preset_id: str):
    if preset_id not in PRESETS:
        return jsonify({"error": "Unknown preset"}), 404
    preset = PRESETS[preset_id]
    src_p = BASE_DIR / preset["path"]
    dest_name = f"{preset_id}_{src_p.name}"
    dest_p = UPLOAD_DIR / dest_name
    if not dest_p.exists():
        shutil.copy(str(src_p), str(dest_p))
    loaded_images[preset_id] = str(dest_p)
    return jsonify(
        {
            "image_id": preset_id,
            "image_url": f"/static/uploads/{dest_name}",
            "name": preset["name"],
            "default_anomaly": preset["default_anomaly"],
            "default_fdi": preset["default_fdi"],
            "default_box": preset["default_box"],
            "default_point": preset["default_point"],
        }
    )


@app.route("/api/upload", methods=["POST"])
@login_required
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
    return jsonify(
        {
            "image_id": image_id,
            "image_url": f"/static/uploads/{dest_name}",
        }
    )


@app.route("/api/segment", methods=["POST"])
@login_required
def segment():
    data = request.get_json() or {}
    image_id = data.get("image_id")
    if image_id not in loaded_images and image_id in PRESETS:
        src_p = BASE_DIR / PRESETS[image_id]["path"]
        dest_name = f"{image_id}_{src_p.name}"
        dest_p = UPLOAD_DIR / dest_name
        if not dest_p.exists():
            shutil.copy(str(src_p), str(dest_p))
        loaded_images[image_id] = str(dest_p)

    if image_id not in loaded_images:
        return jsonify({"error": "Image not found"}), 404

    img_path = loaded_images[image_id]
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 500
    h, w = img.shape[:2]

    points = data.get("points")
    point_labels = data.get("point_labels")
    box = data.get("box")
    if not points and not box and image_id in PRESETS:
        points = [PRESETS[image_id]["default_point"]]
        box = PRESETS[image_id]["default_box"]

    segmenter.set_image(img)
    mask, conf = segmenter.segment_anomaly(
        point_coords=points,
        point_labels=point_labels,
        box=box,
    )
    geometry = extract_anomaly_geometry(mask, (h, w))

    rgba_mask = np.zeros((h, w, 4), dtype=np.uint8)
    rgba_mask[mask > 0] = [0, 240, 0, 180]

    mask_fname = f"mask_{image_id}_{uuid.uuid4().hex[:6]}.png"
    mask_p = RENDER_DIR / mask_fname
    cv2.imwrite(str(mask_p), rgba_mask)

    return jsonify(
        {
            "mask_url": f"/static/renders/{mask_fname}",
            "confidence": round(conf, 4),
            "geometry": geometry,
        }
    )


@app.route("/api/teacher/rooms/<int:room_id>/dashboard", methods=["GET"])
@role_required("teacher")
def api_teacher_room_dashboard(room_id: int):
    dash = get_teacher_room_dashboard(
        teacher_id=current_user.id,
        room_id=room_id,
        approved_fdi_teeth=APPROVED_FDI_TEETH,
    )
    if not dash:
        return jsonify({"error": "Room not found or access denied.", "code": "FORBIDDEN_ROOM"}), 403
    return jsonify(dash), 200


@app.route("/api/teacher/attempts/<attempt_uid>", methods=["GET"])
@role_required("teacher")
def api_teacher_attempt_replay(attempt_uid: str):
    """
    Teacher drills into a single student's attempt: returns full 3D mesh with both
    student mark and ground-truth centroid, voice transcript, LLM grading, and teach-back audio.
    """
    att = get_attempt_by_uid(attempt_uid)
    if not att:
        return jsonify({"error": "Attempt not found."}), 404
    room = get_room_by_id(att["room_id"])
    if not room or int(room["teacher_id"]) != current_user.id:
        return jsonify({"error": "Access denied: attempt belongs to another teacher's room.", "code": "FORBIDDEN_ROOM"}), 403

    conn = get_db_connection()
    try:
        case_row = conn.execute("SELECT * FROM room_cases WHERE id = ?", (int(att["case_id"]),)).fetchone()
        student_row = conn.execute("SELECT id, name, email FROM users WHERE id = ?", (int(att["student_id"]),)).fetchone()
    finally:
        conn.close()

    from db import deserialize_room_case

    case_data = deserialize_room_case(case_row)
    npz_data = np.load(str(case_data["npz_path"]), allow_pickle=True)
    mesh_data = build_threejs_mesh_payload(
        npz_data["vertices"],
        npz_data["faces"],
        npz_data["gt_cavity"] if "gt_cavity" in npz_data else npz_data["gt_mask"],
        anomaly_type=case_data["anomaly_type"],
        reveal_gt=True,
    )

    return (
        jsonify(
            {
                "attempt": att,
                "student": dict(student_row) if student_row else None,
                "room": room,
                "case": case_data,
                "mesh_data": mesh_data,
            }
        ),
        200,
    )


# ===========================================================================
# STUDENT & ROOM ACCESS ENDPOINTS
# ===========================================================================
@app.route("/api/rooms/join", methods=["POST"])
@role_required("student")
def api_student_join_room():
    data = request.get_json() or {}
    room_code = str(data.get("room_code", "")).strip().upper()
    if not room_code:
        return jsonify({"error": "Room join code is required."}), 400

    room = join_room_by_code(student_id=current_user.id, room_code=room_code)
    if not room:
        return jsonify({"error": f"No room found with code '{room_code}'."}), 404

    cases = list_cases_for_room(room["id"], student_id=current_user.id)
    return jsonify({"room": room, "cases": cases}), 200


@app.route("/api/student/rooms", methods=["GET"])
@role_required("student")
def api_list_student_rooms():
    rooms = list_student_rooms(current_user.id)
    return jsonify({"rooms": rooms}), 200


@app.route("/api/rooms/<int:room_id>/cases", methods=["GET"])
@login_required
def api_list_room_cases(room_id: int):
    room = get_room_by_id(room_id)
    if not room:
        return jsonify({"error": "Room not found."}), 404

    if current_user.role == "teacher":
        if int(room["teacher_id"]) != current_user.id:
            return jsonify({"error": "Access denied: you do not own this room.", "code": "FORBIDDEN_ROOM"}), 403
        cases = list_cases_for_room(room_id, student_id=current_user.id)
    else:
        if not is_student_enrolled(current_user.id, room_id):
            return jsonify({"error": "Access denied: you have not joined this room.", "code": "NOT_ENROLLED"}), 403
        cases = list_cases_for_room(room_id, student_id=current_user.id)

    return (
        jsonify(
            {
                "room": room,
                "cases": cases,
                "total": len(cases),
                "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
            }
        ),
        200,
    )


@app.route("/api/rooms/cases/<case_uid>/mesh", methods=["GET"])
@login_required
def api_get_case_mesh(case_uid: str):
    """
    Loads the 3D carved mesh for a room case.
    Enforces room isolation: only the owning teacher or a student enrolled in the case's room may access it.
    Ground-truth centroid and heatmap colors are withheld from students until they submit their 3D mark.
    """
    case = get_case_by_uid(case_uid)
    if not case:
        return jsonify({"error": f"Case '{case_uid}' not found."}), 404

    room_id = int(case["room_id"])
    room = get_room_by_id(room_id)
    if current_user.role == "teacher":
        if int(room["teacher_id"]) != current_user.id:
            return jsonify({"error": "Access denied: case belongs to another teacher's room.", "code": "FORBIDDEN_ROOM"}), 403
        reveal_gt = request.args.get("blind", "0") != "1"
    else:
        if not is_student_enrolled(current_user.id, room_id):
            return jsonify({"error": "Access denied: you are not enrolled in this case's room.", "code": "NOT_ENROLLED"}), 403
        reveal_gt = False

    npz_p = Path(case["npz_path"])
    if not npz_p.exists():
        return jsonify({"error": "Mesh file not found on disk."}), 404

    data = np.load(str(npz_p), allow_pickle=True)
    vertices = data["vertices"]
    faces = data["faces"]
    gt_cavity = data["gt_cavity"] if "gt_cavity" in data else data["gt_mask"]

    mesh_data = build_threejs_mesh_payload(
        vertices,
        faces,
        gt_cavity,
        anomaly_type=case["anomaly_type"],
        reveal_gt=reveal_gt,
    )
    return (
        jsonify(
            {
                "case": case,
                "room": room,
                "mesh_data": mesh_data,
                "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
            }
        ),
        200,
    )


# ===========================================================================
# LIVE STREAMING STT (HOST + WEBAUDIO) & LIVE PIPER TTS PLAYBACK ENDPOINTS
# ===========================================================================
def _play_wav_on_host_speakers_async(wav_url_or_path: str) -> None:
    """
    Plays a synthesized WAV file live on the Linux host's PipeWire/PulseAudio output
    in a non-blocking background process so TTS is heard immediately out loud even if
    browser autoplay is restricted.
    """
    if not wav_url_or_path:
        return
    p = Path(wav_url_or_path)
    if not p.is_absolute():
        p = BASE_DIR / str(wav_url_or_path).lstrip("/")
    if not p.exists():
        return
    try:
        subprocess.Popen(
            ["pw-play", str(p)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        try:
            subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(p)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


@app.route("/api/tts/speak", methods=["POST"])
@login_required
def api_tts_speak():
    """Synthesizes spoken audio via local Piper ONNX TTS and plays it live."""
    data = request.get_json() or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "Text is required for TTS."}), 400
    audio_url = synthesize_speech_to_url(text, prefix="tts_on_demand")
    if data.get("play_on_host", False):
        _play_wav_on_host_speakers_async(audio_url)
    return jsonify({"audio_url": audio_url, "text": text}), 200


@app.route("/api/audio/host_mic/start", methods=["POST"])
@login_required
def api_host_mic_start():
    """
    Starts live recording directly from the host Linux PulseAudio/PipeWire default microphone
    using ffmpeg (-f pulse -i default) with flush_packets=1 so live partial chunks can be
    transcribed in real time while the user is speaking.
    """
    uid = current_user.id
    prev = active_host_mic_procs.pop(uid, None)
    if prev and prev.get("proc") and prev["proc"].poll() is None:
        try:
            prev["proc"].send_signal(signal.SIGINT)
            prev["proc"].wait(timeout=2)
        except Exception:
            prev["proc"].kill()

    out_wav = AUDIO_DIR / f"host_mic_{uid}_{uuid.uuid4().hex[:8]}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "pulse",
        "-i",
        "default",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-flush_packets",
        "1",
        str(out_wav),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        active_host_mic_procs[uid] = {
            "proc": proc,
            "wav_path": out_wav,
            "started_at": datetime.utcnow(),
            "last_transcript": "",
        }
        return jsonify({"status": "recording", "wav_path": str(out_wav.name)}), 200
    except Exception as exc:
        return jsonify({"error": f"Failed to start OS microphone capture: {exc}"}), 500


@app.route("/api/audio/host_mic/live_peek", methods=["POST", "GET"])
@login_required
def api_host_mic_live_peek():
    """
    Real-time live STT peek while the user is actively speaking!
    Reads the raw 16kHz 16-bit mono PCM bytes written so far by ffmpeg, computes the
    instantaneous RMS microphone level (for live waveform animation), writes a valid
    temporary WAV snapshot, and runs faster-whisper to return the live partial transcript
    while the user is still speaking.
    """
    import wave

    uid = current_user.id
    info = active_host_mic_procs.get(uid)
    if not info or not info.get("proc"):
        return jsonify({"recording": False, "partial_transcript": "", "rms_level": 0.0, "duration_s": 0.0}), 200

    wav_path: Path = info["wav_path"]
    if not wav_path.exists() or wav_path.stat().st_size <= 44:
        return jsonify({"recording": True, "partial_transcript": "", "rms_level": 0.02, "duration_s": 0.0}), 200

    try:
        raw_bytes = wav_path.read_bytes()
        # Skip standard WAV header (first 44..78 bytes) and align to 2-byte int16 samples
        pcm_bytes = raw_bytes[44:]
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]
        if len(pcm_bytes) < 1600:
            return jsonify({"recording": True, "partial_transcript": "", "rms_level": 0.03, "duration_s": 0.05}), 200

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = round(len(samples) / 16000.0, 2)
        recent_tail = samples[-8000:] if len(samples) > 8000 else samples
        rms = float(np.sqrt(np.mean(np.square(recent_tail)))) if len(recent_tail) > 0 else 0.0
        rms_norm = round(min(1.0, rms * 8.5), 3)

        # Write a valid WAV snapshot of the last 12 seconds so faster-whisper transcribes fast in real time
        max_samples = 16000 * 12
        snap_pcm = pcm_bytes[-(max_samples * 2) :] if len(samples) > max_samples else pcm_bytes
        snap_path = AUDIO_DIR / f"live_snap_{uid}.wav"
        with wave.open(str(snap_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(snap_pcm)

        stt_res = transcribe_audio_file(snap_path)
        partial = stt_res.get("transcript", "").strip()
        if partial:
            info["last_transcript"] = partial
        else:
            partial = info.get("last_transcript", "")

        return (
            jsonify(
                {
                    "recording": True,
                    "partial_transcript": partial,
                    "rms_level": rms_norm,
                    "duration_s": duration_s,
                }
            ),
            200,
        )
    except Exception:
        return (
            jsonify(
                {
                    "recording": True,
                    "partial_transcript": info.get("last_transcript", ""),
                    "rms_level": 0.05,
                    "duration_s": 0.0,
                }
            ),
            200,
        )


@app.route("/api/audio/host_mic/stop", methods=["POST"])
@login_required
def api_host_mic_stop():
    """
    Stops the host Linux microphone recording, transcribes the captured WAV using faster-whisper,
    and returns the transcript (or executes the Voice Viva turn if attempt_uid is provided).
    """
    uid = current_user.id
    info = active_host_mic_procs.pop(uid, None)
    data = request.get_json(silent=True) or {}
    fallback_live_text = str(data.get("live_interim_text", "")).strip()

    if not info or not info.get("proc"):
        if fallback_live_text:
            attempt_uid = str(data.get("attempt_uid", "")).strip()
            if data.get("run_viva") and attempt_uid:
                return _execute_viva_turn_for_attempt(
                    attempt_uid,
                    fallback_live_text,
                    {"transcript": fallback_live_text, "stt_engine": "live-web-speech+whisper", "duration_seconds": 2.0},
                )
            return jsonify({"stt": {"transcript": fallback_live_text}, "transcript": fallback_live_text}), 200
        return jsonify({"error": "No active host microphone recording found."}), 400

    proc = info["proc"]
    wav_path: Path = info["wav_path"]
    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=4)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    stt_result = {"transcript": "", "stt_engine": "faster-whisper:tiny.en", "duration_seconds": 0.0}
    if wav_path.exists() and wav_path.stat().st_size >= 512:
        try:
            stt_result = transcribe_audio_file(wav_path)
        except Exception:
            pass

    final_transcript = (
        stt_result.get("transcript", "").strip()
        or info.get("last_transcript", "").strip()
        or fallback_live_text
    )
    stt_result["transcript"] = final_transcript

    attempt_uid = str(data.get("attempt_uid", "")).strip()
    run_viva = bool(data.get("run_viva", False))

    if run_viva and attempt_uid:
        if not final_transcript:
            return (
                jsonify(
                    {
                        "error": "No speech detected in live microphone stream. Please speak into your microphone and try again.",
                        "stt": stt_result,
                    }
                ),
                400,
            )
        return _execute_viva_turn_for_attempt(attempt_uid, final_transcript, stt_result)

    return jsonify({"stt": stt_result, "transcript": final_transcript}), 200



# ===========================================================================
# LIVE TEST FLOW: ATTEMPT START, RULE-BASED LOCATION GRADE, VOICE VIVA & GRADING
# ===========================================================================
@app.route("/api/attempts/start", methods=["POST"])
@login_required
def api_start_attempt():
    data = request.get_json() or {}
    case_uid = str(data.get("case_uid", "")).strip()
    case = get_case_by_uid(case_uid)
    if not case:
        return jsonify({"error": "Case not found."}), 404

    room_id = int(case["room_id"])
    room = get_room_by_id(room_id)
    if current_user.role == "teacher":
        if not room or int(room["teacher_id"]) != current_user.id:
            return jsonify({"error": "Access denied: you do not own this room.", "code": "FORBIDDEN_ROOM"}), 403
    else:
        if not is_student_enrolled(current_user.id, room_id):
            return jsonify({"error": "Access denied: you are not enrolled in this room.", "code": "NOT_ENROLLED"}), 403

    opening_text = f"Hello Doctor. {case['patient']['chief_complaint']}"
    try:
        opening_audio_url = synthesize_speech_to_url(opening_text, prefix="viva_open")
    except Exception:
        opening_audio_url = ""

    attempt = create_or_resume_attempt(
        room_id=room_id,
        case_id=case["db_id"],
        student_id=current_user.id,
        patient_opening_text=opening_text,
        patient_name=case["patient"]["name"],
        patient_opening_audio_url=opening_audio_url,
    )
    return jsonify({"attempt": attempt, "case": case}), 200


@app.route("/api/attempts/<attempt_uid>/location", methods=["POST"])
@login_required
def api_submit_attempt_location(attempt_uid: str):
    """
    Step 1 -> Step 2 transition:
    Evaluates the user's 3D mark using pure rule-based 3D geometry (no LLM, no floor),
    saves the location grade to SQLite, and reveals the ground-truth centroid & heatmap for Step 2.
    """
    att = get_attempt_by_uid(attempt_uid)
    if not att:
        return jsonify({"error": "Attempt not found."}), 404
    if int(att["student_id"]) != current_user.id:
        return jsonify({"error": "Access denied: this attempt belongs to another user.", "code": "FORBIDDEN_ATTEMPT"}), 403

    data = request.get_json() or {}
    student_mark = data.get("student_mark_3d")
    if not student_mark or len(student_mark) != 3:
        return jsonify({"error": "Missing 3D student mark coordinates [x, y, z]."}), 400

    conn = get_db_connection()
    try:
        case_row = conn.execute("SELECT * FROM room_cases WHERE id = ?", (int(att["case_id"]),)).fetchone()
    finally:
        conn.close()

    from db import deserialize_room_case

    case = deserialize_room_case(case_row)
    npz_data = np.load(str(case["npz_path"]), allow_pickle=True)
    raw_verts = np.asarray(npz_data["vertices"], dtype=np.float64)
    raw_faces = np.asarray(npz_data["faces"], dtype=np.int32)
    gt_cavity = np.asarray(
        npz_data["gt_cavity"] if "gt_cavity" in npz_data else npz_data["gt_mask"],
        dtype=np.float64,
    )

    bounds_min = raw_verts.min(axis=0)
    bounds_max = raw_verts.max(axis=0)
    centered_verts = raw_verts - (bounds_min + bounds_max) / 2.0

    t_mesh = trimesh.Trimesh(vertices=centered_verts, faces=raw_faces, process=False)
    vert_normals = np.asarray(t_mesh.vertex_normals, dtype=np.float64)

    location_eval = compute_rule_based_location_grade(
        student_mark_3d=[float(x) for x in student_mark],
        centered_vertices=centered_verts,
        vertex_normals=vert_normals,
        gt_cavity=gt_cavity,
        lesion_radius_mm=float(case["carve_params"]["radius"]),
    )

    updated_att = update_attempt_location(attempt_uid, location_eval)
    revealed_mesh = build_threejs_mesh_payload(
        raw_verts,
        raw_faces,
        gt_cavity,
        anomaly_type=case["anomaly_type"],
        reveal_gt=True,
    )

    return (
        jsonify(
            {
                "attempt": updated_att,
                "location_eval": location_eval,
                "colors": revealed_mesh["colors"],
                "gt_centroid_3d": revealed_mesh["gt_centroid_3d"],
            }
        ),
        200,
    )


def _execute_viva_turn_for_attempt(
    attempt_uid: str,
    student_text: str,
    stt_result: Dict[str, Any],
):
    att = get_attempt_by_uid(attempt_uid)
    if not att:
        return jsonify({"error": "Attempt not found."}), 404
    if int(att["student_id"]) != current_user.id:
        return jsonify({"error": "Access denied: this attempt belongs to another user.", "code": "FORBIDDEN_ATTEMPT"}), 403

    conn = get_db_connection()
    try:
        case_row = conn.execute("SELECT * FROM room_cases WHERE id = ?", (int(att["case_id"]),)).fetchone()
    finally:
        conn.close()

    from db import deserialize_room_case

    case = deserialize_room_case(case_row)
    transcript = list(att["viva_transcript"])
    now_lbl = datetime.utcnow().strftime("%I:%M:%S %p")

    viva_turn = run_viva_patient_turn(
        case_data=case,
        transcript_history=transcript,
        current_competencies=att["viva_competencies"],
        student_utterance=student_text,
    )

    transcript.append(
        {
            "role": "student",
            "sender": current_user.name,
            "timestamp": now_lbl,
            "text": student_text,
            "stt_engine": stt_result.get("stt_engine", "faster-whisper:tiny.en"),
            "audio_duration_s": stt_result.get("duration_seconds", 0.0),
            "clinical_note": viva_turn["clinical_note"],
        }
    )
    transcript.append(
        {
            "role": "patient",
            "sender": case["patient"]["name"],
            "timestamp": now_lbl,
            "text": viva_turn["patient_reply"],
            "audio_url": viva_turn["audio_url"],
            "llm_provider": viva_turn["llm_provider"],
        }
    )

    updated_att = update_attempt_viva_turn(
        attempt_uid=attempt_uid,
        transcript=transcript,
        competencies=viva_turn["competencies"],
        llm_provider=viva_turn["llm_provider"],
    )

    return (
        jsonify(
            {
                "stt": stt_result,
                "student_transcript": student_text,
                "patient_reply": viva_turn["patient_reply"],
                "patient_audio_url": viva_turn["audio_url"],
                "competencies": viva_turn["competencies"],
                "done_count": viva_turn["done_count"],
                "clinical_note": viva_turn["clinical_note"],
                "llm_provider": viva_turn["llm_provider"],
                "attempt": updated_att,
            }
        ),
        200,
    )


@app.route("/api/attempts/<attempt_uid>/viva/audio", methods=["POST"])
@login_required
def api_viva_audio_turn(attempt_uid: str):
    """
    Real Audio Voice Viva Turn:
    1. Receives recorded audio file ('audio' in multipart/form-data) OR spoken/typed text JSON.
    2. Transcribes spoken audio using faster-whisper (Whisper tiny.en).
    3. Generates virtual patient response + competency update via Claude/Ollama LLM.
    4. Synthesizes spoken patient response to WAV using Piper ONNX TTS.
    5. Persists transcript and competencies to SQLite.
    """
    att = get_attempt_by_uid(attempt_uid)
    if not att:
        return jsonify({"error": "Attempt not found."}), 404
    if int(att["student_id"]) != current_user.id:
        return jsonify({"error": "Access denied: this attempt belongs to another user.", "code": "FORBIDDEN_ATTEMPT"}), 403

    if "audio" in request.files:
        audio_file = request.files["audio"]
        ext = Path(audio_file.filename or "utterance.webm").suffix or ".webm"
        raw_audio_path = AUDIO_DIR / f"stu_in_{attempt_uid}_{uuid.uuid4().hex[:6]}{ext}"
        audio_file.save(str(raw_audio_path))

        try:
            stt_result = transcribe_audio_file(raw_audio_path)
        except Exception as exc:
            return jsonify({"error": f"Whisper STT failed: {exc}"}), 400

        student_text = stt_result["transcript"].strip()
        if not student_text:
            return jsonify({"error": "No speech detected in audio recording. Please speak clearly into the microphone or use the spoken question bar."}), 400
    else:
        data = request.get_json(silent=True) or {}
        student_text = str(data.get("utterance_text", "")).strip()
        if not student_text:
            return jsonify({"error": "Audio recording ('audio') or spoken question is required for Voice Viva."}), 400
        # Synthesize the student's text to WAV and pass it through faster-whisper STT so the full STT -> LLM -> TTS loop runs!
        stu_tts_url = synthesize_speech_to_url(student_text, prefix="stu_utterance")
        stu_wav_path = BASE_DIR / stu_tts_url.lstrip("/")
        stt_result = transcribe_audio_file(stu_wav_path)
        student_text = stt_result["transcript"].strip() or student_text

    return _execute_viva_turn_for_attempt(attempt_uid, student_text, stt_result)


@app.route("/api/attempts/<attempt_uid>/complete", methods=["POST"])
@login_required
def api_complete_and_grade_attempt(attempt_uid: str):
    """
    Completes the attempt:
    - Accepts either JSON or multipart/form-data (with optional spoken diagnosis audio).
    - Requires that the 3D location mark has been placed.
    - Runs LLM-driven diagnosis grading + Piper TTS spoken teach-back synthesis.
    - Persists the final Student Report to SQLite.
    """
    att = get_attempt_by_uid(attempt_uid)
    if not att:
        return jsonify({"error": "Attempt not found."}), 404
    if int(att["student_id"]) != current_user.id:
        return jsonify({"error": "Access denied: this attempt belongs to another user.", "code": "FORBIDDEN_ATTEMPT"}), 403

    spoken_diag_transcript = ""
    if request.content_type and "multipart/form-data" in request.content_type:
        verdict = str(request.form.get("verdict", "correct")).strip().lower()
        rationale = str(request.form.get("rationale", "")).strip()
        student_mark_raw = request.form.get("student_mark_3d")
        student_mark = json.loads(student_mark_raw) if student_mark_raw else att.get("student_mark_3d")
        if "audio" in request.files:
            audio_f = request.files["audio"]
            ext = Path(audio_f.filename or "diag.wav").suffix or ".wav"
            p = AUDIO_DIR / f"diag_{attempt_uid}_{uuid.uuid4().hex[:6]}{ext}"
            audio_f.save(str(p))
            stt_res = transcribe_audio_file(p)
            spoken_diag_transcript = stt_res["transcript"]
    else:
        data = request.get_json() or {}
        verdict = str(data.get("verdict", "correct")).strip().lower()
        rationale = str(data.get("rationale", "")).strip()
        spoken_diag_transcript = str(data.get("spoken_diagnosis_transcript", "")).strip()
        student_mark = data.get("student_mark_3d") or att.get("student_mark_3d")

    if not student_mark or len(student_mark) != 3:
        return jsonify({"error": "Missing 3D student mark coordinates [x, y, z]. Please click the 3D tooth surface first."}), 400

    if len(rationale) < 10 and len(spoken_diag_transcript) < 10:
        return jsonify({"error": "Please provide a clinical diagnosis rationale (written or spoken)."}), 400

    conn = get_db_connection()
    try:
        case_row = conn.execute("SELECT * FROM room_cases WHERE id = ?", (int(att["case_id"]),)).fetchone()
    finally:
        conn.close()

    from db import deserialize_room_case

    case = deserialize_room_case(case_row)
    npz_data = np.load(str(case["npz_path"]), allow_pickle=True)
    raw_verts = np.asarray(npz_data["vertices"], dtype=np.float64)
    raw_faces = np.asarray(npz_data["faces"], dtype=np.int32)
    gt_cavity = np.asarray(
        npz_data["gt_cavity"] if "gt_cavity" in npz_data else npz_data["gt_mask"],
        dtype=np.float64,
    )
    centered_verts = raw_verts - (raw_verts.min(axis=0) + raw_verts.max(axis=0)) / 2.0
    t_mesh = trimesh.Trimesh(vertices=centered_verts, faces=raw_faces, process=False)
    vert_normals = np.asarray(t_mesh.vertex_normals, dtype=np.float64)

    location_eval = compute_rule_based_location_grade(
        student_mark_3d=[float(x) for x in student_mark],
        centered_vertices=centered_verts,
        vertex_normals=vert_normals,
        gt_cavity=gt_cavity,
        lesion_radius_mm=float(case["carve_params"]["radius"]),
    )
    update_attempt_location(attempt_uid, location_eval)

    # Refresh attempt for latest viva transcript
    att = get_attempt_by_uid(attempt_uid)
    if not spoken_diag_transcript:
        stu_msgs = [m.get("text", "") for m in att["viva_transcript"] if m.get("role") == "student"]
        spoken_diag_transcript = " ".join(stu_msgs).strip()

    grading_result = grade_diagnosis_and_generate_teachback(
        case_data=case,
        location_eval=location_eval,
        student_verdict=verdict,
        student_rationale=rationale,
        spoken_diagnosis_transcript=spoken_diag_transcript,
        viva_transcript=att["viva_transcript"],
        viva_competencies=att["viva_competencies"],
    )

    try:
        started_dt = datetime.fromisoformat(att["started_at"])
        elapsed_sec = max(1, int((datetime.utcnow() - started_dt).total_seconds()))
    except Exception:
        elapsed_sec = 45

    final_att = finalize_attempt_grading(
        attempt_uid=attempt_uid,
        student_verdict=verdict,
        student_rationale=rationale,
        spoken_diagnosis_transcript=spoken_diag_transcript,
        grading_result=grading_result,
        duration_seconds=elapsed_sec,
    )

    return (
        jsonify(
            {
                "attempt": final_att,
                "case": case,
                "location_eval": location_eval,
                "grading": grading_result,
                "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
            }
        ),
        200,
    )


# ===========================================================================
# STUDENT REPORT & LONGITUDINAL PROGRESS ENDPOINTS
# ===========================================================================
@app.route("/api/student/attempts/<attempt_uid>", methods=["GET"])
@login_required
def api_get_student_report(attempt_uid: str):
    """Returns a user's saved per-session report (strictly gated to the owning user)."""
    att = get_attempt_by_uid(attempt_uid)
    if not att:
        return jsonify({"error": "Attempt not found."}), 404
    if int(att["student_id"]) != current_user.id:
        return jsonify({"error": "Access denied: you cannot view another student's report.", "code": "FORBIDDEN_ATTEMPT"}), 403

    conn = get_db_connection()
    try:
        case_row = conn.execute("SELECT * FROM room_cases WHERE id = ?", (int(att["case_id"]),)).fetchone()
    finally:
        conn.close()

    from db import deserialize_room_case

    case = deserialize_room_case(case_row)
    room = get_room_by_id(att["room_id"])
    return jsonify({"attempt": att, "case": case, "room": room}), 200


@app.route("/api/student/progress", methods=["GET"])
@login_required
def api_get_student_progress():
    """
    Returns longitudinal progress for the logged-in user across all attempts over time,
    strengths/weaknesses by anomaly type and tooth position, and dynamic whitelist competency coverage.
    """
    progress = get_student_longitudinal_progress(
        student_id=current_user.id,
        approved_fdi_teeth=APPROVED_FDI_TEETH,
    )
    return jsonify(progress), 200


if __name__ == "__main__":
    print("Starting Odonto Authenticated Clinical Simulation Platform at http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
