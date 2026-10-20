"""
db.py

Persistent SQLite database layer for Odonto:
- Users (bcrypt-hashed passwords, roles: 'teacher' | 'student')
- Dynamic Rooms (created by teachers, joined by students via unique room_code, isolated per room)
- Room Cases (whitelist-validated 3D tooth x anomaly cases belonging to a specific room)
- Student Attempts (per-student, per-case persistent session state: 3D mark, rule-based location grade,
  full voice viva transcript, LLM diagnosis grading, AI teach-back explanation + TTS audio URL, duration)

Zero hardcoded users, rooms, cases, or attempts.
"""

import json
import secrets
import sqlite3
import string
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import bcrypt

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "odonto.db"


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target = db_path or DB_PATH
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_db_connection(db_path)
    with conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('teacher', 'student')),
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_code TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                teacher_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS room_enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                room_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                joined_at TEXT NOT NULL,
                UNIQUE (room_id, student_id),
                FOREIGN KEY (room_id) REFERENCES rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS room_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_uid TEXT UNIQUE NOT NULL,
                room_id INTEGER NOT NULL,
                teacher_id INTEGER NOT NULL,
                case_number TEXT NOT NULL,
                tooth_fdi INTEGER NOT NULL,
                tooth_name TEXT NOT NULL,
                short_name TEXT NOT NULL,
                section TEXT NOT NULL,
                quadrant TEXT NOT NULL,
                arch TEXT NOT NULL,
                tooth_type TEXT NOT NULL,
                anatomical_group TEXT NOT NULL,
                target_surface TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                anomaly_badge TEXT NOT NULL,
                rigor TEXT NOT NULL,
                npz_path TEXT NOT NULL,
                obj_url TEXT NOT NULL,
                npz_url TEXT NOT NULL,
                thumbnail_url TEXT NOT NULL,
                carve_params_json TEXT NOT NULL,
                patient_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (room_id) REFERENCES rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (teacher_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS student_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_uid TEXT UNIQUE NOT NULL,
                room_id INTEGER NOT NULL,
                case_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                duration_seconds INTEGER NOT NULL DEFAULT 0,
                student_mark_3d_json TEXT,
                gt_centroid_3d_json TEXT,
                centroid_distance_mm REAL,
                lesion_radius_mm REAL,
                within_margin INTEGER NOT NULL DEFAULT 0,
                surface_alignment_dot REAL NOT NULL DEFAULT 0.0,
                location_score INTEGER NOT NULL DEFAULT 0,
                margin_precision INTEGER NOT NULL DEFAULT 0,
                depth_agreement INTEGER NOT NULL DEFAULT 0,
                student_verdict TEXT NOT NULL DEFAULT '',
                student_rationale TEXT NOT NULL DEFAULT '',
                spoken_diagnosis_transcript TEXT NOT NULL DEFAULT '',
                viva_transcript_json TEXT NOT NULL DEFAULT '[]',
                viva_competencies_json TEXT NOT NULL DEFAULT '{"onset": false, "triggers": false, "duration": false, "medical_history": false}',
                diagnosis_score INTEGER NOT NULL DEFAULT 0,
                history_score INTEGER NOT NULL DEFAULT 0,
                communication_score INTEGER NOT NULL DEFAULT 0,
                composite_score REAL NOT NULL DEFAULT 0.0,
                grade_letter TEXT NOT NULL DEFAULT 'F',
                llm_grading_json TEXT NOT NULL DEFAULT '{}',
                teachback_text TEXT NOT NULL DEFAULT '',
                teachback_audio_url TEXT NOT NULL DEFAULT '',
                llm_provider TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'in_progress',
                FOREIGN KEY (room_id) REFERENCES rooms(id) ON DELETE CASCADE,
                FOREIGN KEY (case_id) REFERENCES room_cases(id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )
    conn.close()


# ---------------------------------------------------------------------------
# User Authentication Helpers (bcrypt)
# ---------------------------------------------------------------------------
def hash_password(plain_password: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except Exception:
        return False


def create_user(
    email: str,
    name: str,
    password: str,
    role: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    clean_email = email.strip().lower()
    clean_role = role.strip().lower()
    if clean_role not in ("teacher", "student"):
        raise ValueError("Role must be 'teacher' or 'student'.")
    if not clean_email or "@" not in clean_email:
        raise ValueError("A valid email address is required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    pw_hash = hash_password(password)
    now_str = datetime.utcnow().isoformat()

    conn = get_db_connection(db_path)
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO users (email, name, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (clean_email, name.strip(), pw_hash, clean_role, now_str),
            )
            user_id = cur.lastrowid
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(email: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dynamic Room Helpers
# ---------------------------------------------------------------------------
def generate_unique_room_code(conn: sqlite3.Connection) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "OD-" + "".join(secrets.choice(alphabet) for _ in range(5))
        exists = conn.execute(
            "SELECT 1 FROM rooms WHERE room_code = ?", (code,)
        ).fetchone()
        if not exists:
            return code


def create_room(
    teacher_id: int,
    name: str,
    description: str = "",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        with conn:
            room_code = generate_unique_room_code(conn)
            now_str = datetime.utcnow().isoformat()
            cur = conn.execute(
                """
                INSERT INTO rooms (room_code, name, description, teacher_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (room_code, name.strip(), description.strip(), int(teacher_id), now_str),
            )
            room_id = cur.lastrowid
            row = conn.execute("SELECT * FROM rooms WHERE id = ?", (room_id,)).fetchone()
            return dict(row)
    finally:
        conn.close()


def add_case_to_room(
    room_id: int,
    teacher_id: int,
    case_data: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        with conn:
            now_str = datetime.utcnow().isoformat()
            cur = conn.execute(
                """
                INSERT INTO room_cases (
                    case_uid, room_id, teacher_id, case_number, tooth_fdi,
                    tooth_name, short_name, section, quadrant, arch,
                    tooth_type, anatomical_group, target_surface,
                    anomaly_type, anomaly_badge, rigor,
                    npz_path, obj_url, npz_url, thumbnail_url,
                    carve_params_json, patient_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_data["case_uid"],
                    int(room_id),
                    int(teacher_id),
                    case_data["case_number"],
                    int(case_data["tooth_fdi"]),
                    case_data["tooth_name"],
                    case_data["short_name"],
                    case_data["section"],
                    case_data["quadrant"],
                    case_data["arch"],
                    case_data["tooth_type"],
                    case_data["anatomical_group"],
                    case_data["target_surface"],
                    case_data["anomaly_type"],
                    case_data["anomaly_badge"],
                    case_data["rigor"],
                    case_data["npz_path"],
                    case_data["obj_url"],
                    case_data["npz_url"],
                    case_data["thumbnail_url"],
                    json.dumps(case_data["carve_params"]),
                    json.dumps(case_data["patient"]),
                    now_str,
                ),
            )
            cid = cur.lastrowid
            row = conn.execute("SELECT * FROM room_cases WHERE id = ?", (cid,)).fetchone()
            return deserialize_room_case(row)
    finally:
        conn.close()


def deserialize_room_case(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["case_id"] = d["case_uid"]
    d["db_id"] = d["id"]
    d["carve_params"] = json.loads(d.pop("carve_params_json"))
    d["patient"] = json.loads(d.pop("patient_json"))
    d["hud_metric"] = f"DEPTH: {d['carve_params'].get('depth', 0.35):.2f}mm"
    return d


def deserialize_attempt(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["student_mark_3d"] = json.loads(d.pop("student_mark_3d_json")) if d.get("student_mark_3d_json") else None
    d["gt_centroid_3d"] = json.loads(d.pop("gt_centroid_3d_json")) if d.get("gt_centroid_3d_json") else None
    d["viva_transcript"] = json.loads(d.pop("viva_transcript_json") or "[]")
    d["viva_competencies"] = json.loads(d.pop("viva_competencies_json") or "{}")
    d["llm_grading"] = json.loads(d.pop("llm_grading_json") or "{}")
    d["within_margin"] = bool(d.get("within_margin"))
    return d


def get_room_by_id(room_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT r.*, u.name AS teacher_name, u.email AS teacher_email
            FROM rooms r
            JOIN users u ON u.id = r.teacher_id
            WHERE r.id = ?
            """,
            (int(room_id),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_room_by_code(room_code: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT r.*, u.name AS teacher_name, u.email AS teacher_email
            FROM rooms r
            JOIN users u ON u.id = r.teacher_id
            WHERE UPPER(r.room_code) = ?
            """,
            (room_code.strip().upper(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def join_room_by_code(
    student_id: int,
    room_code: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    room = get_room_by_code(room_code, db_path)
    if not room:
        return None
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO room_enrollments (room_id, student_id, joined_at)
                VALUES (?, ?, ?)
                """,
                (room["id"], int(student_id), datetime.utcnow().isoformat()),
            )
        return room
    finally:
        conn.close()


def is_student_enrolled(student_id: int, room_id: int, db_path: Optional[Path] = None) -> bool:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM room_enrollments WHERE room_id = ? AND student_id = ?",
            (int(room_id), int(student_id)),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def list_teacher_rooms(teacher_id: int, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT r.*,
                   (SELECT COUNT(*) FROM room_cases rc WHERE rc.room_id = r.id) AS case_count,
                   (SELECT COUNT(*) FROM room_enrollments re WHERE re.room_id = r.id) AS student_count,
                   (SELECT COUNT(*) FROM student_attempts sa WHERE sa.room_id = r.id AND sa.status = 'completed') AS completed_attempts_count
            FROM rooms r
            WHERE r.teacher_id = ?
            ORDER BY r.id DESC
            """,
            (int(teacher_id),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_student_rooms(student_id: int, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT r.*, u.name AS teacher_name, re.joined_at,
                   (SELECT COUNT(*) FROM room_cases rc WHERE rc.room_id = r.id) AS case_count,
                   (SELECT COUNT(*) FROM student_attempts sa WHERE sa.room_id = r.id AND sa.student_id = ? AND sa.status = 'completed') AS my_completed_count
            FROM room_enrollments re
            JOIN rooms r ON r.id = re.room_id
            JOIN users u ON u.id = r.teacher_id
            WHERE re.student_id = ?
            ORDER BY re.id DESC
            """,
            (int(student_id), int(student_id)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_cases_for_room(
    room_id: int,
    student_id: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM room_cases WHERE room_id = ? ORDER BY id ASC",
            (int(room_id),),
        ).fetchall()
        cases = [deserialize_room_case(r) for r in rows]
        if student_id is not None:
            for c in cases:
                att = conn.execute(
                    """
                    SELECT attempt_uid, composite_score, grade_letter, status
                    FROM student_attempts
                    WHERE room_id = ? AND case_id = ? AND student_id = ? AND status = 'completed'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (int(room_id), int(c["db_id"]), int(student_id)),
                ).fetchone()
                if att:
                    c["status"] = "Solved"
                    c["latest_attempt"] = dict(att)
                else:
                    c["status"] = "Unsolved"
                    c["latest_attempt"] = None
        else:
            for c in cases:
                c["status"] = "Active"
        return cases
    finally:
        conn.close()


def get_case_by_uid(case_uid: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM room_cases WHERE case_uid = ?",
            (case_uid.strip(),),
        ).fetchone()
        return deserialize_room_case(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Student Attempt Lifecycle Helpers
# ---------------------------------------------------------------------------
def create_or_resume_attempt(
    room_id: int,
    case_id: int,
    student_id: int,
    patient_opening_text: str,
    patient_name: str,
    patient_opening_audio_url: str = "",
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        with conn:
            # Always create a fresh attempt or resume an in_progress one for this student + case
            existing = conn.execute(
                """
                SELECT * FROM student_attempts
                WHERE room_id = ? AND case_id = ? AND student_id = ? AND status = 'in_progress'
                ORDER BY id DESC LIMIT 1
                """,
                (int(room_id), int(case_id), int(student_id)),
            ).fetchone()
            if existing:
                att = deserialize_attempt(existing)
                if (
                    patient_opening_audio_url
                    and att.get("viva_transcript")
                    and not att["viva_transcript"][0].get("audio_url")
                ):
                    att["viva_transcript"][0]["audio_url"] = patient_opening_audio_url
                    conn.execute(
                        "UPDATE student_attempts SET viva_transcript_json = ? WHERE id = ?",
                        (json.dumps(att["viva_transcript"]), int(att["id"])),
                    )
                return att

            attempt_uid = f"att_{secrets.token_hex(6)}"
            now_str = datetime.utcnow().isoformat()
            initial_transcript = [
                {
                    "role": "patient",
                    "sender": patient_name,
                    "timestamp": datetime.utcnow().strftime("%I:%M:%S %p"),
                    "text": patient_opening_text,
                    "audio_url": patient_opening_audio_url,
                }
            ]
            initial_comp = {
                "onset": False,
                "triggers": False,
                "duration": False,
                "medical_history": False,
            }
            cur = conn.execute(
                """
                INSERT INTO student_attempts (
                    attempt_uid, room_id, case_id, student_id, started_at,
                    viva_transcript_json, viva_competencies_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress')
                """,
                (
                    attempt_uid,
                    int(room_id),
                    int(case_id),
                    int(student_id),
                    now_str,
                    json.dumps(initial_transcript),
                    json.dumps(initial_comp),
                ),
            )
            row = conn.execute("SELECT * FROM student_attempts WHERE id = ?", (cur.lastrowid,)).fetchone()
            return deserialize_attempt(row)
    finally:
        conn.close()


def get_attempt_by_uid(attempt_uid: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM student_attempts WHERE attempt_uid = ?",
            (attempt_uid.strip(),),
        ).fetchone()
        return deserialize_attempt(row) if row else None
    finally:
        conn.close()


def update_attempt_location(
    attempt_uid: str,
    location_eval: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE student_attempts
                SET student_mark_3d_json = ?,
                    gt_centroid_3d_json = ?,
                    centroid_distance_mm = ?,
                    lesion_radius_mm = ?,
                    within_margin = ?,
                    surface_alignment_dot = ?,
                    location_score = ?,
                    margin_precision = ?,
                    depth_agreement = ?
                WHERE attempt_uid = ?
                """,
                (
                    json.dumps(location_eval["student_mark_3d"]),
                    json.dumps(location_eval["gt_centroid_3d"]),
                    float(location_eval["centroid_distance_mm"]),
                    float(location_eval["lesion_radius_mm"]),
                    1 if location_eval["within_margin"] else 0,
                    float(location_eval["surface_alignment_dot"]),
                    int(location_eval["location_score"]),
                    int(location_eval["margin_precision"]),
                    int(location_eval["depth_agreement"]),
                    attempt_uid,
                ),
            )
            row = conn.execute(
                "SELECT * FROM student_attempts WHERE attempt_uid = ?", (attempt_uid,)
            ).fetchone()
            return deserialize_attempt(row)
    finally:
        conn.close()


def update_attempt_viva_turn(
    attempt_uid: str,
    transcript: List[Dict[str, Any]],
    competencies: Dict[str, bool],
    llm_provider: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                UPDATE student_attempts
                SET viva_transcript_json = ?,
                    viva_competencies_json = ?,
                    llm_provider = ?
                WHERE attempt_uid = ?
                """,
                (
                    json.dumps(transcript),
                    json.dumps(competencies),
                    llm_provider,
                    attempt_uid,
                ),
            )
            row = conn.execute(
                "SELECT * FROM student_attempts WHERE attempt_uid = ?", (attempt_uid,)
            ).fetchone()
            return deserialize_attempt(row)
    finally:
        conn.close()


def finalize_attempt_grading(
    attempt_uid: str,
    student_verdict: str,
    student_rationale: str,
    spoken_diagnosis_transcript: str,
    grading_result: Dict[str, Any],
    duration_seconds: int,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        with conn:
            now_str = datetime.utcnow().isoformat()
            conn.execute(
                """
                UPDATE student_attempts
                SET completed_at = ?,
                    duration_seconds = ?,
                    student_verdict = ?,
                    student_rationale = ?,
                    spoken_diagnosis_transcript = ?,
                    diagnosis_score = ?,
                    history_score = ?,
                    communication_score = ?,
                    composite_score = ?,
                    grade_letter = ?,
                    llm_grading_json = ?,
                    teachback_text = ?,
                    teachback_audio_url = ?,
                    llm_provider = ?,
                    status = 'completed'
                WHERE attempt_uid = ?
                """,
                (
                    now_str,
                    max(1, int(duration_seconds)),
                    student_verdict,
                    student_rationale,
                    spoken_diagnosis_transcript,
                    int(grading_result["diagnosis_score"]),
                    int(grading_result["history_score"]),
                    int(grading_result["communication_score"]),
                    float(grading_result["composite_score"]),
                    grading_result["grade_letter"],
                    json.dumps(grading_result),
                    grading_result["teachback_text"],
                    grading_result["teachback_audio_url"],
                    grading_result["llm_provider"],
                    attempt_uid,
                ),
            )
            row = conn.execute(
                "SELECT * FROM student_attempts WHERE attempt_uid = ?", (attempt_uid,)
            ).fetchone()
            return deserialize_attempt(row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Longitudinal Progress & Dynamic Whitelist Competency-Coverage Engine
# ---------------------------------------------------------------------------
def build_competency_coverage_from_attempts(
    attempts: List[Dict[str, Any]],
    approved_fdi_teeth: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Dynamically computes which whitelist-approved tooth x anomaly combinations
    have been tested vs. untested by cross-referencing actual completed attempts
    against the live whitelist (APPROVED_FDI_TEETH).
    """
    # 1. Per-FDI x Anomaly combinations (40 total approved combinations across 28 teeth)
    fdi_matrix: List[Dict[str, Any]] = []
    for fdi, meta in sorted(approved_fdi_teeth.items()):
        for anom in meta["allowed_anomalies"]:
            matching = [
                a for a in attempts
                if int(a["tooth_fdi"]) == int(fdi) and a["anomaly_type"] == anom
            ]
            best_score = max((float(a["composite_score"]) for a in matching), default=None)
            latest_score = float(matching[0]["composite_score"]) if matching else None
            fdi_matrix.append(
                {
                    "tooth_fdi": int(fdi),
                    "tooth_name": meta["name"],
                    "short_name": meta["short_name"],
                    "tooth_type": meta["tooth_type"],
                    "anatomical_group": meta["anatomical_group"],
                    "quadrant": meta["quadrant"],
                    "target_surface": meta["surface"],
                    "anomaly_type": anom,
                    "tested": len(matching) > 0,
                    "attempts_count": len(matching),
                    "best_score": round(best_score, 1) if best_score is not None else None,
                    "latest_score": round(latest_score, 1) if latest_score is not None else None,
                }
            )

    # 2. Canonical Tooth-Type x Anomaly Families derived dynamically from whitelist
    seen_families = []
    for fdi, meta in sorted(approved_fdi_teeth.items()):
        for anom in meta["allowed_anomalies"]:
            pair = (meta["tooth_type"], anom)
            if pair not in seen_families:
                seen_families.append(pair)

    family_matrix: List[Dict[str, Any]] = []
    for t_type, anom in seen_families:
        matching = [
            a for a in attempts
            if a["tooth_type"] == t_type and a["anomaly_type"] == anom
        ]
        best_score = max((float(a["composite_score"]) for a in matching), default=None)
        avg_score = (
            round(sum(float(a["composite_score"]) for a in matching) / len(matching), 1)
            if matching
            else None
        )
        status = (
            "untested"
            if not matching
            else ("mastered" if (avg_score or 0) >= 80.0 else "needs_practice")
        )
        family_matrix.append(
            {
                "tooth_type": t_type,
                "anomaly_type": anom,
                "label": f"{t_type.capitalize()} × {anom}",
                "tested": len(matching) > 0,
                "attempts_count": len(matching),
                "best_score": round(best_score, 1) if best_score is not None else None,
                "avg_score": avg_score,
                "status": status,
            }
        )

    fdi_tested = sum(1 for item in fdi_matrix if item["tested"])
    fam_tested = sum(1 for item in family_matrix if item["tested"])

    return {
        "family_matrix": family_matrix,
        "fdi_matrix": fdi_matrix,
        "tested_fdi_combinations": [item for item in fdi_matrix if item["tested"]],
        "untested_fdi_combinations": [item for item in fdi_matrix if not item["tested"]],
        "summary": {
            "families_tested": fam_tested,
            "families_total": len(family_matrix),
            "family_coverage_pct": round(100.0 * fam_tested / max(1, len(family_matrix)), 1),
            "fdi_combinations_tested": fdi_tested,
            "fdi_combinations_total": len(fdi_matrix),
            "fdi_coverage_pct": round(100.0 * fdi_tested / max(1, len(fdi_matrix)), 1),
        },
    }


def compute_progress_from_attempt_rows(
    attempt_rows: List[Dict[str, Any]],
    approved_fdi_teeth: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes longitudinal performance trends, anomaly/tooth-position breakdowns,
    strengths/weaknesses, and dynamic whitelist competency coverage from a list
    of enriched attempt dictionaries.
    """
    total = len(attempt_rows)
    if total == 0:
        return {
            "summary": {
                "total_attempts": 0,
                "avg_location_score": 0.0,
                "avg_diagnosis_score": 0.0,
                "avg_composite_score": 0.0,
                "avg_centroid_distance_mm": 0.0,
                "avg_duration_seconds": 0,
            },
            "trend": [],
            "by_anomaly_type": {},
            "by_tooth_type": {},
            "by_anatomical_group": {},
            "strengths": [],
            "weaknesses": [],
            "competency_coverage": build_competency_coverage_from_attempts([], approved_fdi_teeth),
            "attempts": [],
        }

    avg_loc = round(sum(float(a["location_score"]) for a in attempt_rows) / total, 1)
    avg_diag = round(sum(float(a["diagnosis_score"]) for a in attempt_rows) / total, 1)
    avg_comp = round(sum(float(a["composite_score"]) for a in attempt_rows) / total, 1)
    avg_dist = round(sum(float(a["centroid_distance_mm"] or 0.0) for a in attempt_rows) / total, 2)
    avg_dur = int(round(sum(int(a["duration_seconds"] or 0) for a in attempt_rows) / total))

    # Chronological trend (oldest -> newest)
    chronological = list(reversed(attempt_rows))
    trend = [
        {
            "attempt_uid": a["attempt_uid"],
            "completed_at": a["completed_at"],
            "case_number": a["case_number"],
            "tooth_fdi": a["tooth_fdi"],
            "anomaly_type": a["anomaly_type"],
            "location_score": a["location_score"],
            "diagnosis_score": a["diagnosis_score"],
            "composite_score": a["composite_score"],
            "centroid_distance_mm": a["centroid_distance_mm"],
        }
        for a in chronological
    ]

    def group_stats(key_fn):
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for a in attempt_rows:
            k = str(key_fn(a))
            buckets.setdefault(k, []).append(a)
        out = {}
        for k, items in buckets.items():
            n = len(items)
            out[k] = {
                "count": n,
                "avg_location_score": round(sum(float(x["location_score"]) for x in items) / n, 1),
                "avg_diagnosis_score": round(sum(float(x["diagnosis_score"]) for x in items) / n, 1),
                "avg_composite_score": round(sum(float(x["composite_score"]) for x in items) / n, 1),
                "avg_centroid_distance_mm": round(sum(float(x["centroid_distance_mm"] or 0.0) for x in items) / n, 2),
            }
        return out

    by_anomaly = group_stats(lambda a: a["anomaly_type"])
    by_tooth_type = group_stats(lambda a: a["tooth_type"])
    by_group = group_stats(lambda a: a["anatomical_group"])

    strengths: List[str] = []
    weaknesses: List[str] = []
    for k, st in by_anomaly.items():
        if st["avg_composite_score"] >= 80:
            strengths.append(f"{k}: Strong concordance ({st['avg_composite_score']}% avg across {st['count']} attempt(s))")
        else:
            weaknesses.append(f"{k}: Needs improvement ({st['avg_composite_score']}% avg across {st['count']} attempt(s))")

    for k, st in by_tooth_type.items():
        if st["avg_composite_score"] >= 80:
            strengths.append(f"{k.capitalize()} teeth: High accuracy ({st['avg_composite_score']}% avg, mean 3D error {st['avg_centroid_distance_mm']}mm)")
        else:
            weaknesses.append(f"{k.capitalize()} teeth: Lower accuracy ({st['avg_composite_score']}% avg, mean 3D error {st['avg_centroid_distance_mm']}mm)")

    return {
        "summary": {
            "total_attempts": total,
            "avg_location_score": avg_loc,
            "avg_diagnosis_score": avg_diag,
            "avg_composite_score": avg_comp,
            "avg_centroid_distance_mm": avg_dist,
            "avg_duration_seconds": avg_dur,
        },
        "trend": trend,
        "by_anomaly_type": by_anomaly,
        "by_tooth_type": by_tooth_type,
        "by_anatomical_group": by_group,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "competency_coverage": build_competency_coverage_from_attempts(attempt_rows, approved_fdi_teeth),
        "attempts": attempt_rows,
    }


def get_student_longitudinal_progress(
    student_id: int,
    approved_fdi_teeth: Dict[int, Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    conn = get_db_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT sa.*,
                   rc.case_uid, rc.case_number, rc.tooth_fdi, rc.tooth_name, rc.short_name,
                   rc.tooth_type, rc.anatomical_group, rc.quadrant, rc.target_surface,
                   rc.anomaly_type, rc.anomaly_badge,
                   r.name AS room_name, r.room_code
            FROM student_attempts sa
            JOIN room_cases rc ON rc.id = sa.case_id
            JOIN rooms r ON r.id = sa.room_id
            WHERE sa.student_id = ? AND sa.status = 'completed'
            ORDER BY sa.id DESC
            """,
            (int(student_id),),
        ).fetchall()
        enriched = [deserialize_attempt(r) for r in rows]
        return compute_progress_from_attempt_rows(enriched, approved_fdi_teeth)
    finally:
        conn.close()


def get_teacher_room_dashboard(
    teacher_id: int,
    room_id: int,
    approved_fdi_teeth: Dict[int, Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    try:
        room_row = conn.execute(
            "SELECT * FROM rooms WHERE id = ? AND teacher_id = ?",
            (int(room_id), int(teacher_id)),
        ).fetchone()
        if not room_row:
            return None
        room = dict(room_row)

        cases = list_cases_for_room(room_id, db_path=db_path)
        enrolled_rows = conn.execute(
            """
            SELECT u.id AS student_id, u.name AS student_name, u.email AS student_email, re.joined_at
            FROM room_enrollments re
            JOIN users u ON u.id = re.student_id
            WHERE re.room_id = ?
            ORDER BY re.joined_at ASC
            """,
            (int(room_id),),
        ).fetchall()

        attempt_rows = conn.execute(
            """
            SELECT sa.*,
                   u.name AS student_name, u.email AS student_email,
                   rc.case_uid, rc.case_number, rc.tooth_fdi, rc.tooth_name, rc.short_name,
                   rc.tooth_type, rc.anatomical_group, rc.quadrant, rc.target_surface,
                   rc.anomaly_type, rc.anomaly_badge,
                   r.name AS room_name, r.room_code
            FROM student_attempts sa
            JOIN users u ON u.id = sa.student_id
            JOIN room_cases rc ON rc.id = sa.case_id
            JOIN rooms r ON r.id = sa.room_id
            WHERE sa.room_id = ? AND sa.status = 'completed'
            ORDER BY sa.id DESC
            """,
            (int(room_id),),
        ).fetchall()
        attempts = [deserialize_attempt(r) for r in attempt_rows]

        students_summary = []
        for st in enrolled_rows:
            sid = int(st["student_id"])
            st_attempts = [a for a in attempts if int(a["student_id"]) == sid]
            completed_case_ids = {int(a["case_id"]) for a in st_attempts}
            n_att = len(st_attempts)
            students_summary.append(
                {
                    "student_id": sid,
                    "student_name": st["student_name"],
                    "student_email": st["student_email"],
                    "joined_at": st["joined_at"],
                    "completed_cases_count": len(completed_case_ids),
                    "total_cases_count": len(cases),
                    "attempts_count": n_att,
                    "avg_location_score": round(sum(float(a["location_score"]) for a in st_attempts) / n_att, 1) if n_att else None,
                    "avg_diagnosis_score": round(sum(float(a["diagnosis_score"]) for a in st_attempts) / n_att, 1) if n_att else None,
                    "avg_composite_score": round(sum(float(a["composite_score"]) for a in st_attempts) / n_att, 1) if n_att else None,
                    "total_time_seconds": sum(int(a["duration_seconds"] or 0) for a in st_attempts),
                    "attempts": st_attempts,
                }
            )

        class_progress = compute_progress_from_attempt_rows(attempts, approved_fdi_teeth)

        return {
            "room": room,
            "cases": cases,
            "enrolled_students": students_summary,
            "attempts": attempts,
            "class_progress": class_progress,
        }
    finally:
        conn.close()

