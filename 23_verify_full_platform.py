"""
23_verify_full_platform.py

Comprehensive End-to-End Verification of:
1. Unauthenticated access rejection (HTTP 401 on all protected routes)
2. Flask-Login + bcrypt authentication & strict role gating (HTTP 403 on cross-role access)
3. Dynamic Room Creation by Teachers, Shareable Join Codes, Whitelist Enforcement (HTTP 422),
   and Strict Room Isolation between Teacher A and Teacher B (HTTP 403)
4. Blind 3D Mesh Delivery (no gt_centroid_3d or heatmap leak before Step 1 submission)
5. Generic Grading Validation:
   - Deliberately BAD 3D mark + wrong diagnosis ("tooth is normal") -> scores LOW (Location 0-15%, Diagnosis 0-25%, Grade F)
   - Accurate 3D mark + real audio Voice Viva (Whisper STT -> LLM -> Piper TTS) + accurate spoken/written diagnosis -> scores HIGH (Grade A/B)
6. Persistent Student Report, Dynamic Longitudinal Progress & Whitelist Competency-Coverage Matrix,
   and Teacher Dashboard + Single-Attempt Replay Isolation
"""

import io
import json
import os
import wave
from pathlib import Path

import numpy as np
from piper.voice import PiperVoice

import db
from app import app
from voice_llm_service import PIPER_MODEL_PATH


def make_spoken_wav_bytes(text: str) -> bytes:
    """Generates a real 16kHz PCM WAV audio buffer using Piper TTS for audio-endpoint testing."""
    voice = PiperVoice.load(str(PIPER_MODEL_PATH))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf)
    return buf.getvalue()


def run_verification():
    # Use an isolated test database file so we never touch the user's live odonto.db
    test_db = Path(__file__).resolve().parent / "odonto_test_verify.db"
    if test_db.exists():
        test_db.unlink()
    db.DB_PATH = test_db
    db.init_db(test_db)

    client_anon = app.test_client()
    client_teacher_a = app.test_client()
    client_teacher_b = app.test_client()
    client_student_1 = app.test_client()
    client_student_2 = app.test_client()

    checks = []

    def check(name: str, cond: bool, detail: str = ""):
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {name}: {detail}")
        checks.append((name, cond, detail))
        if not cond:
            raise AssertionError(f"Check failed: {name} -> {detail}")

    print("=" * 84)
    print("ODONTO AUTHENTICATED DYNAMIC PLATFORM — VERIFICATION SUITE")
    print("=" * 84)

    # -------------------------------------------------------------------------
    # 1. Unauthenticated Requests Must Be Rejected (HTTP 401)
    # -------------------------------------------------------------------------
    for ep, method in [
        ("/api/auth/me", "GET"),
        ("/api/whitelist", "GET"),
        ("/api/teacher/rooms", "GET"),
        ("/api/rooms", "POST"),
        ("/api/rooms/join", "POST"),
        ("/api/student/rooms", "GET"),
        ("/api/student/progress", "GET"),
        ("/api/attempts/start", "POST"),
    ]:
        r = client_anon.get(ep) if method == "GET" else client_anon.post(ep, json={})
        check(
            f"Unauthenticated Guard [{method} {ep}]",
            r.status_code == 401 and r.get_json().get("code") == "UNAUTHENTICATED",
            f"HTTP {r.status_code}",
        )

    # -------------------------------------------------------------------------
    # 2. Register Teachers & Students (bcrypt password verification)
    # -------------------------------------------------------------------------
    r_ta = client_teacher_a.post(
        "/api/auth/register",
        json={
            "email": "dr.chen@dental.edu",
            "name": "Dr. Sarah Chen",
            "password": "TeacherPass123!",
            "role": "teacher",
        },
    )
    r_tb = client_teacher_b.post(
        "/api/auth/register",
        json={
            "email": "dr.patel@dental.edu",
            "name": "Dr. Raj Patel",
            "password": "TeacherPass456!",
            "role": "teacher",
        },
    )
    r_s1 = client_student_1.post(
        "/api/auth/register",
        json={
            "email": "alex.rivera@dental.edu",
            "name": "Alex Rivera",
            "password": "StudentPass123!",
            "role": "student",
        },
    )
    r_s2 = client_student_2.post(
        "/api/auth/register",
        json={
            "email": "sam.lee@dental.edu",
            "name": "Sam Lee",
            "password": "StudentPass456!",
            "role": "student",
        },
    )

    user_ta_row = db.get_user_by_email("dr.chen@dental.edu")
    check(
        "User Registration & bcrypt Password Hashing",
        r_ta.status_code == 201
        and r_tb.status_code == 201
        and r_s1.status_code == 201
        and r_s2.status_code == 201
        and user_ta_row["password_hash"].startswith("$2b$12$"),
        f"bcrypt hash prefix='{user_ta_row['password_hash'][:10]}...'",
    )

    # Role gating: Student 1 cannot create a room or view teacher rooms
    r_s1_create_room = client_student_1.post("/api/rooms", json={"name": "Illegal Student Room"})
    r_ta_join_room = client_teacher_a.post("/api/rooms/join", json={"room_code": "OD-12345"})
    check(
        "Strict Role Gating (Teacher vs Student)",
        r_s1_create_room.status_code == 403
        and r_s1_create_room.get_json().get("code") == "FORBIDDEN_ROLE"
        and r_ta_join_room.status_code == 403
        and r_ta_join_room.get_json().get("code") == "FORBIDDEN_ROLE",
        f"Student->create_room: HTTP {r_s1_create_room.status_code}, Teacher->join_room: HTTP {r_ta_join_room.status_code}",
    )

    # -------------------------------------------------------------------------
    # 3. Dynamic Room Creation + Server-Side Whitelist Enforcement
    # -------------------------------------------------------------------------
    # 3a. Teacher A attempts to create a room with Posterior Abrasion (#34 Premolar Abrasion) -> HTTP 422
    r_bad_room = client_teacher_a.post(
        "/api/rooms",
        json={
            "name": "Invalid Posterior Abrasion Room",
            "cases": [{"tooth_fdi": 34, "anomaly_type": "Abrasion"}],
        },
    )
    check(
        "Room Creation Whitelist Guard (Posterior Abrasion Rejected)",
        r_bad_room.status_code == 422
        and r_bad_room.get_json().get("code") == "POSTERIOR_ABRASION_EXCLUDED",
        f"HTTP {r_bad_room.status_code} ({r_bad_room.get_json().get('code')})",
    )

    # 3b. Teacher A creates Room A with 2 whitelisted cases (#36 Molar Cavity, #31 Incisor Abrasion)
    r_room_a = client_teacher_a.post(
        "/api/rooms",
        json={
            "name": "Operative Cariology Cohort A",
            "description": "Mandibular Molar Cavity & Incisor Labial Abrasion",
            "cases": [
                {"tooth_fdi": 36, "anomaly_type": "Cavity", "rigor": "Intermediate", "patient_name": "Maya Lin"},
                {"tooth_fdi": 31, "anomaly_type": "Abrasion", "rigor": "Beginner", "patient_name": "David Chen"},
            ],
        },
    )
    room_a_data = r_room_a.get_json()
    room_a_id = room_a_data["room"]["id"]
    room_a_code = room_a_data["room"]["room_code"]
    room_a_cases = room_a_data["cases"]

    # 3c. Teacher B creates Room B with 1 whitelisted case (#34 Premolar Cavity)
    r_room_b = client_teacher_b.post(
        "/api/rooms",
        json={
            "name": "Bicuspid Restorative Cohort B",
            "description": "Premolar Occlusal Fossa Caries",
            "cases": [
                {"tooth_fdi": 34, "anomaly_type": "Cavity", "rigor": "Advanced", "patient_name": "Elena Rostova"},
            ],
        },
    )
    room_b_data = r_room_b.get_json()
    room_b_id = room_b_data["room"]["id"]
    room_b_code = room_b_data["room"]["room_code"]

    check(
        "Dynamic Room Creation & Distinct Join Codes",
        r_room_a.status_code == 201
        and r_room_b.status_code == 201
        and room_a_code != room_b_code
        and len(room_a_cases) == 2,
        f"Room A='{room_a_code}' (2 cases), Room B='{room_b_code}' (1 case)",
    )

    # 3d. Cross-teacher isolation: Teacher A cannot view Teacher B's room dashboard or cases
    r_ta_on_b = client_teacher_a.get(f"/api/teacher/rooms/{room_b_id}/dashboard")
    check(
        "Cross-Teacher Room Isolation",
        r_ta_on_b.status_code == 403 and r_ta_on_b.get_json().get("code") == "FORBIDDEN_ROOM",
        f"Teacher A accessing Room B dashboard -> HTTP {r_ta_on_b.status_code}",
    )

    # -------------------------------------------------------------------------
    # 4. Student Room Join & Cross-Room Isolation
    # -------------------------------------------------------------------------
    # Student 1 joins Room A via room_a_code
    r_join_s1 = client_student_1.post("/api/rooms/join", json={"room_code": room_a_code})
    r_s1_on_b = client_student_1.get(f"/api/rooms/{room_b_id}/cases")
    check(
        "Student Room Join via Code & Unenrolled Room Isolation",
        r_join_s1.status_code == 200
        and len(r_join_s1.get_json()["cases"]) == 2
        and r_s1_on_b.status_code == 403
        and r_s1_on_b.get_json().get("code") == "NOT_ENROLLED",
        f"Student 1 joined {room_a_code} (2 cases); access to Room B -> HTTP {r_s1_on_b.status_code}",
    )

    # Student 2 also joins Room A to test bad-input grading & inter-student report isolation
    client_student_2.post("/api/rooms/join", json={"room_code": room_a_code})

    case_36_uid = room_a_cases[0]["case_uid"]
    case_31_uid = room_a_cases[1]["case_uid"]

    # -------------------------------------------------------------------------
    # 5. Blind 3D Mesh Delivery (No Ground-Truth Leak in Step 1)
    # -------------------------------------------------------------------------
    r_mesh_blind = client_student_1.get(f"/api/rooms/cases/{case_36_uid}/mesh")
    md_blind = r_mesh_blind.get_json()["mesh_data"]
    check(
        "Blind 3D Mesh Delivery in Step 1 (No GT Centroid or Heatmap Leak)",
        r_mesh_blind.status_code == 200
        and md_blind["gt_revealed"] is False
        and "gt_centroid_3d" not in md_blind
        and "colors" not in md_blind
        and len(md_blind["enamel_colors"]) > 3000,
        f"gt_revealed={md_blind['gt_revealed']}, vertices={md_blind['num_vertices']}",
    )

    # -------------------------------------------------------------------------
    # 6. DELIBERATELY BAD INPUT TEST (Student 2 clicks wrong location + wrong diagnosis)
    # -------------------------------------------------------------------------
    r_start_s2 = client_student_2.post("/api/attempts/start", json={"case_uid": case_36_uid})
    att_s2_uid = r_start_s2.get_json()["attempt"]["attempt_uid"]

    # Student 2 places a mark far off / on the opposite side of the tooth: [8.0, 8.0, -8.0]
    r_loc_bad = client_student_2.post(
        f"/api/attempts/{att_s2_uid}/location",
        json={"student_mark_3d": [8.0, 8.0, -8.0]},
    )
    loc_bad_eval = r_loc_bad.get_json()["location_eval"]

    # Student 2 submits a deliberately wrong rationale claiming the tooth is completely normal
    r_comp_bad = client_student_2.post(
        f"/api/attempts/{att_s2_uid}/complete",
        json={
            "student_mark_3d": [8.0, 8.0, -8.0],
            "verdict": "correct",
            "rationale": "This tooth looks completely normal and healthy with no cavity or pathology at all.",
        },
    )
    bad_result = r_comp_bad.get_json()
    bad_att = bad_result["attempt"]

    check(
        "Deliberately Bad Input Scores Low (No 35% Location Floor, No 92% Verdict Floor, Grade F)",
        r_comp_bad.status_code == 200
        and bad_att["location_score"] <= 15
        and bad_att["diagnosis_score"] <= 25
        and bad_att["composite_score"] <= 20.0
        and bad_att["grade_letter"] == "F",
        f"Bad mark [8,8,-8] + 'tooth is normal' -> Location={bad_att['location_score']}%, Diagnosis={bad_att['diagnosis_score']}%, Composite={bad_att['composite_score']}%, Grade={bad_att['grade_letter']}",
    )

    # -------------------------------------------------------------------------
    # 7. GOOD INPUT + REAL AUDIO VOICE VIVA LOOP TEST (Student 1)
    # -------------------------------------------------------------------------
    r_start_s1 = client_student_1.post("/api/attempts/start", json={"case_uid": case_36_uid})
    att_s1_uid = r_start_s1.get_json()["attempt"]["attempt_uid"]

    # Verify all 4 Viva competencies start at False!
    init_comp = r_start_s1.get_json()["attempt"]["viva_competencies"]
    check(
        "Initial Viva Competencies All False",
        all(v is False for v in init_comp.values()),
        f"initial_competencies={init_comp}",
    )

    # Obtain ground-truth centroid from Teacher A's view so we can test a realistic accurate click on the occlusal fossa
    r_mesh_teacher = client_teacher_a.get(f"/api/rooms/cases/{case_36_uid}/mesh")
    gt_centroid_36 = r_mesh_teacher.get_json()["mesh_data"]["gt_centroid_3d"]
    good_mark_36 = [
        round(gt_centroid_36[0] + 0.18, 3),
        round(gt_centroid_36[1] - 0.14, 3),
        round(gt_centroid_36[2] + 0.08, 3),
    ]

    r_loc_good = client_student_1.post(
        f"/api/attempts/{att_s1_uid}/location",
        json={"student_mark_3d": good_mark_36},
    )
    loc_good_eval = r_loc_good.get_json()["location_eval"]
    check(
        "Rule-Based Location Grading on Accurate 3D Mark",
        r_loc_good.status_code == 200
        and loc_good_eval["within_margin"] is True
        and loc_good_eval["location_score"] >= 85,
        f"dist={loc_good_eval['centroid_distance_mm']}mm, location_score={loc_good_eval['location_score']}%",
    )

    # Real Audio Voice Viva Turn 1: Student asks about onset, cold/sweet triggers, and duration via WAV audio
    wav_q1 = make_spoken_wav_bytes(
        "When did the pain first start, does cold water or sweet food trigger it, and how many seconds does the sensitivity last?"
    )
    r_viva_1 = client_student_1.post(
        f"/api/attempts/{att_s1_uid}/viva/audio",
        data={"audio": (io.BytesIO(wav_q1), "question1.wav")},
        content_type="multipart/form-data",
    )
    viva1_json = r_viva_1.get_json()
    patient_wav_path = Path(__file__).resolve().parent / viva1_json["patient_audio_url"].lstrip("/")
    check(
        "Voice Viva Audio Turn 1 (Whisper STT -> LLM -> Piper TTS WAV)",
        r_viva_1.status_code == 200
        and "cold" in viva1_json["student_transcript"].lower()
        and len(viva1_json["patient_reply"]) > 15
        and patient_wav_path.exists()
        and patient_wav_path.stat().st_size > 1000,
        f"STT='{viva1_json['student_transcript']}', Reply='{viva1_json['patient_reply'][:60]}...', TTS WAV={patient_wav_path.stat().st_size} bytes ({viva1_json['llm_provider']})",
    )

    # Real Audio Voice Viva Turn 2: Student asks about medical history and drug allergies via WAV audio
    wav_q2 = make_spoken_wav_bytes(
        "Do you have any medical conditions, daily medications, or allergies to dental local anesthetics or antibiotics?"
    )
    r_viva_2 = client_student_1.post(
        f"/api/attempts/{att_s1_uid}/viva/audio",
        data={"audio": (io.BytesIO(wav_q2), "question2.wav")},
        content_type="multipart/form-data",
    )
    viva2_json = r_viva_2.get_json()
    check(
        "Voice Viva Audio Turn 2 (Competencies Updated Dynamically)",
        r_viva_2.status_code == 200 and viva2_json["done_count"] >= 3,
        f"STT='{viva2_json['student_transcript']}', competencies={viva2_json['competencies']} ({viva2_json['done_count']}/4)",
    )

    # Complete Attempt with Spoken Diagnosis Audio + Written Rationale
    wav_diag = make_spoken_wav_bytes(
        "My diagnosis is a localized occlusal fossa cavity on mandibular left first molar tooth thirty six with reversible pulpitis. I recommend minimally invasive caries removal and composite resin restoration."
    )
    r_comp_good = client_student_1.post(
        f"/api/attempts/{att_s1_uid}/complete",
        data={
            "student_mark_3d": json.dumps(good_mark_36),
            "verdict": "correct",
            "rationale": "Occlusal fossa carious cavity on FDI Tooth #36 (Mandibular Left First Molar) extending into outer dentin with reversible pulpitis; minimally invasive adhesive composite restoration indicated.",
            "audio": (io.BytesIO(wav_diag), "spoken_diagnosis.wav"),
        },
        content_type="multipart/form-data",
    )
    good_result = r_comp_good.get_json()
    good_att = good_result["attempt"]
    teachback_wav_path = Path(__file__).resolve().parent / good_att["teachback_audio_url"].lstrip("/")

    check(
        "Accurate Diagnosis Graded High + Spoken Teach-Back WAV Synthesized",
        r_comp_good.status_code == 200
        and good_att["location_score"] >= 85
        and good_att["diagnosis_score"] >= 80
        and good_att["composite_score"] >= 82.0
        and good_att["grade_letter"] in ("A", "B")
        and teachback_wav_path.exists()
        and teachback_wav_path.stat().st_size > 2000,
        f"Location={good_att['location_score']}%, Diagnosis={good_att['diagnosis_score']}%, Composite={good_att['composite_score']}% ({good_att['grade_letter']}), TeachBack WAV={teachback_wav_path.stat().st_size} bytes",
    )

    # -------------------------------------------------------------------------
    # 8. Student Report Isolation & Dynamic Longitudinal Progress / Whitelist Coverage
    # -------------------------------------------------------------------------
    # Student 2 tries to view Student 1's report -> HTTP 403
    r_s2_steal = client_student_2.get(f"/api/student/attempts/{att_s1_uid}")
    r_s1_report = client_student_1.get(f"/api/student/attempts/{att_s1_uid}")
    check(
        "Student Report Persistence & Inter-Student Isolation",
        r_s2_steal.status_code == 403
        and r_s1_report.status_code == 200
        and len(r_s1_report.get_json()["attempt"]["viva_transcript"]) >= 5,
        f"Student 2 accessing Student 1 report -> HTTP {r_s2_steal.status_code}; Student 1 report transcript turns={len(r_s1_report.get_json()['attempt']['viva_transcript'])}",
    )

    # Check Student 1 Longitudinal Progress & Dynamic Whitelist Coverage after 1 case (#36 Molar Cavity)
    r_prog_1 = client_student_1.get("/api/student/progress").get_json()
    cov_1 = r_prog_1["competency_coverage"]["summary"]
    check(
        "Dynamic Whitelist Coverage After 1 Attempt (#36 Molar Cavity)",
        r_prog_1["summary"]["total_attempts"] == 1
        and cov_1["families_tested"] == 1
        and cov_1["families_total"] == 6
        and cov_1["fdi_combinations_tested"] == 1
        and cov_1["fdi_combinations_total"] == 40,
        f"Families={cov_1['families_tested']}/{cov_1['families_total']}, FDI combos={cov_1['fdi_combinations_tested']}/{cov_1['fdi_combinations_total']}",
    )

    # Student 1 now completes the second room case (#31 Incisor Abrasion) to verify longitudinal update
    r_start_31 = client_student_1.post("/api/attempts/start", json={"case_uid": case_31_uid})
    att_31_uid = r_start_31.get_json()["attempt"]["attempt_uid"]
    gt_centroid_31 = client_teacher_a.get(f"/api/rooms/cases/{case_31_uid}/mesh").get_json()["mesh_data"]["gt_centroid_3d"]
    client_student_1.post(
        f"/api/attempts/{att_31_uid}/location",
        json={"student_mark_3d": gt_centroid_31},
    )
    client_student_1.post(
        f"/api/attempts/{att_31_uid}/complete",
        json={
            "student_mark_3d": gt_centroid_31,
            "verdict": "correct",
            "rationale": "Cervical-middle labial toothbrush abrasion facet on FDI Tooth #31 (Mandibular Left Central Incisor); recommend soft-bristle brushing technique and glass-ionomer or composite restoration.",
        },
    )

    r_prog_2 = client_student_1.get("/api/student/progress").get_json()
    cov_2 = r_prog_2["competency_coverage"]["summary"]
    check(
        "Dynamic Whitelist Coverage Updates After 2nd Attempt (#31 Incisor Abrasion)",
        r_prog_2["summary"]["total_attempts"] == 2
        and cov_2["families_tested"] == 2
        and cov_2["fdi_combinations_tested"] == 2
        and "Cavity" in r_prog_2["by_anomaly_type"]
        and "Abrasion" in r_prog_2["by_anomaly_type"],
        f"Families={cov_2['families_tested']}/{cov_2['families_total']}, FDI combos={cov_2['fdi_combinations_tested']}/{cov_2['fdi_combinations_total']}",
    )

    # -------------------------------------------------------------------------
    # 9. Teacher Dashboard & Single-Attempt Drill-Down Replay
    # -------------------------------------------------------------------------
    r_dash_a = client_teacher_a.get(f"/api/teacher/rooms/{room_a_id}/dashboard").get_json()
    r_replay_a = client_teacher_a.get(f"/api/teacher/attempts/{att_s1_uid}")
    r_replay_b_blocked = client_teacher_b.get(f"/api/teacher/attempts/{att_s1_uid}")

    check(
        "Teacher Dashboard & Attempt Replay Isolation",
        len(r_dash_a["enrolled_students"]) == 2
        and len(r_dash_a["attempts"]) == 3
        and r_replay_a.status_code == 200
        and r_replay_a.get_json()["attempt"]["attempt_uid"] == att_s1_uid
        and r_replay_b_blocked.status_code == 403,
        f"Room A students={len(r_dash_a['enrolled_students'])}, attempts={len(r_dash_a['attempts'])}; Teacher B replay blocked with HTTP {r_replay_b_blocked.status_code}",
    )

    print("=" * 84)
    print(f"ALL {len(checks)} VERIFICATION CHECKS PASSED.")
    print("=" * 84)


if __name__ == "__main__":
    run_verification()
