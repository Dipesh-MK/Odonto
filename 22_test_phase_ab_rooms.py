"""
22_test_phase_ab_rooms.py

End-to-End Verification Suite for Phase A/B:
1. Phase 0 Whitelist Enforcement (GET /api/whitelist, POST /api/carve, POST /api/rooms/cases)
2. Supported Tooth x Anomaly Generation & Zero Bleed-Through Verification
3. Rejected Tooth x Anomaly Combinations (Posterior Abrasion on Premolars/Molars, 3rd Molars, Unvalidated Anomalies -> HTTP 422)
4. Room Case Library & 3D Mesh Payload Verification (GET /api/rooms/cases, GET /api/rooms/cases/<id>/mesh)
5. 3-Step AI Diagnostic Evaluation & Spatial Concordance (POST /api/rooms/evaluate)
6. Voice Viva Session Dialogue & Examiner Grading (POST /api/viva/chat, POST /api/viva/grade)
7. Stitch 5-View UI Template Verification (GET /)
"""

import json
import sys
import numpy as np

from app import app, ROOM_CASES, SESSION_STATE
from whitelist import validate_tooth_anomaly_combination, APPROVED_FDI_TEETH


def run_tests():
    client = app.test_client()
    results = []

    def record(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")
        results.append((name, passed, detail))
        if not passed:
            raise AssertionError(f"Test failed: {name} -> {detail}")

    print("=" * 80)
    print("PHASE A/B END-TO-END VERIFICATION SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Stitch 5-View HTML & Navigation Verification
    # -------------------------------------------------------------------------
    res_index = client.get("/")
    html = res_index.get_data(as_text=True)
    expected_views = [
        'id="view-case-library"',
        'id="view-3d-viewer-and-sim"',
        'id="view-evaluate-the-ai"',
        'id="view-voice-viva-session"',
        'id="view-results-and-performance"',
        'data-path="case-library"',
        'data-path="3d-viewer-and-sim"',
        'data-path="evaluate-the-ai"',
        'data-path="voice-viva-session"',
        'data-path="results-and-performance"',
        'Build v4.2.1-ClinicalCore',
    ]
    missing_views = [v for v in expected_views if v not in html]
    record(
        "Stitch 5-View HTML Structure",
        res_index.status_code == 200 and len(missing_views) == 0,
        f"HTTP {res_index.status_code}, all 5 Stitch views & nav data-paths present",
    )

    # -------------------------------------------------------------------------
    # 2. Whitelist Endpoint & Authoritative Rules
    # -------------------------------------------------------------------------
    res_wl = client.get("/api/whitelist")
    wl_data = res_wl.get_json()
    record(
        "GET /api/whitelist",
        res_wl.status_code == 200
        and len(wl_data["teeth"]) == 28
        and wl_data["summary"]["posterior_abrasion_enabled"] is False
        and "posterior_abrasion" in wl_data["excluded_notes"],
        f"28 approved FDI teeth (#11-#47), posterior_abrasion_enabled=False, depth calibration note present",
    )

    # -------------------------------------------------------------------------
    # 3. Pre-Seeded Room Cases & 3D Mesh Payloads
    # -------------------------------------------------------------------------
    res_cases = client.get("/api/rooms/cases")
    cases_payload = res_cases.get_json()
    cases_list = cases_payload["cases"]
    record(
        "GET /api/rooms/cases (Pre-seeded Phase 0 Cases)",
        res_cases.status_code == 200 and len(cases_list) == 6,
        f"{len(cases_list)} pre-seeded cases loaded",
    )

    for c in cases_list:
        ok_combo, _ = validate_tooth_anomaly_combination(c["tooth_fdi"], c["anomaly_type"])
        res_mesh = client.get(f"/api/rooms/cases/{c['case_id']}/mesh")
        mesh_json = res_mesh.get_json()
        md = mesh_json["mesh_data"]
        record(
            f"Room Case Mesh Payload ({c['case_id']}: FDI #{c['tooth_fdi']} {c['anomaly_type']})",
            res_mesh.status_code == 200
            and ok_combo
            and md["num_vertices"] > 1000
            and md["num_faces"] > 2000
            and len(md["gt_centroid_3d"]) == 3
            and c["carve_params"]["opposite_bleed_vertices"] == 0,
            f"V={md['num_vertices']}, F={md['num_faces']}, gt_centroid={md['gt_centroid_3d']}, bleed=0",
        )

    # -------------------------------------------------------------------------
    # 4. Rejected Combinations (Whitelist Guard -> HTTP 422)
    # -------------------------------------------------------------------------
    rejected_test_cases = [
        (34, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Lower Premolar #34 Abrasion"),
        (14, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Upper Premolar #14 Abrasion"),
        (24, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Upper Premolar #24 Abrasion"),
        (35, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Lower Premolar #35 Abrasion"),
        (44, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Lower Premolar #44 Abrasion"),
        (45, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Lower Premolar #45 Abrasion"),
        (36, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Lower Molar #36 Abrasion"),
        (46, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Lower Molar #46 Abrasion"),
        (16, "Abrasion", "POSTERIOR_ABRASION_EXCLUDED", "Upper Molar #16 Abrasion"),
        (48, "Cavity", "THIRD_MOLAR_EXCLUDED", "Third Molar #48 Cavity"),
        (18, "Cavity", "THIRD_MOLAR_EXCLUDED", "Third Molar #18 Cavity"),
        (36, "Crack", "UNSUPPORTED_ANOMALY_TYPE", "Unvalidated Anomaly Crack"),
        (31, "Diastema", "UNSUPPORTED_ANOMALY_TYPE", "Unvalidated Anomaly Diastema"),
        (22, "Rotation", "UNSUPPORTED_ANOMALY_TYPE", "Unvalidated Anomaly Rotation"),
        (99, "Cavity", "UNSUPPORTED_FDI_TOOTH", "Invalid FDI #99"),
    ]

    initial_case_count = len(ROOM_CASES)
    for fdi, anom, expected_code, label in rejected_test_cases:
        res_c = client.post(
            "/api/carve",
            json={
                "tooth_fdi": fdi,
                "anomaly_type": anom,
                "geometry": {
                    "centroid": [500, 500],
                    "relative_centroid": [0.5, 0.5],
                    "area": 4500.0,
                    "radius": 37.8,
                    "relative_radius": 0.045,
                    "aspect_ratio": 1.3,
                    "orientation": 0.0,
                    "width": 80,
                    "height": 70,
                },
            },
        )
        body_c = res_c.get_json()
        res_r = client.post(
            "/api/rooms/cases",
            json={"tooth_fdi": fdi, "anomaly_type": anom},
        )
        body_r = res_r.get_json()
        record(
            f"Whitelist Reject [{label}]",
            res_c.status_code == 422
            and body_c.get("code") == expected_code
            and res_r.status_code == 422
            and body_r.get("code") == expected_code
            and len(ROOM_CASES) == initial_case_count,
            f"HTTP {res_c.status_code} ({body_c.get('code')}) — no silent substitution",
        )

    # -------------------------------------------------------------------------
    # 5. Supported Combinations End-to-End Carve & Zero Bleed-Through Check
    # -------------------------------------------------------------------------
    supported_live_tests = [
        (31, "Cavity", "Incisor #31 Cavity"),
        (31, "Abrasion", "Incisor #31 Labial Abrasion"),
        (33, "Cavity", "Canine #33 Cavity"),
        (33, "Abrasion", "Canine #33 Labial Abrasion"),
        (34, "Cavity", "Premolar #34 Occlusal Cavity"),
        (14, "Cavity", "Upper Premolar #14 Occlusal Cavity"),
        (36, "Cavity", "Molar #36 Occlusal Cavity"),
    ]

    for fdi, anom, label in supported_live_tests:
        res_ok = client.post(
            "/api/rooms/cases",
            json={
                "tooth_fdi": fdi,
                "anomaly_type": anom,
                "patient_name": f"Test Patient FDI-{fdi}",
                "rigor": "Intermediate",
            },
        )
        payload = res_ok.get_json()
        cp = payload["case"]["carve_params"]
        record(
            f"Supported Live Synthesis [{label}]",
            res_ok.status_code == 201
            and cp["opposite_bleed_vertices"] == 0
            and cp["is_finite"] is True
            and cp["vf_preserved"] is True
            and cp["depth_calibration_source"] == "clinical_heuristic_ratio_on_measured_mesh",
            f"HTTP 201, surface={cp['target_surface']}, r={cp['radius']}mm, d={cp['depth']}mm, wall={cp['local_wall_thickness_mm']}mm, bleed={cp['opposite_bleed_vertices']}",
        )

    # -------------------------------------------------------------------------
    # 6. Evaluate the AI (3-Step Spatial Concordance & Rationale Validation)
    # -------------------------------------------------------------------------
    # 6a. Short rationale (<25 chars) must be rejected with HTTP 400
    res_short = client.post(
        "/api/rooms/evaluate",
        json={
            "case_id": "case_1042",
            "student_mark_3d": [0.0, 0.0, 1.0],
            "verdict": "correct",
            "rationale": "Too short",
        },
    )
    record(
        "POST /api/rooms/evaluate (Short Rationale Guard)",
        res_short.status_code == 400,
        f"HTTP {res_short.status_code} ({res_short.get_json().get('error')})",
    )

    # 6b. Valid 3D mark near ground-truth centroid of case_1042
    mesh_1042 = client.get("/api/rooms/cases/case_1042/mesh").get_json()["mesh_data"]
    gt_c = mesh_1042["gt_centroid_3d"]
    student_mark = [gt_c[0] + 0.25, gt_c[1] - 0.20, gt_c[2] + 0.10]
    history_before = len(SESSION_STATE["history"])

    res_eval = client.post(
        "/api/rooms/evaluate",
        json={
            "case_id": "case_1042",
            "student_mark_3d": student_mark,
            "verdict": "correct",
            "rationale": "Localized carious cavitation in the central occlusal fossa of Tooth #36; conservative composite restoration indicated.",
        },
    )
    eval_json = res_eval.get_json()
    record(
        "POST /api/rooms/evaluate (Valid 3D Mark & Score Computation)",
        res_eval.status_code == 200
        and eval_json["within_margin"] is True
        and eval_json["score_a"] >= 85
        and eval_json["score_b"] >= 90
        and len(SESSION_STATE["history"]) == history_before + 1,
        f"dist={eval_json['centroid_distance_mm']}mm, Score A={eval_json['score_a']}%, Score B={eval_json['score_b']}%, Composite={eval_json['composite_score']}%",
    )

    # -------------------------------------------------------------------------
    # 7. Voice Viva Session Chat & Examiner Grading
    # -------------------------------------------------------------------------
    res_chat = client.post(
        "/api/viva/chat",
        json={
            "case_id": "case_1042",
            "message": "Do you have any medical conditions, medications, or drug allergies?",
        },
    )
    chat_json = res_chat.get_json()
    record(
        "POST /api/viva/chat (Competency Tracking)",
        res_chat.status_code == 200
        and chat_json["competencies"]["medical_history"] is True
        and chat_json["done_count"] == 4,
        f"done_count={chat_json['done_count']}/4, finding='{chat_json['finding_registered']}'",
    )

    res_grade = client.post("/api/viva/grade", json={"case_id": "case_1042"})
    grade_json = res_grade.get_json()
    record(
        "POST /api/viva/grade (Examiner Grade Card)",
        res_grade.status_code == 200 and grade_json["overall_score"] >= 90,
        f"{grade_json['grade_label']} ({grade_json['overall_score']}%)",
    )

    print("=" * 80)
    print(f"ALL {len(results)} PHASE A/B VERIFICATION CHECKS PASSED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
