"""
whitelist.py

Single Source of Truth for Phase 0 Validated Tooth × Anomaly Combinations
in the Odonto Phase A/B Room & Diagnostic Simulation Platform.

Enforces the strict distinction between numerical mesh integrity and clinical shape validity:
- Unconditional GO (Approved for Room Cases):
  1. Cavity / Caries across all 4 validated tooth types:
     - Incisors (#11, #12, #21, #22, #31, #32, #41, #42) -> labial_facial
     - Canines  (#13, #23, #33, #43)                     -> labial_facial
     - Premolars (#14, #15, #24, #25, #34, #35, #44, #45) -> occlusal_table
     - Molars   (#16, #17, #26, #27, #36, #37, #46, #47) -> occlusal_table
  2. Anterior / Labial Abrasion ONLY on validated anterior tooth types:
     - Incisors (#11, #12, #21, #22, #31, #32, #41, #42) -> labial_facial
     - Canines  (#13, #23, #33, #43)                     -> labial_facial

- Explicitly Excluded / Blocked in Phase A/B:
  - Posterior Abrasion on premolars (#14, #15, #24, #25, #34, #35, #44, #45) and
    molars (#16-#18, #26-#28, #36-#38, #46-#48): Radial inward displacement on
    posterior occlusal/cuspal surfaces creates saucer-like depressions rather than
    planar wear facets. Tracked as a separate future engineering task (planar half-space clipping).
  - Third molars (#18, #28, #38, #48): Excluded due to sparse scan representation in Teeth3DS+
    (7-11 scans) and unvalidated status in Phase 0.
  - Unvalidated anomaly categories (Crack, Crowding, Diastema, Rotation, Agenesis,
    Periapical Lesion, Plaque, Other): Not silently relabeled or substituted.
"""

from typing import Dict, Any, List, Tuple

DEPTH_CALIBRATION_NOTE = (
    "Target surface span (S_surf) and local wall thickness (T_wall) are measured dynamically "
    "from each Teeth3DS+ tooth mesh, whereas the depth-ratio multipliers (0.20r and 0.08*T_wall "
    "for Abrasion; 0.42r and 0.15*T_wall for Cavity) and profile exponents are clinical heuristics "
    "rather than empirically regressed ground-truth depth measurements."
)

# Canonical anomaly definitions
SUPPORTED_ANOMALIES: Dict[str, Dict[str, Any]] = {
    "Cavity": {
        "canonical": "Cavity",
        "aliases": ["cavity", "caries", "cavity / caries", "occlusal cavity", "deep dentinal caries", "labial caries"],
        "label": "Cavity / Caries",
        "allowed_tooth_types": ["incisor", "canine", "premolar", "molar"],
        "allowed_surfaces": {
            "incisor": "labial_facial",
            "canine": "labial_facial",
            "premolar": "occlusal_table",
            "molar": "occlusal_table",
        },
        "profile": "cavity",
        "description": "Concave carious cavitation (parabolic profile) on labial enamel (anterior) or concave occlusal pits/fossae (posterior).",
    },
    "Abrasion": {
        "canonical": "Abrasion",
        "aliases": ["abrasion", "cervical abrasion", "labial abrasion", "anterior abrasion"],
        "label": "Abrasion (Anterior Labial)",
        "allowed_tooth_types": ["incisor", "canine"],
        "allowed_surfaces": {
            "incisor": "labial_facial",
            "canine": "labial_facial",
        },
        "profile": "abrasion",
        "description": "Horizontal saucer-shaped cervical/labial wear facet on anterior teeth (incisors and canines only).",
    },
}

# Explicit registry of all 28 approved FDI permanent teeth (excluding 3rd molars 18, 28, 38, 48)
APPROVED_FDI_TEETH: Dict[int, Dict[str, Any]] = {
    # Q1: Maxillary Right
    11: {"fdi": 11, "name": "Maxillary Right Central Incisor", "short_name": "Upper R Central Incisor", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    12: {"fdi": 12, "name": "Maxillary Right Lateral Incisor", "short_name": "Upper R Lateral Incisor", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    13: {"fdi": 13, "name": "Maxillary Right Canine", "short_name": "Upper R Canine", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "canine", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 33},
    14: {"fdi": 14, "name": "Maxillary Right First Premolar", "short_name": "Upper R 1st Premolar", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 14},
    15: {"fdi": 15, "name": "Maxillary Right Second Premolar", "short_name": "Upper R 2nd Premolar", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 15},
    16: {"fdi": 16, "name": "Maxillary Right First Molar", "short_name": "Upper R 1st Molar", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    17: {"fdi": 17, "name": "Maxillary Right Second Molar", "short_name": "Upper R 2nd Molar", "quadrant": "Q1", "arch": "Maxillary Right (Q1)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    # Q2: Maxillary Left
    21: {"fdi": 21, "name": "Maxillary Left Central Incisor", "short_name": "Upper L Central Incisor", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    22: {"fdi": 22, "name": "Maxillary Left Lateral Incisor", "short_name": "Upper L Lateral Incisor", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    23: {"fdi": 23, "name": "Maxillary Left Canine", "short_name": "Upper L Canine", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "canine", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 33},
    24: {"fdi": 24, "name": "Maxillary Left First Premolar", "short_name": "Upper L 1st Premolar", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 24},
    25: {"fdi": 25, "name": "Maxillary Left Second Premolar", "short_name": "Upper L 2nd Premolar", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 25},
    26: {"fdi": 26, "name": "Maxillary Left First Molar", "short_name": "Upper L 1st Molar", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    27: {"fdi": 27, "name": "Maxillary Left Second Molar", "short_name": "Upper L 2nd Molar", "quadrant": "Q2", "arch": "Maxillary Left (Q2)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    # Q3: Mandibular Left
    31: {"fdi": 31, "name": "Mandibular Left Central Incisor", "short_name": "Lower L Central Incisor", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    32: {"fdi": 32, "name": "Mandibular Left Lateral Incisor", "short_name": "Lower L Lateral Incisor", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    33: {"fdi": 33, "name": "Mandibular Left Canine", "short_name": "Lower L Canine", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "canine", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 33},
    34: {"fdi": 34, "name": "Mandibular Left First Premolar", "short_name": "Lower L 1st Premolar", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 34},
    35: {"fdi": 35, "name": "Mandibular Left Second Premolar", "short_name": "Lower L 2nd Premolar", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 35},
    36: {"fdi": 36, "name": "Mandibular Left First Molar", "short_name": "Lower L 1st Molar", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    37: {"fdi": 37, "name": "Mandibular Left Second Molar", "short_name": "Lower L 2nd Molar", "quadrant": "Q3", "arch": "Mandibular Left (Q3)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    # Q4: Mandibular Right
    41: {"fdi": 41, "name": "Mandibular Right Central Incisor", "short_name": "Lower R Central Incisor", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    42: {"fdi": 42, "name": "Mandibular Right Lateral Incisor", "short_name": "Lower R Lateral Incisor", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "incisor", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 31},
    43: {"fdi": 43, "name": "Mandibular Right Canine", "short_name": "Lower R Canine", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "canine", "anatomical_group": "anterior", "surface": "labial_facial", "allowed_anomalies": ["Cavity", "Abrasion"], "phase0_exemplar": 33},
    44: {"fdi": 44, "name": "Mandibular Right First Premolar", "short_name": "Lower R 1st Premolar", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 44},
    45: {"fdi": 45, "name": "Mandibular Right Second Premolar", "short_name": "Lower R 2nd Premolar", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "premolar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 45},
    46: {"fdi": 46, "name": "Mandibular Right First Molar", "short_name": "Lower R 1st Molar", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
    47: {"fdi": 47, "name": "Mandibular Right Second Molar", "short_name": "Lower R 2nd Molar", "quadrant": "Q4", "arch": "Mandibular Right (Q4)", "tooth_type": "molar", "anatomical_group": "posterior", "surface": "occlusal_table", "allowed_anomalies": ["Cavity"], "phase0_exemplar": 36},
}

EXCLUDED_COMBINATION_NOTES: Dict[str, str] = {
    "posterior_abrasion": (
        "Posterior Abrasion on premolars and molars is excluded from Phase A/B room cases. "
        "While numerically stable with zero bleed-through, inward radial displacement on "
        "posterior occlusal/cuspal surfaces produces saucer-like depressions rather than "
        "clinically representative planar wear facets. Planar half-space clipping is tracked "
        "as a separate future engineering task."
    ),
    "third_molar": (
        "Third molars (FDI #18, #28, #38, #48) are excluded from Phase A/B due to sparse "
        "scan representation in Teeth3DS+ (7–11 scans) and unvalidated Phase 0 status."
    ),
    "unsupported_anomaly": (
        "Only Phase 0 validated anomalies ('Cavity' / 'Caries' and anterior labial 'Abrasion') "
        "are permitted. Unvalidated anomalies are blocked and never silently relabeled."
    ),
}


def normalize_anomaly_type(raw_anomaly: str) -> str | None:
    """
    Normalizes an input anomaly string to its canonical Phase 0 name ('Cavity' or 'Abrasion'),
    or returns None if the anomaly is not part of the Phase 0 validated whitelist.
    """
    if not raw_anomaly or not isinstance(raw_anomaly, str):
        return None
    cleaned = raw_anomaly.strip().lower()
    for canonical, spec in SUPPORTED_ANOMALIES.items():
        if cleaned == canonical.lower() or cleaned in spec["aliases"]:
            return canonical
    return None


def validate_tooth_anomaly_combination(tooth_fdi: Any, anomaly_type: Any) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates whether (tooth_fdi, anomaly_type) is approved in the Phase 0 whitelist.

    Returns:
        (is_valid, details_dict)
    """
    try:
        fdi = int(tooth_fdi)
    except (ValueError, TypeError):
        return False, {
            "code": "INVALID_FDI_FORMAT",
            "error": f"Invalid FDI tooth number '{tooth_fdi}'. Must be a valid two-digit ISO-3950 integer.",
            "tooth_fdi": tooth_fdi,
            "anomaly_type": anomaly_type,
        }

    if fdi in (18, 28, 38, 48):
        return False, {
            "code": "THIRD_MOLAR_EXCLUDED",
            "error": f"FDI #{fdi} is a third molar (wisdom tooth). {EXCLUDED_COMBINATION_NOTES['third_molar']}",
            "tooth_fdi": fdi,
            "anomaly_type": anomaly_type,
        }

    if fdi not in APPROVED_FDI_TEETH:
        return False, {
            "code": "UNSUPPORTED_FDI_TOOTH",
            "error": f"FDI #{fdi} is not in the Phase 0 validated tooth registry (supported: #11-#17, #21-#27, #31-#37, #41-#47).",
            "tooth_fdi": fdi,
            "anomaly_type": anomaly_type,
        }

    tooth_meta = APPROVED_FDI_TEETH[fdi]
    canonical_anom = normalize_anomaly_type(str(anomaly_type))

    if canonical_anom is None:
        return False, {
            "code": "UNSUPPORTED_ANOMALY_TYPE",
            "error": (
                f"Anomaly '{anomaly_type}' is not supported in Phase A/B rooms. "
                f"{EXCLUDED_COMBINATION_NOTES['unsupported_anomaly']}"
            ),
            "tooth_fdi": fdi,
            "tooth_name": tooth_meta["name"],
            "tooth_type": tooth_meta["tooth_type"],
            "anomaly_type": anomaly_type,
            "allowed_anomalies_for_tooth": tooth_meta["allowed_anomalies"],
        }

    if canonical_anom not in tooth_meta["allowed_anomalies"]:
        # Specifically catch Posterior Abrasion (Premolar / Molar x Abrasion)
        return False, {
            "code": "POSTERIOR_ABRASION_EXCLUDED",
            "error": (
                f"Unsupported combination: FDI #{fdi} ({tooth_meta['name']}, {tooth_meta['tooth_type']}) "
                f"with '{canonical_anom}'. {EXCLUDED_COMBINATION_NOTES['posterior_abrasion']}"
            ),
            "tooth_fdi": fdi,
            "tooth_name": tooth_meta["name"],
            "tooth_type": tooth_meta["tooth_type"],
            "anatomical_group": tooth_meta["anatomical_group"],
            "anomaly_type": canonical_anom,
            "allowed_anomalies_for_tooth": tooth_meta["allowed_anomalies"],
        }

    return True, {
        "code": "APPROVED",
        "tooth_fdi": fdi,
        "tooth_name": tooth_meta["name"],
        "short_name": tooth_meta["short_name"],
        "quadrant": tooth_meta["quadrant"],
        "arch": tooth_meta["arch"],
        "tooth_type": tooth_meta["tooth_type"],
        "anatomical_group": tooth_meta["anatomical_group"],
        "target_surface": tooth_meta["surface"],
        "anomaly_type": canonical_anom,
        "anomaly_label": SUPPORTED_ANOMALIES[canonical_anom]["label"],
        "profile": SUPPORTED_ANOMALIES[canonical_anom]["profile"],
        "allowed_anomalies_for_tooth": tooth_meta["allowed_anomalies"],
        "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
    }


def get_whitelist_payload() -> Dict[str, Any]:
    """
    Returns the complete JSON-serializable Phase 0 whitelist configuration for the frontend.
    """
    return {
        "teeth": APPROVED_FDI_TEETH,
        "supported_anomalies": SUPPORTED_ANOMALIES,
        "excluded_notes": EXCLUDED_COMBINATION_NOTES,
        "depth_calibration_note": DEPTH_CALIBRATION_NOTE,
        "summary": {
            "total_approved_teeth": len(APPROVED_FDI_TEETH),
            "cavity_supported_tooth_types": ["incisor", "canine", "premolar", "molar"],
            "abrasion_supported_tooth_types": ["incisor", "canine"],
            "posterior_abrasion_enabled": False,
        },
    }
