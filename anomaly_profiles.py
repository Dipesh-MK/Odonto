"""
anomaly_profiles.py

Centralized Anomaly Profile Registry for Odonto 3D Carving Pipeline.

Defines mathematical sculpting profiles, parameter schemas, and manual-only gates
for pathological dental lesion synthesis across all 32 FDI teeth.

Strict Project Isolation & Safety Rules:
- 'radial_bowl' (Cavity) & 'planar_facet' (Abrasion) are validated for both autonomous 2D-to-3D
  conversion (via whitelist.py) and manual placement.
- 'v_groove_path' (Crack) & 'face_removal_recap' (Chipped Cusp) are strictly marked manual_only=True.
  They are NEVER whitelisted for autonomous student pipeline use and exist solely for
  dev-facing QA, testing, and manual scenario authoring.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class AnomalyProfileDefinition:
    profile_id: str
    display_name: str
    description: str
    geometry_type: str  # 'displacement' | 're-meshing_slice'
    manual_only: bool
    whitelisted: bool
    default_surface: str
    allowed_surfaces: List[str]
    parameters: Dict[str, Any]
    experimental_warning: Optional[str] = None


ANOMALY_REGISTRY: Dict[str, AnomalyProfileDefinition] = {
    "radial_bowl": AnomalyProfileDefinition(
        profile_id="radial_bowl",
        display_name="Cavity / Carious Demineralization (Radial Bowl)",
        description="Elliptical radial bowl depression with smooth falloff and normal-gated inward displacement.",
        geometry_type="displacement",
        manual_only=False,
        whitelisted=True,
        default_surface="occlusal",
        allowed_surfaces=["occlusal", "cervical", "buccal", "lingual", "interproximal"],
        parameters={
            "radius": {"type": "float", "default": 1.5, "min": 0.3, "max": 4.5, "unit": "mm"},
            "depth": {"type": "float", "default": 0.8, "min": 0.1, "max": 2.5, "unit": "mm"},
            "aspect_ratio": {"type": "float", "default": 1.0, "min": 0.5, "max": 3.0, "unit": "ratio"},
            "orientation": {"type": "float", "default": 0.0, "min": -90.0, "max": 90.0, "unit": "deg"},
        },
        experimental_warning=None,
    ),
    "planar_facet": AnomalyProfileDefinition(
        profile_id="planar_facet",
        display_name="Incisal / Occlusal Abrasion (Planar Facet)",
        description="Beveled planar wear facet with flat-bottomed saucer geometry simulating masticatory attrition.",
        geometry_type="displacement",
        manual_only=False,
        whitelisted=True,
        default_surface="incisal",
        allowed_surfaces=["incisal", "occlusal"],
        parameters={
            "radius": {"type": "float", "default": 2.0, "min": 0.5, "max": 5.0, "unit": "mm"},
            "depth": {"type": "float", "default": 0.6, "min": 0.1, "max": 2.0, "unit": "mm"},
            "aspect_ratio": {"type": "float", "default": 1.8, "min": 1.0, "max": 4.0, "unit": "ratio"},
            "orientation": {"type": "float", "default": 0.0, "min": -90.0, "max": 90.0, "unit": "deg"},
        },
        experimental_warning=None,
    ),
    "v_groove_path": AnomalyProfileDefinition(
        profile_id="v_groove_path",
        display_name="Hairline Crack / Structural Fracture (V-Groove Polyline)",
        description="Narrow, sharp V-notch polyline cross-section following a multi-point surface path without cratering.",
        geometry_type="displacement",
        manual_only=True,
        whitelisted=False,
        default_surface="occlusal",
        allowed_surfaces=["occlusal", "buccal", "labial", "lingual", "palatal"],
        parameters={
            "depth": {"type": "float", "default": 0.35, "min": 0.1, "max": 1.2, "unit": "mm"},
            "width": {"type": "float", "default": 0.45, "min": 0.15, "max": 1.2, "unit": "mm"},
            "path_points": {"type": "list_3d_points", "default": [], "min_points": 2},
        },
        experimental_warning="⚠️ MANUAL PLACEMENT ONLY (Experimental). Crack 2D detection has 11.1% recall and is NOT whitelisted for autonomous student pipeline use.",
    ),
    "face_removal_recap": AnomalyProfileDefinition(
        profile_id="face_removal_recap",
        display_name="Chipped Cusp / Traumatic Coronal Fracture (Planar Cap)",
        description="True structural material loss from cusp tip via angled plane slicing and sealed polygon re-capping.",
        geometry_type="re-meshing_slice",
        manual_only=True,
        whitelisted=False,
        default_surface="occlusal",
        allowed_surfaces=["occlusal", "incisal"],
        parameters={
            "cusp_target": {"type": "str", "default": "mesiobuccal", "options": [
                "mesiobuccal", "distobuccal", "mesiolingual", "distolingual", "buccal", "lingual", "incisal_corner"
            ]},
            "chip_height": {"type": "float", "default": 1.4, "min": 0.5, "max": 3.5, "unit": "mm"},
            "chip_angle": {"type": "float", "default": 35.0, "min": 15.0, "max": 65.0, "unit": "deg"},
        },
        experimental_warning="⚠️ MANUAL PLACEMENT ONLY (Experimental). Chipped Cusp is a structural fracture model requiring manual clinician/instructor placement.",
    ),
}


def get_profile(profile_id: str) -> Optional[AnomalyProfileDefinition]:
    return ANOMALY_REGISTRY.get(profile_id)


def list_profiles() -> List[Dict[str, Any]]:
    return [
        {
            "profile_id": p.profile_id,
            "display_name": p.display_name,
            "description": p.description,
            "geometry_type": p.geometry_type,
            "manual_only": p.manual_only,
            "whitelisted": p.whitelisted,
            "default_surface": p.default_surface,
            "allowed_surfaces": p.allowed_surfaces,
            "parameters": p.parameters,
            "experimental_warning": p.experimental_warning,
        }
        for p in ANOMALY_REGISTRY.values()
    ]
