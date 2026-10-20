# 09_batch_generate_cavities.py
import os
import json
import random
import numpy as np
import pandas as pd
import trimesh
from parameter_converter import convert_2d_geometry_to_3d_params
from tooth_carver import carve_cavity

os.makedirs("synthetic", exist_ok=True)

manifest = pd.read_csv("dental_manifest_part1.csv")

N_CASES = 200
made = 0
attempts = 0
skipped_nonfinite = 0
skipped_degenerate = 0

while made < N_CASES and attempts < N_CASES * 5:
    attempts += 1
    row = manifest.sample(1).iloc[0]

    try:
        tmesh = trimesh.load(row["obj_path"], process=False)
        with open(row["json_path"]) as f:
            lbl = np.array(json.load(f)["labels"])
    except Exception as e:
        print("Skipping, load failed:", e)
        continue

    tooth_choices = [t for t in np.unique(lbl) if t != 0]
    if not tooth_choices:
        continue
    target_tooth = int(random.choice(tooth_choices))
    anomaly_type = random.choice(["Cavity", "Abrasion"])

    vertex_mask = lbl == target_tooth
    face_mask = vertex_mask[tmesh.faces].all(axis=1)
    tooth = tmesh.submesh([face_mask], append=True)

    if tooth.vertices.shape[0] < 50:
        skipped_degenerate += 1
        continue

    # Sample anatomically valid 2D relative geometry on the crown
    synthetic_geom = {
        "relative_centroid": [random.uniform(0.25, 0.75), random.uniform(0.25, 0.75)],
        "relative_radius": random.uniform(0.08, 0.18),
        "aspect_ratio": random.uniform(1.0, 2.0) if anomaly_type == "Abrasion" else random.uniform(1.0, 1.4),
        "orientation": random.uniform(-60.0, 60.0)
    }

    params = convert_2d_geometry_to_3d_params(
        synthetic_geom,
        tooth,
        tooth_fdi=target_tooth,
        anomaly_type=anomaly_type
    )

    carved, gt_mask = carve_cavity(
        tooth,
        center_idx=params["center_idx"],
        radius=params["radius"],
        depth=params["depth"],
        aspect_ratio=params["aspect_ratio"],
        orientation=params["orientation"],
        profile=params["profile"],
        tangent_u=params["tangent_u"]
    )

    # Hard gate: reject anything non-finite before it ever touches disk
    if not np.isfinite(carved.vertices).all():
        skipped_nonfinite += 1
        continue
    if carved.vertices.shape[0] != len(gt_mask):
        skipped_degenerate += 1
        continue

    case_id = f"case_{made:04d}"

    # Single atomic file: vertices, faces, and mask always stay in lockstep.
    np.savez(
        f"synthetic/{case_id}_tooth{target_tooth}_{anomaly_type.lower()}.npz",
        vertices=carved.vertices,
        faces=carved.faces,
        gt_mask=gt_mask,
        patient_id=row["patient_id"],
        jaw=row["jaw"],
        tooth=target_tooth,
        anomaly_type=anomaly_type
    )

    made += 1
    if made % 20 == 0:
        print(f"{made}/{N_CASES} generated")

print(f"Done. {made} synthetic cases in ./synthetic/ (from {attempts} attempts)")
print(f"Skipped (non-finite): {skipped_nonfinite}")
print(f"Skipped (degenerate): {skipped_degenerate}")