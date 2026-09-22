import os, json, random
import numpy as np
import pandas as pd
import trimesh

os.makedirs("synthetic", exist_ok=True)

manifest = pd.read_csv("dental_manifest_part1.csv")

def carve_cavity(mesh, center_idx, radius, depth):
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals
    center = verts[center_idx]
    dists = np.linalg.norm(verts - center, axis=1)
    influence = np.clip(1 - (dists / radius), 0, 1) ** 2
    displacement = -normals * (depth * influence[:, None])
    carved = mesh.copy()
    carved.vertices = verts + displacement
    return carved, influence

N_CASES = 200
made = 0
attempts = 0
skipped_nonfinite = 0
skipped_degenerate = 0

while made < N_CASES and attempts < N_CASES * 5:
    attempts += 1
    row = manifest.sample(1).iloc[0]

    try:
        tmesh = trimesh.load(row['obj_path'], process=False)
        with open(row['json_path']) as f:
            lbl = np.array(json.load(f)['labels'])
    except Exception as e:
        print("Skipping, load failed:", e)
        continue

    tooth_choices = [t for t in np.unique(lbl) if t != 0]
    if not tooth_choices:
        continue
    target_tooth = int(random.choice(tooth_choices))

    vertex_mask = lbl == target_tooth
    face_mask = vertex_mask[tmesh.faces].all(axis=1)
    tooth = tmesh.submesh([face_mask], append=True)

    if tooth.vertices.shape[0] < 10:
        skipped_degenerate += 1
        continue

    tooth_size = np.linalg.norm(tooth.bounds[1] - tooth.bounds[0])
    radius = random.uniform(0.1, 0.25) * tooth_size
    depth = random.uniform(0.05, 0.15) * tooth_size
    center_idx = random.randrange(tooth.vertices.shape[0])

    carved, gt_mask = carve_cavity(tooth, center_idx, radius, depth)

    # Hard gate: reject anything non-finite before it ever touches disk
    if not np.isfinite(carved.vertices).all():
        skipped_nonfinite += 1
        continue
    if carved.vertices.shape[0] != len(gt_mask):
        # shouldn't be possible given how gt_mask is constructed, but guard anyway
        skipped_degenerate += 1
        continue

    case_id = f"case_{made:04d}"

    # Single atomic file: vertices, faces, and mask always stay in lockstep.
    # No export/reload round-trip, no OBJ parser involved, no way to drift apart.
    np.savez(
        f"synthetic/{case_id}_tooth{target_tooth}.npz",
        vertices=carved.vertices,
        faces=carved.faces,
        gt_mask=gt_mask,
        patient_id=row['patient_id'],
        jaw=row['jaw'],
        tooth=target_tooth
    )

    made += 1
    if made % 20 == 0:
        print(f"{made}/{N_CASES} generated")

print(f"Done. {made} synthetic cavity cases in ./synthetic/ (from {attempts} attempts)")
print(f"Skipped (non-finite): {skipped_nonfinite}")
print(f"Skipped (degenerate): {skipped_degenerate}")