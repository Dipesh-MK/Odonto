import json
import random
import numpy as np
import pandas as pd
import trimesh

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

for attempt in range(200):
    row = manifest.sample(1).iloc[0]
    tmesh = trimesh.load(row['obj_path'], process=False)
    lbl = np.array(json.load(open(row['json_path']))['labels'])

    tooth_choices = [t for t in np.unique(lbl) if t != 0]
    target_tooth = int(random.choice(tooth_choices))

    vertex_mask = lbl == target_tooth
    face_mask = vertex_mask[tmesh.faces].all(axis=1)
    tooth = tmesh.submesh([face_mask], append=True)

    if tooth.vertices.shape[0] < 10:
        continue

    tooth_size = np.linalg.norm(tooth.bounds[1] - tooth.bounds[0])
    radius = random.uniform(0.1, 0.25) * tooth_size
    depth = random.uniform(0.05, 0.15) * tooth_size
    center_idx = random.randrange(tooth.vertices.shape[0])

    carved, gt_mask = carve_cavity(tooth, center_idx, radius, depth)

    # Check BEFORE export
    if carved.vertices.shape[0] != len(gt_mask):
        print("MISMATCH BEFORE EXPORT — bug is in carve_cavity or submesh, not I/O")
        print("carved.vertices:", carved.vertices.shape[0], "gt_mask:", len(gt_mask))
        break

    carved.export("debug_live.obj")
    reloaded = trimesh.load("debug_live.obj", process=False)

    if reloaded.vertices.shape[0] != len(gt_mask):
        print(f"MISMATCH FOUND on attempt {attempt}")
        print("target_tooth:", target_tooth, "patient:", row['patient_id'], "jaw:", row['jaw'])
        print("center_idx:", center_idx, "radius:", radius, "depth:", depth)
        print("tooth.vertices (pre-carve):", tooth.vertices.shape[0])
        print("carved.vertices (in-memory, post-carve):", carved.vertices.shape[0])
        print("gt_mask length:", len(gt_mask))
        print("reloaded.vertices (after export+reload):", reloaded.vertices.shape[0])
        print("n_faces original submesh:", tooth.faces.shape[0])
        print("n_faces carved:", carved.faces.shape[0])
        print("n_faces reloaded:", reloaded.faces.shape[0])

        # check for genuinely duplicate vertex positions, which trimesh's OBJ writer might dedupe on export
        unique_verts = np.unique(carved.vertices.round(decimals=8), axis=0)
        print("unique vertex positions in carved (rounded):", unique_verts.shape[0], "vs total:", carved.vertices.shape[0])
        break
else:
    print("No mismatch found in 200 attempts — bug may be intermittent, rerun this script")