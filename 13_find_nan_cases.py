import json
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
    carved_verts = verts + displacement
    return carved_verts, normals

bad_count = 0
for i in range(min(30, len(manifest))):
    row = manifest.iloc[i]
    tmesh = trimesh.load(row['obj_path'], process=False)
    lbl = np.array(json.load(open(row['json_path']))['labels'])

    for target_tooth in [t for t in np.unique(lbl) if t != 0]:
        vertex_mask = lbl == target_tooth
        face_mask = vertex_mask[tmesh.faces].all(axis=1)
        tooth = tmesh.submesh([face_mask], append=True)

        if tooth.vertices.shape[0] < 10:
            continue

        normals = tooth.vertex_normals
        n_nan = np.isnan(normals).any(axis=1).sum()
        n_zero = (np.linalg.norm(normals, axis=1) < 1e-8).sum()

        if n_nan > 0 or n_zero > 0:
            bad_count += 1
            print(f"row {i} tooth {target_tooth}: verts={tooth.vertices.shape[0]}  NaN normals={n_nan}  zero-length normals={n_zero}")

print(f"\nBad tooth extractions found: {bad_count}")