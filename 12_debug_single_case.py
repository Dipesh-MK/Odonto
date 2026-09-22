import json
import random
import numpy as np
import trimesh

manifest_row = None
import pandas as pd
manifest = pd.read_csv("dental_manifest_part1.csv")

# Reproduce case_0000 generation manually, checking vertex count at every stage
row = manifest.iloc[0]  # adjust if case_0000 didn't use row 0 — doesn't matter, we just want to see the pattern

tmesh = trimesh.load(row['obj_path'], process=False)
lbl = np.array(json.load(open(row['json_path']))['labels'])

tooth_choices = [t for t in np.unique(lbl) if t != 0]
target_tooth = tooth_choices[0]

vertex_mask = lbl == target_tooth
face_mask = vertex_mask[tmesh.faces].all(axis=1)

tooth = tmesh.submesh([face_mask], append=True)
print("STAGE 1 - after submesh:", tooth.vertices.shape[0])

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

center_idx = 0
tooth_size = np.linalg.norm(tooth.bounds[1] - tooth.bounds[0])
carved, gt_mask = carve_cavity(tooth, center_idx, 0.15 * tooth_size, 0.1 * tooth_size)
print("STAGE 2 - after carve, carved.vertices:", carved.vertices.shape[0])
print("STAGE 2 - gt_mask length:", len(gt_mask))

carved.export("debug_case.obj")
print("STAGE 3 - exported")

reloaded = trimesh.load("debug_case.obj", process=False)
print("STAGE 4 - after reload, process=False:", reloaded.vertices.shape[0])

reloaded_default = trimesh.load("debug_case.obj")
print("STAGE 4b - after reload, default process:", reloaded_default.vertices.shape[0])