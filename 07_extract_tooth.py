# 07_extract_tooth.py
import json
import numpy as np
import pandas as pd
import trimesh

manifest = pd.read_csv("dental_manifest_part1.csv")
manifest_row = manifest.iloc[0]   # pick a specific patient/jaw row

tmesh = trimesh.load(manifest_row['obj_path'], process=False)
with open(manifest_row['json_path']) as f:
    lbl = np.array(json.load(f)['labels'])

TARGET_TOOTH = 36  # e.g. lower-left first molar, common cavity site

if TARGET_TOOTH not in lbl:
    print(f"Tooth {TARGET_TOOTH} not present in this scan. Available teeth: {sorted(np.unique(lbl))}")
else:
    vertex_mask = lbl == TARGET_TOOTH
    face_mask = vertex_mask[tmesh.faces].all(axis=1)  # keep faces where all 3 verts belong to this tooth

    tooth = tmesh.submesh([face_mask], append=True)
    print("Isolated tooth vertices:", tooth.vertices.shape[0])
    tooth.export("tooth_36_isolated.obj")