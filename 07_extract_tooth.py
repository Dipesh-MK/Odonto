# 07_extract_tooth.py
import sys
import json
import numpy as np
import pandas as pd
import trimesh
from parameter_converter import classify_fdi_tooth

manifest = pd.read_csv("dental_manifest_part1.csv")
TARGET_TOOTH = int(sys.argv[1]) if len(sys.argv) > 1 else 36
tooth_info = classify_fdi_tooth(TARGET_TOOTH)

# Select appropriate jaw scan from manifest
jaw_rows = manifest[manifest["jaw"] == tooth_info["jaw"]]
if jaw_rows.empty:
    jaw_rows = manifest
manifest_row = jaw_rows.iloc[0]

tmesh = trimesh.load(manifest_row["obj_path"], process=False)
with open(manifest_row["json_path"]) as f:
    lbl = np.array(json.load(f)["labels"])

if TARGET_TOOTH not in lbl:
    print(f"Tooth {TARGET_TOOTH} not present in this scan. Available teeth: {sorted(np.unique(lbl))}")
else:
    vertex_mask = lbl == TARGET_TOOTH
    face_mask = vertex_mask[tmesh.faces].all(axis=1)  # keep faces where all 3 verts belong to this tooth

    tooth = tmesh.submesh([face_mask], append=True)
    out_file = f"tooth_{TARGET_TOOTH}_isolated.obj"
    print(
        f"Isolated FDI #{TARGET_TOOTH} ({tooth_info['tooth_type']}, {tooth_info['jaw']} jaw) "
        f"vertices: {tooth.vertices.shape[0]}"
    )
    tooth.export(out_file)