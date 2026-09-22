import json
import numpy as np
import pandas as pd
import trimesh

manifest = pd.read_csv("dental_manifest_part1.csv")
row = manifest.iloc[0]

tmesh = trimesh.load(row['obj_path'], process=False)
with open(row['json_path']) as f:
    lbl_data = json.load(f)

lbl = np.array(lbl_data['labels'])
print("Vertices:", tmesh.vertices.shape[0], "| Labels:", len(lbl), "| Match:", tmesh.vertices.shape[0] == len(lbl))
print("Unique tooth labels:", np.unique(lbl))