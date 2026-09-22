# 11_check_mask_mismatches.py
import glob
import numpy as np
import trimesh

gt_files = sorted(glob.glob("synthetic/*_gt.npy"))
mismatches = []

for gt_path in gt_files:
    prefix = gt_path.split("_gt.npy")[0]
    obj_matches = glob.glob(prefix + "_tooth*.obj")
    if not obj_matches:
        mismatches.append((prefix, "no obj file found"))
        continue

    mask = np.load(gt_path)
    tmesh = trimesh.load(obj_matches[0], process=False)

    if len(mask) != tmesh.vertices.shape[0]:
        mismatches.append((prefix, f"mask={len(mask)} verts={tmesh.vertices.shape[0]}"))

print(f"Total cases checked: {len(gt_files)}")
print(f"Mismatches: {len(mismatches)}")
for m in mismatches:
    print(m)