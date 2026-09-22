import os, glob
import pandas as pd

path = os.path.expanduser(
    "~/.cache/kagglehub/datasets/aracknab69/dental-data-part-1-stl/versions/1"
)

records = []
for jaw in ['lower', 'upper']:
    jaw_dir = os.path.join(path, jaw)
    if not os.path.isdir(jaw_dir):
        continue
    for patient_dir in sorted(os.listdir(jaw_dir)):
        pdir = os.path.join(jaw_dir, patient_dir)
        if not os.path.isdir(pdir):
            continue
        obj_files = glob.glob(os.path.join(pdir, "*.obj"))
        json_files = glob.glob(os.path.join(pdir, "*.json"))
        if obj_files and json_files:
            records.append({
                "patient_id": patient_dir,
                "jaw": jaw,
                "obj_path": obj_files[0],
                "json_path": json_files[0]
            })

manifest = pd.DataFrame(records)
manifest.to_csv("dental_manifest_part1.csv", index=False)
print("Total pairs found:", len(manifest))
print(manifest.head())