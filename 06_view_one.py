import json
import numpy as np
import pandas as pd
import pyvista as pv
import trimesh

manifest = pd.read_csv("dental_manifest_part1.csv")
row = manifest.iloc[0]

tmesh = trimesh.load(row['obj_path'], process=False)
with open(row['json_path']) as f:
    lbl_data = json.load(f)

mesh = pv.wrap(tmesh)
mesh['tooth_label'] = np.array(lbl_data['labels'])

plotter = pv.Plotter(title=f"Odonto - Jaw 3D Model ({row['patient_id']} - {row['jaw']})")
plotter.add_mesh(mesh, scalars='tooth_label', cmap='tab20', show_scalar_bar=True, scalar_bar_args={'title': 'FDI Tooth Number'})
plotter.add_text(f"Patient: {row['patient_id']} | Jaw: {row['jaw'].capitalize()}", font_size=12, color='white')
plotter.set_background('#1e293b')
plotter.camera_position = 'iso'

# Launch interactive 3D window and capture screenshot
print("Opening interactive 3D PyVista window...")
plotter.show(screenshot='jaw_3d_view.png')
print("Saved 3D jaw render to jaw_3d_view.png")