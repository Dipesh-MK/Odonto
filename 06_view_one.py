import json
import numpy as np
import pandas as pd
import pyvista as pv

manifest = pd.read_csv("dental_manifest_part1.csv")
row = manifest.iloc[0]

mesh = pv.read(row['obj_path'])
with open(row['json_path']) as f:
    lbl_data = json.load(f)

mesh['tooth_label'] = np.array(lbl_data['labels'])

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='tooth_label', cmap='tab20', show_scalar_bar=True)
plotter.show()