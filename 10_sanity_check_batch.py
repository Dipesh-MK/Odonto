import glob
import numpy as np
import pandas as pd
import pyvista as pv
import random

npz_files = sorted(glob.glob("synthetic/*.npz"))
print("Total cases:", len(npz_files))

stats = []
for path in npz_files:
    data = np.load(path)
    mask = data['gt_mask']
    affected = (mask > 0.05).sum()
    frac = affected / len(mask)
    stats.append({
        "case": path,
        "n_vertices": len(mask),
        "affected_vertices": affected,
        "affected_frac": frac,
        "max_influence": mask.max()
    })

df = pd.DataFrame(stats)
df.to_csv("synthetic_batch_stats.csv", index=False)
print(df['affected_frac'].describe())

sample_cases = random.sample(npz_files, min(4, len(npz_files)))
plotter = pv.Plotter(shape=(2, 2))
for i, path in enumerate(sample_cases):
    data = np.load(path)
    verts, faces, mask = data['vertices'], data['faces'], data['gt_mask']
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64)
    mesh_pv = pv.PolyData(verts, faces_pv)
    mesh_pv['cavity'] = mask

    plotter.subplot(i // 2, i % 2)
    plotter.add_mesh(mesh_pv, scalars='cavity', cmap='coolwarm', show_scalar_bar=False)
    plotter.add_text(path.split('/')[-1], font_size=8)

plotter.show()