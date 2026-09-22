import numpy as np
import trimesh
import pyvista as pv

tooth = trimesh.load("tooth_36_isolated.obj", process=False)

def carve_cavity(mesh, center_idx, radius=1.5, depth=0.8):
    verts = mesh.vertices.copy()
    normals = mesh.vertex_normals
    center = verts[center_idx]

    dists = np.linalg.norm(verts - center, axis=1)
    influence = np.clip(1 - (dists / radius), 0, 1) ** 2  # smooth falloff, 0 outside radius

    displacement = -normals * (depth * influence[:, None])  # push inward along normal
    new_verts = verts + displacement

    carved = mesh.copy()
    carved.vertices = new_verts
    return carved, influence  # influence > 0 IS your ground-truth cavity mask

# Pick a point near the occlusal (chewing) surface.
# Rough heuristic for now: highest Z-value vertex = topmost point of the crown.
occlusal_candidate = int(np.argmax(tooth.vertices[:, 2]))
print("Carving at vertex index:", occlusal_candidate, "position:", tooth.vertices[occlusal_candidate])

carved, gt_mask = carve_cavity(tooth, occlusal_candidate, radius=1.5, depth=0.8)
carved.export("tooth_36_cavity.obj")
np.save("tooth_36_cavity_gt.npy", gt_mask)
print("Cavity vertices affected (influence > 0.05):", (gt_mask > 0.05).sum())

# Immediate visual check
carved_pv = pv.read("tooth_36_cavity.obj")
plotter = pv.Plotter()
plotter.add_mesh(carved_pv, smooth_shading=True)
plotter.show()