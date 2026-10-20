# 08_carve_cavity.py
import sys
import numpy as np
import pyvista as pv
from parameter_converter import convert_2d_geometry_to_3d_params
from tooth_carver import load_or_extract_tooth_mesh, carve_cavity, save_atomic_case

target_tooth = int(sys.argv[1]) if len(sys.argv) > 1 else 36
anomaly_type = sys.argv[2] if len(sys.argv) > 2 else "Cavity"

tooth = load_or_extract_tooth_mesh(target_tooth=target_tooth)

# Derive target crown vertex, radius, and depth dynamically from tooth mesh
default_geom = {
    "relative_centroid": [0.5, 0.5],
    "relative_radius": 0.12,
    "aspect_ratio": 1.0,
    "orientation": 0.0
}
params = convert_2d_geometry_to_3d_params(
    default_geom,
    tooth,
    tooth_fdi=target_tooth,
    anomaly_type=anomaly_type
)

print(
    f"Carving {anomaly_type} on FDI #{target_tooth} ({params['tooth_type']}, {params['target_surface']}) "
    f"at vertex {params['center_idx']}: radius={params['radius']}mm, depth={params['depth']}mm"
)

carved, gt_mask = carve_cavity(
    tooth,
    center_idx=params["center_idx"],
    radius=params["radius"],
    depth=params["depth"],
    aspect_ratio=params["aspect_ratio"],
    orientation=params["orientation"],
    profile=params["profile"],
    tangent_u=params["tangent_u"]
)

npz_path, _ = save_atomic_case(
    f"tooth_{target_tooth}_{anomaly_type.lower()}",
    carved,
    gt_mask,
    {"tooth_fdi": target_tooth, "anomaly_type": anomaly_type, "carve_params": params}
)
print("Saved atomic case:", npz_path)
print("Affected vertices (influence > 0.05):", int((gt_mask > 0.05).sum()))

# Immediate visual check using pv.wrap (never pv.read on OBJ!)
carved_pv = pv.wrap(carved)
carved_pv["cavity_influence"] = gt_mask
plotter = pv.Plotter()
plotter.add_mesh(carved_pv, scalars="cavity_influence", cmap="YlOrRd", smooth_shading=True)
plotter.show()