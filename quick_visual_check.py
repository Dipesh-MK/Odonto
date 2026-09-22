# quick visual check
import pyvista as pv
tooth_pv = pv.read("tooth_36_isolated.obj")
plotter = pv.Plotter()
plotter.add_mesh(tooth_pv, smooth_shading=True)
plotter.show()