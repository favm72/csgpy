"""The smallest csgpy program: a plate with a hole through it."""

from csgpy import box, cylinder
from csgpy.pyvista import show

plate = box(60, 40, 8) - cylinder(10, 20).translate(30, 20, -5)
show(plate, color="steelblue")
plate.save_stl("plate.stl")  # ready for a slicer
