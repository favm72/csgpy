"""A quick check that an installed csgpy wheel works - run by CI against
every built wheel on every supported Python. No window is opened, so it
runs headless.

    python tests/smoke_test.py
"""

import sys
import tempfile
from pathlib import Path

from csgpy import box, bspline, circle, cylinder, sphere
from csgpy.path import path


def area_facing_up(solid):
    vertices, triangles = solid.to_mesh()
    total = 0.0
    for i, j, k in triangles:
        (ax, ay, _), (bx, by, _), (cx, cy, _) = vertices[i], vertices[j], vertices[k]
        total += max(0.0, ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2)
    return total


# Booleans, transforms and evaluation.
plate = box(60, 40, 8) - cylinder(10, 20).translate(30, 20, -5)
vertices, triangles = plate.to_mesh()
assert vertices and triangles, "plate produced an empty mesh"

# 2D -> 3D.
profile = path(0, 0).line(5, 0).soft(3, 5).line(0, 10).close().build()
assert profile.revolve(360, segments=16).to_mesh()[1]
assert circle(1).sweep([(0, 0, 0), (0, 0, 5), (2, 0, 10)]).to_mesh()[1]

# Regression: a closed bspline outline intersected with a dome used to lose
# every upward-facing face (the sampler spiked to the origin).
blob = bspline([(40, 0), (30, 30), (0, 38), (-32, 28), (-42, -5), (-20, -36), (15, -30), (40, 0)])
island = blob.extrude(20).translate(0, 0, -1) & sphere(1, 48, 24).scale(46, 46, 12).translate(0, 0, -2)
assert area_facing_up(island) > 1000, "bspline & sphere lost its top faces"

# Regression: booleans on a finely tessellated sphere used to need memory
# quadratic in its face count (csgrs's BSP tree) and could exhaust the RAM.
# cube - ball and cube & ball must add back up to the cube.
ball = sphere(10, 256, 128)
cube = box(14, 14, 14).translate(-7, -7, -7)


def volume(solid):
    vertices, triangles = solid.to_mesh()
    total = 0.0
    for i, j, k in triangles:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = vertices[i], vertices[j], vertices[k]
        total += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return total / 6


assert abs(volume(cube - ball) + volume(cube & ball) - 14**3) < 1, "sphere/cube booleans lost volume"

# A chain of subtractions, then transformed copies of the (shared) result:
# every copy must come out identical in volume to the original.
holey = cube
for i in range(8):
    holey = holey - sphere(2, 32, 16).translate(-6 + i * 1.7, 0, 7)
copies = [holey.rotate(0, 0, 30 * i).translate(40 * i, 0, 0) for i in range(4)]
assert all(abs(volume(c) - volume(holey)) < 1e-6 * volume(holey) for c in copies), "transformed copies differ"

# Export: binary STL (80-byte header + count + 50 bytes per triangle),
# ASCII STL, and OBJ with one face line per triangle.
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    plate.save_stl(tmp / "plate.stl")
    assert (tmp / "plate.stl").stat().st_size == 84 + 50 * len(triangles)
    plate.save_stl(str(tmp / "ascii.stl"), ascii=True)
    assert (tmp / "ascii.stl").read_text().startswith("solid csgpy")
    plate.save_obj(tmp / "plate.obj")
    lines = (tmp / "plate.obj").read_text().splitlines()
    assert sum(line.startswith("f ") for line in lines) == len(triangles)
    assert sum(line.startswith("v ") for line in lines) < 3 * len(triangles), "OBJ vertices not shared"

# The optional renderer's mesh/texture/normal helpers (skipped if the
# pyvista extra isn't installed).
try:
    from csgpy.pyvista import mesh, smooth_normals, texture_coordinates
except ImportError:
    print("pyvista extra not installed - skipping renderer checks")
else:
    poly = mesh(plate)
    for mapping in ("box", "cylinder", "sphere"):
        assert texture_coordinates(poly, mapping).shape == (poly.n_points, 2)
    assert smooth_normals(poly).shape == (poly.n_points, 3)

print(f"csgpy OK on Python {sys.version.split()[0]}")
