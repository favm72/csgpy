"""A handful of dice - lots of cube-minus-sphere booleans.

Each die is a cube rounded off by intersecting it with a sphere, with 21
spherical pips subtracted from its faces: 22 booleans per die against
finely tessellated spheres, five dice in the scene.
"""

from pathlib import Path

from csgpy import box, sphere
from csgpy.pyvista import Scene

HERE = Path(__file__).parent

SIZE = 20  # edge length
D = 5  # pip offset from the face center

# Pip layouts on a face, in the face's own (u, v) coordinates.
LAYOUTS = {
    1: [(0, 0)],
    2: [(-D, -D), (D, D)],
    3: [(-D, -D), (0, 0), (D, D)],
    4: [(-D, -D), (-D, D), (D, -D), (D, D)],
    5: [(-D, -D), (-D, D), (0, 0), (D, -D), (D, D)],
    6: [(-D, -D), (-D, 0), (-D, D), (D, -D), (D, 0), (D, D)],
}

# Where each face's (u, v) lands in 3D, `h` being the distance out from the
# center. Opposite faces add up to 7, like real dice.
FACES = {
    1: lambda u, v, h: (u, v, h),
    6: lambda u, v, h: (u, v, -h),
    2: lambda u, v, h: (h, u, v),
    5: lambda u, v, h: (-h, u, v),
    3: lambda u, v, h: (u, h, v),
    4: lambda u, v, h: (u, -h, v),
}


def die():
    half = SIZE / 2
    cube = box(SIZE, SIZE, SIZE).translate(-half, -half, -half)
    body = cube & sphere(half * 1.42, segments=96, stacks=48)  # round off edges and corners
    for value, place in FACES.items():
        for u, v in LAYOUTS[value]:
            # Center just outside the face, so each pip is a shallow dimple.
            body = body - sphere(2.2, segments=32, stacks=16).translate(*place(u, v, half + 0.8))
    return body


one = die()
scene = Scene(background=("white", "#d5dde6"))
for (x, y, z), (rx, ry, rz), color in [
    ((0, 0, 0), (0, 0, 20), "ivory"),
    ((30, -8, 0), (90, 0, -35), "crimson"),
    ((-28, 10, 0), (0, -90, 60), "royalblue"),
    ((8, 32, 0), (180, 0, 10), "seagreen"),
    ((-12, -34, 0), (-90, 0, 45), "gold"),
]:
    scene.add(one.rotate(rx, ry, rz).translate(x, y, z), color, specular=0.6, specular_power=30)
scene.show(camera=[(70, -110, 90), (0, 0, 0), (0, 0, 1)], screenshot=HERE / "dice.png")
