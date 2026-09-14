"""A gold trophy on a marble plinth.

A `path` profile revolved into the cup, handles swept along arcs, a plinth
unioned from two boxes and wrapped in a PNG texture.
"""

from pathlib import Path

from csgpy import box, circle
from csgpy.path import path
from csgpy.pyvista import Scene

HERE = Path(__file__).parent

# Half a cross-section: x = distance from the axis, y = height. Up the
# outside (foot, stem, bowl), back down the inside, so the cup is hollow.
profile = (
    path(0, 0).line(8, 0).line(8, 1.5)
    .soft(2.5, 4).soft(2, 13).soft(3, 17).soft(9, 21).soft(12, 28).soft(12.5, 36)
    .line(11.5, 36).soft(10.5, 28).soft(0, 23)
    .close()
)
cup = profile.build().revolve(360, segments=64).rotate(90, 0, 0)

# One handle: a small circle swept along a curved 3D path (a 2D `path`
# stood up in the XZ plane), then a copy turned round to the other side.
ear = path(8.5, 22).soft(15, 23.5).soft(17, 29).soft(15.5, 34).soft(12, 35.5).points()
handle = circle(0.9, segments=16).sweep([(x, 0, z) for x, z in ear])
handles = (handle + handle.rotate(0, 0, 180)).rotate(0, 0, 40)

plinth = box(26, 26, 5).translate(-13, -13, -8) + box(20, 20, 3).translate(-10, -10, -3)

scene = Scene(background=("white", "#c9d6e3"))
for part in (cup, handles):  # two parts, not `cup + handles`: cleaner shading
    scene.add(part, "#f2c230", specular=1.0, specular_power=50)
scene.add(plinth, texture=HERE / "textures" / "marble.png", texture_scale=20)
scene.show(camera=[(60, -75, 45), (0, 0, 14), (0, 0, 1)], screenshot=HERE / "trophy.png")
