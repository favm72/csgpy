"""A castle tower textured with PNG images (bricks and wood planks).

Booleans in a loop carve the battlements and arrow slits; `Scene.add`
wraps each part in an image file from `textures/`.
"""

from pathlib import Path

from csgpy import box, cone, cylinder
from csgpy.pyvista import Scene

HERE = Path(__file__).parent
TEXTURES = HERE / "textures"

# A round tower, hollowed out at the top.
tower = cylinder(10, 40, segments=64) - cylinder(8, 20, segments=64).translate(0, 0, 32)
for i in range(8):  # battlements: 8 notches round the rim
    tower = tower - box(4, 12, 6).translate(-2, 0, 36).rotate(0, 0, i * 45 + 22.5)
for i in range(3):  # arrow slits, one facing the camera
    tower = tower - box(1.5, 12, 8).translate(-0.75, 0, 16).rotate(0, 0, i * 120 - 142)

turret = cylinder(4, 14, segments=32).translate(0, 0, 32)
roof = cone(5.5, 9, segments=32).translate(0, 0, 46)
flag = box(0.4, 0.4, 8).translate(-0.2, -0.2, 54) + box(0.2, 6, 3.5).translate(0, 0, 58)
ground = box(44, 44, 3).translate(-22, -22, -3)

scene = Scene(background=("#e8f4fb", "#8fc1e3"))
scene.add(tower, texture=TEXTURES / "bricks.png", mapping="cylinder", texture_scale=14)
scene.add(turret, texture=TEXTURES / "bricks.png", mapping="cylinder", texture_scale=10)
scene.add(ground, texture=TEXTURES / "wood.png", texture_scale=22)
scene.add(roof, color="royalblue")
scene.add(flag, color="crimson")
scene.show(camera=[(85, -110, 70), (0, 0, 27), (0, 0, 1)], screenshot=HERE / "castle.png")
