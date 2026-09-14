"""A lighthouse on a little island - a colorful, textured tour of `csgpy`.

- primitives: `cylinder`, `cone`, `sphere`, `circle`, `rectangle`,
  `polygon`, `bezier`, `bspline`
- the fluent `path` builder: `.line`, `.arc`, `.close`
- 2D -> 3D: `.revolve` (tower), `.sweep` (railing), `.extrude` (almost
  everything else)
- booleans: `-` (door, windows, cockpit), `+` (railing posts, rocks),
  `&` (the island's domed shape)
- `.translate` / `.rotate` / `.scale`
- rendering with procedural textures (`Scene.add(..., texture=...)`):
  stripes, bricks, sand, grass, water ripples and wood planks - numpy
  arrays generated below instead of image files (see `castle.py` for PNGs).

Run it with `python examples/advanced/lighthouse.py` (needs `pip install
csgpy[pyvista]`); `lighthouse.png` alongside this file is what it renders.
"""

import math
from functools import reduce
from operator import add
from pathlib import Path

import numpy as np
from csgpy import bezier, bspline, circle, cone, cylinder, polygon, rectangle, sphere
from csgpy.path import path
from csgpy.pyvista import Scene

# -- Textures: tiny numpy images (height x width x RGB) that tile seamlessly --
rng = np.random.default_rng(7)
N = 128
Y, X = np.mgrid[0:N, 0:N] / N  # 0..1 across one tile


def rgb(hex_color):
    return np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def paint(mix, dark, light):
    """Blend two colors by `mix` (0..1 per pixel) into a uint8 image."""
    mix = np.clip(mix, 0, 1)[..., None]
    return ((1 - mix) * rgb(dark) + mix * rgb(light)).astype(np.uint8)


def noise(amount):
    return rng.uniform(-amount, amount, (N, N))


stripes = paint(Y < 0.5, "#d62828", "#f8f4e8")  # red / white bands
sail = paint((Y * 3) % 1 < 0.5, "#f77f00", "#fff1c1")  # orange / cream bands
sand = paint(0.5 + noise(0.5), "#d9b26f", "#f3dca2")
grass = paint(0.5 + noise(0.5), "#3a7d2c", "#6fbf4a")
ripple = np.sin(2 * math.pi * (4 * X + 0.15 * np.sin(2 * math.pi * 3 * Y)))
ripple += 0.5 * np.sin(2 * math.pi * (2 * X - 5 * Y))  # integer frequencies tile
water = paint(0.45 + 0.2 * ripple + noise(0.08), "#1c6ea4", "#5fc0e8")
row = (Y * 4).astype(int)  # 4 brick rows, every other one offset by half
brick_mix = 0.6 + noise(0.25) + 0.2 * np.sin(row * 5 + ((X * 2 + row % 2 * 0.5) // 1) * 3)
mortar = ((Y * 4) % 1 < 0.1) | (((X * 2 + row % 2 * 0.5) % 1) < 0.05)
bricks = paint(np.where(mortar, 1.0, brick_mix * 0.7), "#8c2f1b", "#e0d6c8")
planks = paint(  # 4 planks with wavy grain
    0.55 + 0.3 * np.sin(2 * math.pi * (6 * X + 0.3 * np.sin(2 * math.pi * Y * 4)) + (Y * 4 // 1) * 2)
    - 0.5 * ((Y * 4) % 1 < 0.06),
    "#5c3a1e",
    "#b07a45",
)

# -- The sea and the island ------------------------------------------------
SEA = 4.0  # water level - high enough to hide the island's steep edge
sea = circle(260, segments=64).extrude(2).translate(0, 0, SEA - 2)

# A soft blob outline (a B-spline, closed because its last control point
# repeats the first), extruded up and intersected with a flattened sphere:
# a domed island.
outline = bspline(
    [(40, 0), (30, 30), (0, 38), (-32, 28), (-42, -5), (-20, -36), (15, -30), (40, 0)]
)
# (The sphere is sunk a little so its equator isn't flush with the
# extrusion's bottom face - coplanar faces are where booleans get fragile.)
dome = sphere(1, segments=48, stacks=24).scale(46, 46, 12).translate(0, 0, -2)
island = outline.extrude(20).translate(0, 0, -1) & dome
meadow = outline.scale(0.7).extrude(20).translate(0, 0, -1) & dome.translate(0, 0, 0.6)

rocks = reduce(add, (  # five low-poly pebbles, unioned into one solid
    sphere(r, segments=7, stacks=5).scale(1, 0.8, 0.6).rotate(0, 0, a * 57).translate(
        48 * math.cos(math.radians(a)), 42 * math.sin(math.radians(a)), SEA
    )
    for a, r in [(10, 5), (140, 4), (200, 6), (260, 3.5), (320, 4.5)]
))

# -- The lighthouse --------------------------------------------------------
plinth = cylinder(13, 4, segments=8).translate(0, 0, 8)
BASE = 12.0  # tower starts on top of the plinth

# Profile: x = distance from the axis, y = height. A slight flare at the
# foot, then a straight taper; revolve() spins it round the Y axis, and
# rotate(90, 0, 0) stands it up along Z.
tower = (
    path(0, 0).line(9.5, 0).line(9, 2).line(6.5, 48).line(0, 48).close()
    .build().revolve(360, segments=48).rotate(90, 0, 0).translate(0, 0, BASE)
)


def facing_out(sketch, depth, radius, z, turn=0.0):
    """Stand a 2D shape up on the tower's surface, facing outward: extrude
    it `depth` deep, tip it upright, push it out to `radius`, turn it round
    the tower."""
    return (
        sketch.extrude(depth)
        .rotate(90, 0, 0)
        .translate(0, -radius + depth / 2, z)
        .rotate(0, 0, turn)
    )


arch = path(-2.5, 0).line(2.5, 0).line(2.5, 5).arc(-2.5, 5, center=(0, 5)).close().build()
window = rectangle(2.4, 3.6).translate(-1.2, 0)
window_spots = [(24, 0), (34, 180), (44, 0), (52, 180)]  # (height, angle)

tower = tower - facing_out(arch, 6, 9.5, BASE + 0.5)
door = facing_out(arch, 1, 8.2, BASE + 0.5)
panes = []
for z, turn in window_spots:
    r = 9 - 2.5 * (z - BASE - 2) / 46  # the taper's radius at that height
    tower = tower - facing_out(window, 6, r, z, turn)
    panes.append(facing_out(window, 1, r - 0.8, z, turn))
glass = reduce(add, panes)

TOP = BASE + 48
deck = cylinder(10.5, 1.5, segments=48).translate(0, 0, TOP)

# Railing: a thin circle swept round a closed loop, plus 16 posts unioned in.
ring = [
    (9.8 * math.cos(t), 9.8 * math.sin(t), TOP + 4.5)
    for t in (i * 2 * math.pi / 48 for i in range(49))
]
railing = circle(0.35, segments=8).sweep(ring)
for i in range(16):
    t = i * 2 * math.pi / 16
    railing = railing + cylinder(0.3, 3, segments=6).translate(
        9.8 * math.cos(t), 9.8 * math.sin(t), TOP + 1.5
    )

lantern = cylinder(4.5, 7, segments=8).translate(0, 0, TOP + 1.5)
roof = cone(6.5, 5, segments=8).translate(0, 0, TOP + 8.5)
finial = sphere(1, segments=12, stacks=8).translate(0, 0, TOP + 14)

# A beam of light: a long cone tipped on its side, apex in the lantern.
beam = cone(10, 90).rotate(0, 90, 0).translate(-90, 0, TOP + 5).rotate(0, 0, 25)

# -- A little sailboat -----------------------------------------------------
hull_outline = polygon([(-8, -3), (5, -3), (10, 0), (5, 3), (-8, 3)])
hull = hull_outline.extrude(3) - hull_outline.scale(0.8).extrude(3).translate(-0.5, 0, 1)
mast = cylinder(0.3, 16, segments=8).translate(0, 0, 1)
# A closed bezier (first control point == last) makes a curved sail shape;
# it's drawn in XY, so tip it upright into the XZ plane.
sail_shape = bezier([(0, 0), (16, 0), (4, 6), (0, 20), (0, 0)], segments=40)
sail_cloth = sail_shape.extrude(0.3).rotate(90, 0, 0).translate(0.5, 0.15, 3)
hull, mast, sail_cloth = (
    part.rotate(0, 0, 20).translate(38, -62, SEA - 1) for part in (hull, mast, sail_cloth)
)

# -- Render: flat colors where that reads best, textures everywhere else ------
scene = Scene(background=("#bfe3f5", "#4a90c8"), size=(1100, 850))
scene.add(sea, texture=water, texture_scale=40)
scene.add(island, texture=sand, texture_scale=15)
scene.add(meadow, texture=grass, texture_scale=15)
scene.add(rocks, "#6c757d", smooth=False)  # low-poly on purpose
scene.add(plinth, texture=bricks, texture_scale=8)
scene.add(tower, texture=stripes, mapping="cylinder", texture_scale=16)
scene.add(door, texture=planks, texture_scale=3)
scene.add(glass, "#1d3557")
scene.add(deck, "#2b2d42")
scene.add(railing, "white")
scene.add(lantern, "#ffe066", ambient=0.7)  # a glowing lamp
scene.add(roof, "#d62828")
scene.add(finial, "gold", specular=0.8)
scene.add(beam, "#fff3b0", opacity=0.25, lighting=False)
scene.add(hull, texture=planks, texture_scale=4)
scene.add(mast, "#5c3a1e")
scene.add(sail_cloth, texture=sail, texture_scale=5)
scene.show(
    camera=[(150, -190, 90), (10, -15, 26), (0, 0, 1)],
    screenshot=Path(__file__).with_name("lighthouse.png"),
)
