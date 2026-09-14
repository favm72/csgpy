"""The PyVista render provider for `csgpy.Solid`.

This is an *optional* part of `csgpy`: the core package (building and
evaluating a lazy CSG tree via `Solid.to_mesh()`) has no rendering code and
no rendering dependency. Backends like this one live as their own
`csgpy.<backend>` submodule, each importable independently -
`from csgpy.pyvista import Scene`, and later e.g. `from csgpy.pyrender
import Scene` - without pulling in every other backend's dependencies.

Install it with the matching extra: `pip install csgpy[pyvista]` (or add
`csgpy[pyvista]` to your own project's dependencies).

    >>> scene = Scene()
    >>> scene.add(cup, color="gold")
    >>> scene.add(table, texture="wood.png")
    >>> scene.show()
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

try:
    import numpy as np
    import pyvista as pv
    from pyvista import Plotter
except ImportError as exc:  # pragma: no cover - exercised only when missing
    raise ImportError(
        "csgpy.pyvista requires the 'pyvista' extra. Install it with: "
        "pip install 'csgpy[pyvista]'"
    ) from exc

if TYPE_CHECKING:
    from . import Solid

# `Plotter` is the real `pyvista.Plotter`, re-exported (and listed in
# `__all__` to mark it intentional, for type checkers and ruff alike) so
# anyone who outgrows `Scene` can drop down to plain PyVista without a
# separate `import pyvista`. `Scene.plotter` is the one a scene draws into.
__all__ = ["Plotter", "Scene", "mesh", "show", "smooth_normals", "texture_coordinates"]

Mapping = Literal["box", "cylinder", "sphere"]
TextureLike = pv.Texture | np.ndarray | str | PathLike[str]
Camera = str | Sequence[Sequence[float]]


def mesh(solid: Solid) -> pv.PolyData:
    """Evaluate `solid` (see `Solid.to_mesh`) and wrap it as a `pyvista.PolyData`."""
    vertices, triangles = solid.to_mesh()

    points = np.asarray(vertices, dtype=float).reshape(-1, 3)
    faces = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)

    if faces.size == 0:
        return pv.PolyData(points)

    # VTK's flat face format: one leading vertex-count per face, here always
    # 3 since `Solid.to_mesh()` triangulates before returning.
    counts = np.full((faces.shape[0], 1), 3, dtype=np.int64)
    vtk_faces = np.hstack([counts, faces]).ravel()
    return pv.PolyData(points, vtk_faces)


def _face_normals(poly: pv.PolyData) -> np.ndarray:
    """One area-weighted (un-normalized) normal per vertex, taken from the
    triangle it belongs to. Relies on `Solid.to_mesh()`'s layout: every
    triangle has its own three vertices, stored in order."""
    tri = np.asarray(poly.points, dtype=float).reshape(-1, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return np.repeat(normals, 3, axis=0)


def texture_coordinates(
    poly: pv.PolyData, mapping: Mapping = "box", scale: float = 10.0
) -> np.ndarray:
    """Per-vertex `(u, v)` texture coordinates for a mesh from `mesh()`.

    `scale` is roughly how many world units one copy of the texture covers
    (textures repeat). `mapping` picks the projection:

    - `"box"`: each triangle is projected along its dominant normal axis -
      good for anything boxy, and a decent default for everything else.
    - `"cylinder"`: `u` wraps around the Z axis, `v` runs up it (flat caps
      fall back to a top-down projection) - towers, columns, bottles.
    - `"sphere"`: `u` wraps around Z, `v` runs pole to pole - balls, domes.

    `Solid.to_mesh()` never shares vertices between triangles, so each
    triangle gets its own coordinates and there are no smeared seams.
    """
    pts = np.asarray(poly.points, dtype=float)
    axis = np.abs(_face_normals(poly)).argmax(axis=1)  # dominant normal axis
    x, y, z = pts.T

    if mapping == "box":
        u = np.choose(axis, [y, x, x])
        v = np.choose(axis, [z, z, y])
        return np.column_stack([u, v]) / scale

    cx, cy, cz = poly.center
    angle = np.arctan2(y - cy, x - cx).reshape(-1, 3)
    # A triangle straddling the -X seam would otherwise span the whole
    # texture backwards; push its negative angles round by a full turn.
    wraps = angle.max(axis=1, keepdims=True) - angle.min(axis=1, keepdims=True) > np.pi
    angle = np.where(wraps & (angle < 0), angle + 2 * np.pi, angle).ravel()
    radius = max(np.hypot(x - cx, y - cy).max(), 1e-9)
    # A whole number of repeats around, so the texture meets itself at the seam.
    u = angle / (2 * np.pi) * max(1, round(2 * np.pi * radius / scale))

    if mapping == "cylinder":
        v = z / scale
        cap = axis == 2
        u = np.where(cap, x / scale, u)
        v = np.where(cap, y / scale, v)
    elif mapping == "sphere":
        polar = np.arctan2(np.hypot(x - cx, y - cy), z - cz)
        v = polar / np.pi * max(1, round(np.pi * radius / scale))
    else:
        raise ValueError(f"unknown mapping {mapping!r}")
    return np.column_stack([u, v])


def smooth_normals(poly: pv.PolyData, crease_angle: float = 40.0) -> np.ndarray:
    """Per-vertex normals that blend across soft edges but keep creases.

    Each vertex averages the normals of every triangle meeting at the same
    point whose surface bends less than `crease_angle` degrees away from its
    own - so a sphere or a revolved profile shades smoothly while a box
    keeps its sharp corners. (PyVista's own `smooth_shading` can't do this
    for `Solid` meshes: it never merges their unshared vertices, so every
    triangle stays flat.)
    """
    raw = _face_normals(poly)
    unit = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)

    # Group vertices by position, then list every (vertex, neighbor) pair
    # within each group.
    pts = np.asarray(poly.points, dtype=float)
    _, group = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
    group = group.ravel()
    order = np.argsort(group, kind="stable")
    sizes = np.bincount(group)[group[order]]
    starts = np.searchsorted(group[order], group[order])
    first = np.repeat(np.arange(len(order)), sizes)
    offset = np.arange(len(first)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    a, b = order[first], order[starts[first] + offset]

    keep = np.einsum("ij,ij->i", unit[a], unit[b]) > np.cos(np.radians(crease_angle))
    normals = np.zeros_like(raw)
    np.add.at(normals, a[keep], raw[b[keep]])
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def _as_texture(texture: TextureLike) -> pv.Texture:
    if isinstance(texture, pv.Texture):
        tex = texture
    elif isinstance(texture, np.ndarray):
        tex = pv.Texture(texture)
    else:
        tex = pv.read_texture(str(texture))
    tex.repeat = True
    tex.interpolate = True
    tex.mipmap = True
    return tex


class Scene:
    """A window you add solids to one at a time, each with its own look.

        scene = Scene(background="white")
        scene.add(walls, texture="bricks.png", texture_scale=8)
        scene.add(roof, color="firebrick")
        scene.show(camera="iso", screenshot="house.png")

    `add` returns the scene, so calls can also be chained. For anything
    this doesn't cover, `scene.plotter` is the underlying `pyvista.Plotter`.
    """

    def __init__(
        self,
        *,
        background: str | tuple[str, str] = "white",
        size: tuple[int, int] = (1000, 800),
        off_screen: bool = False,
        **plotter_kwargs: Any,
    ) -> None:
        """`background` is one color, or a `(bottom, top)` pair for a
        vertical gradient. `off_screen=True` renders without opening a
        window - for scripts that only save a `screenshot`."""
        self.plotter = pv.Plotter(window_size=list(size), off_screen=off_screen, **plotter_kwargs)
        bottom, top = (background, None) if isinstance(background, str) else background
        # `Plotter.set_background` is a `functools.wraps` alias that type
        # checkers mis-read; the `renderers` method it forwards to is the same.
        self.plotter.renderers.set_background(bottom, top=top)
        self.plotter.enable_anti_aliasing()

    def add(
        self,
        solid: Solid,
        color: str = "lightgray",
        *,
        texture: TextureLike | None = None,
        mapping: Mapping = "box",
        texture_scale: float = 10.0,
        smooth: bool = True,
        edges: bool = False,
        **add_mesh_kwargs: Any,
    ) -> Scene:
        """Add `solid` in a flat `color`, or wrapped in a `texture`.

        - `texture`: an image file path (e.g. a `.png`), a
          `(height, width, 3)` uint8 numpy array, or a `pyvista.Texture`.
          Textures repeat; `texture_scale` is how many world units one tile
          covers and `mapping` how it's projected - see
          `texture_coordinates`.
        - `smooth`: shade curved surfaces smoothly while keeping sharp edges
          sharp (see `smooth_normals`); `False` for a faceted, low-poly look.
        - `edges`: draw the triangle wireframe on top.
        - anything else (`opacity`, `specular`, `ambient`, ...) goes straight
          to `pyvista.Plotter.add_mesh`.
        """
        poly = mesh(solid)
        if poly.n_cells == 0:
            return self
        if texture is not None:
            poly.active_texture_coordinates = texture_coordinates(poly, mapping, texture_scale)
            add_mesh_kwargs["texture"] = _as_texture(texture)
        if smooth:
            poly.point_data["Normals"] = smooth_normals(poly)
            poly.point_data.active_normals_name = "Normals"
        actor = self.plotter.add_mesh(
            poly, color=color, show_edges=edges, smooth_shading=False, **add_mesh_kwargs
        )
        if smooth:
            # Our own normals, interpolated per pixel (PyVista's
            # `smooth_shading=True` would recompute - and flatten - them).
            actor.prop.interpolation = "phong"
        return self

    def show(self, *, camera: Camera | None = "iso", screenshot: str | PathLike[str] | None = None) -> None:
        """Open the window (or just render, if `off_screen`).

        `camera` is a PyVista preset (`"iso"`, `"xz"`, ...), an
        `[eye, focus, up]` list of points, or `None` to keep the current
        view. `screenshot` saves the rendered image to that path.
        """
        if camera is not None:
            self.plotter.camera_position = camera
        self.plotter.show(screenshot=str(screenshot) if screenshot else None)


def show(solid: Solid, color: str = "lightgray", **add_kwargs: Any) -> None:
    """Show a single solid in a window - `Scene().add(solid, ...).show()`.

    Takes the same options as `Scene.add` (`texture=`, `edges=`, ...).
    """
    Scene().add(solid, color, **add_kwargs).show()
