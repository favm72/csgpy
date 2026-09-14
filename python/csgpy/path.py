"""A fluent 2D path builder: chain `.line(..)` / `.arc(..)` / `.soft(..)` /
`.angled(..)` calls, then `.build()` straight into a `Sketch` (or `.points()`
for the raw coordinate list, if you want to feed it to `polygon()` yourself
or splice it into a bigger point list).

Not a Rust binding - every method here just records a *waypoint*, and the
actual line-segment/arc-tessellation/curve-fitting math runs once, in
Python, when you call `.points()`/`.build()`. That's the same technique
`modern_house.py` uses by hand for its pool and rounded corners; this module
just gives it a name and a chainable notation instead of a one-off helper
function per shape.

    >>> from csgpy.path import path
    >>> rounded_square = (
    ...     path(4, 0)
    ...     .line(16, 0).arc(20, 4, center=(16, 4))
    ...     .line(20, 16).arc(16, 20, center=(16, 16))
    ...     .line(4, 20).arc(0, 16, center=(4, 16))
    ...     .line(0, 4).arc(4, 0, center=(4, 4))
    ...     .close()
    ...     .build()
    ... )
    >>> blob = (
    ...     path(25, -18)
    ...     .soft(28, 2).soft(18, 20).soft(-2, 24).soft(-10, 6)
    ...     .soft(-24, 14).soft(-30, -6).soft(-10, -22).soft(12, -24)
    ...     .close(smooth=True)
    ...     .build()
    ... )
"""

import math

from csgpy import polygon

__all__ = ["Path", "path"]


def _tessellate_arc(p_from, p_to, center, ccw, segments):
    """Points along the arc centered at `center` from `p_from` to `p_to`
    (inclusive of both ends). The radius is taken from `p_from`; `p_to` is
    projected onto that same radius first, so a slightly-off endpoint
    doesn't distort the rest of the path."""
    cx, cy = center
    r = math.hypot(p_from[0] - cx, p_from[1] - cy)
    a0 = math.atan2(p_from[1] - cy, p_from[0] - cx)
    a1 = math.atan2(p_to[1] - cy, p_to[0] - cx)
    if ccw:
        while a1 <= a0:
            a1 += 2 * math.pi
    else:
        while a1 >= a0:
            a1 -= 2 * math.pi
    return [
        (cx + r * math.cos(a), cy + r * math.sin(a))
        for a in (a0 + (a1 - a0) * i / segments for i in range(segments + 1))
    ]


def _catmull_rom_segment(p0, p1, p2, p3, samples):
    """Points from `p1` to `p2` (inclusive) along the cubic Bezier that
    matches a Catmull-Rom spline through `p0, p1, p2, p3` - i.e. the curve
    is tangent-continuous with its neighbors at both ends, not just heading
    toward `p2` in a straight line."""
    c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
    c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
    pts = []
    for s in range(samples + 1):
        t = s / samples
        mt = 1.0 - t
        pts.append((
            mt**3 * p1[0] + 3 * mt**2 * t * c1[0] + 3 * mt * t**2 * c2[0] + t**3 * p2[0],
            mt**3 * p1[1] + 3 * mt**2 * t * c1[1] + 3 * mt * t**2 * c2[1] + t**3 * p2[1],
        ))
    return pts


class Path:
    """Build with `path(x, y)`, not `Path(...)` directly."""

    def __init__(self, x, y):
        self._waypoints = [(x, y)]  # anchor points, start included
        self._segments = []         # one entry per waypoint after the first
        self._heading = 0.0         # degrees, for .angled(); 0 = +X
        self._closed = False

    def _advance(self, x, y, kind, **extra):
        self._waypoints.append((x, y))
        self._segments.append({"kind": kind, **extra})
        return self

    # -- straight lines ----------------------------------------------------
    def line(self, x, y):
        """Straight line to the absolute point `(x, y)`."""
        return self._advance(x, y, "line")

    def next(self, dx, dy):
        """Straight line to `(dx, dy)` *relative* to the current point."""
        x, y = self._waypoints[-1]
        return self.line(x + dx, y + dy)

    def angled(self, turn_deg, distance):
        """Turtle-style move: turn `turn_deg` degrees (positive =
        counter-clockwise) from the current heading, then go `distance`
        straight ahead. Heading starts at 0 (+X) and persists between
        calls, so a chain of `.angled(..)` traces a turning polyline
        without you having to compute each absolute point by hand."""
        self._heading += turn_deg
        rad = math.radians(self._heading)
        x, y = self._waypoints[-1]
        return self.line(x + distance * math.cos(rad), y + distance * math.sin(rad))

    # -- arcs ----------------------------------------------------------------
    def arc(self, x, y, center, ccw=True, segments=16):
        """Arc to `(x, y)` about `center=(cx, cy)`. `ccw` picks which way
        around the circle to go (there are always two arcs between two
        points on a circle); `segments` controls tessellation smoothness."""
        return self._advance(x, y, "arc", center=center, ccw=ccw, segments=segments)

    # -- tangent-continuous points --------------------------------------
    def soft(self, x, y):
        """A point the path curves *through* smoothly (tangent-continuous
        with its neighbors on both sides) instead of turning a sharp
        corner - a waypoint on a Catmull-Rom spline. The curve is shaped by
        the points *before and after* this one, so the very first point of
        an open (non-`close()`d) path can't itself be `soft()` - there's
        nothing before it to blend from."""
        return self._advance(x, y, "soft")

    def tangent(self, x, y):
        """Alias for `.soft(x, y)`, for when "aim the curve tangent to
        here" reads better than "pass through here softly"."""
        return self.soft(x, y)

    def close(self, smooth=False):
        """Mark the path closed - `polygon()` connects the last point back
        to the first with a straight line. Pass `smooth=True` to make
        *that* closing connection a `.soft(..)` curve too, so a fully
        `.soft(..)`-built loop comes back around without one sharp seam at
        the start. Call this last, before `.points()`/`.build()`."""
        self._closed = True
        if smooth:
            self._advance(*self._waypoints[0], "soft")
        return self

    # -- materialize -----------------------------------------------------
    def points(self, smoothness=12):
        """Tessellate every segment into a flat `[(x, y), ...]` list - feed
        it to `csgpy.polygon(...)` yourself, or just call `.build()`."""
        anchors = self._waypoints
        n = len(anchors)
        result = [anchors[0]]
        for i in range(1, n):
            seg = self._segments[i - 1]
            p_prev, p_cur = anchors[i - 1], anchors[i]
            kind = seg["kind"]
            if kind == "line":
                result.append(p_cur)
            elif kind == "arc":
                result.extend(
                    _tessellate_arc(p_prev, p_cur, seg["center"], seg["ccw"], seg["segments"])[1:]
                )
            elif kind == "soft":
                p0 = anchors[i - 2] if i - 2 >= 0 else p_prev
                p3 = anchors[i + 1] if i + 1 < n else (anchors[1] if self._closed and n > 1 else p_cur)
                result.extend(_catmull_rom_segment(p0, p_prev, p_cur, p3, smoothness)[1:])
            else:  # pragma: no cover - only reachable via internal misuse
                raise ValueError(f"unknown segment kind {kind!r}")
        return result

    def build(self, smoothness=12):
        """`polygon(self.points(smoothness))` - a ready-to-`.extrude(..)`/
        `.revolve(..)`/`.sweep(..)` `Sketch`."""
        return polygon(self.points(smoothness))


def path(x, y):
    """Start a new fluent `Path` at `(x, y)`."""
    return Path(x, y)
