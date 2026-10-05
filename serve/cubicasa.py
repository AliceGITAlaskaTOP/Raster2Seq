"""Raster2Seq output → CubiCasa5K SVG, the input of Floorplan2Walkthru.

Raster2Seq (CubiCasa5K checkpoint) gives room polygons plus doors and
windows as short segments, all in image pixels. Floorplan2Walkthru builds
its 3D from a CubiCasa ``model.svg``:

    <g class="Space Kitchen"><polygon/></g>                  rooms
    <g class="Wall [External]"><polygon 4 points/>           walls
        <g class="Door Swing Beside"><polygon/></g>          openings nested
        <g class="Window Regular"><polygon/></g>             in their wall
    </g>

SVG units are centimetres (``scaleMetersPerUnit = 0.01``), the same as the
CubiCasa5K annotations. Walls are not predicted by the model: an interior
wall is the gap between two facing room edges, an exterior wall is a room
edge with nobody on the other side (its thickness is measured on the image).
"""

import math
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Raster2Seq CubiCasa5K label → CubiCasa SVG class, Russian name
ROOM_CLASSES = {
    "Outdoor": ("Outdoor", "Улица"),
    "Kitchen": ("Kitchen", "Кухня"),
    "Living Room": ("LivingRoom", "Гостиная"),
    "Bed Room": ("Bedroom", "Спальня"),
    "Bath": ("Bath", "Санузел"),
    "Entry": ("Entry", "Прихожая"),
    "Storage": ("Storage", "Кладовая"),
    "Garage": ("Garage", "Гараж"),
    "Undefined": ("Undefined", "Комната"),
}

DOOR_WIDTH_M = 0.85  # typical door leaf + frame: the scale when nothing else is known
INNER_WALL_M = (0.08, 0.35)  # interior wall thickness bounds
INNER_WALL_DEFAULT_M = 0.12
MIN_ROOM_M2 = 1.5  # an unrecognized hole inside the house bigger than this is a room too
OUTER_WALL_M = (0.15, 0.7)  # exterior wall thickness bounds
OUTER_WALL_DEFAULT_M = 0.3
DOOR_HEIGHT_M = 2.0
WINDOW_HEIGHT_M = 1.2
WINDOW_SILL_M = 0.8
AXIS_DEG = 15  # an edge closer than this to horizontal/vertical becomes exactly so


# ── geometry helpers ──────────────────────────────────────────────


def _dedup(pts, tol):
    out = []
    for p in pts:
        if not out or math.dist(p, out[-1]) > tol:
            out.append(list(p))
    if len(out) > 1 and math.dist(out[0], out[-1]) <= tol:
        out.pop()
    return out


def _drop_collinear(pts, tol):
    changed = True
    while changed and len(pts) > 3:
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if _point_line(b, a, c) < tol:
                del pts[i]
                changed = True
                break
    return pts


def _point_line(p, a, b):
    ax, ay = a
    bx, by = b
    l = math.hypot(bx - ax, by - ay)
    if l < 1e-9:
        return math.dist(p, a)
    return abs((bx - ax) * (ay - p[1]) - (ax - p[0]) * (by - ay)) / l


def _axis(a, b):
    """'v', 'h' or None for an edge."""
    ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180
    if min(ang, 180 - ang) < AXIS_DEG:
        return "h"
    if abs(ang - 90) < AXIS_DEG:
        return "v"
    return None


def _manhattan(pts):
    """Straighten near-axis edges: a vertex takes x from its vertical edge, y from its horizontal one."""
    n = len(pts)
    out = [list(p) for p in pts]
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        ax = _axis(a, b)
        if ax == "v":
            x = (a[0] + b[0]) / 2
            out[i][0] = out[(i + 1) % n][0] = x
        elif ax == "h":
            y = (a[1] + b[1]) / 2
            out[i][1] = out[(i + 1) % n][1] = y
    return out


def _cluster(values, tol):
    """Map each value to the mean of its cluster (sorted, split at gaps > tol)."""
    if not values:
        return {}
    vs = sorted(set(values))
    groups, cur = [], [vs[0]]
    for v in vs[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return {v: sum(g) / len(g) for g in groups for v in g}


def _largest(geom):
    geom = make_valid(geom)
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(polys, key=lambda g: g.area) if polys else None


def _ring(poly):
    pts = [list(p) for p in poly.exterior.coords[:-1]]
    if poly.exterior.is_ccw:  # image y points down: keep one winding for everyone
        pts.reverse()
    return pts


# ── rooms ─────────────────────────────────────────────────────────


def _clean_rooms(rooms, tol):
    """Straight, snapped, non-overlapping room polygons (pixels)."""
    polys = []
    for r in rooms:
        pts = _dedup(r["polygon"], tol)
        if len(pts) < 3:
            continue
        pts = _dedup(_manhattan(pts), tol)
        if len(pts) >= 3:
            polys.append([r["type"], pts])

    # One x for every vertical line that is the same line in two rooms, one y likewise
    xs, ys = [], []
    for _, pts in polys:
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            ax = _axis(a, b)
            if ax == "v":
                xs += [a[0], b[0]]
            elif ax == "h":
                ys += [a[1], b[1]]
    mx, my = _cluster(xs, tol), _cluster(ys, tol)
    for _, pts in polys:
        for p in pts:
            p[0] = mx.get(p[0], p[0])
            p[1] = my.get(p[1], p[1])

    # Overlaps: a smaller room keeps its area, the larger one loses it
    shapes = []
    for typ, pts in polys:
        pts = _drop_collinear(_dedup(pts, tol / 2), tol / 2)
        if len(pts) < 3:
            continue
        g = _largest(Polygon(pts))
        if g is not None and g.area > (tol * 3) ** 2:
            shapes.append((typ, g))
    shapes.sort(key=lambda s: s[1].area)
    taken, out = None, []
    for typ, g in shapes:
        if taken is not None:
            g = _largest(g.difference(taken))
            if g is None or g.area < (tol * 3) ** 2:
                continue
        taken = g if taken is None else unary_union([taken, g])
        # Opening cuts the spikes clipping leaves behind; mitre keeps right angles
        g = _largest(g.buffer(-tol, join_style=2).buffer(tol, join_style=2))
        if g is not None and g.area > (tol * 3) ** 2:
            out.append((typ, g.simplify(tol / 2)))
    return out


def _outline(rooms, px_per_m):
    """The house as one filled shape: rooms closed over wall gaps, holes filled."""
    indoor = [g for t, g in rooms if t != "Outdoor"]
    if not indoor:
        return None
    c = INNER_WALL_M[1] * px_per_m
    u = unary_union(indoor).buffer(c, join_style=2).buffer(-c, join_style=2)
    parts = [u] if u.geom_type == "Polygon" else [g for g in getattr(u, "geoms", []) if g.geom_type == "Polygon"]
    return unary_union([Polygon(g.exterior) for g in parts])


def _holes_as_rooms(rooms, px_per_m):
    """Space enclosed by rooms that the model left unlabelled (often a hallway) becomes a room."""
    indoor = [g for t, g in rooms if t != "Outdoor"]
    if not indoor:
        return rooms
    c = INNER_WALL_M[1] * px_per_m
    u = unary_union(indoor).buffer(c, join_style=2).buffer(-c, join_style=2)
    parts = [u] if u.geom_type == "Polygon" else [g for g in getattr(u, "geoms", []) if g.geom_type == "Polygon"]
    taken = unary_union(indoor).buffer(INNER_WALL_DEFAULT_M / 2 * px_per_m, join_style=2)
    added = []
    for part in parts:
        for ring in part.interiors:
            g = _largest(Polygon(ring).difference(taken))
            if g is not None and g.area > MIN_ROOM_M2 * px_per_m**2:
                added.append(("Undefined", g))
    return rooms + added


# ── walls ─────────────────────────────────────────────────────────


class _Edge:
    def __init__(self, room, a, b, ext):
        self.room = room
        self.a = np.array(a, float)
        self.b = np.array(b, float)
        d = self.b - self.a
        self.len = float(np.hypot(*d))
        self.u = d / self.len
        # outward normal: the side of the edge that is not inside its own room
        self.n = np.array([self.u[1], -self.u[0]])
        if ext is not None and ext.contains(_P(self.a + self.u * self.len / 2 + self.n * 0.5)):
            self.n = -self.n
        self.free = [(0.0, self.len)]  # parts of the edge with no wall yet

    def take(self, t0, t1):
        rest = []
        for s, e in self.free:
            if e <= t0 or s >= t1:
                rest.append((s, e))
                continue
            if s < t0:
                rest.append((s, t0))
            if e > t1:
                rest.append((t1, e))
        self.free = [(s, e) for s, e in rest if e - s > 1e-6]


def _P(p):
    return Point(float(p[0]), float(p[1]))


def _dark_run(gray, p, n, max_px):
    """Pixels of dark ink along n starting at p (a wall drawn in black)."""
    h, w = gray.shape
    run, started, miss = 0, False, 0
    for k in range(int(max_px)):
        x, y = int(round(p[0] + n[0] * k)), int(round(p[1] + n[1] * k))
        if not (0 <= x < w and 0 <= y < h):
            break
        if gray[y, x] < 110:
            started = True
            run = k + 1
            miss = 0
        elif started:
            miss += 1
            if miss > 2:
                break
        elif k > max_px / 4:
            break
    return run


def _walls(rooms, px_per_m, gray, tol):
    indoor = [(t, g) for t, g in rooms if t != "Outdoor"]
    building = _outline(rooms, px_per_m)
    edges = []
    for idx, (typ, g) in enumerate(indoor):
        pts = _ring(g)
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            if math.dist(a, b) > tol:
                edges.append(_Edge(idx, a, b, g))

    max_gap = INNER_WALL_M[1] * px_per_m * 1.6
    min_len = max(tol, 0.25 * px_per_m)
    walls = []

    # Interior: two edges of different rooms, parallel, facing each other across a gap
    for i, e in enumerate(edges):
        for f in edges[i + 1 :]:
            if f.room == e.room or abs(float(e.u @ f.u)) < math.cos(math.radians(8)):
                continue
            if float(e.n @ f.n) > -0.9:
                continue
            gap = float((f.a - e.a) @ e.n)
            if gap < -tol or gap > max_gap:
                continue
            ta = sorted([float((f.a - e.a) @ e.u), float((f.b - e.a) @ e.u)])
            for s, t in list(e.free):
                lo, hi = max(s, ta[0]), min(t, ta[1])
                if hi - lo < min_len:
                    continue
                thick = min(max(gap, INNER_WALL_M[0] * px_per_m), INNER_WALL_M[1] * px_per_m)
                mid = max(gap, 0) / 2
                p0 = e.a + e.u * lo + e.n * mid
                p1 = e.a + e.u * hi + e.n * mid
                walls.append({"a": p0, "b": p1, "t": thick, "ext": False})
                e.take(lo, hi)
                fs = sorted([float((p0 - f.a) @ f.u), float((p1 - f.a) @ f.u)])
                f.take(fs[0] - tol, fs[1] + tol)

    # What is left of each edge has no room facing it: an exterior wall if the
    # outside of the house is behind it, an interior one otherwise. Thickness
    # is the dark ink measured on the image, within the bounds of its kind.
    for e in edges:
        for s, t in e.free:
            if t - s < min_len:
                continue
            outward = e.n
            probe = e.a + e.u * (s + t) / 2 + outward * (OUTER_WALL_M[1] * px_per_m)
            ext = building is None or not building.contains(_P(probe))
            lo_m, hi_m = OUTER_WALL_M if ext else INNER_WALL_M
            runs = []
            if gray is not None:
                for k in np.linspace(s + (t - s) * 0.2, t - (t - s) * 0.2, 7):
                    r = _dark_run(gray, e.a + e.u * k, outward, hi_m * px_per_m * 1.3)
                    if r:
                        runs.append(r)
            thick = float(np.median(runs)) if len(runs) >= 3 else (OUTER_WALL_DEFAULT_M if ext else INNER_WALL_DEFAULT_M) * px_per_m
            thick = min(max(thick, lo_m * px_per_m), hi_m * px_per_m)
            p0 = e.a + e.u * s + outward * thick / 2
            p1 = e.a + e.u * t + outward * thick / 2
            walls.append({"a": p0, "b": p1, "t": thick, "ext": ext})

    # Corners: every wall runs half a thickness past its ends, so neighbours overlap instead of gapping
    for w in walls:
        u = (w["b"] - w["a"]) / max(np.hypot(*(w["b"] - w["a"])), 1e-9)
        w["a"] = w["a"] - u * w["t"] / 2
        w["b"] = w["b"] + u * w["t"] / 2
    return walls


def _wall_poly(w):
    a, b, t = w["a"], w["b"], w["t"]
    d = b - a
    u = d / max(np.hypot(*d), 1e-9)
    n = np.array([-u[1], u[0]]) * t / 2
    return [a + n, b + n, b - n, a - n]


def _attach(openings, walls, tol):
    """Opening segment → (wall index, t0, t1 along the wall)."""
    out = []
    for seg in openings:
        p, q = np.array(seg[0], float), np.array(seg[1], float)
        if np.hypot(*(q - p)) < tol:
            continue
        c = (p + q) / 2
        best = None
        for i, w in enumerate(walls):
            d = w["b"] - w["a"]
            L = float(np.hypot(*d))
            u = d / L
            if abs(float(u @ ((q - p) / np.hypot(*(q - p))))) < math.cos(math.radians(25)):
                continue
            dist = abs(float((c - w["a"]) @ np.array([-u[1], u[0]])))
            t = float((c - w["a"]) @ u)
            if t < 0 or t > L or dist > w["t"] / 2 + tol * 2:
                continue
            if best is None or dist < best[0]:
                ts = sorted([float((p - w["a"]) @ u), float((q - w["a"]) @ u)])
                best = (dist, i, max(ts[0], 0), min(ts[1], L))
        if best:
            out.append(best[1:])
    return out


# ── scale ─────────────────────────────────────────────────────────


def px_per_meter(result, scale=None):
    """Pixels in one metre: given by the caller, or from the doors, or a CubiCasa-like guess."""
    if scale:
        return float(scale)
    lens = [math.dist(d[0], d[1]) for d in result.get("doors", [])]
    lens = [l for l in lens if l > 2]
    if len(lens) >= 2:
        return float(np.median(lens)) / DOOR_WIDTH_M
    side = max(result["width"], result["height"])
    return side / 12.0  # an average flat or house is about 12 m across


# ── main entry ────────────────────────────────────────────────────


def to_cubicasa(result, name="План", gray=None, scale=None):
    """Recognized pixels → {plan, svg, config} for Floorplan2Walkthru.

    ``plan`` repeats Floorplan2Walkthru's own Plan type (metres), ``svg`` is
    its import format (CubiCasa model.svg, centimetres), ``config`` its
    per-plan config.json.
    """
    ppm = px_per_meter(result, scale)
    side = max(result["width"], result["height"])
    tol = max(side / 256 * 1.25, 1.0)  # one model cell and a bit: Raster2Seq sees 256 px

    rooms = _holes_as_rooms(_clean_rooms(result["rooms"], tol), ppm)
    walls = _walls(rooms, ppm, gray, tol)
    doors = _attach(result.get("doors", []), walls, tol)
    windows = _attach(result.get("windows", []), walls, tol)
    named = [(*ROOM_CLASSES.get(typ, ("Undefined", "Комната")), g) for typ, g in rooms]
    return _write(named, walls, doors, windows, ppm, tol, name)


ROOM_TYPES = {cls for cls, _ in ROOM_CLASSES.values()}


def from_plan(plan, name="План"):
    """A Floorplan2Walkthru Plan drawn or edited by hand (metres) → the same
    {plan, svg, config}: wall footprints, areas and outline are rebuilt here,
    so a drawn plan and a recognized one come out of one writer.

    Walls are centrelines (start, end, thickness, isExterior); doors and windows
    sit on a wall by id (position 0..1 along it, width); rooms are polygons with
    a CubiCasa type and a name. ``origin`` in the answer is the shift (metres)
    applied so the plan starts at (0, 0).
    """
    walls, index = [], {}
    for w in plan.get("walls") or []:
        try:
            a = np.array(w["start"], float)[:2]
            b = np.array(w["end"], float)[:2]
            t = float(w.get("thickness") or 0.12)
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(a).all() and np.isfinite(b).all()) or np.hypot(*(b - a)) < 0.01:
            continue
        index[str(w.get("id"))] = len(walls)
        walls.append({"a": a, "b": b, "t": min(max(t, 0.03), 1.0), "ext": bool(w.get("isExterior"))})
    starts = [w["a"].copy() for w in walls]
    lengths = [float(np.hypot(*(w["b"] - w["a"]))) for w in walls]

    # A drawn corner: two centrelines meet at one point. Each wall runs half its
    # thickness past such an end, so the corner is filled instead of notched.
    ends = [(i, k, w[k]) for i, w in enumerate(walls) for k in ("a", "b")]
    for i, k, p in ends:
        if any(j != i and np.hypot(*(p - q)) < 0.01 for j, _, q in ends):
            w = walls[i]
            u = (w["b"] - w["a"]) / max(np.hypot(*(w["b"] - w["a"])), 1e-9)
            w[k] = p - u * w["t"] / 2 if k == "a" else p + u * w["t"] / 2
    # openings are placed on the wall as drawn: shift them by the extension of its start
    shift = {i: float(np.hypot(*(w["a"] - a0))) for i, (w, a0) in enumerate(zip(walls, starts))}

    def openings(items):
        out = []
        for o in items or []:
            wi = index.get(str(o.get("wallId")))
            if wi is None:
                continue
            w = walls[wi]
            L = float(np.hypot(*(w["b"] - w["a"])))
            try:
                c = float(o.get("position", 0.5)) * lengths[wi] + shift[wi]
                half = float(o.get("width", 0.9)) / 2
            except (TypeError, ValueError):
                continue
            t0, t1 = max(c - half, 0.0), min(c + half, L)
            if t1 - t0 > 0.05:
                out.append((wi, t0, t1))
        return out

    named = []
    for r in plan.get("rooms") or []:
        try:
            g = _largest(Polygon([(float(x), float(y)) for x, y in r["polygon"]]))
        except (KeyError, TypeError, ValueError):
            continue
        if g is None or g.area < 0.05:
            continue
        cls = r.get("type") if r.get("type") in ROOM_TYPES else "Undefined"
        ru = str(r.get("name") or dict(ROOM_CLASSES.values()).get(cls, "Комната"))[:60]
        named.append((cls, ru, g))
    return _write(named, walls, openings(plan.get("doors")), openings(plan.get("windows")), 1.0, 0.02, name)


def _write(rooms, walls, doors, windows, ppm, tol, name):
    """rooms [(cls, name, Polygon)], walls [{a, b, t, ext}], openings
    [(wall index, t0, t1)] in source units (ppm per metre) → the answer."""
    cm = 100.0 / ppm  # pixels → SVG units (cm)
    m = 1.0 / ppm

    # Everything shifted so the plan starts at (0, 0)
    allpts = [p for _, _, g in rooms for p in g.exterior.coords] + [p for w in walls for p in _wall_poly(w)]
    if allpts:
        x0 = min(p[0] for p in allpts)
        y0 = min(p[1] for p in allpts)
    else:
        x0 = y0 = 0.0

    def P(p, k):
        return [round((float(p[0]) - x0) * k, 3), round((float(p[1]) - y0) * k, 3)]

    def pts_attr(pts):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)

    plan_rooms, svg_rooms = [], []
    for i, (cls, ru, g) in enumerate(rooms):
        ring = _ring(g)
        poly_m = [P(p, m) for p in ring]
        plan_rooms.append(
            {"id": f"room_{i}", "name": ru, "type": cls, "polygon": poly_m, "area": round(g.area * m * m, 2)}
        )
        svg_rooms.append(
            f'<g id="Space" class="Space {cls}"><polygon points="{pts_attr([P(p, cm) for p in ring])}"/></g>'
        )

    plan_walls, svg_walls = [], []
    by_wall = {}
    for kind, items in (("door", doors), ("window", windows)):
        for wi, t0, t1 in items:
            by_wall.setdefault(wi, []).append((kind, t0, t1))
    plan_doors, plan_windows = [], []
    for i, w in enumerate(walls):
        wid = f"wall_{i}"
        L = float(np.hypot(*(w["b"] - w["a"])))
        u = (w["b"] - w["a"]) / max(L, 1e-9)
        nrm = np.array([-u[1], u[0]]) * w["t"] / 2
        plan_walls.append(
            {
                "id": wid,
                "polygon": [P(p, m) for p in _wall_poly(w)],
                "start": P(w["a"], m),
                "end": P(w["b"], m),
                "thickness": round(w["t"] * m, 3),
                "isExterior": w["ext"],
            }
        )
        inner = []
        for k, (kind, t0, t1) in enumerate(by_wall.get(i, [])):
            a, b = w["a"] + u * t0, w["a"] + u * t1
            quad = [P(a + nrm, cm), P(b + nrm, cm), P(b - nrm, cm), P(a - nrm, cm)]
            width = round((t1 - t0) * m, 3)
            pos = round((t0 + t1) / 2 / L, 4)
            if kind == "door":
                plan_doors.append(
                    {"id": f"door_{wid}_{k}", "wallId": wid, "position": pos, "width": width, "height": DOOR_HEIGHT_M}
                )
                inner.append(
                    f'<g id="Door" class="Door Swing Beside"><g class="Threshold">'
                    f'<polygon points="{pts_attr(quad)}"/></g></g>'
                )
            else:
                plan_windows.append(
                    {
                        "id": f"win_{wid}_{k}",
                        "wallId": wid,
                        "position": pos,
                        "width": width,
                        "height": WINDOW_HEIGHT_M,
                        "sillHeight": WINDOW_SILL_M,
                    }
                )
                inner.append(
                    f'<g id="Window" class="Window Regular"><g class="Glass">'
                    f'<polygon points="{pts_attr(quad)}"/></g></g>'
                )
        cls = "Wall External" if w["ext"] else "Wall"
        svg_walls.append(
            f'<g id="Wall" class="{cls}"><polygon points="{pts_attr([P(p, cm) for p in _wall_poly(w)])}"/>'
            + "".join(inner)
            + "</g>"
        )

    # Outline of the building (rooms and walls, without the outdoors)
    shapes = [g for cls, _, g in rooms if cls != "Outdoor"] + [Polygon(_wall_poly(w)) for w in walls]
    outer = []
    if shapes:
        u = _largest(unary_union([s.buffer(tol / 2) for s in shapes]).buffer(-tol / 2))
        if u is not None:
            outer = [P(p, m) for p in _ring(u.simplify(tol))]

    if allpts:
        W = (max(p[0] for p in allpts) - x0) * cm
        H = (max(p[1] for p in allpts) - y0) * cm
    else:
        W = H = 100.0
    svg = (
        '<?xml version="1.0"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" height="{H:.2f}" viewBox="0 0 {W:.2f} {H:.2f}">'
        '<g id="Model" class="Model v1-1"><g class="Floor"><g id="Floor-1" class="Floorplan Floor-1">'
        + "".join(svg_rooms)
        + "".join(svg_walls)
        + "</g></g></g></svg>"
    )
    start = max(plan_rooms, key=lambda r: r["area"] if r["type"] != "Outdoor" else -1)["type"] if plan_rooms else None
    config = {"name": name, "scaleMetersPerUnit": 0.01, "startRoom": start, "outerPerimeter": outer}
    plan = {
        "id": "plan",
        "name": name,
        "scaleMetersPerUnit": 0.01,
        "walls": plan_walls,
        "doors": plan_doors,
        "windows": plan_windows,
        "rooms": plan_rooms,
        "outerPerimeter": outer,
    }
    return {
        "pxPerMeter": round(ppm, 4),
        "origin": [round(x0, 4), round(y0, 4)],
        "plan": plan,
        "svg": svg,
        "config": config,
    }
