"""Rooftops a mech can jet onto and hold.

A candidate is a flat building prism big enough to stand well back from its edges (so ground-level
enemies near the building can't see us) and low enough to climb with the jets (8-40 m above the
ground; 100 energy climbs about 60 m). For each we keep the stand point (the interior point
farthest from the edges), its inset, and a climb point: the nearest walkable ground cell, from which
the mech jumps and jets up the wall.
"""
import math

import numpy as np


def _area(r):
    xs, ys = r[0::2], r[1::2]
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2.0


def _inside(x, y, ring):
    xs, ys = ring[0::2], ring[1::2]
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        if (ys[i] > y) != (ys[j] > y) and x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]:
            inside = not inside
        j = i
    return inside


def _inside_np(px, py, xs, ys):
    res = np.zeros(px.shape, bool)
    j = len(xs) - 1
    for i in range(len(xs)):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        if yi != yj:
            res ^= ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        j = i
    return res


def _edge_dist_np(px, py, rings):
    best = np.full(px.shape, 1e9)
    for r in rings:
        xs, ys = r[0::2], r[1::2]
        for i in range(len(xs)):
            ax, ay, bx, by = xs[i - 1], ys[i - 1], xs[i], ys[i]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = np.zeros(px.shape) if L2 < 1e-9 else np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
            best = np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _edge_dist(x, y, rings):
    best = 1e9
    for r in rings:
        xs, ys = r[0::2], r[1::2]
        n = len(xs)
        for i in range(n):
            ax, ay, bx, by = xs[i - 1], ys[i - 1], xs[i], ys[i]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
            d = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
            if d < best:
                best = d
    return best


class Roofs:
    def __init__(self, m, nav, min_area=500.0, min_h=8.0, max_h=40.0, min_inset=6.0):
        self.m = m
        self.nav = nav
        self.list = []
        for idx, p in enumerate(m.prisms):
            if p.get("kind") != "building":
                continue
            ring = p["ring"]
            if len(ring) < 6 or _area(ring) < min_area:
                continue
            holes = [h for h in (p.get("holes") or []) if len(h) >= 6]
            xs, ys = ring[0::2], ring[1::2]
            gx = np.arange(min(xs) + 1.0, max(xs), 2.0)
            gy = np.arange(min(ys) + 1.0, max(ys), 2.0)
            if len(gx) == 0 or len(gy) == 0:
                continue
            X, Y = np.meshgrid(gx, gy)
            mask = _inside_np(X, Y, xs, ys)
            for h in holes:
                mask &= ~_inside_np(X, Y, h[0::2], h[1::2])
            if not mask.any():
                continue
            px, py = X[mask], Y[mask]
            ci = np.clip(np.floor(px - m.x0).astype(int), 0, m.nx - 1)
            cj = np.clip(np.floor(py - m.y0).astype(int), 0, m.ny - 1)
            flat = np.abs(m.solid_top[cj, ci] - p["top"]) <= 0.3
            if not flat.any():
                continue
            px, py = px[flat], py[flat]
            d = _edge_dist_np(px, py, [ring] + holes)
            k = int(np.argmax(d))
            best = (float(d[k]), float(px[k]), float(py[k]))
            if best is None or best[0] < min_inset:
                continue
            inset, sx, sy = best
            h = p["top"] - m.ground_z(sx, sy)
            if not (min_h <= h <= max_h):
                continue
            cx, cy = nav.nearest_walkable(sx, sy, comp=nav.main_component)
            climb_h = p["top"] - m.ground_z(cx, cy)
            if climb_h > 55.0:
                continue
            self.list.append({"idx": idx, "stand": (sx, sy), "top": float(p["top"]), "inset": inset,
                              "climb": (cx, cy), "ring": ring, "holes": holes, "height": h})

    def contains(self, roof, x, y):
        return _inside(x, y, roof["ring"]) and not any(_inside(x, y, h) for h in roof["holes"])

    def on(self, roof, pos):
        return pos[2] >= roof["top"] - 1.0 and self.contains(roof, pos[0], pos[1])

    def pick(self, pos, zone_center, zone_radius, bad=()):
        """Closest usable roof whose stand point is well inside the next circle."""
        best = None
        for r in self.list:
            if r["idx"] in bad:
                continue
            sx, sy = r["stand"]
            if zone_center is not None and math.hypot(sx - zone_center[0], sy - zone_center[1]) > zone_radius - 15.0:
                continue
            d = math.hypot(r["climb"][0] - pos[0], r["climb"][1] - pos[1])
            s = d - 2.0 * min(r["inset"], 15.0)
            if best is None or s < best[0]:
                best = (s, r)
        return best[1] if best else None
