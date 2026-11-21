"""Static arena model: terrain heights, building solids, tree canopy, and line of sight.

Frame: x east, y north, z up, meters (protocol.md section 5). The terrain file is a vertex grid
(heights at x = x0 + i, y = y0 + j). Solids and trees are rasterized into 1 m cells: cell (j, i)
covers x in [x0 + i, x0 + i + 1) and y in [y0 + j, y0 + j + 1); a cell belongs to a prism when its
center lies inside the prism footprint (and outside its holes). Each cell keeps up to two solid
z-intervals (the map never stacks more than two prisms over one cell), so bridges can be walked
under.
"""
import json
import math
import os

import numpy as np

EMPTY_BASE = np.float32(1e9)
EMPTY_TOP = np.float32(-1e9)


def _inside(px, py, xs, ys):
    """Even-odd point-in-polygon for arrays of points against one ring."""
    res = np.zeros(px.shape, bool)
    n = len(xs)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        if yi != yj:
            cond = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            res ^= cond
        j = i
    return res


class Map:
    def __init__(self, map_dir, cfg=None):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        t = w["terrain"]
        self.x0 = float(t["minX"])
        self.y0 = float(t["minY"])
        self.nx = int(t["nx"])
        self.ny = int(t["ny"])
        raw = np.fromfile(os.path.join(map_dir, "v2", t.get("file", "terrain.i16")), dtype="<i2")
        self.ground = raw.reshape(self.ny, self.nx).astype(np.float32) * float(t.get("scale", 0.01))
        b = w["bounds"]
        self.bounds = (float(b["minX"]), float(b["maxX"]), float(b["minY"]), float(b["maxY"]))
        rules = w.get("rules", {})
        dens = float(rules.get("canopyDensity", 0.35))
        block_t = float(rules.get("canopyBlockT", 0.25))
        self.canopy_block_len = math.log(1.0 / block_t) / dens if dens > 0 else 1e9

        shape = (self.ny, self.nx)
        self.ib0 = np.full(shape, EMPTY_BASE, np.float32)
        self.it0 = np.full(shape, EMPTY_TOP, np.float32)
        self.ib1 = np.full(shape, EMPTY_BASE, np.float32)
        self.it1 = np.full(shape, EMPTY_TOP, np.float32)
        self.prisms = w["prisms"]
        for p in self.prisms:
            self._raster_prism(p)
        self.solid_top = np.maximum(self.it0, self.it1)  # highest solid top per cell (-1e9 if none)

        self.crown_lo = np.full(shape, EMPTY_BASE, np.float32)
        self.crown_hi = np.full(shape, EMPTY_TOP, np.float32)
        self.trunk_lo = np.full(shape, EMPTY_BASE, np.float32)
        self.trunk_hi = np.full(shape, EMPTY_TOP, np.float32)
        tr = w.get("trees", {})
        fields = tr.get("fields", [])
        data = tr.get("data", [])
        if fields:
            n = len(fields)
            ix = {k: fields.index(k) for k in fields}
            for k in range(0, len(data) - n + 1, n):
                row = data[k:k + n]
                self._raster_tree(row, ix)

    # ------------------------------------------------------------------ rasterization
    def _bbox(self, xs, ys, pad=0.0):
        i0 = int(math.floor(min(xs) - pad - self.x0))
        i1 = int(math.ceil(max(xs) + pad - self.x0))
        j0 = int(math.floor(min(ys) - pad - self.y0))
        j1 = int(math.ceil(max(ys) + pad - self.y0))
        if i1 < 0 or j1 < 0 or i0 >= self.nx or j0 >= self.ny:
            return None
        return max(i0, 0), min(i1, self.nx - 1), max(j0, 0), min(j1, self.ny - 1)

    def _raster_prism(self, p):
        ring = p["ring"]
        xs, ys = ring[0::2], ring[1::2]
        bb = self._bbox(xs, ys)
        if bb is None:
            return
        i0, i1, j0, j1 = bb
        cx = self.x0 + np.arange(i0, i1 + 1, dtype=np.float64) + 0.5
        cy = self.y0 + np.arange(j0, j1 + 1, dtype=np.float64) + 0.5
        X, Y = np.meshgrid(cx, cy)
        m = _inside(X, Y, xs, ys)
        for h in p.get("holes", []) or []:
            if len(h) >= 6:
                m &= ~_inside(X, Y, h[0::2], h[1::2])
        if not m.any():
            return
        base, top = np.float32(p["base"]), np.float32(p["top"])
        sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        b0, t0, b1, t1 = self.ib0[sl], self.it0[sl], self.ib1[sl], self.it1[sl]
        empty0 = m & (b0 >= EMPTY_BASE)
        b0[empty0] = base
        t0[empty0] = top
        rest = m & ~empty0
        if not rest.any():
            return
        # merge with slot 0 when touching (gap < 0.5 m), else use slot 1 (merging if it is taken)
        touch0 = rest & (base <= t0 + 0.5) & (top >= b0 - 0.5)
        b0[touch0] = np.minimum(b0[touch0], base)
        t0[touch0] = np.maximum(t0[touch0], top)
        rest &= ~touch0
        empty1 = rest & (b1 >= EMPTY_BASE)
        b1[empty1] = base
        t1[empty1] = top
        rest &= ~empty1
        b1[rest] = np.minimum(b1[rest], base)
        t1[rest] = np.maximum(t1[rest], top)

    def _raster_tree(self, row, ix):
        tz = row[ix["z"]]
        h = row[ix["height"]]
        cb = row[ix["crownBase"]]
        r = row[ix["crownR"]]
        cxw = row[ix["crownX"]] if "crownX" in ix else row[ix["x"]]
        cyw = row[ix["crownY"]] if "crownY" in ix else row[ix["y"]]
        # trunk: the cell holding the trunk center, from the ground to the crown base
        ti = int(math.floor(row[ix["x"]] - self.x0))
        tj = int(math.floor(row[ix["y"]] - self.y0))
        if 0 <= ti < self.nx and 0 <= tj < self.ny:
            self.trunk_lo[tj, ti] = min(self.trunk_lo[tj, ti], tz)
            self.trunk_hi[tj, ti] = max(self.trunk_hi[tj, ti], tz + cb)
        # crown: an ellipsoid of horizontal radius r between tz + cb and tz + h
        if r <= 0 or h <= cb:
            return
        bb = self._bbox([cxw], [cyw], pad=r)
        if bb is None:
            return
        i0, i1, j0, j1 = bb
        cx = self.x0 + np.arange(i0, i1 + 1) + 0.5
        cy = self.y0 + np.arange(j0, j1 + 1) + 0.5
        X, Y = np.meshgrid(cx, cy)
        d2 = ((X - cxw) ** 2 + (Y - cyw) ** 2) / (r * r)
        m = d2 < 1.0
        if not m.any():
            return
        mid = tz + (cb + h) / 2.0
        half = (h - cb) / 2.0 * np.sqrt(np.clip(1.0 - d2, 0.0, 1.0))
        lo = (mid - half).astype(np.float32)
        hi = (mid + half).astype(np.float32)
        sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        clo, chi = self.crown_lo[sl], self.crown_hi[sl]
        clo[m] = np.minimum(clo[m], lo[m])
        chi[m] = np.maximum(chi[m], hi[m])

    # ------------------------------------------------------------------ point queries
    def cell(self, x, y):
        i = int(math.floor(x - self.x0))
        j = int(math.floor(y - self.y0))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def ground_z(self, x, y):
        fx = x - self.x0
        fy = y - self.y0
        i = min(max(int(math.floor(fx)), 0), self.nx - 2)
        j = min(max(int(math.floor(fy)), 0), self.ny - 2)
        tx = min(max(fx - i, 0.0), 1.0)
        ty = min(max(fy - j, 0.0), 1.0)
        g = self.ground
        a = float(g[j, i]) * (1 - tx) + float(g[j, i + 1]) * tx
        b = float(g[j + 1, i]) * (1 - tx) + float(g[j + 1, i + 1]) * tx
        return a * (1 - ty) + b * ty

    def ground_many(self, x, y):
        fx = np.asarray(x, np.float64) - self.x0
        fy = np.asarray(y, np.float64) - self.y0
        i = np.clip(np.floor(fx).astype(np.int64), 0, self.nx - 2)
        j = np.clip(np.floor(fy).astype(np.int64), 0, self.ny - 2)
        tx = np.clip(fx - i, 0.0, 1.0)
        ty = np.clip(fy - j, 0.0, 1.0)
        g = self.ground
        a = g[j, i] * (1 - tx) + g[j, i + 1] * tx
        b = g[j + 1, i] * (1 - tx) + g[j + 1, i + 1] * tx
        return a * (1 - ty) + b * ty

    def surface_z(self, x, y, z_feet, step):
        """Highest walkable surface under (x, y) at or below z_feet + step (terrain if nothing else)."""
        best = self.ground_z(x, y)
        i, j = self.cell(x, y)
        lim = z_feet + step
        for t in (float(self.it0[j, i]), float(self.it1[j, i])):
            if best < t <= lim:
                best = t
        return best

    def head_clear(self, x, y, z_feet, height):
        """Lowest solid base above z_feet under (x, y), or +inf (for jets and jumps)."""
        i, j = self.cell(x, y)
        lo = 1e9
        for b in (float(self.ib0[j, i]), float(self.ib1[j, i])):
            if z_feet < b < lo:
                lo = b
        return lo

    def blocked(self, x, y, z_feet, radius, height, step):
        """True if a standing cylinder at (x, y, z_feet) would overlap a wall or a terrain rise."""
        i0 = int(math.floor(x - radius - self.x0 - 0.5))
        i1 = int(math.floor(x + radius - self.x0 - 0.5)) + 1
        j0 = int(math.floor(y - radius - self.y0 - 0.5))
        j1 = int(math.floor(y + radius - self.y0 - 0.5)) + 1
        if i0 < 0 or j0 < 0 or i1 >= self.nx or j1 >= self.ny:
            return True
        sl = (slice(j0, j1 + 1), slice(i0, i1 + 1))
        cx = self.x0 + np.arange(i0, i1 + 1) + 0.5
        cy = self.y0 + np.arange(j0, j1 + 1) + 0.5
        near = ((cx[None, :] - x) ** 2 + (cy[:, None] - y) ** 2) <= radius * radius
        lo = z_feet + step
        hi = z_feet + height
        wall = ((self.it0[sl] > lo) & (self.ib0[sl] < hi)) | ((self.it1[sl] > lo) & (self.ib1[sl] < hi))
        if (wall & near).any():
            return True
        g = self.ground[j0:j1 + 2, i0:i1 + 2]
        gc = (g[:-1, :-1] + g[:-1, 1:] + g[1:, :-1] + g[1:, 1:]) * 0.25
        return bool(((gc > lo) & near).any())

    # ------------------------------------------------------------------ line of sight
    def los(self, a, b, step=0.5):
        """Shot line of sight: True if nothing solid (buildings, bridges, terrain) is between a and b."""
        return bool(self.los_many(a, [b], step)[0])

    def los_eye(self, a, b, step=0.5):
        """Vision line of sight: also blocked by tree trunks and by dense canopy."""
        return bool(self.los_many(a, [b], step, eye=True)[0])

    def los_many(self, a, pts, step=0.5, eye=False):
        P = np.asarray(pts, np.float64).reshape(-1, 3)
        if len(P) == 0:
            return np.zeros(0, bool)
        A = np.asarray(a, np.float64).reshape(1, 3)
        D = P - A
        L = np.sqrt((D * D).sum(axis=1))
        k = max(2, int(L.max() / step) + 1)
        t = (np.arange(k, dtype=np.float64) + 0.5) / k
        S = A[:, None, :] + D[:, None, :] * t[None, :, None]
        x, y, z = S[..., 0], S[..., 1], S[..., 2]
        fx = x - self.x0
        fy = y - self.y0
        ci = np.clip(np.floor(fx).astype(np.int64), 0, self.nx - 1)
        cj = np.clip(np.floor(fy).astype(np.int64), 0, self.ny - 1)
        vi = np.minimum(ci, self.nx - 2)
        vj = np.minimum(cj, self.ny - 2)
        tx = np.clip(fx - vi, 0.0, 1.0)
        ty = np.clip(fy - vj, 0.0, 1.0)
        g = self.ground
        gz = (g[vj, vi] * (1 - tx) + g[vj, vi + 1] * tx) * (1 - ty) + (g[vj + 1, vi] * (1 - tx) + g[vj + 1, vi + 1] * tx) * ty
        hit = z < gz
        hit |= (self.ib0[cj, ci] <= z) & (z <= self.it0[cj, ci])
        hit |= (self.ib1[cj, ci] <= z) & (z <= self.it1[cj, ci])
        if eye:
            hit |= (self.trunk_lo[cj, ci] <= z) & (z <= self.trunk_hi[cj, ci])
            in_crown = (self.crown_lo[cj, ci] <= z) & (z <= self.crown_hi[cj, ci])
            crown_len = in_crown.sum(axis=1) * (L / k)
            return ~hit.any(axis=1) & (crown_len <= self.canopy_block_len)
        return ~hit.any(axis=1)
