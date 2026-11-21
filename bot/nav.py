"""Navigation: a 2 m walkability grid over the static map, line walk checks, and budgeted A*.

A 1 m cell blocks walking when a solid occupies body height above the local ground (from
step_up to the mech's height) or when the terrain is steeper than max_slope. The blocked mask is
then grown by the mech radius, so a center line that avoids it keeps the whole cylinder clear.
Roofs and walkways count as blocked: a mech on a roof walks off the edge toward the nearest
walkable cell and drops down.
"""
import heapq
import math
import time

import numpy as np

SQRT2 = math.sqrt(2.0)


class NavGrid:
    def __init__(self, m, step_up=1.2, height=9.0, radius=2.5, max_slope_deg=50.0, inflate=None):
        self.m = m
        ny, nx = m.ny, m.nx
        g = m.ground
        gc = np.empty((ny, nx), np.float32)
        gc[:-1, :-1] = (g[:-1, :-1] + g[:-1, 1:] + g[1:, :-1] + g[1:, 1:]) * 0.25
        gc[-1, :] = gc[-2, :]
        gc[:, -1] = gc[:, -2]
        lo = gc + step_up
        hi = gc + height
        wall = ((m.it0 > lo) & (m.ib0 < hi)) | ((m.it1 > lo) & (m.ib1 < hi))
        tan = math.tan(math.radians(max_slope_deg))
        dx = np.abs(g[:, 1:] - g[:, :-1]) > tan
        dy = np.abs(g[1:, :] - g[:-1, :]) > tan
        steep = np.zeros((ny, nx), bool)
        steep[:, :-1] |= dx
        steep[:-1, :-1] |= dx[1:, :]
        steep[:-1, :] |= dy
        steep[:-1, :-1] |= dy[:, 1:]
        blocked = wall | steep
        # the map edge is a wall
        bx0, bx1, by0, by1 = m.bounds
        cx = m.x0 + np.arange(nx) + 0.5
        cy = m.y0 + np.arange(ny) + 0.5
        blocked |= (cx[None, :] < bx0 + 1) | (cx[None, :] > bx1 - 1)
        blocked |= (cy[:, None] < by0 + 1) | (cy[:, None] > by1 - 1)
        r = radius + 0.4 if inflate is None else inflate
        ri = int(math.ceil(r))
        grown = blocked.copy()
        for di in range(-ri, ri + 1):
            for dj in range(-ri, ri + 1):
                if (di or dj) and di * di + dj * dj <= r * r:
                    src = blocked[max(0, -dj):ny - max(0, dj), max(0, -di):nx - max(0, di)]
                    grown[max(0, dj):ny - max(0, -dj), max(0, di):nx - max(0, -di)] |= src
        self.blocked1 = grown
        self._b1 = grown.ravel().tolist()
        self.NX = nx // 2
        self.NY = ny // 2
        b2 = grown[:self.NY * 2, :self.NX * 2].reshape(self.NY, 2, self.NX, 2).any(axis=(1, 3))
        self.walk = ~b2
        self._walk = self.walk.ravel().tolist()
        self._search = None
        self.relabel()

    # ------------------------------------------------------------------ connected areas
    def relabel(self):
        """Label 8-connected walkable areas (diagonals need both sides open, as in A*)."""
        NX, NY = self.NX, self.NY
        walk = self._walk
        lab = [-1] * (NX * NY)
        sizes = []
        for s0 in range(NX * NY):
            if not walk[s0] or lab[s0] >= 0:
                continue
            k = len(sizes)
            lab[s0] = k
            stack = [s0]
            size = 0
            while stack:
                cur = stack.pop()
                size += 1
                cj, ci = divmod(cur, NX)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    ni, nj = ci + di, cj + dj
                    if ni < 0 or nj < 0 or ni >= NX or nj >= NY:
                        continue
                    n = nj * NX + ni
                    if lab[n] >= 0 or not walk[n]:
                        continue
                    if di and dj and not (walk[cj * NX + ni] and walk[nj * NX + ci]):
                        continue
                    lab[n] = k
                    stack.append(n)
            sizes.append(size)
        self._lab = lab
        self.labels = np.array(lab, np.int32).reshape(NY, NX)
        self.comp_sizes = sizes
        self.main_component = int(np.argmax(sizes)) if sizes else -1
        self._dirty = False

    def component(self, x, y):
        if self._dirty:
            self.relabel()
        ci, cj = self.cell_of(x, y)
        return self._lab[cj * self.NX + ci]

    # ------------------------------------------------------------------ cells
    def cell_of(self, x, y):
        ci = int(math.floor((x - self.m.x0) / 2.0))
        cj = int(math.floor((y - self.m.y0) / 2.0))
        return min(max(ci, 0), self.NX - 1), min(max(cj, 0), self.NY - 1)

    def center(self, ci, cj):
        return (self.m.x0 + 2.0 * ci + 1.0, self.m.y0 + 2.0 * cj + 1.0)

    def is_walkable(self, x, y):
        ci, cj = self.cell_of(x, y)
        return self._walk[cj * self.NX + ci]

    def set_walkable(self, ci, cj, value):
        """Override one 2 m cell (tests and dynamic blocking)."""
        self._walk[cj * self.NX + ci] = bool(value)
        self.walk[cj, ci] = bool(value)
        self._dirty = True
        for dj in (0, 1):
            for di in (0, 1):
                j, i = 2 * cj + dj, 2 * ci + di
                self._b1[j * self.m.nx + i] = not value
                self.blocked1[j, i] = not value
        self._search = None

    def nearest_walkable(self, x, y, max_r=600, comp=None):
        """Center of the nearest walkable cell (inside area `comp` when given)."""
        if self._dirty:
            self.relabel()
        ci, cj = self.cell_of(x, y)
        here = self._lab[cj * self.NX + ci]
        if here >= 0 and (comp is None or here == comp):
            return self.center(ci, cj)
        for r in (6, 20, 60, 200, max_r):
            i0, i1 = max(ci - r, 0), min(ci + r + 1, self.NX)
            j0, j1 = max(cj - r, 0), min(cj + r + 1, self.NY)
            win = self.walk[j0:j1, i0:i1] if comp is None else (self.labels[j0:j1, i0:i1] == comp)
            if win.any():
                jj, ii = np.nonzero(win)
                d = (ii + i0 - ci) ** 2 + (jj + j0 - cj) ** 2
                k = int(np.argmin(d))
                return self.center(int(ii[k] + i0), int(jj[k] + j0))
        return (x, y)

    # ------------------------------------------------------------------ line checks
    def walk_clear(self, a, b, step=0.7):
        """True if a mech can walk the straight line a -> b without touching a wall."""
        ax, ay = a[0] - self.m.x0, a[1] - self.m.y0
        bx, by = b[0] - self.m.x0, b[1] - self.m.y0
        dist = math.hypot(bx - ax, by - ay)
        n = max(1, int(dist / step))
        nx, ny = self.m.nx, self.m.ny
        b1 = self._b1
        for k in range(n + 1):
            t = k / n
            i = int(ax + (bx - ax) * t)
            j = int(ay + (by - ay) * t)
            if i < 0 or j < 0 or i >= nx or j >= ny or b1[j * nx + i]:
                return False
        return True

    def smooth(self, path):
        if len(path) <= 2:
            return list(path)
        out = [path[0]]
        anchor = path[0]
        for i in range(1, len(path) - 1):
            if not self.walk_clear(anchor, path[i + 1]):
                out.append(path[i])
                anchor = path[i]
        out.append(path[-1])
        return out

    # ------------------------------------------------------------------ A*
    def plan(self, start, goal, budget_s, weight=1.5):
        """Path from start to goal as a list of (x, y), None if unreachable, or "pending" when the
        budget ran out (call again with the same goal to continue)."""
        t_end = time.perf_counter() + budget_s
        if self._dirty:
            self.relabel()
        sx, sy = start
        if not self.is_walkable(sx, sy):
            sx, sy = self.nearest_walkable(sx, sy)
        comp = self.component(sx, sy)
        gx, gy = goal
        if self.component(gx, gy) != comp:
            gx, gy = self.nearest_walkable(gx, gy, comp=comp)
        goal_c = self.cell_of(gx, gy)
        s = self._search
        if s is None or s["goal"] != goal_c or math.dist(s["start_xy"], start) > 8.0:
            if self.is_walkable(start[0], start[1]) and self.walk_clear(start, (gx, gy)):
                self._search = None
                return [(start[0], start[1]), (gx, gy)]
            sc = self.cell_of(sx, sy)
            NX = self.NX
            s0 = sc[1] * NX + sc[0]
            s = {"goal": goal_c, "start_xy": tuple(start), "goal_xy": (gx, gy), "start_c": sc,
                 "open": [(0.0, 0.0, s0)], "g": {s0: 0.0}, "parent": {s0: -1}, "closed": set()}
            self._search = s
        NX, NY = self.NX, self.NY
        walk = self._walk
        gci, gcj = s["goal"]
        goal_i = gcj * NX + gci
        openh, gs, parent, closed = s["open"], s["g"], s["parent"], s["closed"]
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))
        count = 0
        while openh:
            count += 1
            if (count & 63) == 0 and time.perf_counter() > t_end:
                return "pending"
            f, gcost, cur = heapq.heappop(openh)
            if cur in closed:
                continue
            if cur == goal_i:
                cells = []
                while cur != -1:
                    cells.append(cur)
                    cur = parent[cur]
                cells.reverse()
                pts = [self.center(c % NX, c // NX) for c in cells]
                pts[0] = (start[0], start[1]) if self.is_walkable(start[0], start[1]) else pts[0]
                pts[-1] = s["goal_xy"]
                self._search = None
                path = self.smooth(pts)
                if not self.is_walkable(start[0], start[1]):
                    path.insert(0, (start[0], start[1]))
                return path
            closed.add(cur)
            cj, ci = divmod(cur, NX)
            for di, dj, c in nbrs:
                ni, nj = ci + di, cj + dj
                if ni < 0 or nj < 0 or ni >= NX or nj >= NY:
                    continue
                n = nj * NX + ni
                if not walk[n] or n in closed:
                    continue
                if di and dj and not (walk[cj * NX + ni] and walk[nj * NX + ci]):
                    continue
                ng = gcost + c
                if ng < gs.get(n, 1e18):
                    gs[n] = ng
                    parent[n] = cur
                    ddx, ddy = abs(ni - gci), abs(nj - gcj)
                    h = (ddx + ddy) + (SQRT2 - 2.0) * min(ddx, ddy)
                    heapq.heappush(openh, (ng + weight * h, ng, n))
        self._search = None
        return None
