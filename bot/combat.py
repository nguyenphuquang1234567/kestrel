"""Combat math: leading shots, mortar arcs, aim error, and dodge directions.

Angles follow protocol.md: yaw from +x counter-clockwise, pitch up-positive. Shots leave
`offset` meters (radius + 0.5) out along the aim from the muzzle at the moment of firing.
"""
import math

TAU = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TAU - math.pi


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def yaw_pitch(frm, to):
    dx, dy, dz = to[0] - frm[0], to[1] - frm[1], to[2] - frm[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def angle_err(aim, want):
    """Angle between two [yaw, pitch] directions (radians)."""
    dy = wrap(want[0] - aim[0]) * math.cos((aim[1] + want[1]) * 0.5)
    dp = want[1] - aim[1]
    return math.hypot(dy, dp)


def ang_size(dist, radius):
    return math.atan2(radius, max(dist, 0.1))


def lead_aim(muzzle, target, vel, speed, offset=0.0, pitch_lim=None):
    """Aim for a straight shot at `speed` that meets a target moving at constant `vel`.
    Returns (yaw, pitch, time_to_hit) or None if the shot can't catch it."""
    dx, dy, dz = target[0] - muzzle[0], target[1] - muzzle[1], target[2] - muzzle[2]
    vx, vy, vz = vel
    a = vx * vx + vy * vy + vz * vz - speed * speed
    b = 2.0 * (dx * vx + dy * vy + dz * vz - offset * speed)
    c = dx * dx + dy * dy + dz * dz - offset * offset
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return None
        t = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        r = math.sqrt(disc)
        cands = [t for t in ((-b - r) / (2 * a), (-b + r) / (2 * a)) if t > 0]
        if not cands:
            return None
        t = min(cands)
    if t <= 0:
        return None
    px, py, pz = dx + vx * t, dy + vy * t, dz + vz * t
    yaw = math.atan2(py, px)
    pitch = math.atan2(pz, math.hypot(px, py))
    if pitch_lim:
        pitch = clamp(pitch, pitch_lim[0], pitch_lim[1])
    return yaw, pitch, t


def _ballistic_pitch(R, H, v, g, high=False):
    """Launch pitch to cover horizontal R and rise H at speed v under gravity g (None if out of reach)."""
    if R < 1e-6:
        return math.pi / 2 if H > 0 else -math.pi / 2
    v2 = v * v
    disc = v2 * v2 - g * (g * R * R + 2 * H * v2)
    if disc < 0:
        return None
    r = math.sqrt(disc)
    return math.atan((v2 + r) / (g * R)) if high else math.atan((v2 - r) / (g * R))


def mortar_aim(tube, target, vel, speed, gravity, offset=0.0, high=False, pitch_lim=None):
    """Pitch and yaw to lob a shell onto a target moving at `vel` (a few lead iterations).
    Returns (yaw, pitch, flight_time) or None if out of range or beyond the pitch limits."""
    tx, ty, tz = target
    t = 0.0
    yaw = pitch = None
    for _ in range(4):
        px, py, pz = tx + vel[0] * t, ty + vel[1] * t, tz + vel[2] * t
        yaw = math.atan2(py - tube[1], px - tube[0])
        R = math.hypot(px - tube[0], py - tube[1])
        # the shell starts `offset` out along the aim; correct R and H for the previous pitch
        p0 = pitch if pitch is not None else 0.6
        R2 = R - offset * math.cos(p0)
        H2 = pz - tube[2] - offset * math.sin(p0)
        pitch = _ballistic_pitch(max(R2, 0.1), H2, speed, gravity, high)
        if pitch is None:
            return None
        t = max(R2, 0.1) / max(speed * math.cos(pitch), 1e-6)
    if pitch_lim and not (pitch_lim[0] <= pitch <= pitch_lim[1]):
        return None
    return yaw, pitch, t


def dodge_vector(me, shots, gravity, splash, horizon=0.6, mortar_horizon=3.5):
    """Unit (dx, dy) to step out of incoming fire, or None when nothing threatens us.
    `me`: pos (feet), vel, radius, height. `shots`: observed enemy projectiles."""
    mx, my, mz = me["pos"]
    vx, vy = me["vel"][0], me["vel"][1]
    rad, h = me["radius"], me["height"]
    cz = mz + h * 0.5
    ax = ay = 0.0
    for s in shots:
        px, py, pz = s["pos"]
        ux, uy, uz = s["vel"]
        if s["kind"] == "mortar":
            # landing time against our feet height: pz + uz t - g t^2 / 2 = mz
            a, b, c = -0.5 * gravity, uz, pz - mz
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            t = (-b - math.sqrt(disc)) / (2 * a)
            if t <= 0 or t > mortar_horizon:
                continue
            lx, ly = px + ux * t, py + uy * t
            dx, dy = mx - lx, my - ly
            d = math.hypot(dx, dy)
            if d > splash + rad + 1.5:
                continue
            if d < 1e-3:
                n = math.hypot(ux, uy)
                dx, dy, d = (-uy / n, ux / n, 1.0) if n > 1e-3 else (1.0, 0.0, 1.0)
            w = 1.0 / (t + 0.2)
            ax += dx / d * w
            ay += dy / d * w
            continue
        rx, ry, rz = px - mx, py - my, pz - cz
        wx, wy, wz = ux - vx, uy - vy, uz
        ww = wx * wx + wy * wy + wz * wz
        if ww < 1.0:
            continue
        tc = -(rx * wx + ry * wy + rz * wz) / ww
        if tc <= 0 or tc > horizon:
            continue
        qx, qy, qz = rx + wx * tc, ry + wy * tc, rz + wz * tc
        if abs(qz) > h * 0.5 + 0.5:
            continue
        miss = math.hypot(qx, qy)
        if miss > rad + 1.0:
            continue
        if miss < 0.3:
            n = math.hypot(wx, wy)
            dx, dy = -wy / n, wx / n
            if dx * vx + dy * vy < 0:  # keep going the way we already move
                dx, dy = -dx, -dy
        else:
            dx, dy = -qx / miss, -qy / miss
        w = 1.0 / (tc + 0.1)
        ax += dx * w
        ay += dy * w
    n = math.hypot(ax, ay)
    if n < 1e-9:
        return None
    return ax / n, ay / n
