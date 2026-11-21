"""World model: what the bot knows beyond the current observation.

- Enemy mechs and drones: last known state; the server keeps reporting a mech for memory_s after it
  leaves view, so "listed" means known-now, not necessarily in line of sight.
- Items (crates, cubes, weapon pickups): remembered after they leave view; dropped when they should
  be visible (inside sensor range) but are not listed any more.
- Sounds and damage taken: recent threat bearings.
- Mode: "squad" (practice: house bots never kill each other) or "ffa" (others die without us).
"""
import math

FORGET_S = 25.0        # forget an unseen mech after this long
ITEM_FORGET_S = 90.0   # forget an unseen item after this long
SOUND_KEEP_S = 4.0
FFA_DEATHS = 4         # deaths we did not cause ...
FFA_BEFORE_S = 150.0   # ... before this match time switch us to free-for-all play


class Track:
    __slots__ = ("id", "kind", "pos", "vel", "aim", "hp", "max_hp", "cubes", "weapons", "charging",
                 "seen_t", "first_t", "visible", "hurt_us_t", "dmg_to_us", "dmg_from_us", "prev_vel",
                 "acc")

    def __init__(self, i, kind, t):
        self.id = i
        self.kind = kind
        self.first_t = t
        self.hurt_us_t = -1e9
        self.dmg_to_us = 0.0
        self.dmg_from_us = 0.0
        self.prev_vel = None
        self.acc = (0.0, 0.0)
        self.hp = None
        self.max_hp = None
        self.cubes = 0
        self.weapons = []
        self.charging = False

    def update(self, d, t):
        self.pos = tuple(d["pos"])
        v = tuple(d.get("vel", (0.0, 0.0, 0.0)))
        if self.prev_vel is not None and t > self.seen_t:
            dt = t - self.seen_t
            ax = (v[0] - self.prev_vel[0]) / dt
            ay = (v[1] - self.prev_vel[1]) / dt
            self.acc = (0.6 * self.acc[0] + 0.4 * ax, 0.6 * self.acc[1] + 0.4 * ay)
        self.prev_vel = v
        self.vel = v
        self.aim = tuple(d.get("aim", (0.0, 0.0)))
        self.hp = d.get("hp", self.hp)
        self.max_hp = d.get("max_hp", self.max_hp)
        self.cubes = d.get("cubes", self.cubes) or 0
        self.weapons = d.get("weapons", self.weapons) or []
        self.charging = bool(d.get("charging", False))
        self.seen_t = t
        self.visible = True

    def predict(self, t):
        dt = min(max(t - self.seen_t, 0.0), 3.0)
        return (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt, self.pos[2])


class World:
    def __init__(self, hello):
        self.cfg = hello["config"]
        self.my_id = hello.get("slot", -1)
        self.n_mechs = int(self.cfg.get("match", {}).get("mechs", 16))
        self.near_range = float(self.cfg.get("vision", {}).get("near_range", 45.0))
        self.t = 0.0
        self.enemies = {}
        self.npcs = {}
        self.crates = {}
        self.cubes = {}
        self.pickups = {}
        self.sounds = []
        self.damage = []
        self.hits = []
        self.kills = 0
        self.killed_ids = set()
        self.mode = "squad"
        self.ffa_evidence = 0
        self.alive = self.n_mechs
        self.me = None
        self.last_offer_id = None
        self.events = []

    # ------------------------------------------------------------------ update
    def update(self, obs):
        t = float(obs.get("t", self.t))
        self.t = t
        self.me = obs["self"]
        self.alive = int(obs.get("alive", self.alive))
        mx, my = self.me["pos"][0], self.me["pos"][1]
        near = self.near_range + float(self.me.get("stats", {}).get("near_range_bonus", 0.0))

        for tr in self.enemies.values():
            tr.visible = False
        for tr in self.npcs.values():
            tr.visible = False
        for d in obs.get("mechs", []) or []:
            i = d.get("id")
            if i is None or i == self.me.get("id"):
                continue
            table = self.npcs if d.get("kind") == "npc" else self.enemies
            tr = table.get(i)
            if tr is None:
                tr = table[i] = Track(i, d.get("kind", "mech"), t)
            tr.update(d, t)
        for table in (self.enemies, self.npcs):
            for i in [i for i, tr in table.items() if t - tr.seen_t > FORGET_S]:
                del table[i]

        self._items(self.crates, obs.get("crates"), t, mx, my, near)
        self._items(self.cubes, obs.get("cubes"), t, mx, my, near)
        self._items(self.pickups, obs.get("pickups"), t, mx, my, near)

        self.sounds = [s for s in self.sounds if t - s["t"] < SOUND_KEEP_S]
        for s in obs.get("sounds", []) or []:
            self.sounds.append({"t": t - float(s.get("age", 0.0)), "kind": s.get("kind"),
                                "bearing": float(s.get("bearing", 0.0)), "dist": s.get("dist", "far")})

        self.events = obs.get("events", []) or []
        self.damage = [d for d in self.damage if t - d["t"] < 10.0]
        self.hits = [h for h in self.hits if t - h["t"] < 10.0]
        for e in self.events:
            kind = e.get("type")
            if kind == "damaged":
                rec = dict(e)
                rec["t"] = t
                self.damage.append(rec)
                by = e.get("by")
                if by is not None and by in self.enemies:
                    self.enemies[by].hurt_us_t = t
                    self.enemies[by].dmg_to_us += float(e.get("amount", 0.0))
            elif kind == "hit":
                rec = dict(e)
                rec["t"] = t
                self.hits.append(rec)
                tgt = e.get("target")
                if tgt in self.enemies:
                    self.enemies[tgt].dmg_from_us += float(e.get("amount", 0.0))
            elif kind == "kill":
                self.kills += 1
                self.killed_ids.add(e.get("target"))
                self.enemies.pop(e.get("target"), None)

        for b in obs.get("beams", []) or []:
            self._beam_evidence(b)  # kept for diagnostics; too noisy to switch modes on
        # Practice house bots never kill each other, but some die to drones or the zone. Only an
        # early wave of deaths we did not cause means mechs are hunting each other: a free-for-all.
        others_dead = (self.n_mechs - self.alive) - self.kills
        if others_dead >= FFA_DEATHS and t < FFA_BEFORE_S:
            self.mode = "ffa"

    def _items(self, table, listed, t, mx, my, near):
        seen = set()
        for d in listed or []:
            i = d.get("id")
            if i is None:
                continue
            seen.add(i)
            rec = table.get(i)
            if rec is None:
                rec = table[i] = {"id": i, "first_t": t}
            rec["pos"] = tuple(d["pos"])
            rec["t"] = t
            if "kind" in d:
                rec["kind"] = d["kind"]
            if "hp" in d:
                rec["hp"] = d["hp"]
        for i in list(table):
            if i in seen:
                continue
            rec = table[i]
            px, py = rec["pos"][0], rec["pos"][1]
            if math.hypot(px - mx, py - my) < near * 0.85 or t - rec["t"] > ITEM_FORGET_S:
                del table[i]

    def _beam_evidence(self, b):
        """A beam fired by one mech that ends on another mech (not us) means mechs fight each other."""
        by = b.get("by")
        path = b.get("path") or []
        if len(path) < 2 or by is None:
            return
        end = path[-1]
        for i, tr in self.enemies.items():
            if i == by:
                continue
            dz = end[2] - tr.pos[2]
            if math.hypot(end[0] - tr.pos[0], end[1] - tr.pos[1]) < 3.5 and -1.0 < dz < 10.0:
                self.ffa_evidence += 1
                return

    # ------------------------------------------------------------------ queries
    def visible_enemies(self):
        return [tr for tr in self.enemies.values() if tr.visible]

    def recent_damage(self, window=2.0):
        return sum(d.get("amount", 0.0) for d in self.damage if self.t - d["t"] <= window)

    def last_damage_t(self):
        return max((d["t"] for d in self.damage if d.get("weapon") != "zone"), default=-1e9)
