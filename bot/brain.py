"""Decision layer. Each tick: update the world model, then decide separately
- where the legs go (zone, cover, fight spacing, supply drop, loot, roam), and
- where the torso aims and what fires (enemy, crate or drone, or a scan sweep),
then add loot and upgrade actions."""
import gc
import json
import math
import os
import random
import time

import combat
import geo
import model
import nav
import roof
from combat import ang_size, angle_err, clamp, wrap

WEAPON_VALUE = {"lightning": 3.0, "laser": 2.6, "mortar": 1.0, "gun": 0.0}
# fighting strength per carried weapon, for the outnumbered test (fixed: loot values are tuned separately)
POWER_VALUE = {"lightning": 3.0, "laser": 3.2, "mortar": 1.0, "gun": 0.0}
UPGRADE_ORDER = ["plating", "gun_mag", "armor", "gun_damage", "lightning_efficiency", "laser_damage",
                 "nanites", "gun_rate", "laser_charge", "gyros", "servos", "lightning_range",
                 "lightning_chain", "gun_velocity", "capacitor", "reactor", "jets", "sensors",
                 "laser_pierce", "mortar_reload", "mortar_splash", "mortar_velocity"]
SOUND_THREATS = {"gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost", "jet", "reload"}

DEFAULTS = {
    "kite_range": 70.0,       # keep lightning carriers at least this far when we lack lightning
    "kite": True,
    "multi_retreat_hp": 0.4, # with 2+ enemies engaging, take cover below this HP fraction
    "regen_hold_hp": 0.8,     # after a fight, hide until healed to this fraction
    "regen_hold": True,
    "losing_hp": 0.5,
    "gun_range": 110.0,       # preferred spacing with only a gun
    "laser_range": 120.0,
    "lightning_range": 60.0,
    "mortar_value": 1.0,
    "laser_value": 3.2,
    "lightning_value": 3.0,
    "reload_hidden": True,
    "force_mode": None,       # arena only: pin "squad" or "ffa"
    "upgrade_order": None,    # list of upgrade ids, best first (None: UPGRADE_ORDER)
    "strafe_min": 0.35,       # seconds between strafe direction changes in a fight (uniform min..max)
    "strafe_max": 1.1,
    "strafe_flip": 0.7,       # chance a change actually reverses direction
    "empty_slot_mult": 1.0,   # weapon pickups count this much more while slot 1 is empty
    "unarmed_avoid": False,   # early on with only the gun, don't start distant fights (still shoot back)
    "fight_retreat": True,   # with 2+ enemies engaged, back away from them while shooting
    "retreat_boost_d": 90.0,  # boost while retreating when the nearest is closer than this
    "retreat_seen_s": 3.0,    # enemies seen this recently count toward "2+ engaged"
    "retreat_min_enemies": 2, # fighting retreat kicks in at this many engaged enemies
    "explore": True,          # idle inside the zone: go look for loot (else hold position)
    "loot_enemy_r": 50.0,     # skip loot this close to an enemy seen in the last 10 s
    "engage_range": 110.0,    # visible enemies closer than this get fought
    "hold_fire_beyond": False, # don't shoot enemies beyond engage_range unless they attacked us
    "laser_evade": False,     # an enemy charging a laser at us: duck behind the nearest cover (<= 13 m)
    "zone_go": 0.6,           # head into the next circle once the zone urgency reaches this
    "explore_open_w": 80.0,    # exploring: penalty per unit of openness (prefer short sightlines)
    "loot_open_w": 0.0,       # loot score x (1 - w * openness): skip loot lying in the open
    "zone_open_w": 0.0,       # choosing where to stand in the next circle: penalty per unit of openness
    "fight_open_w": 0.0,      # fighting: pull toward nearby spots with shorter sightlines
    "cover_open_w": 0.0,      # choosing cover: penalty per unit of openness
    "rng_salt": 3,            # offsets the random stream (same play style, different coin flips)
    "late_salt": 0,           # from late_salt_t on, switch to another random stream (re-rolls only the rest of the match)
    "late_salt_t": 120.0,
    "late_salt2": 0,          # a second switch of the random stream at late_salt2_t (stage-wise re-rolls)
    "late_salt2_t": 180.0,
    "early_safe_s": 0.0,       # before this match time: no fights or shots unless attacked or within early_safe_d
    "early_safe_d": 50.0,
    "fight_boost": False,     # strafe at boost speed in fights while energy lasts (not when carrying lightning)
    "fight_boost_on": 60.0,   # start boosting above this energy, stop below fight_boost_off
    "fight_boost_off": 25.0,
    "lead_far": 1.0,          # gun lead multiplier at 150 m+ (1 at 50 m, linear between): hedge against strafe reversals
    "farm_heal_hold": False,  # no crate/drone shooting while hurt after a recent fight (keeps regen running)
    "retreat_open_w": 0.0,    # fighting retreat: pull toward nearby spots with shorter sightlines
    "precharge": False,       # charge the laser for a tracked enemy about to come out from behind cover
    "precharge_d": 90.0,
    "roof_snipe": False,      # on a roof: walk to the edge facing the nearest known enemy and shoot down
    "snipe_hp": 0.5,          # below this HP fraction, back off to the middle of the roof to heal
    "quit_after_kills": 0,    # diagnostic probes only: after this many kills, stand still and hold fire (0 = never)
    "quit_after_t": 0.0,      # diagnostic probes only: from this match time on, stand still and hold fire (0 = never)
    "kill_loot": True,       # right after a kill, go pick up the cubes it dropped (if not badly hurt)
    "kill_loot_hp": 0.35,
    "heal_hide": False,       # hide to heal from enemies seen in the last 20 s, not only the last 6 s
    "hide_hp": 0.5,           # with heal_hide: below this HP after a recent fight, hide in a sheltered spot even if no enemy is known
    "travel_boost_e": 40.0,   # boost while travelling only above this energy (keeps a reserve for escapes)
    "hunt_first": False,      # when healthy, go after the last seen enemy before looting
    "cover_commit_s": 0.0,    # once we break off to cover (losing/outnumbered), stay in cover this long
    "weapon_rush": False,     # without a good second weapon, go get a seen one first (still shooting), even mid-fight
    "weapon_rush_d": 150.0,
    "weapon_rush_gain": 1.5,  # minimum value gain of the pickup (empty slot: its value; swap: the difference)
    "weapon_rush_close": 35.0, # unless a visible enemy is this close
    "check_six": False,       # hit from behind by someone we cannot see: turn to look for a moment
    "check_six_s": 0.7,
    "cover_stick": False,     # keep the chosen cover spot while it still hides us (no re-pick every second)
    "peek_reload": False,     # one-on-one: while the gun reloads and the laser cools down, duck behind nearby cover
    "peek_r": 14.0,
    "peek_retreat": False,    # also duck out of sight of all engaged enemies during a fighting retreat
    "laser_engage_range": 0.0, # when holding a laser, fight approaching enemies out to this range (0 = engage_range)
    "roof": False,            # after roof_after seconds, jet onto a roof inside the zone and hold it
    "roof_after": 30.0,
    "roof_after_kills": 0,    # roof mode only once we have this many kills (the late all-out hunt)
    "farm_npcs": True,        # shoot drones for their cubes
    "farm_radius": 70.0,      # shoot crates and drones in sight within this range
    "cube_value": 2.6,        # loot priorities (value / (distance + 25))
    "crate_value": 1.6,
    "npc_value": 2.2,
    "lob": True,             # mortar ready and no line of fire: lob shells at known enemies behind cover
    "mortar_mode": "gun_down",  # "gun_down": lob only while the gun is reloading or empty (gun first);
                              # "commit": hold the lob up to 1 s whenever the mortar is ready
    "target_hp_weight": 60.0, # target choice: bonus for a weakened enemy (x missing HP fraction)
    "finish_hp": 0.0,         # also engage enemies below this HP fraction out to finish_range
    "finish_range": 160.0,
    "engage_filter": "approach",   # "approach": within engage_range, only fight enemies coming at us,
                              # that hurt us recently, that are very close, or that are weak
    "drops": False,            # go capture supply drops when no enemies are near them
    "hunt": False,            # idle and healthy: go after the most recently seen enemy
    "hunt_hp": 0.6,           # ... only above this HP fraction
    "pair": False,            # with laser or lightning held, swap the gun for the other one
    "pair_range": 30.0,       # spacing when fighting with laser + lightning
    "duel": False,            # with 2+ enemies around, fight from a spot only the target can see (arena: worse)
}


def load_params():
    p = dict(DEFAULTS)
    raw = os.environ.get("KESTREL_PARAMS")
    if raw:
        try:
            p.update(json.loads(raw))
        except ValueError:
            pass
    return p


def dist2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Enemy:
    __slots__ = ("tr", "d", "los", "center", "hp_frac")

    def __init__(self, tr, d, los, center):
        self.tr = tr
        self.d = d
        self.los = los
        self.center = center
        self.hp_frac = (tr.hp / tr.max_hp) if tr.hp is not None and tr.max_hp else 1.0


class Brain:
    name = "kestrel"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        mc = cfg["mech"]
        self.mc = mc
        self.map = geo.Map(hello["map"]["dir"])
        self.nav = nav.NavGrid(self.map, step_up=mc.get("step_up", 1.2), height=mc.get("height", 9.0),
                               radius=mc.get("radius", 2.5), max_slope_deg=mc.get("max_slope_deg", 50.0))
        self.world = model.World(hello)
        self.roofs = roof.Roofs(self.map, self.nav)
        self.open = geo.Openness(self.map)
        self.roof_target = None
        self.roof_bad = set()
        self.climb_t = None
        self.no_steer = False
        self.P = load_params()
        WEAPON_VALUE.update({"mortar": self.P["mortar_value"], "laser": self.P["laser_value"],
                             "lightning": self.P["lightning_value"]})
        self.rng = random.Random(int(hello.get("seed", 1)) + 7919 * int(self.P["rng_salt"]))
        self.wcfg = cfg["weapons"]
        self.pickup_r = float(cfg["loot"]["pickup_radius"])
        self.gravity = float(mc.get("gravity", 22.0))
        self.pitch_lim = (float(mc.get("min_pitch", -0.7)), float(mc.get("max_pitch", 1.3)))
        # movement state
        self.path = None
        self.path_goal = None
        self.path_t = -1e9
        self.trail = []
        self.stuck = 0.0
        self.unstick_until = -1e9
        self.unstick_dir = (1.0, 0.0)
        self.jet_until = -1e9
        self.strafe = 1.0
        self.strafe_until = -1e9
        self.kite_boost = False
        self.fight_boosting = False
        self.cover = None
        self.cover_t = -1e9
        self.explore_pt = None
        self.explore_t = -1e9
        self.visited = []
        self.mortar_until = -1e9
        self.mortar_block_until = -1e9
        self.duel = None
        self.duel_t = -1e9
        self.duel_tid = None
        self.goal = ("roam", None)
        self.ignore_items = set()
        self.sweep_phase = 0
        self.sweep_t = 0.0
        self.last_move = (0.0, 0.0)
        self.zone_hist = {}
        self.kill_spots = []
        self.cover_until = -1e9
        self.t = 0.0
        gc.collect()
        gc.freeze()

    # ================================================================== main
    def act(self, obs):
        t0 = time.perf_counter()
        self.deadline = t0 + 0.025
        w = self.world
        for ev in obs.get("events", []) or []:
            if ev.get("type") == "kill" and ev.get("target") in w.enemies:
                tp = w.enemies[ev["target"]].pos
                self.kill_spots.append((float(obs.get("t", self.t)), (tp[0], tp[1])))
        w.update(obs)
        if self.P.get("force_mode"):
            w.mode = self.P["force_mode"]
        me = obs["self"]
        self.me = me
        self.t = float(obs.get("t", self.t))
        st = me.get("stats", {}) or {}
        self.pos = tuple(me["pos"])
        self.vel = tuple(me.get("vel", (0.0, 0.0, 0.0)))
        self.radius = float(st.get("radius", self.mc.get("radius", 2.5)))
        self.height = float(st.get("height", self.mc.get("height", 9.0)))
        self.eye = (self.pos[0], self.pos[1], self.pos[2] + float(st.get("eye_height", 8.0)))
        self.muzzle = (self.pos[0], self.pos[1], self.pos[2] + float(st.get("muzzle_height", 7.0)))
        self.walk_speed = float(st.get("walk_speed", self.mc.get("walk_speed", 11.7)))
        self.hp = float(me.get("hp", 1.0))
        self.max_hp = float(me.get("max_hp", 1.0)) or 1.0
        self.hp_frac = self.hp / self.max_hp
        self.energy = float(me.get("energy", 0.0))
        self.weapons = me.get("weapons") or [None, None]
        self.grounded = bool(me.get("grounded", True))
        self.zone = obs.get("zone") or {}
        self.on_roof = self.pos[2] > self.map.ground_z(self.pos[0], self.pos[1]) + 3.0

        if self.P["late_salt"] and not getattr(self, "_late_salted", False) and self.t >= self.P["late_salt_t"]:
            self._late_salted = True
            self.rng = random.Random(1000003 + 7919 * int(self.P["late_salt"]) + int(self.P["rng_salt"]))
        if self.P["late_salt2"] and not getattr(self, "_late_salted2", False) and self.t >= self.P["late_salt2_t"]:
            self._late_salted2 = True
            self.rng = random.Random(2000003 + 7919 * int(self.P["late_salt2"]) + 31 * int(self.P["late_salt"]))
        if ((self.P["quit_after_kills"] and w.kills >= self.P["quit_after_kills"])
                or (self.P["quit_after_t"] and self.t >= self.P["quit_after_t"])):
            return {"move": [0.0, 0.0], "fire": [False, False]}
        act = {}
        offer = obs.get("offer")
        if offer and offer.get("choices"):
            act["pick"] = self.pick_upgrade(offer)
        self.handle_take_failures()
        self.track_progress()

        enemies, target, goal = [], None, ("roam", None)
        try:
            enemies = self.assess_enemies()
            target = self.choose_target(enemies)
            goal = self.choose_goal(obs, enemies, target)
        except Exception:
            self.fault("decide")
        self.goal = goal
        try:
            move, boost, jump = self.move_for(goal, target, enemies, obs)
        except Exception:
            self.fault("move")
            move, boost, jump = self.fallback_move(), False, False
        try:
            aim, fire, reload = self.aim_and_fire(target, enemies, move)
        except Exception:
            self.fault("aim")
            aim, fire, reload = self.fallback_aim(target, move), [False, False], False
        act["move"] = [round(move[0], 4), round(move[1], 4)]
        act["boost"] = bool(boost)
        act["jump"] = bool(jump)
        act["aim"] = [round(aim[0], 5), round(aim[1], 5)]
        act["fire"] = [bool(fire[0]), bool(fire[1])]
        if reload:
            act["reload"] = True
        try:
            take = self.loot_action()
            if take:
                act.update(take)
        except Exception:
            self.fault("loot")
        self.last_move = move
        return act

    # ================================================================== fault containment
    def fault(self, where):
        """A part of the brain failed this tick: log it (stderr) and let the others carry on."""
        self.faults = getattr(self, "faults", 0) + 1
        if self.faults <= 5 or self.faults % 200 == 0:
            import sys
            import traceback
            print(f"fault in {where} at t={self.t:.1f} ({self.faults}):\n{traceback.format_exc()}", file=sys.stderr, flush=True)

    def fallback_move(self):
        """Head for the zone center in a straight line, sliding off walls."""
        z = self.zone or {}
        c = z.get("next_center") or z.get("center")
        if not c:
            return self.last_move
        d = self.unit(c[0] - self.pos[0], c[1] - self.pos[1])
        try:
            return self.clear_dir(d)
        except Exception:
            return d

    def fallback_aim(self, target, move):
        if target is not None:
            c = target.center
            return (math.atan2(c[1] - self.muzzle[1], c[0] - self.muzzle[0]), 0.0)
        if math.hypot(*move) > 0.1:
            return (math.atan2(move[1], move[0]), 0.0)
        return tuple(self.me.get("aim", (0.0, 0.0)))

    # ================================================================== perception
    def assess_enemies(self):
        w = self.world
        known = [tr for tr in w.enemies.values() if self.t - tr.seen_t < 6.0]
        if not known:
            return []
        centers = []
        for tr in known:
            p = tr.predict(self.t)
            scale = 1.0 + 0.05 * (tr.cubes or 0)
            centers.append((p[0], p[1], p[2] + 4.5 * scale))
        los = self.map.los_many(self.muzzle, centers, step=1.0)
        out = []
        for tr, c, ok in zip(known, centers, los):
            out.append(Enemy(tr, math.dist(self.muzzle, c), bool(ok) and tr.visible, c))
        return out

    def remembered_threats(self, max_age=20.0):
        """Enemies seen in the last max_age seconds, at their predicted positions (for hiding)."""
        out = []
        for tr in self.world.enemies.values():
            if self.t - tr.seen_t < max_age:
                p = tr.predict(self.t)
                c = (p[0], p[1], p[2] + 4.5)
                out.append(Enemy(tr, math.dist(self.muzzle, c), False, c))
        return out

    def worth_fighting(self, e):
        if self.t < self.P["early_safe_s"]:
            return e.d < self.P["early_safe_d"] or self.t - e.tr.hurt_us_t < 6.0
        if self.P["engage_filter"] != "approach":
            return True
        if e.d < 50.0 or self.t - e.tr.hurt_us_t < 6.0 or e.hp_frac < 0.35:
            return True
        dx, dy = self.pos[0] - e.tr.pos[0], self.pos[1] - e.tr.pos[1]
        n = math.hypot(dx, dy) or 1.0
        closing = (e.tr.vel[0] * dx + e.tr.vel[1] * dy) / n
        return closing > 2.0

    def my_power(self):
        return 100.0 + sum(POWER_VALUE.get(wp["kind"], 0) * 40.0 for wp in self.weapons if wp)

    def threat_level(self, e):
        tr = e.tr
        dps = 100.0
        for k in tr.weapons:
            dps += {"lightning": 140.0 if e.d < 50 else 0.0, "laser": 90.0, "mortar": 50.0}.get(k, 0.0)
        recent = 1.6 if self.t - tr.hurt_us_t < 3.0 else 1.0
        return dps * recent * (0.5 + e.hp_frac)

    def choose_target(self, enemies):
        best, best_s = None, -1e9
        for e in enemies:
            if not e.tr.visible:
                continue
            s = -e.d
            if e.los:
                s += 120.0
            s += self.P["target_hp_weight"] * (1.0 - e.hp_frac)
            if self.t - e.tr.hurt_us_t < 2.0:
                s += 60.0
            if best is not None and best.tr.id == e.tr.id:
                s += 25.0
            if self.goal[0] == "fight" and self.goal[1] == e.tr.id:
                s += 25.0
            if s > best_s:
                best, best_s = e, s
        return best

    # ================================================================== goals
    def choose_goal(self, obs, enemies, target):
        w = self.world
        zone_urg, zone_pt = self.zone_need()
        visible = [e for e in enemies if e.tr.visible]
        eng = self.P["engage_range"]
        if self.P["laser_engage_range"] and "laser" in self.slot_kinds():
            eng = self.P["laser_engage_range"]
        close = [e for e in visible if (e.d < eng and self.worth_fighting(e))
                 or (e.hp_frac < self.P["finish_hp"] and e.d < self.P["finish_range"])]
        recent_hit = self.t - w.last_damage_t() < 3.0

        if zone_urg >= 1.5:
            return ("zone", zone_pt)
        if self.P["weapon_rush"] and zone_urg < 1.0:
            wr = self.weapon_rush_goal(visible)
            if wr is not None:
                return ("loot", wr)
        if (self.P["roof"] and self.t >= self.P["roof_after"] and zone_urg < 1.0
                and w.kills >= self.P["roof_after_kills"]):
            rg = self.roof_goal(visible_close=[e for e in enemies if e.tr.visible and e.d < 50.0])
            if rg is not None:
                return rg

        if close:
            power = sum(self.threat_level(e) for e in close)
            mine = self.my_power()
            engaging = [e for e in close if e.los or self.t - e.tr.hurt_us_t < 3.0]
            outnumbered = len(engaging) >= 2 and power > mine * 1.3
            losing = target is not None and self.hp_frac < self.P["losing_hp"] and target.hp_frac > self.hp_frac + 0.1
            if w.mode == "ffa":
                attacked = recent_hit or any(self.t - e.tr.hurt_us_t < 4.0 for e in close)
                weak = target is not None and target.hp_frac < 0.4 and target.d < 110 and self.hp_frac > 0.6
                if (losing or outnumbered and self.hp_frac < 0.7) or not (attacked or weak or (target and target.d < 40)):
                    return ("cover", None)
            elif losing or (outnumbered and self.hp_frac < self.P["multi_retreat_hp"]):
                self.cover_until = self.t + self.P["cover_commit_s"]
                return ("cover", None)
            if self.t < self.cover_until:
                return ("cover", None)  # committed: don't pop back out the moment line of sight breaks
            unarmed = len(self.weapons) < 2 or not self.weapons[1]
            if (target is not None and self.P["unarmed_avoid"] and unarmed and self.t < 150.0
                    and not recent_hit and target.d > 50.0 and self.t - target.tr.hurt_us_t > 5.0):
                target = None  # find a weapon first; the torso still shoots if they come at us
            if target is not None:
                return ("fight", target.tr.id)

        if zone_urg >= self.P["zone_go"]:
            return ("zone", zone_pt)
        if self.P["kill_loot"] and self.hp_frac >= self.P["kill_loot_hp"]:
            kc = self.kill_cubes_goal()
            if kc is not None:
                return ("loot", kc)
        if self.hp_frac < 0.45 and recent_hit:
            return ("cover", None)
        if (self.P["regen_hold"] and self.hp_frac < self.P["regen_hold_hp"]
                and self.t - w.last_damage_t() < 30.0 and any(self.t - e.tr.seen_t < 20.0 for e in enemies)):
            return ("cover", None)
        if self.P["heal_hide"] and self.P["regen_hold"] and self.t - w.last_damage_t() < 30.0:
            if self.hp_frac < self.P["regen_hold_hp"] and self.remembered_threats():
                return ("cover", None)
            if self.hp_frac < self.P["hide_hp"]:
                return ("cover", None)

        drop = self.drop_goal(obs, enemies)
        if drop is not None:
            return ("drop", drop)
        if self.P["hunt_first"] and self.hp_frac >= self.P["hunt_hp"] and zone_urg < 0.4:
            prey = self.hunt_point(enemies)
            if prey is not None:
                return ("hunt", prey)
        item = self.loot_goal(enemies)
        if item is not None:
            return ("loot", item)
        if self.P["hunt"] and self.hp_frac >= self.P["hunt_hp"] and zone_urg < 0.4:
            prey = self.hunt_point(enemies)
            if prey is not None:
                return ("hunt", prey)
        if self.P["explore"] and (zone_pt is None or dist2(self.pos, zone_pt) < 15.0):
            return ("explore", self.explore_point())
        return ("roam", zone_pt)

    def roof_goal(self, visible_close):
        """("roof", roof) while holding or heading for a roof; None to play normally."""
        r = self.roof_target
        if r is not None and self.roofs.on(r, self.pos):
            return ("roof", r)
        nc = self.zone.get("next_center")
        nr = float(self.zone.get("next_radius", 1e4))
        if r is None or r["idx"] in self.roof_bad or (nc and math.hypot(r["stand"][0] - nc[0], r["stand"][1] - nc[1]) > nr - 10.0):
            r = self.roof_target = self.roofs.pick(self.pos, nc, nr, self.roof_bad)
            self.climb_t = None
        if r is None or visible_close:
            return None
        return ("roof", r)

    def snipe_post(self, r):
        """Where to stand on our roof: the edge facing the nearest known enemy, or the middle to heal."""
        sx, sy = r["stand"]
        if self.hp_frac < self.P["snipe_hp"]:
            return (sx, sy)
        known = [tr for tr in self.world.enemies.values() if self.t - tr.seen_t < 8.0]
        snd = [s_ for s_ in self.world.sounds if s_["kind"] in SOUND_THREATS and self.t - s_["t"] < 3.0]
        if known:
            tr = min(known, key=lambda k: dist2(self.pos, k.pos))
            u = self.unit(tr.pos[0] - sx, tr.pos[1] - sy)
        elif snd:
            b_ = snd[-1]["bearing"]
            u = (math.cos(b_), math.sin(b_))
        else:  # look around from a different edge every few seconds
            a_ = (int(self.t // 4.0) % 6) * math.pi / 3.0
            u = (math.cos(a_), math.sin(a_))
        if math.hypot(*u) < 0.5:
            return (sx, sy)
        p = self.roofs.edge_toward(r, (sx, sy), u)
        if p is None or not self.roofs.contains(r, p[0], p[1]):
            return (sx, sy)
        return p

    def roof_move(self, r):
        """Walk to the roof's climb point, jet up the wall, then hold the stand point."""
        x, y = self.pos[0], self.pos[1]
        sx, sy = r["stand"]
        if self.roofs.on(r, self.pos):
            self.climb_t = None
            self.no_steer = True
            post = self.snipe_post(r) if self.P["roof_snipe"] else (sx, sy)
            d = dist2(self.pos, post)
            return (self.unit(post[0] - x, post[1] - y) if d > 2.5 else (0.0, 0.0)), False, False
        cx, cy = r["climb"]
        climbing = self.climb_t is not None or dist2(self.pos, (cx, cy)) < 6.0
        if not climbing:
            return self.follow((cx, cy)), self.energy > 60, False
        if self.climb_t is None:
            self.climb_t = self.t
        if self.t - self.climb_t > 9.0:  # could not get up: try another roof
            self.roof_bad.add(r["idx"])
            self.roof_target = None
            self.climb_t = None
            return self.follow((cx, cy)), False, False
        self.no_steer = True
        jump = self.grounded and self.energy > 40
        return self.unit(sx - x, sy - y), not self.grounded and self.energy > 2, jump

    def weapon_rush_goal(self, visible):
        """("pickup", id, pos) for the best weapon worth rushing to, or None."""
        if any(e.d < self.P["weapon_rush_close"] for e in visible):
            return None
        best = None
        for i, rec in self.world.pickups.items():
            if i in self.ignore_items:
                continue
            gain = self.want_weapon(rec.get("kind"))
            if gain < self.P["weapon_rush_gain"]:
                continue
            p = rec["pos"]
            if p[2] > self.map.ground_z(p[0], p[1]) + 2.5:
                continue  # on a roof
            d = dist2(self.pos, p)
            if d > self.P["weapon_rush_d"]:
                continue
            s_ = gain / (d + 25.0)
            if best is None or s_ > best[0]:
                best = (s_, ("pickup", i, p))
        return best[1] if best else None

    def kill_cubes_goal(self):
        """("cube", id, pos) for cubes dropped where we recently killed someone, or the spot itself
        while we have not seen them yet; None when there is nothing left to collect."""
        self.kill_spots = [(ts, p) for ts, p in self.kill_spots if self.t - ts < 20.0]
        if not self.kill_spots:
            return None
        best = None
        for i, rec in self.world.cubes.items():
            cp = rec["pos"]
            if any(math.hypot(cp[0] - p[0], cp[1] - p[1]) < 20.0 for _, p in self.kill_spots):
                d = dist2(self.pos, cp)
                if d < 150.0 and (best is None or d < best[0]):
                    best = (d, ("cube", i, cp))
        if best is not None:
            return best[1]
        keep = []
        for ts, p in self.kill_spots:
            d = dist2(self.pos, p)
            if d > 15.0 and d < 150.0:
                keep.append((ts, p))
        self.kill_spots = keep
        if keep:
            ts, p = min(keep, key=lambda k: dist2(self.pos, k[1]))
            return ("cube", None, (p[0], p[1], self.map.ground_z(p[0], p[1])))
        return None

    def zone_circle(self):
        z = self.zone
        c, r = z.get("center", (0.0, 0.0)), float(z.get("radius", 1e4))
        return (c[0], c[1]), r

    def zone_need(self):
        """(urgency, point): urgency >= 1 means go now, >= 1.5 means everything else waits."""
        z = self.zone
        if not z:
            return 0.0, None
        (cx, cy), r = self.zone_circle()
        nc = z.get("next_center", (cx, cy))
        nr = float(z.get("next_radius", r))
        ss, se = float(z.get("shrink_start", 0.0)), float(z.get("shrink_end", 0.0))
        x, y = self.pos[0], self.pos[1]
        dn = math.hypot(x - nc[0], y - nc[1])
        margin = min(25.0, max(6.0, nr * 0.2))
        inner = max(nr - margin, 0.0)
        if nr <= 30.0:
            pt = (nc[0], nc[1])
        elif dn > 1e-6:
            k = min(dn, max(inner * 0.75, nr * 0.4)) / dn
            pt = (nc[0] + (x - nc[0]) * k, nc[1] + (y - nc[1]) * k)
        else:
            pt = (nc[0], nc[1])
        if self.P["zone_open_w"] and nr > 30.0:
            pt = self.sheltered_zone_point(pt, nc, inner)
        if not self.me.get("in_zone", True):
            return 2.0, pt
        if dn <= inner:
            return 0.0, pt
        # time until the shrinking edge reaches us, if we stood still
        deadline = se
        if se > ss:
            for k in range(21):
                tau = max(self.t, ss) + (se - max(self.t, ss)) * k / 20.0
                f = clamp((tau - ss) / (se - ss), 0.0, 1.0)
                ccx, ccy = cx + (nc[0] - cx) * f, cy + (nc[1] - cy) * f
                rr = r + (nr - r) * f
                if math.hypot(x - ccx, y - ccy) > rr - 3.0:
                    deadline = tau
                    break
        travel = (dn - inner) * 1.35 / max(self.walk_speed, 1.0) + 6.0
        left = deadline - self.t
        if left <= 0:
            return 1.6, pt
        return travel / left, pt

    def hunt_point(self, enemies):
        """Where the most recently seen enemy probably is (seen in the last 45 s, inside the zone)."""
        best = None
        nc = self.zone.get("next_center")
        nr = float(self.zone.get("next_radius", 1e4))
        for tr in self.world.enemies.values():
            age = self.t - tr.seen_t
            if age > 45.0:
                continue
            p = tr.predict(self.t)
            if nc and math.hypot(p[0] - nc[0], p[1] - nc[1]) > nr * 1.1:
                continue
            if best is None or age < best[0]:
                best = (age, (p[0], p[1]))
        return best[1] if best else None

    def explore_point(self):
        """Somewhere new inside the next safe circle to look for loot (kept until reached or stale)."""
        x, y = self.pos[0], self.pos[1]
        if (self.explore_pt is not None and dist2(self.pos, self.explore_pt) > 12.0
                and self.t - self.explore_t < 30.0 and self.stuck < 0.5):
            return self.explore_pt
        self.visited.append((x, y))
        if self.explore_pt is not None:
            self.visited.append(self.explore_pt)
        self.visited = self.visited[-40:]
        nc = self.zone.get("next_center") or (x, y)
        nr = float(self.zone.get("next_radius", 300.0))
        here = self.nav.component(x, y)
        best, best_s = None, -1e9
        for _ in range(20):
            a = self.rng.uniform(0.0, 2.0 * math.pi)
            r = self.rng.uniform(60.0, 170.0)
            p = (x + r * math.cos(a), y + r * math.sin(a))
            if not self.nav.is_walkable(*p) or (here >= 0 and self.nav.component(*p) != here):
                continue
            if math.hypot(p[0] - nc[0], p[1] - nc[1]) > max(nr * 0.85, 20.0):
                continue
            novelty = min((dist2(p, v) for v in self.visited), default=150.0)
            sc = min(novelty, 150.0) - 0.15 * r - self.P["explore_open_w"] * self.open.at(*p)
            if sc > best_s:
                best, best_s = p, sc
        if best is None:
            best = self.nav.nearest_walkable(nc[0], nc[1])
        self.explore_pt = best
        self.explore_t = self.t
        return best

    def sheltered_zone_point(self, pt, nc, inner):
        """A spot near `pt`, inside the inner circle, with short sightlines (kept for 15 s per circle)."""
        key = (round(nc[0]), round(nc[1]))
        c = getattr(self, "_zone_shelter", None)
        if c and c[0] == key and self.t - c[1] < 15.0:
            return c[2]
        best, best_s = pt, self.P["zone_open_w"] * self.open.at(*pt)
        for _ in range(24):
            a = self.rng.uniform(0.0, 2.0 * math.pi)
            r = self.rng.uniform(0.0, 60.0)
            q = (pt[0] + r * math.cos(a), pt[1] + r * math.sin(a))
            if math.hypot(q[0] - nc[0], q[1] - nc[1]) > inner or not self.nav.is_walkable(*q):
                continue
            sc = 0.3 * r + self.P["zone_open_w"] * self.open.at(*q)
            if sc < best_s:
                best, best_s = q, sc
        self._zone_shelter = (key, self.t, best)
        return best

    def drop_goal(self, obs, enemies):
        if not self.P["drops"]:
            return None
        best = None
        for d in obs.get("drops") or []:
            p = d.get("pos")
            if not p:
                continue
            dist = dist2(self.pos, p)
            if dist > 220.0:
                continue
            eta = dist / max(self.walk_speed, 1.0)
            if d.get("lands_in", 0.0) > eta + 8.0:
                continue
            holder = d.get("holder")
            if holder is not None and holder != self.me.get("id"):
                continue
            if any(dist2(e.tr.pos, p) < 60.0 for e in enemies if e.tr.visible):
                continue
            if best is None or dist < best[1]:
                best = ((p[0], p[1]), dist)
        return best[0] if best else None

    def slot_kinds(self):
        return [wp["kind"] if wp else None for wp in self.weapons]

    def take_plan(self, kind):
        """(value, slot to drop or None) for taking a pickup of this kind; value 0 means leave it."""
        kinds = list(self.slot_kinds()) + [None] * (2 - len(self.slot_kinds()))
        if kind in kinds:
            return 0.0, None
        v = WEAPON_VALUE.get(kind, 0.0)
        if None in kinds:
            return v * self.P["empty_slot_mult"], None
        if self.P["pair"] and "gun" in kinds and kind in ("laser", "lightning"):
            other = "lightning" if kind == "laser" else "laser"
            if other in kinds:
                return v, kinds.index("gun")
        if "gun" in kinds:
            rep = 1 - kinds.index("gun")
        else:
            rep = 0 if WEAPON_VALUE.get(kinds[0], 0.0) <= WEAPON_VALUE.get(kinds[1], 0.0) else 1
        gain = v - WEAPON_VALUE.get(kinds[rep], 0.0)
        return (gain, rep) if gain > 0 else (0.0, None)

    def want_weapon(self, kind):
        return self.take_plan(kind)[0]

    def loot_goal(self, enemies):
        w = self.world
        best, best_s = None, 0.0
        x, y = self.pos[0], self.pos[1]
        nc = self.zone.get("next_center")
        nr = float(self.zone.get("next_radius", 1e4))

        def score(pos, value):
            if value <= 0:
                return 0.0
            g = self.map.ground_z(pos[0], pos[1])
            if pos[2] > g + 2.5:  # on a roof: skip for now
                return 0.0
            d = dist2(self.pos, pos)
            if d > 260.0:
                return 0.0
            s = value / (d + 25.0)
            if nc and math.hypot(pos[0] - nc[0], pos[1] - nc[1]) > nr:
                s *= 0.5
            for e in enemies:
                if self.t - e.tr.seen_t < 10.0 and dist2(e.tr.predict(self.t), pos) < self.P["loot_enemy_r"]:
                    s *= 0.4
            if self.P["loot_open_w"]:
                s *= max(0.05, 1.0 - self.P["loot_open_w"] * self.open.at(pos[0], pos[1]))
            return s

        for i, rec in w.pickups.items():
            if i in self.ignore_items:
                continue
            s = score(rec["pos"], self.want_weapon(rec.get("kind")) * 1.4)
            if s > best_s:
                best, best_s = ("pickup", i, rec["pos"]), s
        for i, rec in w.cubes.items():
            s = score(rec["pos"], self.P["cube_value"])
            if s > best_s:
                best, best_s = ("cube", i, rec["pos"]), s
        for i, rec in w.crates.items():
            s = score(rec["pos"], self.P["crate_value"])
            if s > best_s:
                best, best_s = ("crate", i, rec["pos"]), s
        if self.P["farm_npcs"] and self.hp_frac > 0.7 and self.t < 300.0:
            for i, tr in w.npcs.items():
                if self.t - tr.seen_t > 5.0:
                    continue
                s = score(tr.pos, self.P["npc_value"] * (0.5 + 0.5 * self.hp_frac))
                if s > best_s:
                    best, best_s = ("npc", i, tr.pos), s
        return best

    # ================================================================== movement
    def track_progress(self):
        self.trail.append((self.t, self.pos))
        self.trail = [p for p in self.trail if self.t - p[0] <= 1.2]
        if len(self.trail) >= 2:
            t_old, p_old = self.trail[0]
            moved = dist2(p_old, self.pos)
            commanded = math.hypot(*self.last_move) > 0.5
            if commanded and self.t - t_old > 0.9 and moved < 2.0 and self.t > self.jet_until:
                self.stuck += 0.1
            else:
                self.stuck = max(0.0, self.stuck - 0.2)

    def goal_point(self, goal, target):
        kind, arg = goal
        if kind == "zone" or kind == "roam" or kind == "explore" or kind == "hunt":
            return arg
        if kind == "drop":
            return arg
        if kind == "loot":
            what, _, pos = arg
            if what == "crate" or what == "npc":
                d = dist2(self.pos, pos)
                if d < 35.0:
                    return None  # close enough to shoot it from here
            return (pos[0], pos[1])
        return None

    def follow(self, gp):
        """Unit direction toward gp along a planned path (replanned when needed)."""
        x, y = self.pos[0], self.pos[1]
        need = (self.path is None or self.path_goal is None or dist2(self.path_goal, gp) > 6.0
                or self.t - self.path_t > 4.0 or self.stuck > 0.6)
        if need:
            budget = max(0.002, min(0.010, self.deadline - time.perf_counter()))
            r = self.nav.plan((x, y), gp, budget)
            if isinstance(r, list):
                self.path = r
                self.path_goal = gp
                self.path_t = self.t
            elif r is None:
                self.path = [(x, y), gp]
                self.path_goal = gp
                self.path_t = self.t
        if not self.path:
            return self.unit(gp[0] - x, gp[1] - y)
        # advance along the path
        while len(self.path) > 1:
            nxt = self.path[1]
            if dist2(self.pos, nxt) < 3.0 or (len(self.path) > 2 and self.nav.walk_clear((x, y), self.path[2])):
                self.path.pop(0)
            else:
                break
        nxt = self.path[1] if len(self.path) > 1 else self.path[0]
        return self.unit(nxt[0] - x, nxt[1] - y)

    @staticmethod
    def unit(dx, dy):
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)

    def can_go(self, d, look=6.0):
        """True if a short walk along unit direction d stays clear. From a spot inside the grid's wall
        margin (hugging a building) the start cell itself counts as blocked, so only the points ahead
        are checked there."""
        x, y = self.pos[0], self.pos[1]
        q = (x + look * d[0], y + look * d[1])
        if self.nav.is_walkable(x, y):
            return self.nav.walk_clear((x, y), q)
        mid = (x + 0.6 * look * d[0], y + 0.6 * look * d[1])
        return self.nav.is_walkable(*q) and self.nav.is_walkable(*mid)

    def clear_dir(self, d, look=6.0):
        """Rotate d until a short walk ahead is clear (local steering); pressed against a wall with no
        clear way, head for the nearest walkable spot."""
        if self.on_roof or (d[0] == 0.0 and d[1] == 0.0):
            return d
        x, y = self.pos[0], self.pos[1]
        base = math.atan2(d[1], d[0])
        for off in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6, 2.0, -2.0):
            a = base + off
            c = (math.cos(a), math.sin(a))
            if self.can_go(c, look):
                return c
        if not self.nav.is_walkable(x, y):
            wx, wy = self.nav.nearest_walkable(x, y)
            esc = self.unit(wx - x, wy - y)
            if esc != (0.0, 0.0):
                return esc
        return d

    def laser_cover(self, enemies):
        """Nearest walkable spot within 13 m hidden from an enemy charging a laser at us, or None."""
        x, y = self.pos[0], self.pos[1]
        threats = []
        for e in enemies:
            tr = e.tr
            if not (tr.visible and tr.charging and "laser" in (tr.weapons or []) and e.d < 290.0):
                continue
            to_us = math.atan2(y - tr.pos[1], x - tr.pos[0])
            if abs(wrap(tr.aim[0] - to_us)) < 0.15:
                threats.append(e)
        if not threats:
            return None
        e = min(threats, key=lambda e: e.d)
        cache = getattr(self, "_laser_cover", None)
        if cache and cache[0] == e.tr.id and self.t - cache[1] < 0.5:
            return cache[2]
        start = (x, y) if self.nav.is_walkable(x, y) else self.nav.nearest_walkable(x, y)
        cands = []
        for r in (5.0, 9.0, 13.0):
            for k in range(12):
                a = 2 * math.pi * k / 12
                p = (x + r * math.cos(a), y + r * math.sin(a))
                if self.nav.is_walkable(*p) and self.nav.walk_clear(start, p):
                    cands.append(p)
        best = None
        if cands:
            gz = self.map.ground_many([p[0] for p in cands], [p[1] for p in cands])
            ctr = [(p[0], p[1], float(g) + 4.5) for p, g in zip(cands, gz)]
            tp = e.tr.pos
            seen = self.map.los_many((tp[0], tp[1], tp[2] + 8.0), ctr, step=1.0)
            hidden = [p for p, v in zip(cands, seen) if not v]
            if hidden:
                best = min(hidden, key=lambda p: dist2(self.pos, p))
        self._laser_cover = (e.tr.id, self.t, best)
        return best

    def shelter_dir(self, step=16.0):
        """Unit direction toward the least exposed walkable spot one grid cell away, if clearly better."""
        x, y = self.pos[0], self.pos[1]
        here = self.open.at(x, y)
        best = None
        for k in range(8):
            a = k * math.pi / 4.0
            c, s_ = math.cos(a), math.sin(a)
            q = (x + step * c, y + step * s_)
            if not self.nav.is_walkable(*q):
                continue
            o = self.open.at(*q)
            if o < here - 0.05 and (best is None or o < best[0]):
                best = (o, (c, s_))
        return best[1] if best else None

    def hide_spot(self):
        """A nearby walkable spot with the shortest sightlines (no known threat to hide from)."""
        if getattr(self, "_hide", None) is not None and self.t - self._hide[0] < 3.0:
            return self._hide[1]
        x, y = self.pos[0], self.pos[1]
        start = (x, y) if self.nav.is_walkable(x, y) else self.nav.nearest_walkable(x, y)
        best = ((self.open.at(x, y), 0.0), (x, y))
        for r in (10.0, 20.0, 30.0, 45.0):
            for k in range(12):
                a = 2 * math.pi * k / 12
                p = (x + r * math.cos(a), y + r * math.sin(a))
                if not self.nav.is_walkable(*p) or not self.nav.walk_clear(start, p):
                    continue
                key = (self.open.at(*p) + 0.002 * r, r)
                if key < best[0]:
                    best = (key, p)
        self._hide = (self.t, best[1])
        return best[1]

    def find_cover(self, threats):
        if self.cover is not None and self.t - self.cover_t < 1.0:
            return self.cover
        x, y = self.pos[0], self.pos[1]
        start = (x, y) if self.nav.is_walkable(x, y) else self.nav.nearest_walkable(x, y)
        cands = []
        for r in (9.0, 16.0, 25.0, 36.0):
            for k in range(12):
                a = 2 * math.pi * k / 12 + self.rng.uniform(-0.2, 0.2)
                p = (x + r * math.cos(a), y + r * math.sin(a))
                if self.nav.is_walkable(*p) and self.nav.walk_clear(start, p):
                    cands.append(p)
        keep = None
        if self.P["cover_stick"] and self.cover is not None and dist2(self.pos, self.cover) < 45.0:
            keep = self.cover  # hysteresis: only move to a clearly better spot
            cands.append(keep)
        if not cands or not threats:
            return None
        pts = [(p[0], p[1], self.map.ground_z(p[0], p[1]) + 6.0) for p in cands]
        exposed = [0] * len(cands)
        for e in threats[:4]:
            tp = e.tr.pos
            eye = (tp[0], tp[1], tp[2] + 8.0)
            vis = self.map.los_many(eye, pts, step=1.0)
            for i, v in enumerate(vis):
                if v:
                    exposed[i] += 1
        best, best_s = None, -1e9
        nc = self.zone.get("next_center")
        nr = float(self.zone.get("next_radius", 1e4))
        for p, ex in zip(cands, exposed):
            s = -ex * 100.0 - dist2(self.pos, p) * 0.8 - self.P["cover_open_w"] * self.open.at(p[0], p[1])
            if p is keep:
                s += 30.0
            s += 0.3 * min(dist2(p, e.tr.pos) for e in threats)
            if nc and math.hypot(p[0] - nc[0], p[1] - nc[1]) > nr:
                s -= 40.0
            if s > best_s:
                best, best_s = p, s
        self.cover = best
        self.cover_t = self.t
        return best

    def peek_hide(self, target, others=()):
        """A spot within peek_r that the target (and `others`) cannot see, while none of our weapons can fire
        soon; else None."""
        self._peek_hide = None
        gi, gun = self.weapon("gun")
        zi, lz = self.weapon("laser")
        gun_down = gun is None or gun.get("reload", 0) > 0.6 or gun.get("ammo", 1) == 0
        laser_down = lz is None or lz.get("cooldown", 0.0) > 0.6
        if not (gun_down and laser_down) or target.d < 30.0:
            return None
        x, y = self.pos[0], self.pos[1]
        eyes = [(e.tr.pos[0], e.tr.pos[1], e.tr.pos[2] + 8.0) for e in [target] + [o for o in others if o is not target][:3]]
        cands = []
        for r in (4.0, 8.0, self.P["peek_r"]):
            for k in range(12):
                a = 2 * math.pi * k / 12
                p = (x + r * math.cos(a), y + r * math.sin(a))
                if self.nav.is_walkable(*p) and self.nav.walk_clear((x, y), p):
                    cands.append((r, p))
        if not cands:
            return None
        pts = [(p[0], p[1], self.map.ground_z(p[0], p[1]) + 6.0) for _, p in cands]
        seen = [False] * len(pts)
        for eye in eyes:
            for i, v in enumerate(self.map.los_many(eye, pts, step=1.0)):
                seen[i] = seen[i] or bool(v)
        hidden = [rp for rp, v in zip(cands, seen) if not v]
        if not hidden:
            return None
        r, p = min(hidden, key=lambda rp: rp[0])
        self._peek_hide = p
        return p

    def duel_spot(self, target, enemies):
        """A nearby walkable spot that keeps a line of fire to the target while as few of the other
        known enemies as possible can see it (turns a 2-vs-1 into a 1-vs-1)."""
        if self.duel is not None and self.t - self.duel_t < 1.0 and self.duel_tid == target.tr.id:
            return self.duel
        x, y = self.pos[0], self.pos[1]
        others = [e for e in enemies if e.tr.id != target.tr.id and self.t - e.tr.seen_t < 5.0 and e.d < 220.0]
        cands = [(x, y)]
        for r in (7.0, 13.0, 20.0):
            for k in range(12):
                a = 2 * math.pi * k / 12
                p = (x + r * math.cos(a), y + r * math.sin(a))
                if self.nav.is_walkable(*p) and self.nav.walk_clear((x, y), p):
                    cands.append(p)
        mh = self.muzzle[2] - self.pos[2]
        gz = self.map.ground_many([p[0] for p in cands], [p[1] for p in cands])
        muz = [(p[0], p[1], float(g) + mh) for p, g in zip(cands, gz)]
        ctr = [(p[0], p[1], float(g) + 4.5) for p, g in zip(cands, gz)]
        sees_target = self.map.los_many(target.center, muz, step=1.0)
        exposed = [0] * len(cands)
        for e in others[:4]:
            tp = e.tr.predict(self.t)
            vis = self.map.los_many((tp[0], tp[1], tp[2] + 8.0), ctr, step=1.0)
            for i, v in enumerate(vis):
                if v:
                    exposed[i] += 1
        nc = self.zone.get("next_center")
        nr = float(self.zone.get("next_radius", 1e4))
        best, best_s = None, -1e9
        for i, p in enumerate(cands):
            s = (0.0 if sees_target[i] else -150.0) - 70.0 * exposed[i] - 0.6 * math.hypot(p[0] - x, p[1] - y)
            if nc and math.hypot(p[0] - nc[0], p[1] - nc[1]) > nr:
                s -= 30.0
            if s > best_s:
                best, best_s = p, s
        self.duel, self.duel_t, self.duel_tid = best, self.t, target.tr.id
        return best

    def fight_move(self, target, enemies):
        """Spacing plus a strafe that changes direction at random intervals."""
        x, y = self.pos[0], self.pos[1]
        if self.P["fight_retreat"]:
            engaged = [e for e in enemies if self.t - e.tr.seen_t < self.P["retreat_seen_s"] and e.d < 200.0]
            if len(engaged) >= max(1, int(self.P["retreat_min_enemies"])):
                ax = sum((x - e.tr.pos[0]) / max(e.d, 1.0) ** 2 for e in engaged)
                ay = sum((y - e.tr.pos[1]) / max(e.d, 1.0) ** 2 for e in engaged)
                away = self.unit(ax, ay)
                near = min(engaged, key=lambda e: e.d)
                to = self.unit(near.tr.pos[0] - x, near.tr.pos[1] - y)
                if self.t > self.strafe_until:
                    self.strafe = -self.strafe if self.rng.random() < self.P["strafe_flip"] else self.strafe
                    self.strafe_until = self.t + self.rng.uniform(self.P["strafe_min"], self.P["strafe_max"])
                side = (-to[1] * self.strafe, to[0] * self.strafe)
                self.kite_boost = near.d < self.P["retreat_boost_d"]
                if self.P["peek_retreat"]:
                    hide = self.peek_hide(near, engaged)
                    if hide is not None:
                        return self.unit(hide[0] - x, hide[1] - y) if math.hypot(hide[0] - x, hide[1] - y) > 1.5 else (0.0, 0.0)
                rx, ry = away[0] + side[0] * 0.5, away[1] + side[1] * 0.5
                if self.P["retreat_open_w"]:
                    sh = self.shelter_dir()
                    if sh is not None:
                        rx, ry = self.unit(rx, ry)
                        rx, ry = rx + self.P["retreat_open_w"] * sh[0], ry + self.P["retreat_open_w"] * sh[1]
                return self.unit(rx, ry)
        if self.P["peek_reload"]:
            hide = self.peek_hide(target)
            if hide is not None:
                return self.unit(hide[0] - x, hide[1] - y) if math.hypot(hide[0] - x, hide[1] - y) > 1.5 else (0.0, 0.0)
        if self.P["duel"]:
            others = [e for e in enemies if e.tr.id != target.tr.id and self.t - e.tr.seen_t < 5.0 and e.d < 220.0]
            if others:
                spot = self.duel_spot(target, enemies)
                if spot is not None and math.hypot(spot[0] - x, spot[1] - y) > 3.0:
                    to = self.unit(spot[0] - x, spot[1] - y)
                    return to
        tx, ty = target.tr.pos[0], target.tr.pos[1]
        d = math.hypot(tx - x, ty - y)
        kinds = self.slot_kinds()
        P = self.P
        want = P["gun_range"]
        if "laser" in kinds:
            want = P["laser_range"]
        if "lightning" in kinds and self.energy > 15:
            want = P["lightning_range"]
        if "gun" not in kinds and "laser" in kinds and "lightning" in kinds and self.energy > 20:
            want = P["pair_range"]
        kiting = P["kite"] and "lightning" in target.tr.weapons and ("lightning" not in kinds or self.energy < 15)
        if kiting:
            want = max(want, P["kite_range"])
        if self.hp_frac < 0.4:
            want += 25.0
        to = self.unit(tx - x, ty - y)
        radial = clamp((d - want) / 25.0, -1.0, 1.0)
        if kiting and d < P["kite_range"] + 5.0:
            self.kite_boost = True
            dvec = self.unit(-to[0] + (-to[1]) * self.strafe * 0.35, -to[1] + to[0] * self.strafe * 0.35)
            if self.on_roof or self.can_go(dvec):
                return dvec
        if self.t > self.strafe_until:
            self.strafe = -self.strafe if self.rng.random() < self.P["strafe_flip"] else self.strafe
            self.strafe_until = self.t + self.rng.uniform(self.P["strafe_min"], self.P["strafe_max"])
        if P["fight_boost"] and "lightning" not in kinds:
            if self.energy > P["fight_boost_on"]:
                self.fight_boosting = True
            elif self.energy < P["fight_boost_off"]:
                self.fight_boosting = False
            self.kite_boost = self.kite_boost or self.fight_boosting
        side = (-to[1] * self.strafe, to[0] * self.strafe)
        mx = to[0] * radial + side[0] * 0.9
        my = to[1] * radial + side[1] * 0.9
        dvec = self.unit(mx, my)
        if self.P["fight_open_w"]:
            sh = self.shelter_dir()
            if sh is not None:
                dvec = self.unit(dvec[0] + self.P["fight_open_w"] * sh[0], dvec[1] + self.P["fight_open_w"] * sh[1])
        if not self.on_roof and not self.can_go(dvec):
            self.strafe = -self.strafe
            self.strafe_until = self.t + 0.8
            side = (-to[1] * self.strafe, to[0] * self.strafe)
            dvec = self.unit(to[0] * radial + side[0] * 0.9, to[1] * radial + side[1] * 0.9)
        return dvec

    def move_for(self, goal, target, enemies, obs):
        x, y = self.pos[0], self.pos[1]
        kind = goal[0]
        boost = False
        jump = False
        visible = [e for e in enemies if e.tr.visible]

        evade = self.laser_cover(enemies) if self.P["laser_evade"] else None
        self.no_steer = False
        if kind == "roof":
            d, boost, jump = self.roof_move(goal[1])
            if self.no_steer:
                return d, boost, jump  # pressing into the wall or holding the roof: no steering, no unstick
        if kind == "roof":
            pass
        elif self.t < self.unstick_until:
            d = self.unstick_dir
        elif evade is not None:
            d = self.unit(evade[0] - x, evade[1] - y) if dist2(self.pos, evade) > 1.5 else (0.0, 0.0)
            boost = self.energy > 10
        elif kind == "fight" and target is not None:
            zone_urg, zone_pt = self.zone_need()
            self.kite_boost = False
            if zone_urg >= 0.8 and zone_pt:
                d = self.follow(zone_pt)
            else:
                d = self.fight_move(target, enemies)
                boost = self.kite_boost and self.energy > 8
        elif kind == "cover":
            threats = visible or enemies
            if not threats and self.P["heal_hide"]:
                threats = self.remembered_threats()
            c = self.find_cover(threats) if (threats or not self.P["heal_hide"]) else self.hide_spot()
            if c is not None:
                d = self.follow(c) if dist2(self.pos, c) > 2.5 else (0.0, 0.0)
            else:
                d = self.retreat_dir(visible or enemies)
            boost = self.energy > 25
        else:
            gp = self.goal_point(goal, target)
            if gp is None:
                d = self.wander()
            else:
                d = self.follow(gp)
                far = dist2(self.pos, gp) > 70.0
                boost = (kind == "zone" or (far and not visible)) and self.energy > self.P["travel_boost_e"]
            if kind == "drop" and dist2(self.pos, goal[1]) < 4.0:
                d = (0.0, 0.0)

        d = self.clear_dir(d)
        dodge = combat.dodge_vector(
            {"pos": self.pos, "vel": self.vel, "radius": self.radius, "height": self.height},
            obs.get("projectiles") or [], gravity=self.wcfg.get("mortar", {}).get("gravity", 22.0),
            splash=self.wcfg.get("mortar", {}).get("splash_radius", 7.0))
        if dodge is not None:
            d = self.unit(d[0] * 0.3 + dodge[0], d[1] * 0.3 + dodge[1])

        # stuck: jet over the obstacle when we have energy, else sidestep
        if self.stuck > 0.8 and self.t > self.unstick_until:
            self.stuck = 0.0
            self.path = None
            if self.energy > 45 and self.grounded:
                self.jet_until = self.t + 4.0
                jump = True
            else:
                a = math.atan2(d[1], d[0]) + self.rng.choice((1.6, -1.6))
                self.unstick_dir = (math.cos(a), math.sin(a))
                self.unstick_until = self.t + 0.9
        if self.t < self.jet_until:
            if self.grounded and self.t > self.jet_until - 3.9:
                self.jet_until = -1e9  # landed: done
            else:
                boost = self.energy > 4
        return d, boost, jump

    def retreat_dir(self, threats):
        if not threats:
            return (0.0, 0.0)
        x, y = self.pos[0], self.pos[1]
        ax = sum((x - e.tr.pos[0]) / max(e.d, 1.0) for e in threats)
        ay = sum((y - e.tr.pos[1]) / max(e.d, 1.0) for e in threats)
        return self.unit(ax, ay)

    def wander(self):
        if self.t > self.strafe_until:
            self.strafe = -self.strafe
            self.strafe_until = self.t + self.rng.uniform(1.0, 2.5)
        a = self.t * 0.4 * self.strafe
        return (math.cos(a) * 0.4, math.sin(a) * 0.4)

    # ================================================================== aim and fire
    def weapon(self, kind):
        for i, wp in enumerate(self.weapons):
            if wp and wp.get("kind") == kind:
                return i, wp
        return None, None

    def aim_and_fire(self, target, enemies, move):
        aim_now = tuple(self.me.get("aim", (0.0, 0.0)))
        fire = [False, False]
        reload = False
        gi, gun = self.weapon("gun")
        if self.P["check_six"]:
            six = self.check_six(target)
            if six is not None:
                self.hold_laser(fire)
                return (six, 0.0), fire, False
        if target is not None and target.los:
            aim, fire, reload = self.engage(target, aim_now, fire)
            if self.world.mode == "ffa" and self.goal[0] == "cover" and not self.provoked(target):
                fire = [False, False]  # avoiding this fight: shooting would only draw it to us
            elif (self.t < self.P["early_safe_s"] and target.d > self.P["early_safe_d"]
                  and self.t - self.world.last_damage_t() > 4.0 and self.t - target.tr.hurt_us_t > 6.0):
                fire = [False, False]  # early on, don't start fights we weren't dragged into
                self.hold_laser(fire)
            elif (self.P["hold_fire_beyond"] and target.d > self.P["engage_range"]
                  and self.t - self.world.last_damage_t() > 4.0 and self.t - target.tr.hurt_us_t > 6.0):
                fire = [False, False]  # not our fight (yet): don't provoke it from range
                self.hold_laser(fire)
            return aim, fire, reload
        if self.P["precharge"]:
            pc = self.precharge(enemies)
            if pc is not None:
                return pc
        if self.P["lob"]:
            lob = self.lob_shot(enemies, aim_now)
            if lob is not None:
                return lob
        # farm crates and drones when no mech is in our sights
        farm = self.farm_target()
        if farm is not None:
            pos, ctr = farm
            sol = combat.lead_aim(self.muzzle, ctr, (0.0, 0.0, 0.0), gun["stats"]["speed"] if gun else 252.0,
                                  offset=self.radius + 0.5, pitch_lim=self.pitch_lim)
            if sol:
                aim = (sol[0], sol[1])
                err = angle_err(aim_now, aim)
                d = math.dist(self.muzzle, ctr)
                if gun is not None and err < ang_size(d, 1.2) + 0.004 and gun.get("ammo", 1) > 0:
                    fire[gi] = True
                li, lw = self.weapon("lightning")
                if lw is not None and d < lw["stats"]["range"] * 0.9 and err < 0.2 and self.energy > 55:
                    fire[li] = True
                self.hold_laser(fire)
                return aim, fire, False
        if gun is not None and gun.get("ammo", 99) < gun["stats"].get("mag", 12) * 0.6 and not gun.get("reload"):
            hidden = not any(e.los for e in enemies)
            if (self.P["reload_hidden"] and hidden) or (
                    self.t - self.world.last_damage_t() > 2.0 and not any(e.tr.visible for e in enemies)):
                reload = True
        self.hold_laser(fire)
        return self.scan_aim(enemies, move), fire, reload

    def precharge(self, enemies):
        """(aim, fire, reload) charging the laser at a tracked enemy that is hidden but coming our way, so
        the first shot lands the moment it steps out."""
        zi, lz = self.weapon("laser")
        if lz is None:
            return None
        if float(lz.get("charge", 0.0)) <= 0.0 and lz.get("cooldown", 0.0) > 0.0:
            return None
        best = None
        for e in enemies:
            if not e.tr.visible or e.los or e.d > self.P["precharge_d"]:
                continue
            dx, dy = self.pos[0] - e.tr.pos[0], self.pos[1] - e.tr.pos[1]
            n = math.hypot(dx, dy) or 1.0
            closing = (e.tr.vel[0] * dx + e.tr.vel[1] * dy) / n
            if (closing > 1.0 or e.d < 45.0) and (best is None or e.d < best.d):
                best = e
        if best is None:
            return None
        yaw, pitch = combat.yaw_pitch(self.muzzle, best.center)
        fire = [False, False]
        fire[zi] = True
        return (yaw, clamp(pitch, *self.pitch_lim)), fire, False

    def check_six(self, target):
        """Yaw to look at an unseen attacker that hit us from behind in the last check_six_s seconds."""
        w = self.world
        for e in w.events:
            if e.get("type") == "damaged" and e.get("rear") and e.get("bearing") is not None:
                by = e.get("by")
                if by is not None and by in w.enemies and w.enemies[by].visible:
                    continue  # we can see it: target selection handles it
                self._six = (self.t, float(e["bearing"]))
        six = getattr(self, "_six", None)
        if six is None or self.t - six[0] > self.P["check_six_s"]:
            return None
        if target is not None and target.los and target.hp_frac < 0.25:
            return None  # finish the kill first
        return six[1]

    def lob_shot(self, enemies, aim_now):
        """(aim, fire, reload) lobbing a mortar shell at a recently seen enemy we can't shoot directly."""
        mi, mo = self.weapon("mortar")
        if mo is None or mo.get("cooldown", 1.0) > 0.0:
            return None
        st = mo["stats"]
        tube = (self.pos[0], self.pos[1], self.pos[2] + self.height + 0.5)
        best = None
        for e in enemies:
            if self.t - e.tr.seen_t > 3.0 or e.los:
                continue
            p = e.tr.predict(self.t)
            d = dist2(self.pos, p)
            if not 35.0 <= d <= 220.0:
                continue
            if best is None or d < best[0]:
                best = (d, e, p)
        if best is None:
            return None
        d, e, p = best
        vel = (e.tr.vel[0], e.tr.vel[1], 0.0)
        for high in (False, True):
            sol = combat.mortar_aim(tube, (p[0], p[1], p[2]), vel, st["speed"], st.get("gravity", self.gravity),
                                    offset=self.radius + 0.5, high=high, pitch_lim=self.pitch_lim)
            if sol is None:
                continue
            if not combat.arc_clear(self.map, tube, sol[0], sol[1], st["speed"], st.get("gravity", self.gravity),
                                    sol[2], offset=self.radius + 0.5):
                continue
            aim = (sol[0], sol[1])
            fire = [False, False]
            if angle_err(aim_now, aim) < 0.03:
                fire[mi] = True
            self.hold_laser(fire)
            return aim, fire, False
        return None

    def hold_laser(self, fire):
        """Keep a charge in progress instead of wasting it into a wall when the target ducks away."""
        zi, lz = self.weapon("laser")
        if lz is not None and float(lz.get("charge", 0.0)) > 0.0:
            fire[zi] = True

    def engage(self, e, aim_now, fire):
        tr = e.tr
        lat = 0.05  # half a decision: shots leave during the next two ticks
        vel = (tr.vel[0], tr.vel[1], tr.vel[2] if abs(tr.vel[2]) > 1.0 else 0.0)
        ctr = (e.center[0] + vel[0] * lat, e.center[1] + vel[1] * lat, e.center[2] + vel[2] * lat)
        rad = 2.5 * (1.0 + 0.05 * (tr.cubes or 0))
        muzzle = (self.muzzle[0] + self.vel[0] * lat, self.muzzle[1] + self.vel[1] * lat, self.muzzle[2])
        gi, gun = self.weapon("gun")
        li, lw = self.weapon("lightning")
        zi, lz = self.weapon("laser")
        mi, mo = self.weapon("mortar")
        direct = combat.yaw_pitch(muzzle, ctr)
        direct = (direct[0], clamp(direct[1], *self.pitch_lim))
        aim = direct
        if gun is not None:
            lf = 1.0 + (self.P["lead_far"] - 1.0) * clamp((e.d - 50.0) / 100.0, 0.0, 1.0)
            gvel = (vel[0] * lf, vel[1] * lf, vel[2] * lf)
            sol = combat.lead_aim(muzzle, ctr, gvel, gun["stats"]["speed"], offset=self.radius + 0.5,
                                  pitch_lim=self.pitch_lim)
            if sol:
                aim = (sol[0], sol[1])
        laser_release = False
        if lz is not None:
            charge = float(lz.get("charge", 0.0))
            full = float(lz["stats"].get("charge", 0.75))
            if charge >= full - 0.05:
                aim = direct  # hitscan: no lead
        # mortar takes over the torso briefly when it is ready and the target is far enough
        mortar_aim = None
        if mo is not None and mo.get("cooldown", 1.0) <= 0.0 and e.d > 28.0:
            sol = combat.mortar_aim((self.pos[0], self.pos[1], self.pos[2] + self.height + 0.5),
                                    (tr.pos[0], tr.pos[1], tr.pos[2]), vel, mo["stats"]["speed"],
                                    mo["stats"].get("gravity", self.gravity), offset=self.radius + 0.5,
                                    pitch_lim=self.pitch_lim)
            if sol:
                mortar_aim = (sol[0], sol[1])
        # Commit the torso to a lob for up to 1 s once the mortar is ready (flip-flopping between the
        # gun's aim and the mortar's each tick lined up neither); then let the gun work for a while.
        gun_down = gun is None or gun.get("reload", 0) > 0 or gun.get("ammo", 1) == 0
        if self.P["mortar_mode"] == "gun_down":
            self.mortar_until = self.t + 0.2 if (mortar_aim is not None and gun_down) else -1e9
        elif mortar_aim is None:
            self.mortar_until = -1e9
        elif self.t >= self.mortar_until and self.t >= self.mortar_block_until:
            self.mortar_until = self.t + 1.0
            self.mortar_block_until = self.t + 2.5
        if mortar_aim is not None and self.t < self.mortar_until:
            aim = mortar_aim
            if angle_err(aim_now, aim) < 0.03:
                fire[mi] = True
        d = e.d
        err = angle_err(aim_now, aim)
        tol = ang_size(d, rad) * 0.9 + 0.003
        if gun is not None and aim is not mortar_aim and err < tol and gun.get("ammo", 1) > 0 and not gun.get("reload"):
            fire[gi] = True
        if lw is not None:
            rng = lw["stats"]["range"]
            cone = math.radians(lw["stats"].get("cone_deg", 30.0))
            yaw_err = abs(wrap(direct[0] - aim_now[0]))
            if d < rng + rad * 0.5 and yaw_err < cone * 0.6 and self.energy >= lw["stats"].get("energy", 3.0):
                fire[li] = True
        if lz is not None and aim is not mortar_aim:
            charge = float(lz.get("charge", 0.0))
            full = float(lz["stats"].get("charge", 0.75))
            if lz.get("cooldown", 0.0) <= 0.0 and d < lz["stats"].get("range", 280.0):
                if charge < full - 0.05:
                    fire[zi] = True  # keep charging
                else:
                    derr = angle_err(aim_now, direct)
                    fire[zi] = derr >= ang_size(d, rad) * 0.6  # release when on target
        return aim, fire, False

    def provoked(self, e):
        """In a free-for-all: worth shooting back (attacked, very close, or an easy kill)."""
        t = self.t
        if t - self.world.last_damage_t() < 4.0 or t - e.tr.hurt_us_t < 4.0:
            return True
        return e.d < 40.0 or (e.hp_frac < 0.4 and e.d < 110.0 and self.hp_frac > 0.6)

    def farm_target(self):
        """Nearest crate or drone in line of sight within 70 m (when we chose to loot it, or it is near)."""
        w = self.world
        if (self.P["farm_heal_hold"] and self.hp_frac < self.P["regen_hold_hp"]
                and self.t - w.last_damage_t() < 30.0):
            return None  # shooting anything resets the regen timer
        cands = []
        for i, rec in w.crates.items():
            if self.t - rec["t"] < 0.5:
                p = rec["pos"]
                cands.append((dist2(self.pos, p), p, (p[0], p[1], p[2] + 1.2)))
        if self.P["farm_npcs"] and self.hp_frac > 0.55:
            for i, tr in w.npcs.items():
                if tr.visible:
                    p = tr.pos
                    cands.append((dist2(self.pos, p) - 10.0, p, (p[0], p[1], p[2] + 1.75)))
        cands = [c for c in cands if c[0] < self.P["farm_radius"]]
        if not cands:
            return None
        cands.sort(key=lambda c: c[0])
        for dd, p, ctr in cands[:3]:
            if self.map.los(self.muzzle, ctr, step=1.0):
                return p, ctr
        return None

    def scan_aim(self, enemies, move):
        w = self.world
        t = self.t
        recent = [d for d in w.damage if d.get("bearing") is not None and t - d["t"] < 2.5]
        if recent:
            return (recent[-1]["bearing"], 0.0)
        known = [e for e in enemies if t - e.tr.seen_t < 6.0]
        if known:
            e = min(known, key=lambda e: e.d)
            p = e.tr.predict(t)
            return (math.atan2(p[1] - self.pos[1], p[0] - self.pos[0]), 0.0)
        snd = [s for s in w.sounds if s["kind"] in SOUND_THREATS and t - s["t"] < 1.5]
        if snd:
            s = min(snd, key=lambda s: {"near": 0, "mid": 1, "far": 2}.get(s["dist"], 2))
            return (s["bearing"], 0.0)
        if t - self.sweep_t > 0.7:
            self.sweep_t = t
            self.sweep_phase = (self.sweep_phase + 1) % 6
        base = math.atan2(move[1], move[0]) if math.hypot(*move) > 0.3 else self.me.get("aim", (0.0, 0.0))[0]
        off = (0.0, 1.9, 0.0, -1.9, 0.0, math.pi)[self.sweep_phase]
        return (wrap(base + off), 0.0)

    # ================================================================== loot and upgrades
    def loot_action(self):
        x, y, z = self.pos
        best = None
        for i, rec in self.world.pickups.items():
            if i in self.ignore_items:
                continue
            p = rec["pos"]
            if math.hypot(p[0] - x, p[1] - y) > self.pickup_r - 0.3 or abs(p[2] - z) > 3.8:
                continue
            v = self.want_weapon(rec.get("kind"))
            if v > 0 and (best is None or v > best[0]):
                best = (v, i, rec.get("kind"))
        if best is None:
            return None
        out = {"take": best[1]}
        _, slot = self.take_plan(best[2])
        if slot is not None:
            zi, lz = self.weapon("laser")
            if zi == slot and lz and lz.get("charge", 0) > 0:
                return None  # a charging laser can't be dropped
            out["drop"] = slot
        return out

    def handle_take_failures(self):
        for e in self.world.events:
            if e.get("type") == "take_failed" and e.get("reason") in ("have", "gone", "full"):
                self.ignore_items.add(e.get("id"))

    def pick_upgrade(self, offer):
        choices = offer.get("choices") or []
        order = self.P.get("upgrade_order") or UPGRADE_ORDER
        best, best_r = 0, 1e9
        for k, c in enumerate(choices):
            cid = c.get("id")
            r = order.index(cid) if cid in order else 500
            if r < best_r:
                best, best_r = k, r
        return best
