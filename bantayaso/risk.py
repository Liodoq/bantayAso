"""Transparent rule engine: dogs + hazards + zones + actions/motion -> risk level per dog.

Levels: 0 safe, 1 watch, 2 warning, 3 danger.
A level must hold `persist_seconds` before it becomes current (danger: half); lowering needs
`calm_seconds` of calm. Works without actions too (then hazards/zones only, Batch 2 behaviour).
"""
from __future__ import annotations

import math
import time

from .zones import zone_phrase
from dataclasses import dataclass, field

LEVELS = ["safe", "watch", "warning", "danger"]
ZONE_LEVEL = {"danger": 3, "trash": 2, "nogo": 2, "bed": 0, "food": 0, "play": 0}
ZONE_TEXT = {"danger": "in the danger zone", "trash": "at the trash", "nogo": "in a no-go area",
             "bed": "on the bed"}
CHEW_LABELS = ("chewing something", "eating")
BUSY_LABELS = ("digging", "scratching furniture", "jumping on furniture")
SELF_CARE = ("scratching itself", "licking itself")
REST_LABELS = ("sleeping", "lying down", "sitting", "standing")
# What each behaviour means (editable on the Behaviours page -> config "behaviors").
# key: (shown as, default level 0-3, default seconds before it counts or None)
BEHAVIORS = {
    "chewing":              ("Chewing something unknown", 2, 5.0),
    "long_chewing":         ("Chewing for a long time", 3, 15.0),
    "nose_down":            ("Nose down eating in one spot", 2, 6.0),
    "rough_play":           ("Playing rough with another dog", 2, 3.0),
    "fighting":             ("Fighting", 3, 8.0),
    "digging":              ("Digging", 2, None),
    "scratching furniture": ("Scratching furniture", 2, None),
    "jumping on furniture": ("Jumping on furniture", 2, None),
    "sniffing the floor":   ("Sniffing around", 1, None),
    "licking itself":       ("Licking itself", 0, None),
    "scratching itself":    ("Scratching itself", 0, None),
    "walking":              ("Walking around", 0, None),
    "standing":             ("Standing", 0, None),
    "sitting":              ("Sitting", 0, None),
    "lying down":           ("Lying down", 0, None),
    "sleeping":             ("Sleeping", 0, None),
}

CALM_TEXT = {"sleeping": "sleeping", "lying down": "just lying down", "sitting": "just sitting",
             "standing": "just standing", "walking": "walking around",
             "licking itself": "licking itself", "scratching itself": "scratching itself"}


def box_distance(a, b) -> float:
    """Pixel gap between two boxes (0 if they overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    return math.hypot(dx, dy)


@dataclass
class Assessment:
    track_id: int
    level: int                 # current (persisted) level
    raw_level: int             # this frame's level before persistence
    reason: str
    zone: str | None = None
    near: list = field(default_factory=list)
    in_mouth: str | None = None
    action: str | None = None  # smoothed action label (if actions are enabled)
    chewing: bool = False

    @property
    def name(self) -> str:
        return LEVELS[self.level]


@dataclass
class _Track:
    level: int = 0
    candidate: int = 0
    candidate_since: float = 0.0
    calm_since: float = 0.0
    last_seen: float = 0.0
    zone_since: dict = field(default_factory=dict)
    near_memory: dict = field(default_factory=dict)   # hazard name -> (last_seen_t, times_seen, center)
    disappeared: set = field(default_factory=set)    # confirmed missing on a fresh hazard pass
    chew_since: float | None = None
    chew_last: float = 0.0
    nose_since: float | None = None      # nose-down in one spot (sniffing/eating hidden food)
    nose_last: float = 0.0
    nose_center: tuple | None = None
    fight_since: float | None = None
    fight_last: float = 0.0
    alert_reason: str = ""
    box: tuple | None = None             # last box (used to hand state over on tracker ID switches)


class RiskEngine:
    def __init__(self, cfg: dict):
        r = cfg.get("risk", {})
        self.persist = float(r.get("persist_seconds", 1.0))
        self.calm = float(r.get("calm_seconds", 2.0))
        self.memory = float(r.get("last_seen_memory_seconds", 5))
        self.near_px = float(r.get("muzzle_near_px", 60))
        self.zone_dwell = float(r.get("zone_dwell_seconds", 1.5))
        self.chew_conf = float(r.get("chew_conf", 0.35))
        self.mouth_thr = float(r.get("mouth_threshold", 0.65))       # with head/jaw motion
        self.mouth_alone = float(r.get("mouth_alone_threshold", 0.85))  # without a chew label
        self.chew_min_s = float(r.get("chew_min_seconds", 5.0))      # user: notify after 5-10 s
        self.safe_chew_fn = None        # fn(track_id) -> True when the owner said this chewing is safe
        self.chew_head = float(r.get("chew_head_threshold", 0.35))
        self.eat_thr = float(r.get("eat_score_threshold", 0.36))
        self.selfcare_max = float(r.get("self_care_max_score", 0.45))
        self.nose_down_s = float(r.get("nose_down_seconds", 6.0))
        self.fight_thr = float(r.get("fight_threshold", 0.5))
        self.fight_min_s = float(r.get("fight_min_seconds", 3.0))
        self.fight_danger_s = float(r.get("fight_danger_seconds", 8.0))
        self.chew_danger_s = float(r.get("chew_danger_seconds", 15))
        self.cooldown = {2: float(r.get("cooldown_warning", 20)), 3: float(r.get("cooldown_danger", 5))}
        self.set_behaviors(cfg.get("behaviors") or {})
        self.vocab_tiers = {str(k): int(v) for k, v in (cfg.get("hazards") or {}).items()}
        self.tracks: dict[int, _Track] = {}
        self._last_alert = {2: float("-inf"), 3: float("-inf")}
        self.handovers: list = []

    def set_behaviors(self, beh: dict) -> None:
        """Owner-edited levels/seconds per behaviour (Behaviours page). Missing keys keep the defaults."""
        self.beh = {}
        for k, (_name, lvl, sec) in BEHAVIORS.items():
            o = beh.get(k) or {}
            self.beh[k] = (max(0, min(3, int(o.get("level", lvl)))),
                           float(o["seconds"]) if sec is not None and o.get("seconds") is not None else sec)
        self.chew_min_s = self.beh["chewing"][1]
        self.chew_danger_s = self.beh["long_chewing"][1]
        self.nose_down_s = self.beh["nose_down"][1]
        self.fight_min_s = self.beh["rough_play"][1]
        self.fight_danger_s = self.beh["fighting"][1]
        self._calm_reasons = {"resting", "all calm"} | {
            t for lab, t in CALM_TEXT.items() if self.beh.get(lab, (0,))[0] == 0}

    def lvl(self, key: str) -> int:
        return self.beh.get(key, (0, None))[0]

    # ------------------------------------------------------------------
    def update(self, dogs, hazards, zones, frame_size, now: float | None = None,
               hazards_fresh: bool = True, actions: dict | None = None,
               pairs: dict | None = None) -> list[Assessment]:
        """pairs: track_id -> (other_track_id, fight_score 0..1) for dogs touching each other."""
        now = time.monotonic() if now is None else now
        w, h = frame_size
        owner = {}                                   # hazard index -> nearest dog track_id
        for i, hz in enumerate(hazards):
            best = min(dogs, key=lambda dd: box_distance(dd.box, hz.box), default=None)
            if best is not None:
                owner[i] = best.track_id
        out = []
        self.handovers = []
        present = {d.track_id for d in dogs}
        for d in dogs:
            tid = d.track_id
            if tid not in self.tracks:
                # The tracker often gives the same dog a new ID when dogs overlap (seen in the
                # 5-dog clip: IDs up to 170 in 2 minutes). Inherit the state of a dog that was
                # lost < 2 s ago near the same place, so chewing timers/levels don't reset.
                lost = [(t, tr0) for t, tr0 in self.tracks.items()
                        if t not in present and now - tr0.last_seen < 2.0 and tr0.box is not None]
                if lost:
                    cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
                    bw = max(1, d.box[2] - d.box[0])
                    def dist(item):
                        b = item[1].box
                        return math.hypot(cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2)
                    old_tid, old_tr = min(lost, key=dist)
                    if dist((old_tid, old_tr)) < 0.6 * bw:
                        self.tracks[tid] = self.tracks.pop(old_tid)
                        self.handovers.append((old_tid, tid))
            tr = self.tracks.setdefault(tid, _Track(candidate_since=now, calm_since=now))
            tr.box = d.box
            tr.last_seen = now
            reasons: list[tuple[int, str]] = []

            def add(lvl, text):
                reasons.append((lvl, text))

            # ---------- action / motion ----------
            act = (actions or {}).get(tid)
            label = getattr(act, "label", None) if act is not None else None
            conf = getattr(act, "conf", 0.0) if act is not None else 0.0
            mouth = getattr(act, "mouth", 0.0) if act is not None else 0.0
            motion = getattr(act, "motion", "still") if act is not None else "still"
            chew_p = getattr(act, "chew", 0.0) if act is not None else 0.0
            if isinstance(act, tuple):                       # (label, conf) legacy form
                label, conf = act
            # Calibrated on the user's clips (rec_20261009_173621): eating dogs scored
            # (chew+mouth)/2 ≈ 0.42–0.46 (p25 0.34–0.41) while resting/sitting/scratching dogs
            # scored ≈ 0.18–0.20 (max 0.34). CLIP often *labels* head-down eating as "licking
            # itself" / "scratching itself", so the label alone must not cancel a high score.
            eat_score = 0.5 * (chew_p + mouth)
            self_care_calm = label in SELF_CARE and eat_score < self.selfcare_max
            chewing_now = act is not None and not self_care_calm and (
                eat_score >= self.eat_thr
                or (label in CHEW_LABELS and conf >= self.chew_conf)
                or mouth >= self.mouth_alone)
            if chewing_now:
                tr.chew_last = now
                tr.chew_since = tr.chew_since or now
            elif tr.chew_since and now - tr.chew_last > 1.5:  # 1.5 s grace for flicker
                tr.chew_since = None
            chew_for = now - tr.chew_since if tr.chew_since is not None else 0.0
            chewing = tr.chew_since is not None and chew_for >= self.chew_min_s   # sustained only

            # nose down in ONE spot for a while (sniffing/eating while moving the head) = probably
            # eating something hidden in the blanket/floor. Walking around sniffing does not count.
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            bw = max(1, d.box[2] - d.box[0])
            nose_now = act is not None and label == "sniffing the floor" and chew_p >= 0.30
            if nose_now:
                if tr.nose_center is None or math.hypot(cx - tr.nose_center[0], cy - tr.nose_center[1]) > 0.5 * bw:
                    tr.nose_since, tr.nose_center = now, (cx, cy)    # (re)start: new spot
                tr.nose_last = now
            elif tr.nose_since and now - tr.nose_last > 1.5:
                tr.nose_since, tr.nose_center = None, None
            nose_down = tr.nose_since is not None and now - tr.nose_since >= self.nose_down_s

            # ---------- zones (dog's feet = bottom-center) ----------
            feet = ((d.box[0] + d.box[2]) / 2, d.box[3])
            zone_hit = next((z for z in zones if z.contains(feet, w, h)), None)
            for k in list(tr.zone_since):
                if zone_hit is None or k != zone_hit.name:
                    tr.zone_since.pop(k)
            on_bed = False
            calm_zone = zone_hit is not None and zone_hit.type in ("food", "play")
            if zone_hit:
                since = tr.zone_since.setdefault(zone_hit.name, now)
                if zone_hit.type == "bed":
                    on_bed = True
                elif calm_zone:
                    pass
                else:
                    dwell = now - since >= self.zone_dwell
                    lvl = ZONE_LEVEL.get(zone_hit.type, 1) if dwell else 1
                    if zone_hit.type == "danger" and not (chewing or dwell):
                        lvl = 1
                    add(lvl, zone_phrase(zone_hit))

            # ---------- hazards near this dog (each hazard belongs to its nearest dog) ----------
            near = []
            for i, hz in enumerate(hazards):
                if owner.get(i) == tid and box_distance(d.box, hz.box) <= self.near_px:
                    near.append(hz)
                    if hazards_fresh:
                        _l, n, _c = tr.near_memory.get(hz.name, (0.0, 0, None))
                        tr.near_memory[hz.name] = (now, n + 1, hz.center)
            for hz in near:
                if calm_zone and hz.tier < 2:
                    continue                         # normal toy/food proximity in its designated area
                if act is None:                              # no action model: Batch 2 rule
                    add({3: 3, 2: 2}.get(hz.tier, 1), f"near a {hz.name}")
                elif chewing or mouth >= self.mouth_thr * 0.8 or label == "sniffing the floor":
                    add(3 if hz.tier >= 3 else 2, f"chewing near a {hz.name}" if chewing
                        else f"nosing a {hz.name}")
                else:                                        # just lying/standing beside it
                    add(2 if hz.tier >= 3 else 1, f"near a {hz.name}")

            # ---------- "possibly in mouth": hazard seen near the dog 3+ times just vanished ----------
            in_mouth = None
            names_now = {hz.name for hz in hazards}
            for name, (last, n, _c) in list(tr.near_memory.items()):
                age = now - last
                if age > self.memory:
                    tr.near_memory.pop(name)
                    tr.disappeared.discard(name)
                    continue
                if hazards_fresh:
                    if name in names_now:
                        tr.disappeared.discard(name)
                    elif n >= 3 and age > 0:
                        tr.disappeared.add(name)
                if name in tr.disappeared:
                    tier = self.vocab_tiers.get(name, 2)
                    if tier >= 2:
                        in_mouth = name
                        add(3 if tier >= 3 else 2, f"the {name} disappeared near its mouth")

            # ---------- rough play / fighting (two dogs tangled + vigorous + CLIP says fighting) ----------
            pr = (pairs or {}).get(tid)
            fight_now = pr is not None and pr[1] >= self.fight_thr
            if fight_now:
                tr.fight_last = now
                tr.fight_since = tr.fight_since or now
            elif tr.fight_since and now - tr.fight_last > 1.5:
                tr.fight_since = None
            if tr.fight_since is not None:
                dur = now - tr.fight_since
                if dur >= self.fight_danger_s:
                    add(self.lvl("fighting"), "fighting - separate them")
                elif dur >= self.fight_min_s:
                    add(self.lvl("rough_play"), "playing rough - may turn into a fight")

            # ---------- behaviour rules ----------
            owner_safe = bool(chewing and self.safe_chew_fn and not any(hz.tier >= 2 for hz in near)
                              and not in_mouth and self.safe_chew_fn(tid))
            if chewing and calm_zone and not any(hz.tier >= 2 for hz in near) and not in_mouth:
                pass                                         # eating at the bowl / chewing toys: normal
            elif owner_safe:
                pass                                         # owner pressed "It's safe" for this chewing
            elif chewing:
                named = [hz for hz in near]
                if not named and not in_mouth:
                    add(self.lvl("chewing"), "chewing an unknown object - check what it is")
                if chew_for >= self.chew_danger_s:
                    add(self.lvl("long_chewing"), f"has been chewing for {int(chew_for)} s")
                if zone_hit is not None and zone_hit.type == "danger":
                    add(3, "chewing in the danger zone")
            elif nose_down and not calm_zone:
                add(self.lvl("nose_down"), "nose-down eating something in one spot - check what it is")
            elif label in BUSY_LABELS and motion in ("active", "frantic"):
                add(self.lvl(label), label)
            elif motion == "frantic" and zone_hit is not None and zone_hit.type in ("trash", "nogo"):
                add(2, f"frantic {ZONE_TEXT[zone_hit.type]}")
            elif label == "sniffing the floor" and conf >= 0.3:
                add(self.lvl("sniffing the floor"), "sniffing around")
            elif label in CALM_TEXT and conf >= 0.3 and self.lvl(label) > 0:
                add(self.lvl(label), CALM_TEXT[label])     # owner made a normally-calm action count

            raw = max((r[0] for r in reasons), default=0)
            if on_bed and raw < 3 and not chewing:
                raw = min(raw, 1)                            # bed calms things unless chewing/danger
            level = self._persist(tr, raw, now)
            if reasons and raw > 0:
                reasons.sort(key=lambda r: -r[0])
                reason = reasons[0][1]
                if raw >= level:
                    tr.alert_reason = reason
            elif level >= 2 and tr.alert_reason:
                reason = tr.alert_reason                     # level still cooling down: keep the real cause
            elif chewing and calm_zone:
                reason = ("eating " if zone_hit.type == "food" else "chewing ") + zone_phrase(zone_hit)
            elif owner_safe:
                reason = "chewing something you said is safe"
            elif label in CALM_TEXT:
                reason = CALM_TEXT[label] + (" " + zone_phrase(zone_hit) if zone_hit and zone_hit.type in ("bed", "food", "play") else "")
            elif motion == "still":
                reason = ("resting " + zone_phrase(zone_hit)) if on_bed else "resting"
            else:
                reason = "all calm"
            out.append(Assessment(tid, level, raw, reason, zone_hit.name if zone_hit else None,
                                  sorted(hz.name for hz in near), in_mouth, label, chewing))

        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > 10]:
            self.tracks.pop(tid)
        return out

    def _persist(self, tr: _Track, raw: int, now: float) -> int:
        if raw > tr.level:
            if raw != tr.candidate:
                tr.candidate, tr.candidate_since = raw, now
            need = self.persist / 2 if raw == 3 else self.persist
            if now - tr.candidate_since >= need:
                tr.level = raw
                tr.calm_since = now
        elif raw < tr.level:
            tr.candidate = raw
            if now - tr.calm_since >= self.calm:
                tr.level = raw
                tr.calm_since = now
        else:
            tr.candidate = raw
            tr.calm_since = now
        return tr.level

    # ------------------------------------------------------------------
    def should_alert(self, assessments, now: float | None = None):
        """Top warning/danger Assessment to alert on, respecting cooldowns; else None."""
        now = time.monotonic() if now is None else now
        if not assessments:
            return None
        calm = self._calm_reasons
        live = [a for a in assessments if a.raw_level >= a.level          # only while the cause is still happening
                and a.reason.split(" on ")[0].split(" in ")[0] not in calm]  # never alert with a calm reason
        if not live:
            return None
        top = max(live, key=lambda a: a.level)
        if top.level < 2 or now - self._last_alert[top.level] < self.cooldown[top.level]:
            return None
        self._last_alert[top.level] = now
        if top.level == 3:
            self._last_alert[2] = now
        return top
