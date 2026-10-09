"""Transparent rule engine: dogs + hazards + zones (+ actions/motion later) -> risk level per dog.

Levels: 0 safe, 1 watch, 2 warning, 3 danger.
A level must hold `persist_seconds` before it becomes current (no flicker); lowering needs 2 s calm.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

LEVELS = ["safe", "watch", "warning", "danger"]
ZONE_LEVEL = {"danger": 3, "trash": 2, "nogo": 2, "bed": 0}
ZONE_TEXT = {"danger": "in the danger zone", "trash": "at the trash", "nogo": "in a no-go area",
             "bed": "on the bed"}
TIER_LEVEL = {3: 3, 2: 2, 1: 1}


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
    near: list = field(default_factory=list)      # hazard names near the dog
    in_mouth: str | None = None                    # hazard possibly in mouth (vanished near dog)

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


class RiskEngine:
    def __init__(self, cfg: dict):
        r = cfg.get("risk", {})
        self.persist = float(r.get("persist_seconds", 1.0))
        self.calm = float(r.get("calm_seconds", 2.0))
        self.memory = float(r.get("last_seen_memory_seconds", 5))
        self.near_px = float(r.get("muzzle_near_px", 60))
        self.zone_dwell = float(r.get("zone_dwell_seconds", 1.5))
        self.cooldown = {2: float(r.get("cooldown_warning", 20)), 3: float(r.get("cooldown_danger", 5))}
        self.tracks: dict[int, _Track] = {}
        self.vocab_tiers = {str(k): int(v) for k, v in (cfg.get("hazards") or {}).items()}
        self._last_alert = {2: float("-inf"), 3: float("-inf")}

    # ------------------------------------------------------------------
    def update(self, dogs, hazards, zones, frame_size, now: float | None = None,
               hazards_fresh: bool = True, actions: dict | None = None) -> list[Assessment]:
        """hazards_fresh=False when the hazard detector did not run this frame (reuse last result,
        and do not age the in-mouth memory)."""
        now = time.monotonic() if now is None else now
        w, h = frame_size
        out = []
        owner = {}                                   # hazard index -> nearest dog track_id
        for i, hz in enumerate(hazards):
            best = min(dogs, key=lambda dd: box_distance(dd.box, hz.box), default=None)
            if best is not None:
                owner[i] = best.track_id
        for d in dogs:
            tid = d.track_id
            tr = self.tracks.setdefault(tid, _Track(candidate_since=now, calm_since=now))
            tr.last_seen = now
            raw, reasons = 0, []

            # --- zones (use the dog's feet = bottom-center of the box) ---
            feet = ((d.box[0] + d.box[2]) / 2, d.box[3])
            zone_hit = None
            for z in zones:
                if z.contains(feet, w, h):
                    zone_hit = z
                    break
            for k in list(tr.zone_since):
                if zone_hit is None or k != zone_hit.name:
                    tr.zone_since.pop(k)
            on_bed = False
            if zone_hit:
                since = tr.zone_since.setdefault(zone_hit.name, now)
                if zone_hit.type == "bed":
                    on_bed = True
                elif now - since >= self.zone_dwell:
                    lvl = ZONE_LEVEL.get(zone_hit.type, 1)
                else:
                    lvl = 1                                   # just entered: watch
                if zone_hit.type != "bed":
                    raw = max(raw, lvl)
                    reasons.append((lvl, ZONE_TEXT.get(zone_hit.type, "in a zone")))

            # --- hazards near the dog ---
            near = []
            for i, hz in enumerate(hazards):
                if owner.get(i) == tid and box_distance(d.box, hz.box) <= self.near_px:
                    near.append(hz)
                    if hazards_fresh:
                        last, n, _ = tr.near_memory.get(hz.name, (0.0, 0, None))
                        tr.near_memory[hz.name] = (now, n + 1, hz.center)
            for hz in near:
                lvl = TIER_LEVEL.get(hz.tier, 1)
                raw = max(raw, lvl)
                reasons.append((lvl, f"near a {hz.name}"))

            # --- "possibly in mouth": a hazard seen near the dog twice+ just vanished ---
            in_mouth = None
            near_names = {hz.name for hz in near}
            all_names = {hz.name for hz in hazards}
            for name, (last, n, _c) in list(tr.near_memory.items()):
                age = now - last
                if age > self.memory:
                    tr.near_memory.pop(name)
                    continue
                if hazards_fresh and name not in all_names and n >= 3 and age > 0:
                    tier = self._tier_of(name, hazards)
                    if tier >= 2:
                        in_mouth = name
                        lvl = 3 if tier == 3 else 2
                        raw = max(raw, lvl)
                        reasons.append((lvl, f"the {name} disappeared near its mouth"))
            # --- actions (Batch 3 hook) ---
            act = (actions or {}).get(tid)
            if act:
                label, conf = act
                if label in ("chewing something", "eating") and (near or zone_hit):
                    raw = max(raw, 2)
                    reasons.append((2, label.replace(" something", "")))
                elif label in ("digging", "scratching furniture", "jumping on furniture"):
                    raw = max(raw, 2)
                    reasons.append((2, label))
                elif label == "sniffing the floor":
                    raw = max(raw, 1)
                    reasons.append((1, "sniffing around"))

            if on_bed and raw < 3:
                raw = min(raw, 1)                             # bed biases to safe unless danger
            level = self._persist(tr, raw, now)
            if reasons:
                reasons.sort(key=lambda r: -r[0])
                reason = reasons[0][1]
            else:
                reason = "resting on the bed" if on_bed else "all calm"
            out.append(Assessment(tid, level, raw, reason, zone_hit.name if zone_hit else None,
                                  sorted(near_names), in_mouth))

        # forget dogs not seen for 10 s
        for tid in [t for t, tr in self.tracks.items() if now - tr.last_seen > 10]:
            self.tracks.pop(tid)
        return out

    def _tier_of(self, name: str, hazards) -> int:
        for hz in hazards:
            if hz.name == name:
                return hz.tier
        return self.vocab_tiers.get(name, 1)

    def _persist(self, tr: _Track, raw: int, now: float) -> int:
        if raw > tr.level:
            if raw != tr.candidate:
                tr.candidate, tr.candidate_since = raw, now
            # danger escalates faster (half the persistence time)
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
        """Return the top Assessment to alert on (warning/danger) respecting cooldowns, else None."""
        now = time.monotonic() if now is None else now
        if not assessments:
            return None
        top = max(assessments, key=lambda a: a.level)
        if top.level < 2:
            return None
        if now - self._last_alert[top.level] < self.cooldown[top.level]:
            return None
        self._last_alert[top.level] = now
        if top.level == 3:
            self._last_alert[2] = now            # a danger alert also resets the warning cooldown
        return top
