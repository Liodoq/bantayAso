"""Rolling 10-minute activity memory (one sample per dog per second) for "Ask Bantay"."""
from __future__ import annotations

import collections
import time

CATEGORIES = [            # (keyword in reason, spoken activity) - first match wins
    ("disappeared", "possibly swallowed something"), ("chewing", "chewing something"),
    ("eating", "eating something"), ("nose-down", "eating something"), ("nosing", "nosing a hazard"),
    ("sleeping", "sleeping"), ("lying", "lying down"), ("sitting", "sitting"),
    ("standing", "standing"), ("walking", "walking around"), ("licking", "licking itself"),
    ("scratching itself", "scratching itself"), ("sniffing", "sniffing around"),
    ("digging", "digging"), ("scratching furniture", "scratching furniture"),
    ("jumping", "jumping on furniture"), ("trash", "at the trash"), ("no-go", "in a no-go area"),
    ("danger zone", "in the danger zone"), ("near", "near something risky"), ("resting", "resting"),
]


def category(reason: str) -> str:
    r = reason.lower()
    for key, name in CATEGORIES:
        if key in r:
            return name
    return "moving around"


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} seconds"
    m, s = divmod(sec, 60)
    return f"{m} minute{'s' if m > 1 else ''}" + (f" {s} seconds" if s >= 10 else "")


class ActivityHistory:
    def __init__(self, max_minutes: int = 10):
        self.max_s = max_minutes * 60
        self.samples: collections.deque = collections.deque()   # (t, who, activity, level, zone)
        self.counts: collections.deque = collections.deque()    # (t, number of dogs)
        self._last = 0.0

    def record(self, now: float, assessments, names: dict) -> None:
        if now - self._last < 1.0:
            return
        self._last = now
        self.counts.append((now, len(assessments)))
        for a in assessments:
            who = names.get(a.track_id) or "unnamed"
            self.samples.append((now, who, category(a.reason), a.level, a.zone))
        while self.samples and now - self.samples[0][0] > self.max_s:
            self.samples.popleft()
        while self.counts and now - self.counts[0][0] > self.max_s:
            self.counts.popleft()

    def summary(self, minutes: int, now: float | None = None) -> dict:
        now = now or time.monotonic()
        win = minutes * 60
        rows = [s for s in self.samples if now - s[0] <= win]
        per = collections.defaultdict(collections.Counter)
        for _t, who, act, _lvl, _z in rows:
            per[who][act] += 1
        seen = [c for t, c in self.counts if now - t <= win]
        covered = (now - self.counts[0][0]) if self.counts else 0
        return {"minutes": minutes, "per_dog": {k: dict(v) for k, v in per.items()},
                "max_dogs": max(seen, default=0), "covered_s": min(covered, win)}

    def describe(self, minutes: int, default_name: str = "your dog") -> str:
        s = self.summary(minutes)
        if not s["per_dog"]:
            return f"I haven't seen any dog in the last {fmt_dur(minutes * 60)}."
        parts = []
        multi = len(s["per_dog"]) > 1 or s["max_dogs"] > 1
        for who, acts in sorted(s["per_dog"].items(), key=lambda kv: kv[0] == "unnamed"):
            name = (("your other dogs" if len(s["per_dog"]) > 1 else
                     "your dogs" if multi else default_name) if who == "unnamed" else who)
            total = sum(acts.values())
            top = sorted(acts.items(), key=lambda kv: -kv[1])
            main, sec = top[0]
            if sec >= 0.85 * total:
                parts.append(f"{name} {'were' if name.endswith('dogs') else 'was'} {main} the whole time")
                continue
            bits = [f"{main} for about {fmt_dur(sec / max(1, s['max_dogs'] if who == 'unnamed' else 1))}"]
            for act, n in top[1:3]:
                if n >= 3:
                    bits.append(f"{act} for about {fmt_dur(n / max(1, s['max_dogs'] if who == 'unnamed' else 1))}")
            parts.append(f"{name} {'were' if name.endswith('dogs') else 'was'} " + ", then ".join(bits))
        span = "minute" if minutes == 1 else f"{minutes} minutes"
        note = "" if s["covered_s"] >= minutes * 60 - 5 else f" (I've only been watching for {fmt_dur(s['covered_s'])})"
        return f"In the last {span}{note}, " + ". ".join(parts) + "."
