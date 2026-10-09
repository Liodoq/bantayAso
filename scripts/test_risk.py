"""Offline tests for the risk engine (no camera, no GPU).  Run: python scripts/test_risk.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso.detect_dog import Dog            # noqa: E402
from bantayaso.detect_hazards import Hazard     # noqa: E402
from bantayaso.risk import RiskEngine           # noqa: E402
from bantayaso.zones import Zone                # noqa: E402

CFG = {"risk": {"persist_seconds": 1.0, "calm_seconds": 2.0, "last_seen_memory_seconds": 5,
                "muzzle_near_px": 60, "zone_dwell_seconds": 1.5, "cooldown_warning": 20,
                "cooldown_danger": 5},
       "hazards": {"battery": 3, "slipper": 2, "toy": 1}}
SIZE = (1280, 720)
dog = Dog(1, (500, 300, 700, 500), 0.9)
battery = Hazard("battery", 3, (710, 470, 730, 490), 0.4)
slipper = Hazard("slipper", 2, (690, 480, 760, 520), 0.5)
trash = Zone("trash 1", "trash", [[0.35, 0.6], [0.6, 0.6], [0.6, 0.8], [0.35, 0.8]])
bed = Zone("bed 1", "bed", [[0.3, 0.5], [0.7, 0.5], [0.7, 0.9], [0.3, 0.9]])
fails = 0


def run(engine, frames, hz, zones, t0=0.0, dt=0.1, fresh=True):
    a = None
    for i in range(frames):
        a = engine.update([dog], hz, zones, SIZE, now=t0 + i * dt, hazards_fresh=fresh)[0]
    return a, t0 + frames * dt


def check(name, cond, info=""):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  ({info})" if info else ""))
    fails += 0 if cond else 1


e = RiskEngine(CFG); a, _ = run(e, 30, [], [])
check("calm dog is safe", a.level == 0, a.reason)

e = RiskEngine(CFG); a, _ = run(e, 3, [battery], [])
check("no instant flicker to danger", a.level == 0, f"raw={a.raw_level}")
a, _ = run(e, 10, [battery], [], t0=0.3)
check("battery near dog -> danger after persistence", a.level == 3, a.reason)

e = RiskEngine(CFG); a, _ = run(e, 15, [slipper], [])
check("slipper near dog -> warning", a.level == 2, a.reason)

e = RiskEngine(CFG); a, t = run(e, 30, [], [trash])
check("dwelling in trash zone -> warning", a.level == 2, a.reason)

e = RiskEngine(CFG); a, t = run(e, 25, [slipper], [bed])
check("on bed caps slipper at watch", a.level == 1, a.reason)

e = RiskEngine(CFG); a, t = run(e, 10, [battery], [])
a, t = run(e, 10, [], [], t0=t)
check("battery vanished near dog -> stays danger (in mouth)", a.level == 3 and a.in_mouth == "battery", a.reason)
a, t = run(e, 80, [], [], t0=t)
check("memory expires and level calms down", a.level == 0, a.reason)

e = RiskEngine(CFG); a, _ = run(e, 15, [battery], [])
check("first danger alert fires", e.should_alert([a], now=2.0) is not None)
check("danger cooldown blocks repeat", e.should_alert([a], now=3.0) is None)
check("danger re-alerts after cooldown", e.should_alert([a], now=8.0) is not None)

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
