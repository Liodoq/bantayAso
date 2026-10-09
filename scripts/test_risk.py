"""Offline tests for the risk engine (no camera, no GPU).  Run: python scripts/test_risk.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso.detect_dog import Dog            # noqa: E402
from bantayaso.detect_hazards import Hazard     # noqa: E402
from bantayaso.risk import RiskEngine           # noqa: E402
from bantayaso.zones import Zone                # noqa: E402
from bantayaso.actions import ActionResult      # noqa: E402

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


def run(engine, frames, hz, zones, t0=0.0, dt=0.1, fresh=True, act=None):
    a = None
    acts = {1: act} if act is not None else None
    for i in range(frames):
        a = engine.update([dog], hz, zones, SIZE, now=t0 + i * dt, hazards_fresh=fresh, actions=acts)[0]
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

# ---- Batch 3: actions (values taken from the user's real calibration clip) ----
def A(label, conf, mouth, chew, motion="active"):
    return ActionResult(label, conf, mouth, chew, motion)

cases = [
    ("sleeping dog is safe and says so", A("sleeping", .35, .13, .19), 20, [], [], 0, "sleeping"),
    ("sitting dog says 'just sitting'", A("sitting", .34, .19, .19), 20, [], [], 0, "just sitting"),
    ("lying dog says 'just lying down'", A("lying down", .4, .15, .2), 20, [], [], 0, "just lying down"),
    ("scratching itself (score .18) is calm", A("scratching itself", .41, .23, .16), 60, [], [], 0, "scratching itself"),
    ("real licking (score .30) is calm", A("licking itself", .35, .3, .3), 60, [], [], 0, "licking itself"),
    ("eating labelled 'licking itself' (score .48) -> warning", A("licking itself", .25, .46, .5), 40, [], [], 2, "unknown object"),
    ("eating labelled 'eating' (score .44) -> warning", A("eating", .3, .45, .44), 40, [], [], 2, "unknown object"),
    ("brief eating (<2 s) does not warn yet", A("eating", .3, .45, .44), 15, [], [], 0, None),
    ("eating over 10 s -> danger", A("eating", .3, .45, .44), 120, [], [], 3, "chewing for"),
    ("eating near a battery -> danger", A("eating", .3, .45, .44), 40, [battery], [], 3, "battery"),
    ("sleeping beside a battery -> warning only", A("sleeping", .35, .13, .19), 20, [battery], [], 2, "battery"),
    ("sleeping beside a slipper -> watch only", A("sleeping", .35, .13, .19), 20, [slipper], [], 1, "slipper"),
    ("eating on the bed still warns (comb case)", A("eating", .3, .45, .44), 40, [], [bed], 2, None),
    ("frantic digging -> warning", A("digging", .4, .2, .2, "frantic"), 20, [], [], 2, "digging"),
    ("nose down sniffing in one spot 5 s+ -> warning", A("sniffing the floor", .4, .25, .33), 60, [], [], 2, "nose-down"),
    ("short sniff (2.5 s) -> only watch", A("sniffing the floor", .4, .25, .33), 25, [], [], 1, "sniffing"),
]
for name, act, frames, hz, zs, want, text in cases:
    e = RiskEngine(CFG); a, _ = run(e, frames, hz, zs, act=act)
    check(name, a.level == want and (text is None or text in a.reason), f"level {a.level}: {a.reason}")

# tracker ID switch mid-chew: state carries over to the new ID
e = RiskEngine(CFG); eat = A("eating", .3, .45, .44)
for i in range(15):
    e.update([dog], [], [], SIZE, now=i * 0.1, actions={1: eat})
dog7 = Dog(7, (505, 302, 705, 502), 0.9)
for i in range(15, 40):
    a = e.update([dog7], [], [], SIZE, now=i * 0.1, actions={7: eat})[0]
check("tracker ID switch keeps the chewing timer (warns on time)", a.level == 2, a.reason)

# named zones: food bowl / play area / named no-go
bowl = Zone("Food bowl", "food", [[0.35, 0.6], [0.6, 0.6], [0.6, 0.8], [0.35, 0.8]])
sofa = Zone("Sofa", "nogo", [[0.35, 0.6], [0.6, 0.6], [0.6, 0.8], [0.35, 0.8]])
e = RiskEngine(CFG); a, _ = run(e, 120, [], [bowl], act=A("eating", .3, .45, .44))
check("eating at the food bowl stays calm (even 12 s)", a.level == 0 and "food bowl" in a.reason, a.reason)
e = RiskEngine(CFG); a, _ = run(e, 40, [battery], [bowl], act=A("eating", .3, .45, .44))
check("battery at the food bowl is still danger", a.level == 3, a.reason)
e = RiskEngine(CFG); a, _ = run(e, 30, [], [sofa], act=A("sitting", .4, .2, .2))
check("named no-go zone is spoken by name", a.level == 2 and "Sofa" in a.reason, a.reason)

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
