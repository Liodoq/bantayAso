"""Yes/no questions lead with Yes / No / Not sure and then say why. Also: dog names Whisper mis-hears.

    python scripts\\test_yesno.py
"""
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso.events import EventLog            # noqa: E402
from bantayaso.history import ActivityHistory    # noqa: E402
from bantayaso.qa import answer                  # noqa: E402
from bantayaso.risk import Assessment            # noqa: E402
from bantayaso.voice_in import correct_names     # noqa: E402

fails = 0


def check(name, cond, info=""):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  -> {info}" if info else ""))
    fails += 0 if cond else 1


def make(level_choco=0, reason_choco="just lying down", events=()):
    d = Path(tempfile.mkdtemp())
    hist = ActivityHistory()
    now = time.monotonic()
    for i in range(240):                        # 4 minutes: Pachuchay ate for 60 s, then lay down; Oreo slept
        hist.record(now - 240 + i, [Assessment(1, 0, 0, "eating" if 60 <= i < 120 else "just lying down", zone="Bed 1"),
                                    Assessment(2, 0, 0, "sleeping")], {1: "Pachuchay", 2: "Oreo"})
    log = EventLog(d / "e.db", d / "s")
    for lvl, r in events:
        log.add(Assessment(1, lvl, lvl, r, "Bed 1"), None, "Pachuchay")
    st = types.SimpleNamespace(status="", level=level_choco, dogs=2,
                               assessments=[Assessment(1, level_choco, level_choco, reason_choco, "Bed 1"),
                                            Assessment(2, 0, 0, "sleeping")],
                               boxes=[(1, (100, 300, 300, 500), "Pachuchay"), (2, (900, 300, 1100, 500), "Oreo")],
                               frame_w=1280)
    reg = types.SimpleNamespace(names_by_tid={1: "Pachuchay", 2: "Oreo"}, dogs={"Pachuchay": [], "Oreo": []})
    return types.SimpleNamespace(registry=reg, state=st, dog_name="your dog", history=hist, events=log,
                                 zones=[types.SimpleNamespace(name="Bed 1")], ask_llm=lambda q, f: None)


p = make()
a = answer("Bantay, is Oreo sleeping?", p)
check("is X sleeping (yes) -> 'Yes.' + why", a.startswith("Yes.") and "sleeping" in a, a)
a = answer("Bantay, is Pachuchay eating?", p)
check("is X eating (no) -> 'No.' + what instead + when it last did", a.startswith("No.") and "lying" in a
      and "eating" in a and "ago" in a, a)
a = answer("Bantay, did Pachuchay eat?", p)
check("did X eat -> 'Yes.' + when + how long", a.startswith("Yes.") and "ago" in a and "about" in a, a)
a = answer("Bantay, did Oreo eat anything?", p)
check("did X eat (no) -> 'No.' + what instead", a.startswith("No.") and "sleeping" in a, a)
a = answer("Bantay, is Pachuchay on the bed?", p)
check("is X on the bed -> 'Yes.' + zone", a.startswith("Yes.") and "Bed 1" in a, a)
a = answer("Bantay, is everything okay?", p)
check("is everything okay (calm) -> 'Yes.' + why", a.startswith("Yes.") and "calm" in a.lower(), a)
a = answer("Bantay, should I check on them?", p)
check("should I check (calm) -> 'No.' + why", a.startswith("No.") and "calm" in a.lower(), a)
a = answer("Bantay, natutulog ba si Oreo?", p)
check("Tagalog 'natutulog ba si Oreo' -> 'Yes.'", a.startswith("Yes."), a)

p = make(2, "chewing an unknown object - check what it is", [(2, "chewing an unknown object - check what it is")])
a = answer("Bantay, is everything okay?", p)
check("is everything okay (warning now) -> 'No.' + the reason", a.startswith("No.") and "chewing" in a, a)
a = answer("Bantay, should I check on them?", p)
check("should I check (warning now) -> 'Yes.' + the reason", a.startswith("Yes.") and "chewing" in a, a)
a = answer("Bantay, is Pachuchay eating?", p)
check("chewing counts as eating -> 'Yes.'", a.startswith("Yes."), a)

p = make(events=[(2, "chewing an unknown object - check what it is")])
a = answer("Bantay, is everything okay?", p)
check("calm now but a recent warning -> 'Yes, for now.' + when", a.startswith("Yes, for now.") and "warning" in a, a)

p = make()
p.state.assessments = [p.state.assessments[1]]
p.state.boxes = [p.state.boxes[1]]
a = answer("Bantay, is Pachuchay sleeping?", p)
check("dog not on camera -> 'Not sure.' + why", a.startswith("Not sure.") and "don't see" in a, a)

# app knowledge: who the dogs are, behaviours, things
p = make()
p.registry.dogs["Brownie"] = []
p.cfg = {"hazards": {"battery": 3, "slipper": 2}, "camera": {"index": 1}}
p.hazard_det = None
from bantayaso.risk import RiskEngine   # noqa: E402
p.engine = RiskEngine({"behaviors": {}})
a = answer("Bantay, who are the dogs?", p)
check("who are the dogs -> registered names + who is on camera", "Pachuchay" in a and "Oreo" in a and "Brownie" in a
      and "Not in view: Brownie" in a, a)
a = answer("Bantay, who is Brownie?", p)
check("who is <name> (away) -> registered but not in view", "registered" in a and "don't see" in a, a)
a = answer("Bantay, is a battery dangerous?", p)
check("is a battery dangerous -> 'Yes.' + Things level", a.startswith("Yes.") and "high danger" in a, a)
a = answer("Bantay, how dangerous is digging?", p)
check("behaviour level", "Digging: Warning" in a, a)

names = ["Pachuchay", "Oreo", "Brownie"]
for heard, want in (("Bantay, is Patchouchai sleeping?", "Bantay, is Pachuchay sleeping?"),
                    ("Bantay, where is pa choo chay", "Bantay, where is Pachuchay"),
                    ("Is Pacho Chay okay", "Is Pachuchay okay"),
                    ("Patutsay is sleeping", "Pachuchay is sleeping"),
                    ("Bantay, is Brownee awake?", "Bantay, is Brownie awake?"),
                    ("Bantay, are they okay?", "Bantay, are they okay?"),
                    ("Is Patrick sleeping", "Is Patrick sleeping"),
                    ("is the brown one eating", "is the brown one eating")):
    got = correct_names(heard, names)
    check(f"name fix: {heard!r}", got == want, got)

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
