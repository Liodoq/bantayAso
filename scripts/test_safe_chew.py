"""'It's safe' on a chewing alert stops the alert and is remembered; 'I've got it' keeps reminding.

    python scripts\\test_safe_chew.py
"""
import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso.actions import ActionResult          # noqa: E402
from bantayaso.detect_dog import Dog                # noqa: E402
from bantayaso.detect_hazards import Hazard         # noqa: E402
from bantayaso.risk import RiskEngine               # noqa: E402
from bantayaso.safe_chews import SafeChews          # noqa: E402

fails = 0


def check(name, cond, info=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {info}" if info else ""))
    fails += 0 if cond else 1


CFG = {"risk": {"persist_seconds": 1.0, "calm_seconds": 2.0}, "hazards": {"battery": 3, "toy": 1}}
SIZE = (1280, 720)
dog = Dog(1, (500, 300, 700, 500), 0.9)
eat = ActionResult("eating", 0.3, 0.45, 0.44)


def run(e, frames, t0=0.0, hz=()):
    a = None
    for i in range(frames):
        a = e.update([dog], list(hz), [], SIZE, now=t0 + i * 0.1, actions={1: eat})[0]
    return a, t0 + frames * 0.1


e = RiskEngine(CFG)
a, t = run(e, 70)
check("chewing unknown object for 7 s -> warning", a.level == 2 and "chewing" in a.reason, a.reason)

d = Path(tempfile.mkdtemp())
safe = SafeChews(d / "safe.npy")
emb = {1: np.random.default_rng(0).standard_normal(512).astype(np.float32)}
e.safe_chew_fn = lambda tid: safe.is_safe(tid, emb.get(tid))
check("owner presses It's safe -> look saved", safe.add(1, emb[1]) and (d / "safe.npy").exists())
a, t = run(e, 60, t)
check("same chewing episode -> back to safe, no alert", a.raw_level == 0 and "safe" in a.reason, a.reason)
check("...and it does not alert", e.should_alert([a], now=t + 1000) is None)
a, t = run(e, 50, t, hz=[Hazard("battery", 3, (710, 470, 730, 490), 0.5)])
check("a battery near the mouth still alerts even after It's safe", a.raw_level >= 2, a.reason)

safe2 = SafeChews(d / "safe.npy")                       # app restarted: the look is remembered
check("saved looks reload after restart", len(safe2.rows) == 1)
check("similar look (new episode) counts as safe", safe2.is_safe(9, emb[1] + 0.01))
check("a different look does not", not safe2.is_safe(9, np.random.default_rng(5).standard_normal(512)))

# 'I've got it' (not safe): alerts keep coming while the chewing continues
e2 = RiskEngine(CFG)
a, t = run(e2, 70)
first = e2.should_alert([a], now=t)
a, t = run(e2, 250, t)
again = e2.should_alert([a], now=t)
check("not safe: the warning repeats while chewing continues (until the object is removed)",
      first is not None and again is not None, f"{a.reason}")

# UI: the buttons on the status card
from PySide6.QtWidgets import QApplication             # noqa: E402
app = QApplication.instance() or QApplication(sys.argv)
from bantayaso.ui import app as A                       # noqa: E402
from bantayaso.ui import theme as T                     # noqa: E402
T.apply("light")
A.Worker.start = lambda self, *a, **k: None
w = A.MainWindow(argparse.Namespace(source=None, config="config.yaml", video=None, camera=None,
                                    no_hazards=True, no_actions=True, no_vlm=True))
alert = {"id": 7, "level": 2, "text": "Warning! Oreo is chewing an unknown object.", "reason": "chewing an unknown object",
         "vlm": None, "ts": 0, "track_id": 1, "near": [], "chewing": True}
w._set_status_card(2, "Chewing an unknown object", "", alert)
vis = lambda: sorted(n for n, b in (("safe", w.btn_safe), ("ack", w.btn_ack), ("snooze", w.btn_snooze))   # noqa: E731
                     if not b.isHidden())
check("chewing alert -> It's safe + I've got it", vis() == ["ack", "safe"], vis())
w._set_status_card(3, "In the danger zone", "", dict(alert, chewing=False))
check("other alerts -> I've got it + Snooze", vis() == ["ack", "snooze"], vis())
said = []
pipe = types.SimpleNamespace(state=types.SimpleNamespace(last_alert=alert), voice=True, dnd=False,
                             speaker=types.SimpleNamespace(say=lambda t, **k: said.append(t)),
                             mark_chewing_safe=lambda ev: "Okay. I'll remember that this kind of chewing by Oreo is okay.")
w.worker.pipe = pipe
w.chewing_is_safe()
check("It's safe -> Bantay confirms out loud and on the card", said and "remember" in w.status_note.text(), said)
w.acknowledge()
check("I've got it on a chewing alert -> 'I'll keep reminding you'", "keep reminding" in w.status_note.text())

# by voice
from bantayaso.qa import answer                            # noqa: E402
marked = []
vp = types.SimpleNamespace(state=types.SimpleNamespace(last_alert=dict(alert, ts=__import__("time").time()), assessments=[],
                                                       boxes=[]), registry=None, dog_name="your dog",
                           mark_chewing_safe=lambda ev: marked.append(ev) or "Okay, I'll stop alerting for this chewing.")
check("'Bantay, it's safe' after a chewing alert -> marks it safe", answer("Bantay, it's safe", vp).startswith("Okay")
      and marked)
check("'Bantay, I've got it' -> keeps reminding", "keep reminding" in answer("Bantay, I've got it", vp))

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
