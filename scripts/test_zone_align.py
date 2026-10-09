"""Checks that zones follow a bumped / rotated / zoomed camera (synthetic room picture, no models).

    python scripts\\test_zone_align.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso.zone_align import ZoneAligner   # noqa: E402

fails = 0


def check(name, cond, info=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {info}" if info else ""))
    fails += 0 if cond else 1


rng = np.random.default_rng(1)
room = np.full((720, 1280, 3), 90, np.uint8)
for _ in range(260):                                   # a cluttered "room" with plenty of corners
    x, y = int(rng.integers(0, 1280)), int(rng.integers(0, 720))
    col = tuple(int(c) for c in rng.integers(0, 255, 3))
    if rng.random() < 0.5:
        cv2.rectangle(room, (x, y), (x + int(rng.integers(10, 90)), y + int(rng.integers(10, 90))), col, -1)
    else:
        cv2.circle(room, (x, y), int(rng.integers(5, 40)), col, -1)
d = Path(tempfile.mkdtemp())
a = ZoneAligner(d / "ref.png")
check("no reference yet", not a.has_reference)
a.set_reference(room)
check("reference saved to disk", (d / "ref.png").exists())
zone = [[.2, .55], [.8, .5], [.9, .9], [.15, .92]]
for ang, sc, tx, ty in ((4, 1.05, 40, -25), (-8, 0.95, -60, 30), (0, 1.0, 80, 0)):
    M = cv2.getRotationMatrix2D((640, 360), ang, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    a.M = np.array([[1, 0, 0], [0, 1, 0]], float)
    a.status = "aligned"
    st = a.update(cv2.warpAffine(room, M, (1280, 720)), now=a._last + 10)
    true = np.clip(((np.float64(zone) * [1280, 720]) @ M[:, :2].T + M[:, 2]) / [1280, 720], 0, 1)
    err = float(np.abs(np.array(a.apply(zone)) - true).max())
    check(f"camera rotated {ang} deg, zoom {sc}, shifted ({tx},{ty}) -> zone follows", st == "moved" and err < 0.01,
          f"error {err:.4f}")
a.status = "aligned"
check("same view -> no change (no jitter)", a.update(room, now=a._last + 10) == "aligned" and a.shift == 0.0)
check("completely different view -> 'lost'",
      a.update(rng.integers(0, 255, (720, 1280, 3), np.uint8), now=a._last + 10) == "lost")
b = ZoneAligner(d / "ref.png")
check("reference reloads after restart", b.has_reference)

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
