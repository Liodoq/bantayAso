"""Measure what the action model sees on a recorded clip (no window needed).

  python scripts\\calibrate_actions.py data\\clips\\rec_XXXX.mp4

Prints, about twice a second, each dog's top-3 actions, mouth score and motion, then a summary.
Also writes the full log to data\\calibration_<clip>.csv. Paste the summary back to the AI chat.
"""
from __future__ import annotations

import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso import config  # noqa: E402
from bantayaso.actions import ActionClassifier  # noqa: E402
from bantayaso.detect_dog import DogDetector  # noqa: E402
from bantayaso.motion import MotionMeter  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    clip = Path(sys.argv[1])
    cfg = config.load()
    device = config.resolve_device(cfg)
    det_cfg = cfg.get("detect", {})
    dogs_m = DogDetector(config.MODELS_DIR / cfg["models"]["dog_detector"], device=device,
                         conf=det_cfg.get("dog_conf", 0.25), imgsz=det_cfg.get("imgsz", 960))
    clf = ActionClassifier(cfg.get("actions") or [], config.MODELS_DIR, device=device,
                           model_name=cfg["models"].get("clip_openai_name", "ViT-B/32"))
    mm = MotionMeter()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    step = max(1, int(round(fps / 4)))           # ~4 checks per second, like the live app
    rows, labels, mouths, chews = [], defaultdict(Counter), defaultdict(list), defaultdict(list)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dogs = dogs_m(frame)
        mot = mm.update(frame, dogs)
        if i % step == 0 and dogs:
            res = clf(frame, dogs)
            t = i / fps
            for d in dogs:
                r = res.get(d.track_id)
                if r is None:
                    continue
                e = clf._ema[d.track_id]
                top3 = [(clf.labels[j], float(e[j])) for j in np.argsort(-e)[:3]]
                m_lvl, m_e = mot.get(d.track_id, ("?", 0.0))
                labels[d.track_id][r.label] += 1
                mouths[d.track_id].append(r.mouth)
                chews[d.track_id].append(r.chew)
                rows.append([f"{t:.1f}", d.track_id, *[f"{l}:{p:.2f}" for l, p in top3],
                             f"{r.mouth:.2f}", f"{r.chew:.2f}", m_lvl, f"{m_e:.1f}"])
                if i % (step * 2) == 0:
                    print(f"{t:6.1f}s  dog#{d.track_id:<3} " +
                          "  ".join(f"{l} {p:.2f}" for l, p in top3) +
                          f"  | mouth {r.mouth:.2f} | chew {r.chew:.2f} | motion {m_lvl} {m_e:.1f}")
        i += 1
    out = config.DATA_DIR / f"calibration_{clip.stem}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["t", "dog", "top1", "top2", "top3", "mouth", "chew", "motion", "energy"], *rows])
    print("\n===== SUMMARY (paste this) =====")
    print(f"clip: {clip.name}")
    for tid in labels:
        m = np.array(mouths[tid])
        print(f"dog#{tid}: actions {dict(labels[tid].most_common(4))} | mouth mean {m.mean():.2f} "
              f"max {m.max():.2f} | chew mean {np.mean(chews[tid]):.2f} max {np.max(chews[tid]):.2f}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
