"""Teach Bantay an action from a recorded video, or test it on a video it has never seen.

Teach (adds examples to data/actions/<action>.npy, same place as live teaching):
    python scripts\\teach_from_video.py data\\clips\\rec_x.mp4 --label sitting --start 4 --end 12
Test on a DIFFERENT clip (nothing is saved):
    python scripts\\teach_from_video.py data\\clips\\rec_y.mp4 --label sitting --start 0 --end 8 --test

Why a video is not "the same frame 100 times":
- frames are sampled every --every seconds (default 0.75 s), and
- a frame whose dog looks almost identical (cosine >= --dedupe, default 0.97) to one already
  taken is skipped, so 30 s of a dog lying perfectly still gives a handful of examples, not 40.
Variety (other days, lighting, angles, each of your dogs) is what improves recognition; many
copies of one moment do not. Keep some clips OUT of teaching and use --test on them to see
whether it really got better.

CLOSE BANTAY FIRST: the running app keeps its own copy of the examples in memory and would
overwrite this file the next time you teach live.

Only the segment you give must show the action, and only one dog should be doing it
(the biggest dog box in the frame is used; use --dog left/right to pick another).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso import config                                   # noqa: E402
from bantayaso.examples import reference_gates, save_array    # noqa: E402


def pick(dogs, mode):
    if not dogs:
        return None
    if mode == "left":
        return min(dogs, key=lambda d: d.box[0])
    if mode == "right":
        return max(dogs, key=lambda d: d.box[2])
    return max(dogs, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--label", required=True, help="action name exactly as in config.yaml 'actions'")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds (default: end of video)")
    ap.add_argument("--every", type=float, default=0.75, help="sample one frame every N seconds")
    ap.add_argument("--max", type=int, default=20, help="most examples to add from this clip")
    ap.add_argument("--dedupe", type=float, default=0.97, help="skip frames this similar to one already taken")
    ap.add_argument("--dog", choices=["biggest", "left", "right"], default="biggest")
    ap.add_argument("--test", action="store_true", help="only report what Bantay predicts; save nothing")
    a = ap.parse_args()

    cfg = config.load()
    labels = cfg.get("actions") or []
    if a.label not in labels:
        sys.exit(f"'{a.label}' is not an action. Choose one of: {', '.join(labels)}")
    import torch
    from bantayaso.actions import ActionClassifier
    from bantayaso.detect_dog import DogDetector
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = cfg.get("detect", {})
    dogs_m = DogDetector(config.MODELS_DIR / cfg["models"]["dog_detector"], device=dev,
                         conf=det.get("dog_conf", 0.25), imgsz=det.get("imgsz", 960))
    clf = ActionClassifier(labels, config.MODELS_DIR, device=dev,
                           model_name=cfg["models"].get("clip_openai_name", "ViT-B/32"))

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Can't open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None
    end = a.end if a.end is not None else (total or 1e9)
    t, feats, preds, skipped_same, no_dog = a.start, [], Counter(), 0, 0
    while t <= end and (a.test or len(feats) < a.max):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        t += a.every
        d = pick(dogs_m(frame), a.dog)
        if d is None:
            no_dog += 1
            continue
        clf._ema.clear()                       # judge each frame on its own (no smoothing)
        res = clf(frame, [d], collect=False).get(d.track_id)
        if res is not None:
            preds[res.label] += 1
        f = clf.embeddings.get(d.track_id)
        if f is None or not np.isfinite(f).all():
            continue
        f = f / np.linalg.norm(f)
        if feats and max(float(f @ g) for g in feats) >= a.dedupe:
            skipped_same += 1
            continue
        feats.append(f.astype(np.float32))
    cap.release()

    n = sum(preds.values())
    print(f"\n{a.video}  [{a.start:.1f}s - {min(end, total or end):.1f}s], one frame every {a.every}s")
    print(f"dog found in {n} frames, no dog in {no_dog}; {len(feats)} distinct looks, {skipped_same} near-duplicates skipped")
    if n:
        top = ", ".join(f"{k} {v * 100 // n}%" for k, v in preds.most_common(4))
        print(f"Bantay currently says: {top}")
        print(f"'{a.label}' recognised in {preds[a.label] * 100 // n}% of frames "
              f"(development check on this clip only, not an accuracy claim)")
    if a.test:
        return
    if not feats:
        sys.exit("Nothing to save.")
    path = clf.examples_dir / f"{a.label.replace(' ', '_')}.npy"
    old = clf.examples.get(a.label)
    rows = np.vstack([old, np.asarray(feats)]) if old is not None else np.asarray(feats)
    if len(rows) > 80:
        print(f"Note: keeping the newest 80 of {len(rows)} examples for '{a.label}'.")
        rows = rows[-80:]
    save_array(path, rows.astype(np.float32))
    ex = dict(clf.examples)
    ex[a.label] = rows
    gates = reference_gates(ex)
    state = ("ACTIVE" if a.label in gates else
             "inactive - teach at least one other action too" if len(ex) < 2 else
             "OFF - some examples look like another action (run scripts\\check_examples.py)")
    print(f"Saved {len(feats)} new examples -> {path.name} ({len(rows)} total). Status: {state}")
    print("Start Bantay again to use them (keep it closed while teaching from videos).")


if __name__ == "__main__":
    main()
