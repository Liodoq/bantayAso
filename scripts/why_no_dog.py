"""Why isn't a dog detected? Shows everything the detector sees in one picture or video frame.

    python scripts\\why_no_dog.py data\\snapshots\\snap_x.jpg
    python scripts\\why_no_dog.py data\\clips\\rec_x.mp4 --time 12.5

It runs YOLO with a very low threshold on (1) the frame as-is, (2) the brightened frame Bantay now
uses when the room is dark, and (3) a bigger input size, and lists every object it guesses with its
class and confidence. Typical answers:
- "dog 0.12"           -> it sees the dog but below detect.dog_conf (config.yaml); lower it a little
- "cat/sheep 0.4"      -> sleeping-dog look-alike; Bantay now counts these as a dog above alias_conf
- only with brighten   -> the room is too dark; more light or keep detect.enhance_dark: true
- nothing at all       -> the dog is too small/blended in; move the camera closer or add light
Saves <input>_why.jpg with the boxes drawn so you can see what each guess covers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso import config                                      # noqa: E402
from bantayaso.detect_dog import ALIAS_CLASSES, DOG_CLASS, enhance_dark   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--time", type=float, default=0.0, help="seconds into a video")
    a = ap.parse_args()
    p = Path(a.path)
    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        frame = cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_MSEC, a.time * 1000)
        ok, frame = cap.read()
        cap.release()
        frame = frame if ok else None
    if frame is None:
        sys.exit(f"Can't read a frame from {p}")
    import torch
    from ultralytics import YOLO
    cfg = config.load()
    det = cfg.get("detect", {})
    model = YOLO(str(config.MODELS_DIR / cfg["models"]["dog_detector"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = model.names
    gray = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    print(f"frame {frame.shape[1]}x{frame.shape[0]}, brightness {gray:.0f}/255"
          f"{'  (dark: Bantay brightens frames under 80)' if gray < 80 else ''}")
    print(f"settings: dog_conf {det.get('dog_conf', 0.25)}, alias_conf {det.get('alias_conf', 0.30)}, "
          f"imgsz {det.get('imgsz', 960)}\n")
    draw = frame.copy()
    for title, img, size in (("as-is", frame, det.get("imgsz", 960)),
                             ("brightened", enhance_dark(frame, 256), det.get("imgsz", 960)),
                             ("bigger input", frame, 1280)):
        r = model.predict(img, conf=0.05, imgsz=size, device=dev, verbose=False)[0]
        rows = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            animal = k == DOG_CLASS or k in ALIAS_CLASSES
            rows.append((float(c), names[int(k)], tuple(int(v) for v in b), animal))
            if title == "as-is" or animal:
                cv2.rectangle(draw, rows[-1][2][:2], rows[-1][2][2:], (0, 165, 255) if animal else (160, 160, 160), 2)
                cv2.putText(draw, f"{names[int(k)]} {c:.2f}", (rows[-1][2][0], max(12, rows[-1][2][1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255) if animal else (200, 200, 200), 1)
        rows.sort(reverse=True)
        print(f"[{title}]")
        for c, n, b, animal in rows[:12]:
            used = ("USED as dog" if (n == "dog" and c >= det.get("dog_conf", 0.25)) or
                    (animal and n != "dog" and c >= det.get("alias_conf", 0.30)) else
                    "too weak" if animal else "")
            print(f"  {n:<14}{c:5.2f}  box {b}  {used}")
        if not rows:
            print("  (nothing above 0.05)")
    out = p.with_name(p.stem + "_why.jpg")
    cv2.imwrite(str(out), draw)
    print(f"\nBoxes drawn in {out}")


if __name__ == "__main__":
    main()
