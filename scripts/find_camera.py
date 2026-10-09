"""Find the action camera (USB webcam mode) and save its index to config.yaml.

Run:  python scripts/find_camera.py
Plug the action camera in and switch it to webcam mode first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso import config  # noqa: E402

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


def probe(max_index: int = 6):
    found = []
    for name, be in BACKENDS.items():
        for i in range(max_index):
            cap = cv2.VideoCapture(i, be)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    found.append((i, name, frame.shape[1], frame.shape[0]))
            cap.release()
    return found


def preview(index: int, backend: str, cfg: dict) -> None:
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    print("Preview open. Press Q to close.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost frames.")
            break
        cv2.putText(frame, f"index {index} / {backend}  {frame.shape[1]}x{frame.shape[0]}  (Q to close)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("BantayAso camera test", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    cfg = config.load()
    print("Probing cameras... (the laptop webcam is usually index 0)")
    found = probe()
    if not found:
        print("No camera found. Check the USB cable and that the camera is in webcam mode.")
        return
    for n, (i, be, w, h) in enumerate(found):
        print(f"  [{n}] index {i}  backend {be}  {w}x{h}")
    choice = input("Pick the action camera number to preview (Enter = cancel): ").strip()
    if not choice.isdigit() or int(choice) >= len(found):
        return
    i, be, _, _ = found[int(choice)]
    preview(i, be, cfg)
    if input(f"Save index {i} ({be}) to config.yaml? [y/N]: ").strip().lower() == "y":
        cfg["camera"]["index"] = i
        cfg["camera"]["backend"] = be
        config.save(cfg)
        print("Saved.")


if __name__ == "__main__":
    main()
