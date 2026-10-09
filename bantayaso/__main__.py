"""BantayAso - Batch 1 live view.

  python -m bantayaso                       # action camera from config.yaml
  python -m bantayaso --source 0            # another USB device index
  python -m bantayaso --source data\\clips\\rec_x.mp4   # test on a recorded clip

Keys (click the video window first):
  R  start/stop recording a clip      C  save the last 10 seconds
  S  save a snapshot                  D  show/hide debug info (FPS, device)
  Q  quit
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from . import config, overlay
from .capture import ClipRecorder, FrameSource

WINDOW = "BantayAso"


def parse_args():
    p = argparse.ArgumentParser(prog="bantayaso")
    p.add_argument("--source", help="USB index (e.g. 1) or path to a video file")
    p.add_argument("--no-detect", action="store_true", help="camera only, no AI (for testing)")
    p.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = config.load()
    config.ensure_dirs()
    cam = cfg.get("camera", {})
    det_cfg = cfg.get("detect", {})
    device = config.resolve_device(cfg)
    print(f"[BantayAso] inference device: {device}")

    source = args.source if args.source is not None else cam.get("index", 0)
    src = FrameSource(source, backend=cam.get("backend", "msmf"),
                      width=cam.get("width", 1280), height=cam.get("height", 720),
                      fps=cam.get("fps", 30)).start()
    recorder = ClipRecorder(config.DATA_DIR / "clips",
                            seconds=cfg.get("capture", {}).get("buffer_seconds", 10),
                            fps=cfg.get("capture", {}).get("record_fps", 15))

    detector = None
    if not args.no_detect:
        from .detect_dog import DogDetector
        print("[BantayAso] loading dog detector...")
        detector = DogDetector(config.MODELS_DIR / cfg["models"]["dog_detector"], device=device,
                               conf=det_cfg.get("dog_conf", 0.35), imgsz=det_cfg.get("imgsz", 640))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    show_debug = args.debug
    last_id = 0
    fps, frames, t_fps = 0.0, 0, time.perf_counter()
    infer_ms = 0.0
    toast, toast_until = "", 0.0
    last_seen = 0.0

    def notify(msg: str):
        nonlocal toast, toast_until
        toast, toast_until = msg, time.monotonic() + 2.5
        print(f"[BantayAso] {msg}")

    try:
        while True:
            fid, frame = src.read()
            if frame is None or fid == last_id:
                if not src.connected and frame is None:
                    blank = np.full((720, 1280, 3), overlay.C["bg"], dtype=np.uint8)
                    overlay.draw_status(blank, "Camera disconnected - reconnecting...", "warning")
                    cv2.imshow(WINDOW, blank)
                if (cv2.waitKey(5) & 0xFF) in (ord("q"), ord("Q")):
                    break
                continue
            last_id = fid
            recorder.push(frame)                     # clean frame (no boxes) for test clips

            view = frame.copy()
            if detector is not None:
                t0 = time.perf_counter()
                dogs = detector(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                if dogs:
                    last_seen = time.monotonic()
                    overlay.draw_dogs(view, dogs)
                    overlay.draw_status(view, "Watching your dog", "safe")
                else:
                    gone = time.monotonic() - last_seen if last_seen else None
                    msg = "No dog in view" if gone is None or gone > 3 else "Watching your dog"
                    overlay.draw_status(view, msg, "watch" if msg == "No dog in view" else "safe")

            overlay.draw_recording(view, recorder.is_recording)
            frames += 1
            now = time.perf_counter()
            if now - t_fps >= 1.0:
                fps, frames, t_fps = frames / (now - t_fps), 0, now
            if show_debug:
                overlay.draw_debug(view, [f"FPS {fps:.1f}", f"detect {infer_ms:.0f} ms",
                                          f"device {device}", f"source {source}"])
            if toast and time.monotonic() < toast_until:
                overlay.draw_toast(view, toast)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key in (ord("d"), ord("D")):
                show_debug = not show_debug
            elif key in (ord("r"), ord("R")):
                path = recorder.toggle((frame.shape[1], frame.shape[0]))
                notify(f"Recording... {path.name}" if recorder.is_recording else f"Saved {path.name}")
            elif key in (ord("c"), ord("C")):
                path = recorder.save_last()
                notify(f"Saved last clip: {path.name}" if path else "Nothing to save yet")
            elif key in (ord("s"), ord("S")):
                path = config.DATA_DIR / "snapshots" / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)
                notify(f"Snapshot {path.name}")
    finally:
        recorder.close()
        src.stop()
        cv2.destroyAllWindows()
        print(f"[BantayAso] stopped. Last measured FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
