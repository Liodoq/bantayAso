"""BantayAso - Local AI dog watcher.

  python -m bantayaso                 # desktop app (Batch 6 UI)
  python -m bantayaso --cv            # simple OpenCV window (developer view)
  python -m bantayaso --source data\\clips\\rec_x.mp4   # test on a recorded clip

OpenCV-window keys (click the video first):
  R record clip   C save last 10 s   S snapshot   Z draw zones   H hazard boxes
  M mute voice    N do not disturb   D debug info  Q quit
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
    p.add_argument("--cv", action="store_true", help="use the simple OpenCV window instead of the app UI")
    p.add_argument("--no-actions", action="store_true", help="skip the CLIP action model")
    p.add_argument("--no-hazards", action="store_true", help="dog detection only")
    p.add_argument("--no-vlm", action="store_true", help="skip the local VLM sentence")
    p.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    return p.parse_args()


def open_source(cfg, source):
    cam = cfg.get("camera", {})
    src = source if source is not None else cam.get("index", 0)
    return src, FrameSource(src, backend=cam.get("backend", "msmf"), width=cam.get("width", 1280),
                            height=cam.get("height", 720), fps=cam.get("fps", 30)).start()


def run_cv(args) -> None:
    from .pipeline import Pipeline
    cfg = config.load()
    source, src = open_source(cfg, args.source)
    pipe = Pipeline(cfg, use_hazards=not args.no_hazards, use_actions=not args.no_actions,
                    use_vlm=not args.no_vlm)
    pipe.show_debug = args.debug
    rec = ClipRecorder(config.DATA_DIR / "clips", seconds=cfg.get("capture", {}).get("buffer_seconds", 10),
                       fps=cfg.get("capture", {}).get("record_fps", 15))
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, pipe.editor.mouse)
    toast_msg, toast_until, last_id = "", 0.0, 0

    def notify(msg):
        nonlocal toast_msg, toast_until
        toast_msg, toast_until = msg, time.monotonic() + 2.5
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
            rec.push(frame)
            view, st = pipe.process(frame)
            overlay.draw_status(view, st.status, overlay.LEVEL_KEY[st.level] if st.level else
                                ("watch" if st.status == "No dog in view" else "safe"))
            if st.last_alert and st.last_alert.get("vlm") and time.time() - st.last_alert["ts"] < 12:
                overlay.draw_toast(view, f"Local AI: {st.last_alert['vlm']}")
            overlay.draw_recording(view, rec.is_recording)
            if pipe.show_debug:
                overlay.draw_debug(view, pipe.debug_lines() + [f"voice {'on' if pipe.voice else 'off'}"
                                                               f"  dnd {'on' if pipe.dnd else 'off'}"])
            if toast_msg and time.monotonic() < toast_until:
                overlay.draw_toast(view, toast_msg)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if pipe.editor.key(key):
                continue
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            elif key in (ord("z"), ord("Z")):
                if pipe.editor.active:
                    pipe.editor.finish()
                    pipe.save_zones()
                    notify(f"Zones saved ({len(pipe.zones)})")
                pipe.editor.active = not pipe.editor.active
            elif key in (ord("h"), ord("H")):
                pipe.show_hazards = not pipe.show_hazards
            elif key in (ord("d"), ord("D")):
                pipe.show_debug = not pipe.show_debug
            elif key in (ord("m"), ord("M")):
                pipe.voice = not pipe.voice
                notify("Voice " + ("on" if pipe.voice else "muted"))
            elif key in (ord("n"), ord("N")):
                pipe.dnd = not pipe.dnd
                notify("Do not disturb " + ("on (still logging)" if pipe.dnd else "off"))
            elif key in (ord("r"), ord("R")):
                path = rec.toggle((frame.shape[1], frame.shape[0]))
                notify(f"Recording... {path.name}" if rec.is_recording else f"Saved {path.name}")
            elif key in (ord("c"), ord("C")):
                path = rec.save_last()
                notify(f"Saved last clip: {path.name}" if path else "Nothing to save yet")
            elif key in (ord("s"), ord("S")):
                path = config.DATA_DIR / "snapshots" / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), frame)
                notify(f"Snapshot {path.name}")
    finally:
        rec.close()
        src.stop()
        cv2.destroyAllWindows()
        print(f"[BantayAso] stopped. Last measured FPS: {pipe.state.fps:.1f}")


def main() -> None:
    args = parse_args()
    if args.cv:
        run_cv(args)
        return
    try:
        from .ui.app import run_ui
    except ImportError:
        print("[BantayAso] desktop UI not available yet - using the OpenCV window")
        run_cv(args)
        return
    run_ui(args)


if __name__ == "__main__":
    main()
