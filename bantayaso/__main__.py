"""BantayAso - Batch 1 live view.

  python -m bantayaso                       # action camera from config.yaml
  python -m bantayaso --source 0            # another USB device index
  python -m bantayaso --source data\\clips\\rec_x.mp4   # test on a recorded clip

Keys (click the video window first):
  R  start/stop recording a clip      C  save the last 10 seconds
  S  save a snapshot                  D  show/hide debug info (FPS, device)
  Z  draw zones (see the help bar)     H  show/hide hazard boxes
  Q  quit
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from . import config, overlay
from .capture import ClipRecorder, FrameSource
from .risk import LEVELS, RiskEngine
from .zones import ZoneEditor, load_zones, zones_to_cfg

WINDOW = "BantayAso"


def parse_args():
    p = argparse.ArgumentParser(prog="bantayaso")
    p.add_argument("--source", help="USB index (e.g. 1) or path to a video file")
    p.add_argument("--no-detect", action="store_true", help="camera only, no AI (for testing)")
    p.add_argument("--no-actions", action="store_true", help="skip the CLIP action model")
    p.add_argument("--no-hazards", action="store_true", help="dog detection only")
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

    hazard_det = None
    if not args.no_detect and not args.no_hazards:
        from .detect_hazards import HazardDetector
        print("[BantayAso] loading hazard detector...")
        hazard_det = HazardDetector(config.MODELS_DIR / cfg["models"]["hazard_detector"],
                                    cfg.get("hazards", {}), device=device,
                                    conf=det_cfg.get("hazard_conf", 0.15),
                                    imgsz=det_cfg.get("hazard_imgsz", 640))
    hazard_every = int(det_cfg.get("hazard_every", 5))
    classifier = None
    if not args.no_detect and not args.no_actions:
        from .actions import ActionClassifier
        print("[BantayAso] loading action model (CLIP)...")
        classifier = ActionClassifier(cfg.get("actions") or [], config.MODELS_DIR, device=device,
                                      model_name=cfg["models"].get("clip_openai_name", "ViT-B/32"))
    from .motion import MotionMeter
    motion_meter = MotionMeter()
    action_every_s = float(det_cfg.get("action_every_seconds", 0.25))
    actions, last_action_t, act_ms = {}, 0.0, 0.0
    last_zoom_t, zoom_hits = 0.0, []
    zones = load_zones(cfg)
    editor = ZoneEditor(zones)
    engine = RiskEngine(cfg)
    hazards, show_hazards, n_frame, hz_ms = [], True, 0, 0.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, editor.mouse)
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
            overlay.reset_labels()
            editor.size = (frame.shape[1], frame.shape[0])
            overlay.draw_zones(view, zones, editor)
            if detector is not None:
                t0 = time.perf_counter()
                dogs = detector(frame)
                infer_ms = (time.perf_counter() - t0) * 1000
                fresh = False
                if hazard_det is not None and n_frame % hazard_every == 0:
                    t1 = time.perf_counter()
                    hazards = hazard_det(frame)
                    hz_ms = (time.perf_counter() - t1) * 1000
                    fresh = True
                n_frame += 1
                motions = motion_meter.update(frame, dogs)
                now_m = time.monotonic()
                if classifier is not None and dogs and now_m - last_action_t >= action_every_s:
                    t2 = time.perf_counter()
                    actions = classifier(frame, dogs)
                    act_ms = (time.perf_counter() - t2) * 1000
                    last_action_t = now_m
                for tid, (lvl, en) in motions.items():
                    if tid in actions:
                        actions[tid].motion, actions[tid].energy = lvl, en
                # mouth zoom: a dog seems to be chewing but no hazard is near it -> look closer (1/s)
                if hazard_det is not None and now_m - last_zoom_t >= 1.0:
                    chewers = [d for d in dogs if d.track_id in actions and
                               0.5 * (actions[d.track_id].mouth + actions[d.track_id].chew) >= engine.eat_thr]
                    zoom_hits = []
                    for d in chewers[:2]:
                        zoom_hits += hazard_det.detect_crop(frame, d.box)
                    last_zoom_t = now_m
                all_hz = hazards + [z for z in zoom_hits
                                    if not any(z.name == h.name for h in hazards)]
                found = engine.update(dogs, all_hz, zones, (frame.shape[1], frame.shape[0]),
                                      hazards_fresh=fresh, actions=actions if classifier else None)
                by_id = {a.track_id: a for a in found}
                if show_hazards:
                    overlay.draw_hazards(view, all_hz)
                if dogs:
                    last_seen = time.monotonic()
                    overlay.draw_dogs(view, dogs, by_id)
                    top = max(found, key=lambda a: a.level)
                    if top.level == 0:
                        calm = [a.reason for a in found if a.reason not in ("all calm", "resting")]
                        if len(found) == 1 and calm:
                            msg = f"Your dog is {calm[0]}"
                        elif calm:
                            msg = f"All calm - {len(found)} dogs ({', '.join(sorted(set(calm)))})"
                        else:
                            msg = "All calm"
                        overlay.draw_status(view, msg, "safe")
                    else:
                        overlay.draw_status(view, f"{LEVELS[top.level].upper()}: dog {top.reason}",
                                            overlay.LEVEL_KEY[top.level])
                    alert = engine.should_alert(found)
                    if alert is not None:
                        print(f"[ALERT {LEVELS[alert.level].upper()}] {time.strftime('%H:%M:%S')} "
                              f"dog {alert.reason}")
                else:
                    gone = time.monotonic() - last_seen if last_seen else None
                    msg = "No dog in view" if gone is None or gone > 3 else "All calm"
                    overlay.draw_status(view, msg, "watch" if msg == "No dog in view" else "safe")

            overlay.draw_recording(view, recorder.is_recording)
            frames += 1
            now = time.perf_counter()
            if now - t_fps >= 1.0:
                fps, frames, t_fps = frames / (now - t_fps), 0, now
            if show_debug:
                overlay.draw_debug(view, [f"FPS {fps:.1f}", f"detect {infer_ms:.0f} ms",
                                          f"hazards {hz_ms:.0f} ms every {hazard_every}",
                                          f"actions {act_ms:.0f} ms every {action_every_s:.2f} s",
                                          *[f"#{t}: {r.label} {r.conf:.2f} eat {0.5 * (r.mouth + r.chew):.2f} (mouth {r.mouth:.2f} chew {r.chew:.2f}) {r.motion}"
                                            for t, r in list(actions.items())[:4]],
                                          f"device {device}", f"source {source}"])
            if toast and time.monotonic() < toast_until:
                overlay.draw_toast(view, toast)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(1) & 0xFF
            if editor.key(key):
                continue
            if key in (ord("z"), ord("Z")):
                if editor.active:
                    editor.finish()
                    latest = config.load()            # don't clobber edits made while running
                    latest["zones"] = zones_to_cfg(zones)
                    config.save(latest)
                    notify(f"Zones saved ({len(zones)})")
                editor.active = not editor.active
                continue
            if key in (ord("h"), ord("H")):
                show_hazards = not show_hazards
                continue
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
