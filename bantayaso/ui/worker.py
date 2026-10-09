"""Background thread: camera -> Pipeline -> QImage for the UI. Models load here, not in the UI thread."""
from __future__ import annotations

import time

import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .. import config, overlay
from ..capture import ClipRecorder, FrameSource


class Worker(QThread):
    frame_ready = Signal(QImage, dict)
    message = Signal(str)
    event = Signal(dict)
    ready = Signal()
    teaching = Signal(dict)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args
        self.pipe = None
        self.running = True
        self.debug = bool(getattr(args, "debug", False))
        self._req = set()                       # "record", "last", "snapshot"
        self._new_source = None
        self.source = None
        self.camera_ok = True
        self.recording = False
        self.started_at = time.time()

    def request(self, what: str) -> None:
        self._req.add(what)

    def switch_source(self, source) -> None:
        """Use another camera (USB index, video file or stream URL) without restarting."""
        self._new_source = source

    def stop(self) -> None:
        self.running = False
        self.wait(3000)

    def run(self) -> None:
        from ..pipeline import Pipeline
        cfg = config.load()
        cam = cfg.get("camera", {})
        source = self.args.source if self.args.source is not None else cam.get("index", 0)
        self.source = source

        def open_src(s_):
            return FrameSource(s_, backend=cam.get("backend", "msmf"), width=cam.get("width", 1280),
                               height=cam.get("height", 720), fps=cam.get("fps", 30)).start()
        src = open_src(source)
        self.message.emit("Loading the local AI models...")
        try:
            self.pipe = Pipeline(cfg, use_hazards=not self.args.no_hazards,
                                 use_actions=not self.args.no_actions, use_vlm=not self.args.no_vlm,
                                 log=lambda m: (print(m), self.message.emit(m)))
        except Exception as e:                  # pragma: no cover
            self.message.emit(f"Could not load the AI models: {e}")
            src.stop()
            return
        self.pipe.on_event = lambda ev: self.event.emit(dict(ev))
        self.pipe.on_teaching = lambda ev: self.teaching.emit(dict(ev))
        rec = ClipRecorder(config.DATA_DIR / "clips", seconds=cfg.get("capture", {}).get("buffer_seconds", 10),
                           fps=cfg.get("capture", {}).get("record_fps", 15))
        self.started_at = time.time()
        self.ready.emit()
        last_id, last_frame_t = 0, time.monotonic()
        try:
            while self.running:
                if self._new_source is not None:
                    self.pipe.cancel_teaching('The camera source changed.')
                    new, self._new_source = self._new_source, None
                    src.stop()
                    src = open_src(new)
                    self.source, last_id, last_frame_t = new, 0, time.monotonic()
                    self.message.emit(f"Switched camera to {new}")
                fid, frame = src.read()
                if frame is None or fid == last_id:
                    if time.monotonic() - last_frame_t > 2:
                        self.pipe.cancel_teaching('Camera frames stopped arriving.')
                    if time.monotonic() - last_frame_t > 4 and self.camera_ok:
                        self.camera_ok = False
                        self.message.emit("CAMERA_LOST")
                    self.msleep(5)
                    continue
                if not self.camera_ok:
                    self.camera_ok = True
                    self.message.emit("CAMERA_OK")
                last_frame_t = time.monotonic()
                last_id = fid
                rec.push(frame)
                self._handle_requests(rec, frame)
                try:
                    view, st = self.pipe.process(frame)
                except Exception:                   # one bad frame must never stop the watcher
                    import traceback
                    err = traceback.format_exc()
                    print(err)
                    try:
                        with open(config.DATA_DIR / "crash.log", "a", encoding="utf-8") as f:
                            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} frame error\n{err}")
                    except OSError:
                        pass
                    self.msleep(50)
                    continue
                overlay.draw_recording(view, rec.is_recording)
                if self.debug:
                    overlay.draw_debug(view, self.pipe.debug_lines())
                rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                self.frame_ready.emit(img, {
                    "status": st.status, "level": st.level, "dogs": st.dogs, "fps": st.fps,
                    "last_alert": dict(st.last_alert) if st.last_alert else None,
                    "frame_size": (w, h)})
        finally:
            self.pipe.cancel_teaching('Monitoring stopped.')
            rec.close()
            src.stop()

    def _handle_requests(self, rec, frame) -> None:
        if not self._req:
            return
        reqs, self._req = self._req, set()
        if "record" in reqs:
            path = rec.toggle((frame.shape[1], frame.shape[0]))
            self.recording = rec.is_recording
            self.message.emit(f"Recording {path.name}..." if rec.is_recording else f"Saved clip {path.name}")
        if "last" in reqs:
            path = rec.save_last()
            self.message.emit(f"Saved last 10 s: {path.name}" if path else "Nothing to save yet")
        if "snapshot" in reqs:
            path = config.DATA_DIR / "snapshots" / f"snap_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path), frame)
            self.message.emit(f"Snapshot saved: {path.name}")
