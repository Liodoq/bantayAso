"""Camera / video input with a background reader (always the latest frame) and a clip recorder."""
from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

import cv2
import numpy as np

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}


class FrameSource:
    """Reads frames on a thread. `read()` returns the newest frame, never a backlog.

    source: int (USB device index) or str (path to a video file).
    """

    def __init__(self, source, backend: str = "msmf", width: int = 1280, height: int = 720,
                 fps: int = 30, loop_file: bool = True):
        self.source = source
        self.is_url = isinstance(source, str) and "://" in source        # rtsp:// or http://
        self.is_file = isinstance(source, str) and not source.isdigit() and not self.is_url
        self.backend = backend
        self.size = (width, height)
        self.req_fps = fps
        self.loop_file = loop_file
        self._cap = None
        self._frame = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False
        self.file_fps = 30.0

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "FrameSource":
        self._thread = threading.Thread(target=self._run, name="FrameSource", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def _open(self) -> bool:
        if self._cap:
            self._cap.release()
        if self.is_url:
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.is_file:
            if not Path(self.source).exists():
                raise FileNotFoundError(self.source)
            self._cap = cv2.VideoCapture(self.source)
            self.file_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self._cap = cv2.VideoCapture(int(self.source), BACKENDS.get(self.backend, cv2.CAP_ANY))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.req_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return self._cap.isOpened()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.connected:
                self.connected = self._open()
                if not self.connected:
                    time.sleep(1.0)          # camera unplugged: retry every second
                    continue
            t0 = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok:
                if self.is_file and self.loop_file:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.connected = False
                continue
            if not self.is_file and (frame.shape[1], frame.shape[0]) != self.size and \
                    abs(frame.shape[1] / frame.shape[0] - self.size[0] / self.size[1]) < 0.05:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            with self._lock:
                self._frame = frame
                self._frame_id += 1
            if self.is_file:                 # play files at real speed
                delay = 1.0 / self.file_fps - (time.perf_counter() - t0)
                if delay > 0:
                    time.sleep(delay)

    def read(self):
        """Return (frame_id, frame) or (0, None) if nothing yet."""
        with self._lock:
            if self._frame is None:
                return 0, None
            return self._frame_id, self._frame.copy()


class ClipRecorder:
    """Keeps the last N seconds (JPEG-compressed in RAM) and can record continuously.

    - save_last(): writes the last N seconds to data/clips/
    - toggle(): start/stop a continuous recording to data/clips/
    """

    def __init__(self, out_dir: Path, seconds: float = 10.0, fps: float = 15.0):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._buf = collections.deque(maxlen=int(seconds * fps))
        self._last_push = 0.0
        self._writer = None
        self.recording_path = None

    def push(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_push < 1.0 / self.fps:
            return
        self._last_push = now
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self._buf.append(jpg)
        if self._writer is not None:
            self._writer.write(frame)

    def _new_writer(self, size, tag: str):
        path = self.out_dir / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        base, suffix = path, 1
        while path.exists():   # repeated clicks in one second must not overwrite an earlier lesson
            path = base.with_name(f'{base.stem}_{suffix}{base.suffix}')
            suffix += 1
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        if not writer.isOpened():
            writer.release()
            raise OSError('Could not create the video file. Check the clips folder and video encoder.')
        return path, writer

    def save_last(self) -> Path | None:
        if not self._buf:
            return None
        frames = [cv2.imdecode(j, cv2.IMREAD_COLOR) for j in list(self._buf)]
        h, w = frames[0].shape[:2]
        path, writer = self._new_writer((w, h), "last")
        for f in frames:
            writer.write(f)
        writer.release()
        return path

    def toggle(self, size) -> Path | None:
        """Start recording (returns path) or stop (returns the finished path)."""
        if self._writer is None:
            self.recording_path, self._writer = self._new_writer(size, "rec")
            return self.recording_path
        self._writer.release()
        self._writer = None
        done, self.recording_path = self.recording_path, None
        return done

    @property
    def is_recording(self) -> bool:
        return self._writer is not None

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
