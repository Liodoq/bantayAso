"""Dog detection + tracking with YOLO11 (COCO class 16 = dog). Runs every frame on the GPU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOG_CLASS = 16


@dataclass
class Dog:
    track_id: int          # stable ID across frames (-1 if the tracker has not assigned one yet)
    box: tuple             # (x1, y1, x2, y2) in pixels
    conf: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class DogDetector:
    def __init__(self, weights: Path, device: str = "cuda", conf: float = 0.35, imgsz: int = 640,
                 hold_seconds: float = 0.8, smooth: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = device.startswith("cuda")
        # anti-flicker: keep a dog that the detector missed for a moment (e.g. a white dog curled
        # up on a white blanket hovers around the confidence threshold) and smooth box jitter
        self.hold_seconds, self.smooth = hold_seconds, smooth
        self._last: dict[int, tuple[Dog, float]] = {}
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=device,
                           half=self.half, verbose=False)

    @staticmethod
    def _overlap(a, b) -> float:
        """Intersection over the SMALLER box (1.0 = one box sits inside the other)."""
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
        return inter / small

    def _dedupe(self, dogs: list[Dog]) -> list[Dog]:
        """One dog sometimes gets two boxes (whole body + half body). Keep the bigger/surer one."""
        keep: list[Dog] = []
        for d in sorted(dogs, key=lambda x: (x.conf, (x.box[2] - x.box[0]) * (x.box[3] - x.box[1])), reverse=True):
            if all(self._overlap(d.box, k.box) < 0.6 for k in keep):
                keep.append(d)
        return keep

    def __call__(self, frame: np.ndarray) -> list[Dog]:
        import time
        now = time.monotonic()
        raw = self._dedupe(self._detect(frame))
        out = []
        for d in raw:
            prev = self._last.get(d.track_id)
            if prev is not None and d.track_id >= 0:
                a = self.smooth
                d = Dog(d.track_id, tuple(int(a * n + (1 - a) * o) for n, o in zip(d.box, prev[0].box)), d.conf)
            out.append(d)
            if d.track_id >= 0:
                self._last[d.track_id] = (d, now)
        seen = {d.track_id for d in out}
        for tid, (d, t) in list(self._last.items()):
            if tid in seen:
                continue
            if now - t <= self.hold_seconds and all(self._overlap(d.box, o.box) < 0.5 for o in out):
                out.append(d)                      # briefly missed: keep showing it (if not a duplicate)
            else:
                self._last.pop(tid)
        return out

    def _detect(self, frame: np.ndarray) -> list[Dog]:
        res = self.model.track(frame, persist=True, classes=[DOG_CLASS], conf=self.conf,
                               imgsz=self.imgsz, device=self.device, half=self.half,
                               tracker=str(Path(__file__).parent / "trackers" / "dogs.yaml"), verbose=False)[0]
        dogs = []
        if res.boxes is None or len(res.boxes) == 0:
            return dogs
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(xyxy)
        for b, c, i in zip(xyxy, confs, ids):
            dogs.append(Dog(int(i), tuple(int(v) for v in b), float(c)))
        return dogs
