"""Dog detection + tracking with YOLO11 (COCO class 16 = dog). Runs every frame on the GPU."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DOG_CLASS = 16
# A dog curled up or stretched out asleep on a light blanket is often labelled cat/sheep/cow/bear by the
# COCO detector (it still "sees" an animal). Those classes are accepted as a dog, with a slightly
# higher bar. Teddy bear (77) is NOT included: plush toys on the bed would become dogs.
ALIAS_CLASSES = {15: "cat", 18: "sheep", 19: "cow", 21: "bear"}


def enhance_dark(frame: np.ndarray, mean_below: float = 80.0) -> np.ndarray:
    """Brighten a dim frame for detection only (CLAHE on lightness). Returns the frame unchanged if bright."""
    import cv2
    small = cv2.resize(frame, (64, 36))
    if float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()) >= mean_below:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


@dataclass
class Dog:
    track_id: int          # stable ID across frames (-1 if the tracker has not assigned one yet)
    box: tuple             # (x1, y1, x2, y2) in pixels
    conf: float
    observed: bool = True   # False when briefly held for display after a missed detection

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class DogDetector:
    def __init__(self, weights: Path, device: str = "cuda", conf: float = 0.35, imgsz: int = 640,
                 hold_seconds: float = 0.8, smooth: float = 0.5, aliases: bool = True,
                 alias_conf: float = 0.30, enhance: bool = True):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = device.startswith("cuda")
        self.aliases = dict(ALIAS_CLASSES) if aliases else {}
        self.alias_conf = alias_conf
        self.enhance = enhance
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
                out.append(Dog(d.track_id, d.box, d.conf, observed=False))
            else:
                self._last.pop(tid)
        return out

    def _detect(self, frame: np.ndarray) -> list[Dog]:
        img = enhance_dark(frame) if self.enhance else frame
        res = self.model.track(img, persist=True, classes=[DOG_CLASS, *self.aliases], conf=self.conf,
                               imgsz=self.imgsz, device=self.device, half=self.half,
                               tracker=str(Path(__file__).parent / "trackers" / "dogs.yaml"), verbose=False)[0]
        dogs = []
        if res.boxes is None or len(res.boxes) == 0:
            return dogs
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(xyxy)
        cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else [DOG_CLASS] * len(xyxy)
        for b, c, i, k in zip(xyxy, confs, ids, cls):
            if k != DOG_CLASS and c < self.alias_conf:
                continue                           # a weak "cat/sheep" guess is not enough
            dogs.append(Dog(int(i), tuple(int(v) for v in b), float(c)))
        return dogs
