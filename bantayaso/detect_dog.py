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
    def __init__(self, weights: Path, device: str = "cuda", conf: float = 0.35, imgsz: int = 640):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.half = device.startswith("cuda")
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=device,
                           half=self.half, verbose=False)

    def __call__(self, frame: np.ndarray) -> list[Dog]:
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
