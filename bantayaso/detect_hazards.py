"""Open-vocabulary hazard detection (YOLOE / YOLO-World). Words + tiers come from config.yaml `hazards`."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Hazard:
    name: str
    tier: int              # 1 low, 2 medium, 3 high
    box: tuple             # (x1, y1, x2, y2)
    conf: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


class HazardDetector:
    def __init__(self, weights: Path, vocab: dict, device: str = "cuda", conf: float = 0.15,
                 imgsz: int = 640, confirm: bool = True):
        from ultralytics import YOLO
        self.vocab = {str(k): int(v) for k, v in vocab.items()}
        self.names = list(self.vocab.keys())
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.confirm = confirm
        self._prev: list[Hazard] = []
        self.half = device.startswith("cuda")
        self.model = YOLO(str(weights))
        cwd = os.getcwd()
        os.chdir(Path(weights).parent)           # text encoder is cached next to the weights
        try:
            if "yoloe" in Path(weights).name:
                self.model.set_classes(self.names, self.model.get_text_pe(self.names))
            else:
                self.model.set_classes(self.names)
        finally:
            os.chdir(cwd)
        self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=device,
                           half=self.half, verbose=False)

    def __call__(self, frame: np.ndarray) -> list[Hazard]:
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device,
                                 half=self.half, verbose=False)[0]
        raw = []
        if res.boxes is not None:
            for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                               res.boxes.cls.cpu().numpy().astype(int)):
                name = res.names.get(int(k), self.names[int(k)] if int(k) < len(self.names) else "?")
                tier = self.vocab.get(name, 1)
                if tier <= 0:          # distractor words (dog collar, paw, pillow...) absorb look-alikes
                    continue
                raw.append(Hazard(name, tier, tuple(int(v) for v in b), float(c)))
        if not self.confirm:
            return raw
        # keep only objects also seen (same word, nearby) in the previous pass -> kills one-off ghosts
        out = [h for h in raw if any(p.name == h.name and abs(p.center[0] - h.center[0]) < 60
                                     and abs(p.center[1] - h.center[1]) < 60 for p in self._prev)]
        self._prev = raw
        return out
