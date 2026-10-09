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
        self.weights = Path(weights)
        self._set_vocab()
        self.model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=device,
                           half=self.half, verbose=False)

    def _set_vocab(self) -> None:
        cwd = os.getcwd()
        os.chdir(self.weights.parent)            # text encoder is cached next to the weights
        try:
            if "yoloe" in self.weights.name:
                self.model.set_classes(self.names, self.model.get_text_pe(self.names))
            else:
                self.model.set_classes(self.names)
        finally:
            os.chdir(cwd)

    def update_vocab(self, vocab: dict) -> None:
        """Change the object list live (Things page). Runs on the pipeline thread."""
        self.vocab = {str(k): int(v) for k, v in vocab.items()}
        self.names = list(self.vocab.keys())
        self._prev = []
        self._set_vocab()

    def detect_crop(self, frame: np.ndarray, box, conf: float | None = None) -> list[Hazard]:
        """Mouth zoom: run on an enlarged crop around a dog and map boxes back to the frame.
        No two-pass confirmation here (the caller only uses it while the dog is chewing)."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = int(0.15 * max(x2 - x1, y2 - y1))
        cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []
        res = self.model.predict(crop, conf=conf or self.conf, imgsz=self.imgsz, device=self.device,
                                 half=self.half, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy().astype(int)):
            name = res.names.get(int(k), "?")
            tier = self.vocab.get(name, 1)
            if tier <= 0:
                continue
            bx = (int(b[0]) + cx1, int(b[1]) + cy1, int(b[2]) + cx1, int(b[3]) + cy1)
            out.append(Hazard(name, tier, bx, float(c)))
        return out

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
