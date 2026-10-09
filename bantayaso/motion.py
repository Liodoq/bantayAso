"""Cheap motion energy inside each dog's box (frame differencing on a small grayscale crop)."""
from __future__ import annotations

import cv2
import numpy as np


class MotionMeter:
    def __init__(self, active: float = 14.0, frantic: float = 22.0, alpha: float = 0.3):
        # calibrated: even resting dogs read ~10-12 on the action cam (noise/compression)
        self.active, self.frantic, self.alpha = active, frantic, alpha
        self._prev: dict[int, np.ndarray] = {}
        self._energy: dict[int, float] = {}

    def update(self, frame: np.ndarray, dogs) -> dict[int, tuple[str, float]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = {}
        for d in dogs:
            x1, y1, x2, y2 = d.box
            crop = gray[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
            if crop.size == 0:
                continue
            small = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
            prev = self._prev.get(d.track_id)
            self._prev[d.track_id] = small
            if prev is None:
                continue
            e = float(np.mean(np.abs(small - prev)))
            pe = self._energy.get(d.track_id, e)
            e = self.alpha * e + (1 - self.alpha) * pe
            self._energy[d.track_id] = e
            level = "frantic" if e >= self.frantic else "active" if e >= self.active else "still"
            out[d.track_id] = (level, e)
        alive = {d.track_id for d in dogs}
        for tid in [t for t in self._prev if t not in alive]:
            self._prev.pop(tid, None)
            self._energy.pop(tid, None)
        return out
