"""Zones drawn on the camera view. Stored normalized (0..1) in config.yaml so any resolution works.

Types: trash, danger, nogo, bed
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ZONE_TYPES = ["trash", "danger", "nogo", "bed"]


@dataclass
class Zone:
    name: str
    type: str
    points: list = field(default_factory=list)     # normalized [[x, y], ...]

    def poly(self, w: int, h: int) -> np.ndarray:
        return np.array([[int(x * w), int(y * h)] for x, y in self.points], dtype=np.int32)

    def contains(self, pt, w: int, h: int) -> bool:
        if len(self.points) < 3:
            return False
        return cv2.pointPolygonTest(self.poly(w, h), (float(pt[0]), float(pt[1])), False) >= 0

    def to_dict(self):
        return {"name": self.name, "type": self.type,
                "points": [[round(x, 4), round(y, 4)] for x, y in self.points]}


def load_zones(cfg: dict) -> list[Zone]:
    return [Zone(z.get("name", z["type"]), z["type"], z.get("points", [])) for z in cfg.get("zones") or []]


def zones_to_cfg(zones: list[Zone]) -> list[dict]:
    return [z.to_dict() for z in zones]


class ZoneEditor:
    """Mouse editor inside the OpenCV window.

    Z: start/stop editing   left-click: add point   right-click or ENTER: finish zone
    1-4: zone type (1 trash, 2 danger, 3 no-go, 4 bed)   BACKSPACE: undo point   X: delete last zone
    """

    def __init__(self, zones: list[Zone]):
        self.zones = zones
        self.active = False
        self.current: list = []
        self.type = "trash"
        self.size = (1280, 720)

    def mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        w, h = self.size
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append([x / w, y / h])
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.finish()

    def finish(self) -> None:
        if len(self.current) >= 3:
            n = sum(1 for z in self.zones if z.type == self.type) + 1
            self.zones.append(Zone(f"{self.type} {n}", self.type, self.current))
        self.current = []

    def key(self, k: int) -> bool:
        """Handle a key while editing. Returns True if consumed."""
        if not self.active:
            return False
        if k in (13, 10):
            self.finish()
        elif k in (8, 127) and self.current:
            self.current.pop()
        elif k in (ord("x"), ord("X")) and self.zones:
            self.zones.pop()
        elif ord("1") <= k <= ord("4"):
            self.type = ZONE_TYPES[k - ord("1")]
        else:
            return False
        return True
