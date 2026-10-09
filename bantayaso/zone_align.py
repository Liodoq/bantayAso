"""Keep zones on the right furniture when the camera gets bumped or re-angled.

When zones are saved, a reference picture of the room is saved too (data/zones_ref.png). Every
couple of seconds the current frame is compared with it: ORB feature points are matched and a
similarity transform (shift + rotation + zoom) is fitted with RANSAC, so moving dogs and people
are ignored as outliers. The zones are then drawn and checked through that transform.

Limits (honest): it follows a moved/rotated/zoomed camera looking at the SAME scene. If the camera
now points somewhere else, or the room changed a lot, there are too few matches: status becomes
"lost", the last good mapping is kept and the owner is asked to redraw the zones.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640                      # matching is done on a 640-px-wide grey copy


class ZoneAligner:
    def __init__(self, ref_path: Path | None, every_s: float = 2.0, min_inliers: int = 25):
        self.ref_path = Path(ref_path) if ref_path else None
        self.every_s, self.min_inliers = every_s, min_inliers
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_kp = self.ref_des = None
        self.ref_size = None
        self.M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)   # normalized ref -> normalized now
        self.status = "none"     # none | aligned | moved | lost
        self.shift = 0.0         # how far the zones moved, as a fraction of the picture
        self._last = 0.0
        if self.ref_path and self.ref_path.exists():
            img = cv2.imread(str(self.ref_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._set_ref_gray(img)

    # ------------------------------------------------------------------
    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h, w = g.shape[:2]
        return cv2.resize(g, (WIDTH, int(h * WIDTH / w))) if w != WIDTH else g

    def _set_ref_gray(self, g: np.ndarray) -> None:
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(g, None)
        self.ref_size = (g.shape[1], g.shape[0])
        self.M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
        self.status, self.shift = ("aligned" if self.ref_des is not None and len(self.ref_kp) >= 50 else "lost"), 0.0

    @property
    def has_reference(self) -> bool:
        return self.ref_des is not None

    def set_reference(self, frame: np.ndarray) -> None:
        """Call when zones are saved: this view becomes the one the zones were drawn on."""
        g = self._gray(frame)
        self._set_ref_gray(g)
        if self.ref_path:
            self.ref_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.ref_path), g)

    # ------------------------------------------------------------------
    def estimate(self, frame: np.ndarray):
        """-> (M 2x3 in normalized coords, inliers) or (None, inliers) when it can't tell."""
        if not self.has_reference:
            return None, 0
        g = self._gray(frame)
        kp, des = self.orb.detectAndCompute(g, None)
        if des is None or len(kp) < 30:
            return None, 0
        pairs = self.matcher.knnMatch(self.ref_des, des, k=2)
        good = [m for m, *n in pairs if n and m.distance < 0.75 * n[0].distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        rw, rh = self.ref_size
        cw, ch = g.shape[1], g.shape[0]
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        # fit in pixels (rotation/zoom are only "similar" in real pixels, not in 0..1 coords)
        Mp, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0,
                                              maxIters=2000, confidence=0.99)
        M = None
        if Mp is not None:
            A = np.vstack([Mp, [0, 0, 1]])
            S_ref = np.diag([rw, rh, 1.0])
            S_cur_inv = np.diag([1.0 / cw, 1.0 / ch, 1.0])
            M = (S_cur_inv @ A @ S_ref)[:2]
        n = int(inl.sum()) if inl is not None else 0
        if M is None or n < self.min_inliers:
            return None, n
        scale = float(np.hypot(Mp[0, 0], Mp[0, 1]))
        if not 0.6 < scale < 1.6:                 # implausible zoom -> treat as lost
            return None, n
        return M.astype(np.float64), n

    def update(self, frame: np.ndarray, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        if not self.has_reference or now - self._last < self.every_s:
            return self.status
        self._last = now
        M, _n = self.estimate(frame)
        if M is None:
            self.status = "lost"
            return self.status
        corners = np.float64([[0, 0], [1, 0], [1, 1], [0, 1]])
        moved = float(np.abs(self._apply(M, corners) - corners).max())
        if moved < 0.006:                         # tiny wobble: keep zones still (no jitter)
            M = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
            moved = 0.0
        self.M = 0.5 * self.M + 0.5 * M if moved and self.status in ("moved",) else M
        self.shift = moved
        self.status = "moved" if moved else "aligned"
        return self.status

    @staticmethod
    def _apply(M, pts) -> np.ndarray:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        return pts @ M[:, :2].T + M[:, 2]

    def apply(self, points) -> list:
        """Normalized zone points (as drawn on the reference view) -> where they are now."""
        if not points:
            return []
        out = np.clip(self._apply(self.M, points), 0.0, 1.0)
        return [[float(x), float(y)] for x, y in out]
