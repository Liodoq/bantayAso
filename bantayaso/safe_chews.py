"""Chewing the owner said is safe ("It's safe" on a chewing alert).

Two parts:
- this episode: the dog (track) that was chewing stays quiet until it stops chewing for a while;
- the look: the dog's CLIP picture embedding at that moment is saved (data/safe_chews.npy), and later
  chewing that looks almost the same (cosine >= threshold) is treated as safe too.
Honest limit: the embedding is of the whole dog, so it learns "this dog chewing like this", e.g. its
toy on its bed; a different object in a very similar pose can look alike. Medium/high objects from the
Things list and "disappeared near the mouth" still alert regardless (the risk engine checks those first).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class SafeChews:
    def __init__(self, path: Path | None, threshold: float = 0.93, max_examples: int = 60,
                 episode_quiet_s: float = 20.0):
        self.path = Path(path) if path else None
        self.threshold, self.max_examples, self.quiet_s = threshold, max_examples, episode_quiet_s
        self.rows = np.zeros((0, 512), np.float32)
        self.episodes: dict[int, float] = {}             # track id -> last time it was seen chewing safely
        if self.path and self.path.exists():
            try:
                arr = np.load(self.path, allow_pickle=False)
                if arr.ndim == 2 and len(arr):
                    self.rows = (arr / np.linalg.norm(arr, axis=1, keepdims=True)).astype(np.float32)
            except Exception:
                pass

    def add(self, tid, emb) -> bool:
        if tid is not None:
            self.episodes[int(tid)] = time.monotonic()
        if emb is None or not np.isfinite(emb).all() or np.linalg.norm(emb) == 0:
            return False
        v = (np.asarray(emb, np.float32) / np.linalg.norm(emb))[None]
        if self.rows.shape[1] != v.shape[1]:
            self.rows = np.zeros((0, v.shape[1]), np.float32)
        self.rows = np.vstack([self.rows, v])[-self.max_examples:]
        if self.path:
            try:
                from .examples import save_array
                save_array(self.path, self.rows)
            except Exception:
                pass
        return True

    def is_safe(self, tid, emb) -> bool:
        now = time.monotonic()
        if tid is not None and int(tid) in self.episodes:
            if now - self.episodes[int(tid)] <= self.quiet_s:
                self.episodes[int(tid)] = now              # still the same chewing episode
                return True
            self.episodes.pop(int(tid), None)
        if emb is None or not len(self.rows) or not np.isfinite(emb).all() or np.linalg.norm(emb) == 0:
            return False
        v = np.asarray(emb, np.float32) / np.linalg.norm(emb)
        return bool(len(v) == self.rows.shape[1] and float((self.rows @ v).max()) >= self.threshold)

    def forget(self) -> None:
        self.rows = np.zeros((0, self.rows.shape[1] if self.rows.ndim == 2 else 512), np.float32)
        self.episodes.clear()
        if self.path and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
