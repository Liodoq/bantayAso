"""Dog names (Batch 6c): few-shot recognition with CLIP image embeddings. No training, all local.

Enroll: a name + several crops of that dog (from the live camera or photos).
Match: cosine similarity of the current crop vs each dog's saved embeddings, voted over time
per track so names don't flicker.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np


class DogRegistry:
    def __init__(self, folder: Path, threshold: float = 0.80, margin: float = 0.02):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.threshold, self.margin = threshold, margin
        self.dogs: dict[str, np.ndarray] = {}
        self.load()
        self._votes: dict[int, collections.deque] = {}
        self.names_by_tid: dict[int, str] = {}
        self._enroll: dict[int, tuple[str, int]] = {}      # tid -> (name, samples still needed)

    # ---------------- storage ----------------
    def load(self) -> None:
        self.dogs = {}
        for f in self.folder.glob("*.npy"):
            try:
                arr = np.load(f)
                if arr.ndim == 2 and len(arr):
                    self.dogs[f.stem.replace("_", " ")] = arr
            except Exception:
                pass

    def save(self, name: str) -> None:
        np.save(self.folder / f"{name.strip().replace(' ', '_')}.npy", self.dogs[name])

    def delete(self, name: str) -> None:
        self.dogs.pop(name, None)
        p = self.folder / f"{name.strip().replace(' ', '_')}.npy"
        if p.exists():
            p.unlink()
        for tid in [t for t, n in self.names_by_tid.items() if n == name]:
            self.names_by_tid.pop(tid)

    def add_samples(self, name: str, embs) -> None:
        embs = np.atleast_2d(np.asarray(embs, dtype=np.float32))
        old = self.dogs.get(name)
        self.dogs[name] = embs if old is None else np.vstack([old, embs])[-60:]
        self.save(name)

    # ---------------- enrollment from the live camera ----------------
    def start_enroll(self, tid: int, name: str, samples: int = 10) -> None:
        self._enroll[tid] = (name.strip(), samples)
        self.names_by_tid[tid] = name.strip()

    def enrolling(self) -> dict:
        return {tid: n for tid, (n, _) in self._enroll.items()}

    # ---------------- matching ----------------
    def match(self, emb: np.ndarray) -> tuple[str | None, float]:
        if not self.dogs:
            return None, 0.0
        scores = []
        for name, arr in self.dogs.items():
            sims = arr @ emb
            k = min(3, len(sims))
            scores.append((float(np.sort(sims)[-k:].mean()), name))
        scores.sort(reverse=True)
        best, name = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best >= self.threshold and best - second >= self.margin:
            return name, best
        return None, best

    def update(self, embeddings: dict[int, np.ndarray], alive: set[int]) -> dict[int, str]:
        """embeddings: tid -> normalized CLIP image embedding. Returns tid -> name (stable)."""
        for tid, emb in embeddings.items():
            if tid in self._enroll:                       # collecting samples for a new dog
                name, left = self._enroll[tid]
                self.add_samples(name, emb)
                left -= 1
                if left <= 0:
                    self._enroll.pop(tid)
                else:
                    self._enroll[tid] = (name, left)
                self.names_by_tid[tid] = name
                continue
            name, _ = self.match(emb)
            v = self._votes.setdefault(tid, collections.deque(maxlen=7))
            v.append(name)
            counts = collections.Counter(n for n in v if n)
            if counts:
                top, c = counts.most_common(1)[0]
                if c >= 3:
                    self.names_by_tid[tid] = top
        for tid in [t for t in list(self._votes) if t not in alive]:
            self._votes.pop(tid, None)
        # a name belongs to one dog at a time: keep the most recent
        seen = {}
        for tid in sorted(t for t in self.names_by_tid if t in alive):
            seen[self.names_by_tid[tid]] = tid
        self.names_by_tid = {tid: n for n, tid in seen.items()}
        return dict(self.names_by_tid)

    def handover(self, old_tid: int, new_tid: int) -> None:
        if old_tid in self.names_by_tid:
            self.names_by_tid[new_tid] = self.names_by_tid.pop(old_tid)
