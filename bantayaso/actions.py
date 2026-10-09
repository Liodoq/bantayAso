"""Zero-shot dog actions + "something in its mouth" with OpenAI CLIP (ViT-B/32), fully local.

For each dog we embed 3 crops in one GPU batch: the whole dog, and the upper-left / upper-right
parts (where the head usually is, whichever way the dog faces). Results are smoothed per track.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MOUTH_POS = ["a photo of a dog with an object in its mouth",
             "a photo of a dog chewing on something",
             "a photo of a dog biting a toy or a stick"]
MOUTH_NEG = ["a photo of a dog with its mouth closed",
             "a photo of a dog with nothing in its mouth",
             "a photo of a dog resting its head",
             "a photo of a dog panting with its mouth open",
             "a photo of a dog scratching itself with its leg",
             "a photo of a dog licking its paw or its body",
             "a photo of a dog sitting and looking around"]


@dataclass
class ActionResult:
    label: str            # smoothed top action, e.g. "chewing something"
    conf: float           # its smoothed probability
    mouth: float          # 0..1 "something in its mouth" score (smoothed)
    chew: float = 0.0      # 0..1 best "chewing/eating" probability over whole + head crops
    motion: str = "still"  # filled in from MotionMeter: still / active / frantic
    energy: float = 0.0


class ActionClassifier:
    def __init__(self, labels: list[str], models_dir: Path, device: str = "cuda",
                 model_name: str = "ViT-B/32", alpha: float = 0.4):
        import clip
        import torch
        self.torch = torch
        self.device = device
        self.labels = list(labels)
        self.alpha = alpha
        self.model, self.preprocess = clip.load(model_name, device=device,
                                                download_root=str(Path(models_dir) / "clip"))
        self.model.eval()
        with torch.no_grad():
            def enc(texts):
                t = self.model.encode_text(clip.tokenize(texts).to(device)).float()
                return t / t.norm(dim=-1, keepdim=True)
            self.t_act = enc([f"a photo of a dog {a}" for a in self.labels])
            self.t_mouth = enc(MOUTH_POS + MOUTH_NEG)
        self.chew_idx = [i for i, l in enumerate(self.labels) if l in ("chewing something", "eating")]
        self._chew: dict[int, float] = {}
        self._ema: dict[int, np.ndarray] = {}
        self._mouth: dict[int, float] = {}

    @staticmethod
    def _crops(frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad = int(0.08 * max(bw, bh))
        full = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        hy2 = y1 + int(0.65 * bh)
        left = frame[max(0, y1 - pad):hy2, max(0, x1 - pad):x1 + int(0.6 * bw)]
        right = frame[max(0, y1 - pad):hy2, x2 - int(0.6 * bw):min(w, x2 + pad)]
        return [c for c in (full, left, right)]

    def __call__(self, frame: np.ndarray, dogs) -> dict[int, ActionResult]:
        from PIL import Image
        torch = self.torch
        imgs, owners = [], []
        for d in dogs:
            for k, c in enumerate(self._crops(frame, d.box)):
                if c.size == 0 or min(c.shape[:2]) < 16:
                    c = np.zeros((32, 32, 3), np.uint8)
                imgs.append(self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))))
                owners.append((d.track_id, k))
        if not imgs:
            return {}
        with torch.no_grad():
            x = torch.stack(imgs).to(self.device)
            f = self.model.encode_image(x.type(self.model.dtype)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            act = (100.0 * f @ self.t_act.T).softmax(dim=-1).cpu().numpy()
            mo = (100.0 * f @ self.t_mouth.T).softmax(dim=-1).cpu().numpy()
        out = {}
        npos = len(MOUTH_POS)
        for d in dogs:
            idx = [i for i, (tid, _) in enumerate(owners) if tid == d.track_id]
            full = [i for i in idx if owners[i][1] == 0]
            p = act[full[0]] if full else act[idx[0]]
            mouth_now = max(float(mo[i][:npos].sum()) for i in idx)   # best of whole/head crops
            # a dog eating with its head down looks "lying down" as a whole, so take the best
            # chewing/eating probability over the head crops too
            chew_now = max(float(act[i][self.chew_idx].sum()) for i in idx) if self.chew_idx else 0.0
            tid = d.track_id
            prev = self._ema.get(tid)
            self._ema[tid] = p if prev is None else self.alpha * p + (1 - self.alpha) * prev
            pm = self._mouth.get(tid)
            self._mouth[tid] = mouth_now if pm is None else self.alpha * mouth_now + (1 - self.alpha) * pm
            pc = self._chew.get(tid)
            self._chew[tid] = chew_now if pc is None else self.alpha * chew_now + (1 - self.alpha) * pc
            e = self._ema[tid]
            j = int(e.argmax())
            out[tid] = ActionResult(self.labels[j], float(e[j]), self._mouth[tid], self._chew[tid])
        for tid in [t for t in self._ema if t not in {d.track_id for d in dogs}]:
            self._ema.pop(tid, None)
            self._mouth.pop(tid, None)
            self._chew.pop(tid, None)
        return out
