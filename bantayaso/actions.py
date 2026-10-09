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


# Several clearer phrasings per action (averaged). Avoid scene words (bed, blanket, sofa):
# CLIP matches the scene instead of the action (tested: "digging at a blanket" and "jumping
# onto a bed" won for every dog lying on the bed). The original single prompt is kept in each
# list so the calibrated chewing/eating scores stay comparable.
PROMPTS = {
    "sleeping": ["a photo of a dog sleeping", "a photo of a dog sleeping with its eyes closed",
                 "a photo of a dog curled up asleep"],
    "lying down": ["a photo of a dog lying down", "a photo of a dog lying down awake with its head up",
                   "a photo of a dog lying on its side"],
    "sitting": ["a photo of a dog sitting", "a photo of a dog sitting upright",
                "a photo of a dog sitting and looking around"],
    "standing": ["a photo of a dog standing", "a photo of a dog standing on all four legs"],
    "walking": ["a photo of a dog walking", "a photo of a dog walking around the room"],
    "sniffing the floor": ["a photo of a dog sniffing the floor", "a photo of a dog sniffing the ground with its nose down"],
    "chewing something": ["a photo of a dog chewing something", "a photo of a dog chewing on an object",
                          "a photo of a dog chewing a toy"],
    "eating": ["a photo of a dog eating", "a photo of a dog eating food", "a photo of a dog with its head down eating"],
    "digging": ["a photo of a dog digging", "a photo of a dog digging fast with its front paws"],
    "scratching itself": ["a photo of a dog scratching itself", "a photo of a dog scratching its ear with its back leg",
                          "a photo of a dog scratching an itch"],
    "scratching furniture": ["a photo of a dog scratching furniture", "a photo of a dog clawing with its paws"],
    "jumping on furniture": ["a photo of a dog jumping on furniture", "a photo of a dog leaping in the air"],
    "licking itself": ["a photo of a dog licking itself", "a photo of a dog licking its paw",
                       "a photo of a dog grooming itself"],
}


@dataclass
class ActionResult:
    label: str            # smoothed top action, e.g. "chewing something"
    conf: float           # its smoothed probability
    mouth: float          # 0..1 "something in its mouth" score (smoothed)
    chew: float = 0.0      # 0..1 best "chewing/eating" probability over whole + head crops
    motion: str = "still"  # filled in from MotionMeter: still / active / frantic
    energy: float = 0.0
    label2: str = ""       # second-best action
    pose: str = ""         # best calm pose (sleeping/lying/sitting/standing/licking): used when a
                           # vigorous label (scratching/digging/jumping) wins but the dog is still

MOTION_LABELS = ("scratching itself", "scratching furniture", "digging", "jumping on furniture")
POSE_LABELS = ("sleeping", "lying down", "sitting", "standing", "licking itself", "eating",
               "chewing something", "sniffing the floor", "walking")


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
            rows = []
            for a in self.labels:
                t = enc(PROMPTS.get(a, [f"a photo of a dog {a}"])).mean(dim=0)
                rows.append(t / t.norm())
            self.t_act = torch.stack(rows)
            self.t_mouth = enc(MOUTH_POS + MOUTH_NEG)
        self.chew_idx = [i for i, l in enumerate(self.labels) if l in ("chewing something", "eating")]
        self._chew: dict[int, float] = {}
        self._ema: dict[int, np.ndarray] = {}
        self.embeddings: dict[int, np.ndarray] = {}     # tid -> whole-dog image embedding (names)
        # "Teach Bantay": the owner's own labelled examples per action (data/actions/<label>.npy)
        self.examples_dir = Path(models_dir).parent / "data" / "actions"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.examples: dict[str, np.ndarray] = {}
        for f in self.examples_dir.glob("*.npy"):
            lab = f.stem.replace("_", " ")
            if lab in self.labels:
                self.examples[lab] = np.load(f)
        self._teach: dict[int, tuple[str, int]] = {}
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

    def teach(self, tid: int, label: str, samples: int = 6) -> None:
        """Collect `samples` crops of this dog over the next moments as examples of `label`."""
        if label in self.labels:
            self._teach[tid] = (label, samples)

    def forget_examples(self) -> None:
        self.examples = {}
        for f in self.examples_dir.glob("*.npy"):
            f.unlink()

    def _blend_examples(self, p_text: np.ndarray, feat: np.ndarray) -> np.ndarray:
        if len(self.examples) < 2:
            return p_text
        idx = [j for j, l in enumerate(self.labels) if l in self.examples]
        sims = np.array([float(np.max(self.examples[self.labels[j]] @ feat)) for j in idx])
        e = np.exp(100.0 * (sims - sims.max()))
        p_ex = np.zeros_like(p_text)
        p_ex[idx] = e / e.sum()
        mix = 0.6 * p_ex + 0.4 * p_text          # the owner's examples weigh more than text
        return mix / mix.sum()

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
            feats = f.cpu().numpy()
            mo = (100.0 * f @ self.t_mouth.T).softmax(dim=-1).cpu().numpy()
        out = {}
        npos = len(MOUTH_POS)
        for d in dogs:
            idx = [i for i, (tid, _) in enumerate(owners) if tid == d.track_id]
            full = [i for i in idx if owners[i][1] == 0]
            fi = full[0] if full else idx[0]
            self.embeddings[d.track_id] = feats[fi]
            p = self._blend_examples(act[fi], feats[fi])
            if d.track_id in self._teach:
                lab, left = self._teach[d.track_id]
                old = self.examples.get(lab)
                self.examples[lab] = feats[fi][None] if old is None else np.vstack([old, feats[fi][None]])[-80:]
                np.save(self.examples_dir / f"{lab.replace(' ', '_')}.npy", self.examples[lab])
                if left <= 1:
                    self._teach.pop(d.track_id)
                else:
                    self._teach[d.track_id] = (lab, left - 1)
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
            order = np.argsort(-e)
            j, j2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
            pose_idx = [k for k in order if self.labels[k] in POSE_LABELS]
            pose = self.labels[int(pose_idx[0])] if pose_idx else self.labels[j2]
            out[tid] = ActionResult(self.labels[j], float(e[j]), self._mouth[tid], self._chew[tid],
                                    label2=self.labels[j2], pose=pose)
        for tid in [t for t in self._ema if t not in {d.track_id for d in dogs}]:
            self._ema.pop(tid, None)
            self._mouth.pop(tid, None)
            self._chew.pop(tid, None)
            self.embeddings.pop(tid, None)
        return out
