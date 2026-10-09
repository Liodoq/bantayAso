"""Zero-shot dog actions + "something in its mouth" with OpenAI CLIP (ViT-B/32), fully local.

For each dog we embed 3 crops in one GPU batch: the whole dog, and the upper-left / upper-right
parts (where the head usually is, whichever way the dog faces). Results are smoothed per track.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np
from .examples import blend, reference_gates, save_array

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


PAIR_FIGHT = ["a photo of two dogs fighting aggressively", "a photo of two dogs biting each other",
              "a photo of two dogs wrestling roughly"]
PAIR_CALM = ["a photo of two dogs lying calmly next to each other", "a photo of two dogs sleeping together",
             "a photo of two dogs sitting side by side"]


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


@dataclass
class _Lesson:
    label: str
    needed: int
    started: float
    features: list

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
            self.t_pair = enc(PAIR_FIGHT + PAIR_CALM)
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
                arr = np.load(f, allow_pickle=False)
                if arr.ndim == 2 and arr.shape[1] == self.t_act.shape[1] and len(arr) and np.isfinite(arr).all():
                    norm = np.linalg.norm(arr, axis=1, keepdims=True)
                    if (norm > 0).all():
                        self.examples[lab] = arr / norm
        self._example_gates = reference_gates(self.examples)
        self._teach: dict[int, _Lesson] = {}
        self.on_teaching = None
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

    def pair_score(self, frame, box_a, box_b) -> float:
        """0..1: how much the crop around two touching dogs looks like fighting / rough play."""
        from PIL import Image
        h, w = frame.shape[:2]
        x1, y1 = max(0, min(box_a[0], box_b[0]) - 10), max(0, min(box_a[1], box_b[1]) - 10)
        x2, y2 = min(w, max(box_a[2], box_b[2]) + 10), min(h, max(box_a[3], box_b[3]) + 10)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        torch = self.torch
        with torch.no_grad():
            x = self.preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))[None].to(self.device)
            f = self.model.encode_image(x.type(self.model.dtype)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.t_pair.T).softmax(dim=-1)[0].cpu().numpy()
        return float(p[:len(PAIR_FIGHT)].sum())

    def teach(self, tid: int, label: str, samples: int = 6) -> bool:
        """Collect `samples` crops of this dog over the next moments as examples of `label`."""
        if label not in self.labels or tid < 0 or samples < 1:
            return False
        self.cancel_teaching('Replaced by a new lesson.', tid)
        self._teach[tid] = _Lesson(label, samples, time.monotonic(), [])
        return True

    def _teaching_event(self, tid, label, status, message):
        if self.on_teaching:
            self.on_teaching({'track_id': tid, 'label': label, 'status': status, 'message': message})

    def cancel_teaching(self, reason: str, tid: int | None = None) -> None:
        for key in list(self._teach):
            if tid is None or key == tid:
                lesson = self._teach.pop(key)
                self._teaching_event(key, lesson.label, 'cancelled', f'Lesson cancelled: {reason} No examples saved.')

    def check_teaching(self, visible: set[int], now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for tid, lesson in list(self._teach.items()):
            if tid not in visible:
                self.cancel_teaching('The dog is no longer clearly visible.', tid)
            elif now - lesson.started > 10.0:
                self.cancel_teaching('Capture timed out. Try again while the dog holds the action.', tid)

    def _collect_example(self, tid: int, feat: np.ndarray, valid: bool = True) -> None:
        if tid not in self._teach:
            return
        if not valid or not np.isfinite(feat).all() or np.linalg.norm(feat) <= 0:
            self.cancel_teaching('The dog crop was not usable.', tid)
            return
        lesson = self._teach[tid]
        lesson.features.append(feat.copy() / np.linalg.norm(feat))
        if len(lesson.features) < lesson.needed:
            return
        # Commit the entire lesson only after all frames succeed; interruption saves nothing.
        lab = lesson.label
        rows = np.asarray(lesson.features, dtype=np.float32)
        old = self.examples.get(lab)
        rows = rows if old is None else np.vstack([old, rows])[-80:]
        try:
            save_array(self.examples_dir / f"{lab.replace(' ', '_')}.npy", rows)
        except OSError:
            self.cancel_teaching('Could not write the examples to disk.', tid)
            return
        self.examples[lab] = rows
        self._example_gates = reference_gates(self.examples)
        self._teach.pop(tid)
        note = (' Teach another action too before examples can influence recognition.' if len(self.examples) < 2
                else ' Examples influence recognition only when the match is clear.' if lab in self._example_gates
                else ' These examples are not distinct enough yet to influence recognition.')
        self._teaching_event(tid, lab, 'saved', f'Saved {lesson.needed} examples of {lab}.' + note)

    def forget_examples(self) -> None:
        self.cancel_teaching('Taught actions were reset.')
        self.examples = {}
        self._example_gates = {}
        for f in self.examples_dir.glob("*.npy"):
            f.unlink()

    def _blend_examples(self, p_text: np.ndarray, feat: np.ndarray) -> np.ndarray:
        return blend(self.labels, self.examples, self._example_gates, p_text, feat)

    def collect_teaching(self, frame, dogs) -> None:
        """Called after the pipeline validates the current frame's dog identities."""
        for d in dogs:
            if d.track_id in self.embeddings and getattr(d, 'observed', True):
                crop = self._crops(frame, d.box)[0]
                self._collect_example(d.track_id, self.embeddings[d.track_id],
                                      valid=crop.size > 0 and min(crop.shape[:2]) >= 16)

    def __call__(self, frame: np.ndarray, dogs, collect: bool = True) -> dict[int, ActionResult]:
        from PIL import Image
        torch = self.torch
        self.check_teaching({d.track_id for d in dogs if getattr(d, 'observed', True)})
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
        if collect:
            self.collect_teaching(frame, dogs)
        for tid in [t for t in self._ema if t not in {d.track_id for d in dogs}]:
            self._ema.pop(tid, None)
            self._mouth.pop(tid, None)
            self._chew.pop(tid, None)
            self.embeddings.pop(tid, None)
        return out
