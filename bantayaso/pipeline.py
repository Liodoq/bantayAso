"""The whole per-frame brain, shared by the OpenCV window (__main__) and the desktop UI (ui/).

frame -> dogs (YOLO11) -> hazards (YOLOE, every N frames) -> actions/mouth (CLIP, ~4 Hz)
      -> motion -> risk engine -> alerts (voice/toast/clip) + event log + local VLM sentence
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import config, overlay
from .alerts import Speaker, phrase, play_clip, toast
from .events import EventLog
from .motion import MotionMeter
from .risk import LEVELS, RiskEngine
from .zones import ZoneEditor, load_zones, zones_to_cfg


@dataclass
class State:
    level: int = 0
    status: str = "Starting..."
    assessments: list = field(default_factory=list)
    dogs: int = 0
    fps: float = 0.0
    timings: dict = field(default_factory=dict)
    last_alert: dict | None = None        # {"id", "level", "text", "vlm", "ts"}


class Pipeline:
    def __init__(self, cfg: dict | None = None, use_hazards: bool = True, use_actions: bool = True,
                 use_vlm: bool = True, log=print):
        self.cfg = cfg or config.load()
        config.ensure_dirs()
        self.log = log
        cfg = self.cfg
        self.device = config.resolve_device(cfg)
        det = cfg.get("detect", {})
        al = cfg.get("alerts", {})
        log(f"[BantayAso] inference device: {self.device}")

        from .detect_dog import DogDetector
        log("[BantayAso] loading dog detector...")
        self.dog_det = DogDetector(config.MODELS_DIR / cfg["models"]["dog_detector"], device=self.device,
                                   conf=det.get("dog_conf", 0.25), imgsz=det.get("imgsz", 960))
        self.hazard_det = None
        if use_hazards:
            from .detect_hazards import HazardDetector
            log("[BantayAso] loading hazard detector...")
            self.hazard_det = HazardDetector(config.MODELS_DIR / cfg["models"]["hazard_detector"],
                                             cfg.get("hazards", {}), device=self.device,
                                             conf=det.get("hazard_conf", 0.25),
                                             imgsz=det.get("hazard_imgsz", 640))
        self.classifier = None
        if use_actions:
            from .actions import ActionClassifier
            log("[BantayAso] loading action model (CLIP)...")
            self.classifier = ActionClassifier(cfg.get("actions") or [], config.MODELS_DIR,
                                               device=self.device,
                                               model_name=cfg["models"].get("clip_openai_name", "ViT-B/32"))
        self.vlm = None
        if use_vlm and al.get("vlm", True):
            from .vlm import VLMWorker
            self.vlm = VLMWorker(cfg["models"].get("ollama_url", "http://localhost:11434"),
                                 cfg["models"].get("vlm", "moondream"))

        self.motion = MotionMeter()
        self.zones = load_zones(cfg)
        self.editor = ZoneEditor(self.zones)
        self.engine = RiskEngine(cfg)
        self.events = EventLog(config.DATA_DIR / "events.db", config.DATA_DIR / "snapshots")
        self.speaker = Speaker()
        self.voice = bool(al.get("voice", True))
        self.dnd = bool(al.get("do_not_disturb", False))
        self.toasts = bool(al.get("toast", True))
        self.speak_vlm = bool(al.get("speak_vlm", True))
        self.owner_clip = al.get("owner_voice_clip", "")
        self.dog_name = cfg.get("dog_name", "") or "your dog"

        self.hazard_every = int(det.get("hazard_every", 5))
        self.action_every = float(det.get("action_every_seconds", 0.25))
        self.show_hazards, self.show_debug = True, False
        self.state = State()
        self._hazards, self._actions, self._zoom = [], {}, []
        self._n, self._last_act, self._last_zoom, self._last_seen = 0, 0.0, 0.0, 0.0
        self._fps_n, self._fps_t = 0, time.perf_counter()
        self.on_event = None                # optional callback(event_dict) for the UI

    # ------------------------------------------------------------------
    def save_zones(self) -> None:
        latest = config.load()              # don't clobber edits made while running
        latest["zones"] = zones_to_cfg(self.zones)
        config.save(latest)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, State]:
        st = self.state
        h, w = frame.shape[:2]
        now = time.monotonic()
        t0 = time.perf_counter()
        dogs = self.dog_det(frame)
        st.timings["dogs"] = (time.perf_counter() - t0) * 1000

        fresh = False
        if self.hazard_det is not None and self._n % self.hazard_every == 0:
            t1 = time.perf_counter()
            self._hazards = self.hazard_det(frame)
            st.timings["hazards"] = (time.perf_counter() - t1) * 1000
            fresh = True
        self._n += 1

        motions = self.motion.update(frame, dogs)
        if self.classifier is not None and dogs and now - self._last_act >= self.action_every:
            t2 = time.perf_counter()
            self._actions = self.classifier(frame, dogs)
            st.timings["actions"] = (time.perf_counter() - t2) * 1000
            self._last_act = now
        for tid, (lvl, en) in motions.items():
            if tid in self._actions:
                self._actions[tid].motion, self._actions[tid].energy = lvl, en

        if self.hazard_det is not None and now - self._last_zoom >= 1.0:   # mouth zoom, 1x/s
            chewers = [d for d in dogs if d.track_id in self._actions and
                       0.5 * (self._actions[d.track_id].mouth + self._actions[d.track_id].chew)
                       >= self.engine.eat_thr]
            self._zoom = []
            for d in chewers[:2]:
                self._zoom += self.hazard_det.detect_crop(frame, d.box)
            self._last_zoom = now
        hz = self._hazards + [z for z in self._zoom if not any(z.name == x.name for x in self._hazards)]

        found = self.engine.update(dogs, hz, self.zones, (w, h), hazards_fresh=fresh,
                                   actions=self._actions if self.classifier else None)
        st.assessments, st.dogs = found, len(dogs)

        # ---------- status text ----------
        if dogs:
            self._last_seen = now
            top = max(found, key=lambda a: a.level)
            st.level = top.level
            if top.level == 0:
                calm = [a.reason for a in found if a.reason not in ("all calm", "resting")]
                if len(found) == 1 and calm:
                    st.status = f"{self.dog_name.capitalize()} is {calm[0]}"
                elif calm:
                    st.status = f"All calm - {len(found)} dogs ({', '.join(sorted(set(calm)))})"
                else:
                    st.status = "All calm"
            else:
                verb = "" if top.reason.startswith(("has been", "the ")) else "is "
                who = "" if top.reason.startswith("the ") else f"{self.dog_name} "
                st.status = f"{LEVELS[top.level].upper()}: {who}{verb}{top.reason}"
            alert = self.engine.should_alert(found)
            if alert is not None:
                self._fire(alert, frame, dogs)
        else:
            gone = now - self._last_seen if self._last_seen else None
            st.level = 0
            st.status = "No dog in view" if gone is None or gone > 3 else st.status

        # ---------- drawing ----------
        view = frame.copy()
        overlay.reset_labels()
        self.editor.size = (w, h)
        overlay.draw_zones(view, self.zones, self.editor)
        if self.show_hazards:
            overlay.draw_hazards(view, hz)
        overlay.draw_dogs(view, dogs, {a.track_id: a for a in found})

        self._fps_n += 1
        tn = time.perf_counter()
        if tn - self._fps_t >= 1.0:
            st.fps, self._fps_n, self._fps_t = self._fps_n / (tn - self._fps_t), 0, tn
        return view, st

    def debug_lines(self) -> list[str]:
        t = self.state.timings
        lines = [f"FPS {self.state.fps:.1f}", f"dogs {t.get('dogs', 0):.0f} ms",
                 f"hazards {t.get('hazards', 0):.0f} ms every {self.hazard_every}",
                 f"actions {t.get('actions', 0):.0f} ms", f"device {self.device}"]
        if self.vlm:
            lines.append(f"VLM {'busy' if self.vlm.busy else 'ready' if self.vlm.available else 'off'}"
                         f" last {self.vlm.last_ms:.0f} ms")
        lines += [f"#{tid}: {r.label} {r.conf:.2f} eat {0.5 * (r.mouth + r.chew):.2f} {r.motion}"
                  for tid, r in list(self._actions.items())[:4]]
        return lines

    # ------------------------------------------------------------------
    def _fire(self, a, frame, dogs) -> None:
        text = phrase(a.level, a.reason, self.dog_name)
        ev_id = self.events.add(a, frame)
        ev = {"id": ev_id, "level": a.level, "text": text, "reason": a.reason, "vlm": None,
              "ts": time.time()}
        self.state.last_alert = ev
        self.log(f"[ALERT {LEVELS[a.level].upper()}] {time.strftime('%H:%M:%S')} {text}")
        if not self.dnd:
            if self.voice:
                self.speaker.say(text, urgent=a.level == 3)
            if self.toasts:
                toast(f"BantayAso - {LEVELS[a.level].upper()}", text)
            if a.level == 3 and self.owner_clip:
                play_clip(self.owner_clip)
        if self.vlm is not None:
            box = next((d.box for d in dogs if d.track_id == a.track_id), None)
            if box is not None:
                def done(sentence, ev=ev, level=a.level):
                    ev["vlm"] = sentence
                    self.events.set_vlm(ev["id"], sentence)
                    self.log(f"[LOCAL AI] {sentence}")
                    if self.speak_vlm and self.voice and not self.dnd:
                        self.speaker.say(sentence, urgent=False)
                    if self.on_event:
                        self.on_event(ev)
                self.vlm.describe(frame, box, done)
        if self.on_event:
            self.on_event(ev)
