"""The whole per-frame brain, shared by the OpenCV window (__main__) and the desktop UI (ui/).

frame -> dogs (YOLO11) -> hazards (YOLOE, every N frames) -> actions/mouth (CLIP, ~4 Hz)
      -> motion -> risk engine -> alerts (voice/toast/clip) + event log + local VLM sentence
"""
from __future__ import annotations

import time
import queue
from dataclasses import dataclass, field

import numpy as np

from . import config, overlay
from .alerts import Speaker, phrase, play_clip, toast
from .events import EventLog
from .history import ActivityHistory
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
    boxes: list = field(default_factory=list)   # [(track_id, box, name)]
    frame_w: int = 1280


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

        try:                                   # ultralytics 8.4 warns "'half' is deprecated" every frame
            import logging
            from ultralytics.utils import LOGGER as _ULOG
            _ULOG.setLevel(logging.ERROR)
        except Exception:
            pass
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

        self.registry = None
        if self.classifier is not None:
            from .names import DogRegistry
            nm = cfg.get("names", {})
            self.registry = DogRegistry(config.DATA_DIR / "dogs", threshold=nm.get("threshold", 0.80),
                                        margin=nm.get("margin", 0.02))
        self.history = ActivityHistory(10)
        from .persona import Persona
        self.persona = Persona(config.DATA_DIR, llm_call=self._chat)
        qa = cfg.get("qa", {})
        self.qa_model = qa.get("model", "qwen2.5:1.5b")
        self.ollama_url = cfg["models"].get("ollama_url", "http://localhost:11434")
        self.listener = None
        if qa.get("enabled", True):
            try:
                from .voice_in import Listener
                self.listener = Listener(config.MODELS_DIR, model=qa.get("whisper_model", "base.en"),
                                         language=qa.get("language", "en") or None,
                                         is_speaking=lambda: self.speaker.speaking, log=log)
                import threading as _th
                _th.Thread(target=lambda: self.listener and self.listener._load(), daemon=True).start()
            except Exception as e:                # pragma: no cover
                log(f"[BantayAso] voice questions unavailable: {e}")
        self.motion = MotionMeter()
        self.zones = load_zones(cfg)
        self.editor = ZoneEditor(self.zones)
        self.engine = RiskEngine(cfg)
        self.events = EventLog(config.DATA_DIR / "events.db", config.DATA_DIR / "snapshots")
        self.speaker = Speaker(rate=int(al.get("voice_rate", 1)), voice=al.get("voice_name", ""))
        self.voice = bool(al.get("voice", True))
        self.dnd = bool(al.get("do_not_disturb", False))
        self.toasts = bool(al.get("toast", True))
        self.speak_vlm = bool(al.get("speak_vlm", True))
        self.owner_clip = al.get("owner_voice_clip", "")
        self.dog_name = "your dog"           # each dog's own name comes from the Dogs page

        self.hazard_every = int(det.get("hazard_every", 5))
        self.action_every = float(det.get("action_every_seconds", 0.25))
        self.show_hazards, self.show_debug = True, False
        self.zone_help = False                 # OpenCV window sets True (the app has its own instructions)
        self.state = State()
        self._hazards, self._actions, self._zoom = [], {}, []
        self._n, self._last_act, self._last_zoom, self._last_seen = 0, 0.0, 0.0, 0.0
        self._fps_n, self._fps_t = 0, time.perf_counter()
        self.on_event = None                # optional callback(event_dict) for the UI
        self._pending_vocab = None
        self._teaching_requests = queue.Queue(maxsize=16)
        self._teaching_targets = ()
        self._teaching_frame_at = 0.0
        self._lesson_subjects = {}
        self.on_teaching = None
        if self.classifier:
            self.classifier.on_teaching = self._teaching_event

    # ------------------------------------------------------------------
    def save_zones(self) -> None:
        latest = config.load()              # don't clobber edits made while running
        latest["zones"] = zones_to_cfg(self.zones)
        config.save(latest)

    def _pairs(self, frame, dogs, now) -> dict:
        """Dogs whose boxes touch and are both moving get a CLIP 'fighting?' score (~3 per second)."""
        if self.classifier is None or len(dogs) < 2:
            self._pair_cache = {}
            return {}
        if now - getattr(self, "_last_pair", 0.0) < 0.33:
            return getattr(self, "_pair_cache", {})
        self._last_pair = now
        from .detect_dog import DogDetector
        out = {}
        moving = {tid for tid, r in self._actions.items() if r.motion in ("active", "frantic")}
        for i, a in enumerate(dogs):
            for b in dogs[i + 1:]:
                touching = DogDetector._overlap(a.box, b.box) > 0.15
                if not touching or not ({a.track_id, b.track_id} <= moving):
                    continue
                sc = self.classifier.pair_score(frame, a.box, b.box)
                for x, y in ((a, b), (b, a)):
                    if sc > out.get(x.track_id, (None, 0.0))[1]:
                        out[x.track_id] = (y.track_id, sc)
        self._pair_cache = out
        return out

    def request_vocab(self, vocab: dict) -> None:
        """Things page: apply a new object list on the next frame (pipeline thread)."""
        self._pending_vocab = dict(vocab)

    def _teaching_event(self, event):
        self.log('[TEACH] ' + event['message'])
        if self.on_teaching:
            self.on_teaching(event)
        if event['status'] in ('saved', 'cancelled'):
            self._lesson_subjects.pop(event['track_id'], None)
        if event['status'] == 'saved' and self.voice and not self.dnd:
            self.speaker.say(event['message'])

    def request_teach(self, tid: int, label: str, subject: str | None = None) -> str:
        """Thread-safe request; camera/identity validation and capture happen on process()."""
        if self.classifier is None or label not in self.classifier.labels:
            return 'I cannot teach that action. Choose an action from the teaching menu.'
        if time.monotonic() - self._teaching_frame_at > 2.0:
            return 'I need a current camera view before learning. No examples saved.'
        targets = dict(self._teaching_targets)
        if tid < 0 or tid not in targets or (subject and targets[tid] != subject):
            return 'I cannot clearly identify that dog right now. Please select it again.'
        try:
            self._teaching_requests.put_nowait(('teach', tid, label, subject, time.monotonic()))
        except queue.Full:
            return 'Please wait for the current lesson, then try again.'
        return f"I'll remember {label} after collecting examples. Keep the dog doing it until I say saved."

    def request_reset_examples(self):
        try:
            self._teaching_requests.put_nowait(('reset', None, None, None, time.monotonic()))
            return True
        except queue.Full:
            return False

    def cancel_teaching(self, reason):
        """Processing-thread only, also called when the source disconnects/changes."""
        self._teaching_frame_at = 0.0
        self._teaching_targets = ()
        if self.classifier:
            self.classifier.cancel_teaching(reason)
        while True:
            try:
                kind, tid, label, _, _ = self._teaching_requests.get_nowait()
            except queue.Empty:
                break
            if kind == 'teach':
                self._teaching_event({'track_id': tid, 'label': label, 'status': 'cancelled',
                                      'message': f'Lesson cancelled: {reason} No examples saved.'})
            elif self.classifier:
                self.classifier.forget_examples()

    def _prepare_teaching(self, dogs, now):
        if self.classifier is None:
            return
        visible = {d.track_id for d in dogs if d.track_id >= 0 and getattr(d, 'observed', True)}
        self.classifier.check_teaching(visible, now)
        names = self.registry.names_by_tid if self.registry else {}
        for tid, subject in list(self._lesson_subjects.items()):
            if subject and names.get(tid) != subject:
                self.classifier.cancel_teaching('The dog identity is no longer certain.', tid)
        while True:
            try:
                kind, tid, label, subject, requested = self._teaching_requests.get_nowait()
            except queue.Empty:
                break
            if kind == 'reset':
                self.classifier.forget_examples()
                self._teaching_event({'track_id': -1, 'label': '', 'status': 'reset',
                                      'message': 'Taught actions reset.'})
            elif now - requested > 2.0 or tid not in visible or (subject and names.get(tid) != subject):
                self._teaching_event({'track_id': tid, 'label': label, 'status': 'cancelled',
                                      'message': 'Lesson cancelled: the dog is no longer clear. No examples saved.'})
            elif self.classifier.teach(tid, label):
                self._lesson_subjects[tid] = subject
                self._teaching_event({'track_id': tid, 'label': label, 'status': 'learning',
                                      'message': f'Learning {label}. Keep the dog doing this until examples are saved.'})

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, State]:
        st = self.state
        if self._pending_vocab is not None:
            v, self._pending_vocab = self._pending_vocab, None
            self.engine.vocab_tiers = {str(k): int(x) for k, x in v.items()}
            if self.hazard_det is not None:
                try:
                    self.hazard_det.update_vocab(v)
                    self.log(f"[BantayAso] object list updated ({sum(1 for x in v.values() if x > 0)} things)")
                except Exception as e:
                    self.log(f"[BantayAso] could not update the object list: {e}")
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
        self._prepare_teaching(dogs, now)
        action_updated = False
        if self.classifier is not None and dogs and now - self._last_act >= self.action_every:
            t2 = time.perf_counter()
            self._actions = self.classifier(frame, dogs, collect=False)
            action_updated = True
            st.timings["actions"] = (time.perf_counter() - t2) * 1000
            self._last_act = now
        for tid, (lvl, en) in motions.items():
            if tid in self._actions:
                r = self._actions[tid]
                r.motion, r.energy = lvl, en
                # scratching/digging/jumping are vigorous: if the dog is still, CLIP's vote for
                # them is noise (seen in calibration), so fall back to the best calm pose
                from .actions import MOTION_LABELS
                if r.label in MOTION_LABELS and lvl == "still" and r.pose:
                    r.label = r.pose

        if self.hazard_det is not None and now - self._last_zoom >= 1.0:   # mouth zoom, 1x/s
            chewers = [d for d in dogs if d.track_id in self._actions and
                       0.5 * (self._actions[d.track_id].mouth + self._actions[d.track_id].chew)
                       >= self.engine.eat_thr]
            self._zoom = []
            for d in chewers[:2]:
                self._zoom += self.hazard_det.detect_crop(frame, d.box)
            self._last_zoom = now
        hz = self._hazards + [z for z in self._zoom if not any(z.name == x.name for x in self._hazards)]

        pairs = self._pairs(frame, dogs, now)
        found = self.engine.update(dogs, hz, self.zones, (w, h), hazards_fresh=fresh,
                                   actions=self._actions if self.classifier else None, pairs=pairs)
        names = {}
        if self.registry is not None:
            for old, new in self.engine.handovers:
                self.registry.handover(old, new)
            if st.timings.get("_act_t") != self._last_act:          # new embeddings this frame
                st.timings["_act_t"] = self._last_act
                emb = {d.track_id: self.classifier.embeddings[d.track_id] for d in dogs
                       if d.track_id in self.classifier.embeddings}
                self.registry.update(emb, {d.track_id for d in dogs})
            names = {d.track_id: self.registry.names_by_tid[d.track_id] for d in dogs
                     if d.track_id in self.registry.names_by_tid}
        self.names = names
        if action_updated:
            for tid, subject in list(self._lesson_subjects.items()):
                if subject and names.get(tid) != subject:
                    self.classifier.cancel_teaching('The dog identity is no longer certain.', tid)
            self.classifier.collect_teaching(frame, dogs)
        self._teaching_targets = tuple((d.track_id, names.get(d.track_id)) for d in dogs
                                       if d.track_id >= 0 and getattr(d, 'observed', True))
        self._teaching_frame_at = time.monotonic()
        st.assessments, st.dogs = found, len(dogs)
        st.boxes, st.frame_w = [(d.track_id, d.box, names.get(d.track_id)) for d in dogs], w
        self.history.record(now, found, names)

        # ---------- status text ----------
        if dogs:
            self._last_seen = now
            top = max(found, key=lambda a: a.level)
            st.level = top.level
            if top.level == 0:
                calm = [a.reason for a in found if a.reason not in ("all calm", "resting")]
                if len(found) == 1 and calm:
                    who = names.get(found[0].track_id) or self.dog_name
                    st.status = f"{who[0].upper() + who[1:]} is {calm[0]}"
                elif calm:
                    st.status = f"All calm - {len(found)} dogs ({', '.join(sorted(set(calm)))})"
                else:
                    st.status = "All calm"
            else:
                verb = "" if top.reason.startswith(("has been", "the ")) else "is "
                who = "" if top.reason.startswith("the ") else f"{names.get(top.track_id) or self.dog_name} "
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
        if self.editor.active:                 # zones are only drawn while you edit them
            overlay.draw_zones(view, self.zones, self.editor, help_bar=self.zone_help)
        if self.show_hazards:
            overlay.draw_hazards(view, hz)
        overlay.draw_dogs(view, dogs, {a.track_id: a for a in found}, names)

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

    # ------------------------------------------------------------------ Ask Bantay
    def ask(self, question: str, speak: bool = True) -> str:
        from .qa import answer, guard
        if self.listener is not None and self.registry is not None:     # dog names help Whisper spell
            self.listener.vocab = ["Bantay", "BantayAso", *self.registry.dogs.keys()]
        try:
            text = self.persona.style.command(question)          # "keep it short", "speak Tagalog"...
            if text is None:
                self.persona.style.observe(question)            # learn language / length / formality
                text = answer(question, self)
                text = self.persona.finish(question, text, guard)
        except Exception as e:                     # pragma: no cover
            text = "Sorry, I couldn't work that out."
            self.log(f"[BantayAso] Q&A error: {e}")
        self.log(f"[ASK] {question!r} -> {text}")
        if speak and self.voice:
            self.speaker.say(text, urgent=True)
        return text

    def _chat(self, messages: list, schema: dict | None, max_tokens: int) -> str | None:
        """One local Ollama chat call (structured JSON output when supported)."""
        try:
            import requests
            body = {"model": self.qa_model, "messages": messages, "stream": False, "keep_alive": "30m",
                    "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 2048}}
            if schema:
                body["format"] = schema
            r = requests.post(self.ollama_url.rstrip("/") + "/api/chat", json=body, timeout=20)
            if r.status_code >= 400 and schema:          # older Ollama: no schema support -> plain JSON mode
                body["format"] = "json"
                r = requests.post(self.ollama_url.rstrip("/") + "/api/chat", json=body, timeout=20)
            r.raise_for_status()
            return ((r.json().get("message") or {}).get("content") or "").strip() or None
        except Exception as e:
            self.log(f"[BantayAso] local LLM unavailable ({e.__class__.__name__}); using templates")
            return None

    def ask_llm(self, question: str, facts: str) -> str | None:
        """Open question -> grounded, style-aware answer. Returns text, 'OUT_OF_SCOPE', 'UNKNOWN' or None."""
        res = self.persona.ask_llm(question, facts)
        if res is None:
            return None
        status, ans = res
        if status == "out_of_scope":
            return "OUT_OF_SCOPE"
        if status == "unknown" or not ans:
            return "UNKNOWN"
        return ans

    # ------------------------------------------------------------------
    def _fire(self, a, frame, dogs) -> None:
        who = getattr(self, "names", {}).get(a.track_id)
        text = phrase(a.level, a.reason, who or self.dog_name)
        ev_id = self.events.add(a, frame, dog_label=who)
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
