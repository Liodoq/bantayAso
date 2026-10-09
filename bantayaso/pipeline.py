"""The whole per-frame brain, shared by the OpenCV window (__main__) and the desktop UI (ui/).

frame -> dogs (YOLO11) -> hazards (YOLOE, every N frames) -> actions/mouth (CLIP, ~4 Hz)
      -> motion -> risk engine -> alerts (voice/toast/clip) + event log + local VLM sentence
"""
from __future__ import annotations

import time
import queue
import threading
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
                                   conf=det.get("dog_conf", 0.25), imgsz=det.get("imgsz", 960),
                                   aliases=det.get("dog_aliases", True), alias_conf=det.get("alias_conf", 0.30),
                                   enhance=det.get("enhance_dark", True))
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
        self._ask_lock = threading.Lock()
        self._reply_generation = 0
        self._asking = False
        from .companion import Companion
        self.companion = Companion()
        self.companion_enabled = cfg.get('qa', {}).get('calm_checkins', True)
        self._companion_observations = ()
        self._companion_at = 0.0
        from .scene import Presence
        self.presence = Presence()
        self.entry_alerts = cfg.get('qa', {}).get('entry_alerts', True)
        self.scene = None
        self._objects_at = 0.0
        self.on_notice = None
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
        from .zone_align import ZoneAligner
        self.aligner = ZoneAligner(config.DATA_DIR / "zones_ref.png")   # follows a bumped/re-angled camera
        self._last_frame = None
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
    def reload_zones(self) -> None:
        """Throw away unsaved zone edits: back to what config.yaml has."""
        self.zones[:] = load_zones(config.load())

    def live_zones(self) -> list:
        """Zones moved to where the furniture is now (camera bumped/re-angled since they were drawn)."""
        if self.editor.editing or self.aligner.status not in ("moved", "lost"):
            return self.zones
        from .zones import Zone
        return [Zone(z.name, z.type, self.aligner.apply(z.points)) for z in self.zones]

    def bake_alignment(self) -> None:
        """Before editing: put the zones where they are drawn now, and make this view the new reference."""
        if self.aligner.status in ("moved", "lost"):
            for z in self.zones:
                z.points = self.aligner.apply(z.points)
        if self._last_frame is not None:
            self.aligner.set_reference(self._last_frame)

    def save_zones(self) -> None:
        latest = config.load()              # don't clobber edits made while running
        latest["zones"] = zones_to_cfg(self.zones)
        config.save(latest)
        if self._last_frame is not None:                # this view is what the zones were drawn on
            self.aligner.set_reference(self._last_frame)

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
        self._last_frame = frame
        if self.zones and not self.editor.editing:
            if not self.aligner.has_reference:          # zones from before this feature: today's view is the reference
                self.aligner.set_reference(frame)
            prev = self.aligner.status
            if self.aligner.update(frame) != prev and self.aligner.status in ("moved", "lost"):
                self.log("[BantayAso] camera moved: zones shifted to match" if self.aligner.status == "moved"
                         else "[BantayAso] camera view changed a lot: please check the zones")
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
            self._objects_at = time.monotonic()
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
        found = self.engine.update(dogs, hz, self.live_zones(), (w, h), hazards_fresh=fresh,
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

        if hasattr(self, 'presence'):
            reliable = self.aligner.status != 'lost' and not self.editor.active
            zones = self.live_zones() if reliable else []
            visible = [d for d in dogs if getattr(d, 'observed', True)]
            rows=[]
            for d in visible:
                center=((d.box[0]+d.box[2])/2, (d.box[1]+d.box[3])/2)
                memberships=[{'name':z.name,'type':z.type} for z in zones if z.contains(center,w,h)]
                rows.append({'id':d.track_id,'name':names.get(d.track_id),'box':tuple(d.box),'zones':memberships})
            self.scene={'at':time.monotonic(),'dogs':rows,'width':w,
                        'zones':[{'name':z.name,'type':z.type} for z in zones], 'zones_reliable':reliable,
                        'objects_enabled':self.hazard_det is not None, 'objects_at':self._objects_at,
                        'objects':tuple(getattr(self.hazard_det,'scene_objects',()))}
            counts={'camera':len(rows)}
            if reliable:
                counts.update({z.name:sum(any(m['name']==z.name for m in row['zones']) for row in rows) for z in zones})
            else:
                # Do not invent a zone entry when camera alignment recovers.
                self.presence.stable={k:v for k,v in self.presence.stable.items() if k=='camera'}
                self.presence.pending={k:v for k,v in self.presence.pending.items() if k=='camera'}
            signature=tuple((z.name,tuple(tuple(p) for p in z.points)) for z in self.zones) if reliable else None
            if signature != getattr(self,'_zone_entry_signature',None):
                self.presence.stable={k:v for k,v in self.presence.stable.items() if k=='camera'}
                self.presence.stable.update({k:v for k,v in counts.items() if k!='camera'})
                self.presence.pending={k:v for k,v in self.presence.pending.items() if k=='camera'}
                self._zone_entry_signature=signature
            for area,text in self.presence.update(now, counts):
                if not self.entry_alerts:
                    continue
                self.log('[ENTRY] '+text)
                if self.on_notice:
                    self.on_notice(text)
                if not self.dnd:
                    if self.toasts:
                        toast('Bantay: dog in view',text)
                    if self.voice and not self._asking and not (self.listener and (self.listener.busy or self.listener.recording_speech)):
                        def valid_entry(area=area, minimum=counts.get(area,0)):
                            current=self.scene['dogs']
                            count=len(current) if area=='camera' else sum(any(z['name']==area for z in d['zones']) for d in current)
                            return (self.voice and not self.dnd and self.entry_alerts and not self._asking
                                    and self.state.level < 2 and time.monotonic()-self.scene['at']<2 and count>=minimum
                                    and not (self.listener and (self.listener.busy or self.listener.recording_speech)))
                        self.speaker.say(text,casual=True,valid=valid_entry)

        # Calm check-ins use fresh detections, not held boxes or a language-model guess.
        if hasattr(self, 'companion'):
            observed = {d.track_id for d in dogs if getattr(d, 'observed', True)}
            safe = {a.track_id: a.level == 0 for a in found}
            self._companion_observations = tuple(
                (d.track_id, names.get(d.track_id), getattr(self._actions.get(d.track_id), 'label', ''),
                 d.track_id in observed and safe.get(d.track_id, False)
                 and now - self._last_act < 2
                 and getattr(self._actions.get(d.track_id), 'motion', '') == 'still') for d in dogs)
            self._companion_at = now
            message = self.companion.update(now, self._companion_observations,
                                             blocked=not self._can_check_in())
            if message:
                episode = self.companion.key
                def valid_checkin():
                    return self._can_check_in(ignore_speaker=True) and self.companion.key == episode and time.monotonic() - self._companion_at < 2
                if self.speaker.say(message, casual=True, valid=valid_checkin):
                    self.companion.mark_spoken(now)
                    self.log(f'[CHECK-IN] {message}')

        # ---------- drawing ----------
        view = frame.copy()
        overlay.reset_labels()
        self.editor.size = (w, h)
        if self.editor.active:                 # zones are only drawn while you edit them
            overlay.draw_zones(view, self.zones if self.editor.editing else self.live_zones(), self.editor,
                               help_bar=self.zone_help)
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
    def _can_check_in(self, ignore_speaker=False):
        listener = self.listener
        return (self.companion_enabled and self.voice and not self.dnd and not self._asking
                and self.state.level == 0 and (ignore_speaker or not getattr(self.speaker, "speaking", False))
                and not (listener and (listener.busy or getattr(listener, 'recording_speech', False)
                                        or time.monotonic() < listener._expect_until)))

    def stop_reply(self):
        self._reply_generation += 1
        self.speaker.stop()
        if self.listener:
            self.listener._expect_until = 0.0

    def ask(self, question: str, speak: bool = True) -> str:
        from .qa import answer, guard, strip_wake, OUT_OF_SCOPE, CANT_TELL
        import re
        _, command = strip_wake(question)
        if re.fullmatch(r'(stop( talking| speaking)?|be quiet|quiet|tama na)[.! ]*', command.lower()):
            self.stop_reply()
            return 'Okay.'
        if not self._ask_lock.acquire(blocking=False):
            return "I'm still finishing your previous question."
        generation = self._reply_generation
        self._asking = True
        self.persona.already_styled = False
        self.persona.protected_text = None
        started = time.monotonic()
        try:
            if self.listener is not None and self.registry is not None:
                self.listener.vocab = ['Bantay', 'BantayAso', *self.registry.dogs.keys()]
            text = self.persona.style.command(question)
            if text is None:
                self.persona.style.observe(question)
                text = answer(question, self)
                text = self.persona.finish(question, text, guard)
            if generation != self._reply_generation:
                return ''
            repeat_fallback = False
            if text in (OUT_OF_SCOPE, CANT_TELL):
                stamp=time.monotonic()
                repeat_fallback=(getattr(self,'_fallback_text',None)==text and stamp-getattr(self,'_fallback_at',0)<30)
                if not repeat_fallback:
                    self._fallback_text,self._fallback_at=text,stamp
            self.log(f'[ASK {time.monotonic() - started:.2f}s] {question!r} -> {text}')
            follow_up = self.listener.open_followup if self.listener else None
            if speak and self.voice and not self.dnd and not repeat_fallback:
                self.speaker.say(text, valid=lambda: self.voice and not self.dnd and generation == self._reply_generation,
                                 on_done=follow_up)
            elif follow_up:
                follow_up()
            return text
        except Exception as e:
            self.log(f'[BantayAso] Q&A error: {e}')
            return "Sorry, I couldn't work that out."
        finally:
            self._asking = False
            self._ask_lock.release()

    def _chat(self, messages: list, schema: dict | None, max_tokens: int) -> str | None:
        """One local Ollama chat call (structured JSON output when supported)."""
        try:
            import requests
            body = {"model": self.qa_model, "messages": messages, "stream": False, "keep_alive": "30m",
                    "options": {"temperature": 0.1, "num_predict": max_tokens, "num_ctx": 2048}}
            if schema:
                body["format"] = schema
            r = requests.post(self.ollama_url.rstrip("/") + "/api/chat", json=body, timeout=(2, 8))
            if r.status_code >= 400 and schema:          # older Ollama: no schema support -> plain JSON mode
                body["format"] = "json"
                r = requests.post(self.ollama_url.rstrip("/") + "/api/chat", json=body, timeout=(2, 8))
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
                self.speaker.say(text, urgent=a.level == 3, valid=lambda: self.voice and not self.dnd)
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
                        self.speaker.say(sentence, urgent=False, valid=lambda: self.voice and not self.dnd)
                    if self.on_event:
                        self.on_event(ev)
                self.vlm.describe(frame, box, done)
        if self.on_event:
            self.on_event(ev)
