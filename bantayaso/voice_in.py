"""Local speech recognition for Ask Bantay (OpenAI Whisper, offline after the first download).

- push_to_talk(): record one question (stops after ~1 s of silence, max 8 s), transcribe, callback.
- hands-free: keeps listening; only questions that contain the wake word "Bantay" are answered.
Uses the GPU when available (base.en is small: ~0.3 s per question) and falls back to the CPU.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

RATE = 16000
BLOCK = 480                      # 30 ms


def looks_hallucinated(text: str, prompt: str = "") -> bool:
    import re
    import zlib
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return True
    if prompt and text.strip().lower().rstrip(".") in prompt.lower():
        return True                                       # echo of the prompt
    run = best = 1
    for a, b in zip(words, words[1:]):                    # "six six six six six"
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 4:
        return True
    if len(words) >= 8 and len(set(words)) / len(words) < 0.35:
        return True
    raw = text.encode()
    return len(raw) > 40 and len(raw) / max(1, len(zlib.compress(raw))) > 2.4


class Listener:
    def __init__(self, models_dir: Path, model: str = "base.en", language: str | None = "en",
                 is_speaking=lambda: False, log=print):
        self.models_dir = Path(models_dir) / "whisper"
        self.model_name, self.language = model, language
        self.is_speaking = is_speaking           # pause while our own voice is talking
        self.log = log
        self._model = None
        self._lock = threading.Lock()
        self.busy = False
        self.hands_free = False
        self._hf_thread = None
        self.on_text = None                      # callback(text, wake_word_heard: bool)
        self.on_state = None                     # callback("listening" / "thinking" / "idle")
        self.vocab = ["Bantay", "BantayAso"]     # spelling hints for Whisper (dog names added later)
        self.on_heard = None                     # callback(text) for every hands-free utterance (debug view)
        self.level = 0.0
        self._expect_until = 0.0                 # after a bare "Bantay", the next sentence is the question

    # ---------------- model ----------------
    def _load(self):
        with self._lock:
            if self._model is None:
                import whisper
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
                name = self.model_name
                if self.language not in (None, "en") and name.endswith(".en"):
                    name = name[:-3]                     # .en models only understand English
                self.log(f"[BantayAso] loading speech recognition (Whisper {name} on {dev})...")
                try:
                    self._model = whisper.load_model(name, device=dev, download_root=str(self.models_dir))
                except Exception:                        # e.g. GPU memory full -> CPU
                    self._model = whisper.load_model(name, device="cpu", download_root=str(self.models_dir))
                self._fp16 = next(self._model.parameters()).is_cuda
        return self._model

    def transcribe(self, audio: np.ndarray, hint: bool = True) -> str:
        """Speech -> text, with filters for Whisper's known hallucinations on noise/silence:
        echoing the prompt ("Bantay, Bantay, BantayAso") and repetition loops ("six, six, six...")."""
        if audio is None or len(audio) < RATE * 0.4:
            return ""
        model = self._load()
        kw = dict(language=self.language, fp16=getattr(self, "_fp16", False), beam_size=1,
                  condition_on_previous_text=False, temperature=0.0, no_speech_threshold=0.6)
        if hint:     # dog names help spelling; only for push-to-talk (hands-free would echo it)
            names = [v for v in dict.fromkeys(self.vocab) if v.lower() not in ("bantay", "bantayaso")]
            if names:
                kw["initial_prompt"] = "My dogs are " + ", ".join(names) + "."
        res = model.transcribe(audio.astype(np.float32), **kw)
        segs = res.get("segments") or []
        if segs and all(sg.get("no_speech_prob", 0) > 0.6 for sg in segs):
            return ""                                    # it was noise, not speech
        if segs and sum(sg.get("avg_logprob", 0) for sg in segs) / len(segs) < -1.2:
            return ""                                    # very unsure: probably noise
        text = (res.get("text") or "").strip()
        return "" if looks_hallucinated(text, kw.get("initial_prompt", "")) else text

    # ---------------- recording ----------------
    def record_utterance(self, max_s: float = 8.0, wait_s: float = 6.0) -> np.ndarray | None:
        """Wait up to wait_s for speech, then record until ~1 s of silence (max max_s)."""
        import sounddevice as sd
        frames, started, silence, t0 = [], False, 0, time.monotonic()
        floor = None
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK) as st:
            while True:
                block, _ = st.read(BLOCK)
                block = block[:, 0].copy()
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
                if floor is None:
                    floor = rms
                loud = rms > max(0.008, floor * 2.5)
                if not loud and not started:                 # learn the room noise from quiet blocks only
                    floor = 0.95 * floor + 0.05 * rms
                self.level = rms
                if self.is_speaking():
                    frames, started, silence = [], False, 0
                    if time.monotonic() - t0 > wait_s + max_s:
                        return None
                    continue
                if not started:
                    frames = (frames + [block])[-10:]          # keep 300 ms before speech starts
                    if loud:
                        started, t_start = True, time.monotonic()
                    elif time.monotonic() - t0 > wait_s:
                        return None
                    continue
                frames.append(block)
                silence = 0 if loud else silence + 1
                if silence * BLOCK / RATE > 0.6 or time.monotonic() - t_start > max_s:
                    break
        return np.concatenate(frames)

    # ---------------- modes ----------------
    def _state(self, s):
        if self.on_state:
            self.on_state(s)

    def push_to_talk(self) -> bool:
        if self.busy:
            return False
        self.busy = True

        def run():
            try:
                self._state("listening")
                audio = self.record_utterance(wait_s=5)
                self._state("thinking")
                text = self.transcribe(audio) if audio is not None else ""
                if self.on_text:
                    self.on_text(text, True)
            except Exception as e:
                self.log(f"[BantayAso] microphone error: {e}")
                if self.on_text:
                    self.on_text("", True)
            finally:
                self.busy = False
                self._state("idle")
        threading.Thread(target=run, name="PushToTalk", daemon=True).start()
        return True

    def set_hands_free(self, on: bool) -> None:
        self.hands_free = on
        if on and (self._hf_thread is None or not self._hf_thread.is_alive()):
            self._hf_thread = threading.Thread(target=self._hands_free_loop, name="HandsFree", daemon=True)
            self._hf_thread.start()

    def _hands_free_loop(self):
        from .qa import strip_wake
        self._state("hands_free")
        errors = 0
        while self.hands_free:
            if self.busy:
                time.sleep(0.2)
                continue
            try:
                audio = self.record_utterance(max_s=8, wait_s=30)
                if audio is None or not self.hands_free:
                    continue
                text = self.transcribe(audio, hint=False)
                errors = 0
                if not text:
                    continue
                self.log(f"[BantayAso] heard (ignored unless it starts with \"Bantay\"): {text!r}")
                if self.on_heard:
                    self.on_heard(text)
                woke, rest = strip_wake(text, strict=True)      # wake word must START the sentence
                if woke and not rest:
                    # bare "Bantay": answer "Yes?" and treat the next sentence as the question
                    self._expect_until = time.monotonic() + 12
                    if self.on_text:
                        self.on_text("Bantay", True)
                elif woke or time.monotonic() < self._expect_until:
                    self._expect_until = 0.0
                    if self.on_text:
                        self.on_text(text, True)
            except Exception as e:
                errors += 1
                self.log(f"[BantayAso] hands-free error ({errors}): {e!r}")
                if errors >= 3:
                    self.log("[BantayAso] hands-free listening stopped (microphone problem)")
                    self.hands_free = False
                    self._state("hf_error")
                    return
                time.sleep(1.0)
        self._state("idle")
