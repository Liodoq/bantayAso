"""Local speech recognition for Ask Bantay (OpenAI Whisper, offline after the first download).

- push_to_talk(): record one question (stops after ~1 s of silence, max 8 s), transcribe, callback.
- hands-free: keeps listening; only questions that contain the wake word "Bantay" are answered.
Runs Whisper on the CPU so the 4 GB GPU stays free for the vision models.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

RATE = 16000
BLOCK = 480                      # 30 ms


class Listener:
    def __init__(self, models_dir: Path, model: str = "base", language: str | None = "en",
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

    # ---------------- model ----------------
    def _load(self):
        with self._lock:
            if self._model is None:
                import whisper
                self.log("[BantayAso] loading speech recognition (Whisper)...")
                self._model = whisper.load_model(self.model_name, device="cpu",
                                                 download_root=str(self.models_dir))
        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < RATE * 0.4:
            return ""
        model = self._load()
        res = model.transcribe(audio.astype(np.float32), language=self.language, fp16=False,
                               condition_on_previous_text=False, temperature=0.0)
        return (res.get("text") or "").strip()

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
                floor = rms if floor is None else (0.95 * floor + 0.05 * rms if not started else floor)
                loud = rms > max(0.012, floor * 3.0)
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
                if silence * BLOCK / RATE > 0.9 or time.monotonic() - t_start > max_s:
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
        while self.hands_free:
            if self.busy:
                time.sleep(0.2)
                continue
            try:
                audio = self.record_utterance(max_s=8, wait_s=30)
                if audio is None or not self.hands_free:
                    continue
                text = self.transcribe(audio)
                woke, _ = strip_wake(text)
                if woke and self.on_text:
                    self.on_text(text, True)
            except Exception as e:
                self.log(f"[BantayAso] hands-free listening stopped: {e}")
                self.hands_free = False
                self._state("idle")
                return
