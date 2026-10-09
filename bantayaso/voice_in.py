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


def _sound_key(s: str) -> str:
    """Rough 'how it sounds' spelling, so Whisper's guesses match a dog's name:
    'Patchouchai', 'pa choo chay', 'Pachuchai' -> all close to 'Pachuchay'."""
    import re
    s = re.sub(r"[^a-z]", "", s.lower())
    for a, b in (("tch", "C"), ("ch", "C"), ("ts", "C"), ("sh", "C"), ("j", "C"), ("ph", "f"), ("ck", "k"), ("qu", "k"),
                 ("c", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("oo", "u"), ("ou", "u"), ("ew", "u"),
                 ("ay", "E"), ("ai", "E"), ("ei", "E"), ("ey", "E"), ("ee", "i"), ("ea", "i"), ("ie", "i"),
                 ("y", "i"), ("h", ""), ("w", "u")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


def correct_names(text: str, names) -> str:
    """Replace a mis-heard dog name (1-3 words that SOUND like it) with the real name."""
    import difflib
    import re
    names = [n for n in names if n and len(_sound_key(n)) >= 3 and n.lower() not in ("bantay", "bantayaso")]
    if not names or not text:
        return text
    stop = {"is", "are", "was", "were", "the", "a", "an", "what", "where", "how", "did", "does", "do", "on",
            "in", "at", "my", "your", "and", "or", "it", "he", "she", "they", "si", "ang", "ba", "ni", "kay",
            "sa", "na", "ng", "doing", "bantay", "bantayaso", "okay", "ok", "dog", "dogs", "brown", "white", "black",
            "grey", "gray", "one", "two", "patch", "pooch", "puppy", "bed", "today", "now"}
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*|[^A-Za-z]+", text)
    idx = [i for i, w in enumerate(words) if re.match(r"[A-Za-z]", w)]
    out_spans = []
    i = 0
    while i < len(idx):
        best = None
        for n in names:
            nk = _sound_key(n)
            for span in (1, 2, 3):
                if i + span > len(idx):
                    continue
                parts = [words[idx[k]] for k in range(i, i + span)]
                if any(p.lower() in stop for p in parts):
                    continue                              # "is", "the", "Bantay"... are never part of a name
                cand = "".join(parts)
                ck = _sound_key(cand)
                if not ck or abs(len(ck) - len(nk)) > max(2, len(nk) // 3):
                    continue
                if cand.lower() == n.lower():
                    r = 1.0
                else:
                    r = difflib.SequenceMatcher(None, ck, nk).ratio()
                need = 0.8 if len(nk) >= 5 else 0.9
                if r >= need and (best is None or r > best[0] + 0.02):
                    best = (r, n, span)
        if best:
            out_spans.append((idx[i], idx[i + best[2] - 1], best[1]))
            i += best[2]
        else:
            i += 1
    for a, b, n in reversed(out_spans):
        words[a:b + 1] = [n]
    return "".join(words)


def prepare_audio(audio):
    """Remove DC offset and apply bounded gain; never turn silence into loud noise."""
    audio=np.asarray(audio,dtype=np.float32)
    audio=np.nan_to_num(audio,nan=0.0,posinf=0.0,neginf=0.0)
    audio=audio-float(np.mean(audio)) if audio.size else audio
    rms=float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    if rms<0.001:return audio
    return np.clip(audio*min(3.0,0.08/max(rms,1e-8)),-1,1)


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
        self.names_fn = lambda: [v for v in self.vocab if v.lower() not in ("bantay", "bantayaso")]
        self.beam_size = 5                       # more careful decoding than greedy (slightly slower)
        self.last_raw = ""                       # exactly what Whisper wrote, before name correction
        self.on_heard = None                     # callback(text) for every hands-free utterance (debug view)
        self.level = 0.0
        self.recording_speech = False
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
        audio = prepare_audio(audio)
        if float(np.sqrt(np.mean(audio ** 2))) < 0.001:
            self._state("too_quiet")
            return ""
        model = self._load()
        kw = dict(language=self.language, fp16=getattr(self, "_fp16", False), beam_size=self.beam_size,
                  condition_on_previous_text=False, temperature=0.0, no_speech_threshold=0.6)
        names = list(dict.fromkeys(self.names_fn() or []))
        if hint:     # dog names help spelling; only for push-to-talk (hands-free would echo it)
            if names:
                kw["initial_prompt"] = "My dogs are " + ", ".join(names) + "."
        res = model.transcribe(audio.astype(np.float32), **kw)
        segs = res.get("segments") or []
        if segs and all(sg.get("no_speech_prob", 0) > 0.6 for sg in segs):
            return ""                                    # it was noise, not speech
        if segs and sum(sg.get("avg_logprob", 0) for sg in segs) / len(segs) < -1.2:
            return ""                                    # very unsure: probably noise
        text = (res.get("text") or "").strip()
        if looks_hallucinated(text, kw.get("initial_prompt", "")):
            return ""
        self.last_raw = text
        fixed = correct_names(text, names)
        if fixed != text:
            self.log(f"[BantayAso] heard {text!r} -> understood {fixed!r}")
        return fixed

    # ---------------- recording ----------------
    def record_utterance(self, max_s: float = 12.0, wait_s: float = 6.0, hands_free: bool = False) -> np.ndarray | None:
        """Wait up to wait_s for speech, then record until ~1 s of silence (max max_s)."""
        import sounddevice as sd
        frames, started, silence, t0 = [], False, 0, time.monotonic()
        floor = None
        with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=BLOCK) as st:
            while True:
                if hands_free and (self.busy or not self.hands_free):
                    self.recording_speech = False
                    return None
                block, _ = st.read(BLOCK)
                block = block[:, 0].copy()
                rms = float(np.sqrt(np.mean(block ** 2)) + 1e-9)
                if floor is None:
                    floor = min(rms, 0.003)
                loud = rms > max(0.004, floor * 2.5)
                if not loud and not started:                 # learn the room noise from quiet blocks only
                    floor = 0.95 * floor + 0.05 * rms
                self.level = rms
                self.recording_speech = started or loud
                if self.is_speaking():
                    self.recording_speech = False
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
                if silence * BLOCK / RATE > 0.85 or time.monotonic() - t_start > max_s:
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
                self.recording_speech = False
                self.busy = False
                self._state("idle")
        threading.Thread(target=run, name="PushToTalk", daemon=True).start()
        return True

    def open_followup(self):
        if self.hands_free:
            self._expect_until = time.monotonic() + 8
            self._state('followup')

    def set_hands_free(self, on: bool) -> None:
        self.hands_free = on
        if not on:
            self._expect_until = 0.0
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
                audio = self.record_utterance(max_s=12, wait_s=30, hands_free=True)
                self.recording_speech = False
                if audio is None or not self.hands_free:
                    continue
                text = self.transcribe(audio, hint=False)
                errors = 0
                if not text:
                    continue
                self.log(f"[BantayAso] heard (ignored unless it starts with \"Bantay\"): {text!r}")
                if self.on_heard:
                    self.on_heard(text if text == self.last_raw else f"{text}\x1f{self.last_raw}")
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
