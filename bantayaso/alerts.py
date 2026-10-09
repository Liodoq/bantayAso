"""Alerts: offline voice (Windows SAPI via pyttsx3), Windows toast, optional owner voice clip.

Everything runs on background threads so the video never freezes.
"""
from __future__ import annotations

import queue
import itertools
from .speech_text import spoken_text
import threading
import time
from pathlib import Path


class Speaker:
    """Speaks one message at a time from a queue, using Windows SAPI directly.

    pyttsx3 re-initialised on a background thread speaks once and then goes silent on Windows,
    so we talk to SAPI ("SAPI.SpVoice") ourselves: reliable, and it lets the user pick a voice.
    Falls back to pyttsx3 on other systems.
    """

    def __init__(self, rate: int = 0, voice: str = ""):
        self.q = queue.PriorityQueue(maxsize=8)
        self._sequence = itertools.count()
        self._generation = 0
        self._danger_until = 0.0
        self.current_priority = 2
        self.rate = rate                 # SAPI rate -10 (slow) .. 10 (fast)
        self.voice_name = voice          # part of a voice name, e.g. "Zira"
        self.enabled = True
        self.speaking = False
        self.voices: list[str] = []
        self._apply = True
        threading.Thread(target=self._run, name="Speaker", daemon=True).start()

    def say(self, text: str, urgent: bool = False, *, casual=False, valid=None, on_done=None) -> bool:
        if not self.enabled or not text:
            return False
        priority = 0 if urgent else 2 if casual else 1
        if urgent:
            self._danger_until = time.monotonic() + 15
        # Danger never gets displaced by a reply. Casual messages must be immediate.
        if casual and (self.speaking or not self.q.empty() or time.monotonic() < self._danger_until):
            return False
        item = (priority, next(self._sequence), self._generation,
                time.monotonic() + (2 if casual else 15), spoken_text(text), valid, on_done)
        try:
            self.q.put_nowait(item)
            return True
        except queue.Full:
            if urgent:
                # Evict the least important queued item under the queue's own lock.
                import heapq
                with self.q.mutex:
                    worst = max(range(len(self.q.queue)), key=lambda i: self.q.queue[i][:2])
                    self.q.queue[worst] = item
                    heapq.heapify(self.q.queue)
                return True
            return False

    def _danger_pending(self):
        with self.q.mutex:
            return any(item[0] == 0 for item in self.q.queue)

    def stop(self):
        """Cancel ordinary speech; a safety alert retains priority."""
        self._generation += 1

    def _valid(self, item, playing=False):
        priority, _, generation, expires, _, valid, _ = item
        return ((playing or time.monotonic() <= expires) and (priority == 0 or generation == self._generation)
                and self.enabled and (valid is None or valid()))

    def set_voice(self, name: str = None, rate: int = None) -> None:
        if name is not None:
            self.voice_name = name
        if rate is not None:
            self.rate = int(rate)
        self._apply = True

    def _run(self) -> None:
        sapi = None
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            sapi = win32com.client.Dispatch("SAPI.SpVoice")
        except Exception:
            try:
                import comtypes
                import comtypes.client
                comtypes.CoInitialize()
                sapi = comtypes.client.CreateObject("SAPI.SpVoice")
            except Exception:
                sapi = None
        if sapi is not None:
            try:
                tokens = sapi.GetVoices()
                self.voices = [tokens.Item(i).GetDescription() for i in range(tokens.Count)]
            except Exception:
                self.voices = []
        else:
            print("[BantayAso] Windows voice (SAPI) not available, trying pyttsx3")
        while True:
            item = self.q.get()
            if not self._valid(item):
                continue
            priority, _, generation, _, text, valid, on_done = item
            self.current_priority = priority
            self.speaking = True
            completed = False
            try:
                if sapi is not None:
                    if self._apply:
                        self._apply = False
                        sapi.Rate = max(-10, min(10, int(self.rate)))
                        if self.voice_name:
                            tokens = sapi.GetVoices()
                            for i in range(tokens.Count):
                                if self.voice_name.lower() in tokens.Item(i).GetDescription().lower():
                                    sapi.Voice = tokens.Item(i)
                                    break
                    sapi.Speak(text, 1)  # async, COM remains on this thread
                    while not sapi.WaitUntilDone(50):
                        if (not self._valid(item, playing=True) or
                                (priority > 0 and self._danger_pending())):
                            sapi.Speak('', 3)  # async + purge current speech
                            break
                    else:
                        completed = True
                else:
                    import pyttsx3
                    eng = pyttsx3.init()
                    eng.say(text)
                    eng.runAndWait()
                    completed = self._valid(item, playing=True)
            except Exception as e:                    # pragma: no cover
                print(f"[BantayAso] voice error: {e!r}")
            finally:
                time.sleep(0.25)                      # let the room go quiet before listening again
                self.speaking = self.q.qsize() > 0
                if completed and on_done:
                    try:
                        on_done()
                    except Exception as e:
                        print(f"[BantayAso] speech completion error: {e!r}")


def toast(title: str, message: str, icon: Path | None = None) -> None:
    """Windows notification (non-blocking). Silently does nothing elsewhere."""
    def _show():
        try:
            from winotify import Notification
            n = Notification(app_id="BantayAso", title=title, msg=message,
                             icon=str(icon) if icon and Path(icon).exists() else "")
            n.show()
        except Exception:
            pass
    threading.Thread(target=_show, daemon=True).start()


def play_clip(path: str) -> None:
    """Play a recorded .wav (e.g. the owner saying "No!") without blocking."""
    if not path or not Path(path).exists():
        return
    def _play():
        try:
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        except Exception:
            pass
    threading.Thread(target=_play, daemon=True).start()


TEMPLATES = {
    2: "Warning! {who} is {reason}.",
    3: "Danger! {who} is {reason}. Check now!",
}


def phrase(level: int, reason: str, who: str = "your dog") -> str:
    """Turn a risk reason into a short spoken sentence."""
    r = reason.replace(" - check what it is", ". Check what it is")
    if r.startswith(("near", "in ", "at ", "on ")):
        r = r                                        # "near a battery", "at the trash"
    elif r.startswith("the "):                       # "the battery disappeared near its mouth"
        return f"{'Danger' if level == 3 else 'Warning'}! {r.capitalize()}. Check {who}'s mouth now!"
    elif r.startswith("has been"):
        r = r.replace(" s", " seconds") if r.endswith(" s") else r
        prefix = {2: "Warning! ", 3: "Danger! "}.get(level, "")
        suffix = " Check now!" if level >= 2 else ""
        return f"{prefix}{who.capitalize()} {r}.{suffix}"
    return TEMPLATES.get(level, "{who} is {reason}.").format(who=who.capitalize(), reason=r)
