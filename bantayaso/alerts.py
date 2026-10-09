"""Alerts: offline voice (Windows SAPI via pyttsx3), Windows toast, optional owner voice clip.

Everything runs on background threads so the video never freezes.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path


class Speaker:
    """Speaks one message at a time from a queue. New danger messages jump the queue."""

    def __init__(self, rate: int = 175):
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=3)
        self.rate = rate
        self.enabled = True
        self.speaking = False
        threading.Thread(target=self._run, name="Speaker", daemon=True).start()

    def say(self, text: str, urgent: bool = False) -> None:
        if not self.enabled or not text:
            return
        if urgent:                                    # drop stale messages
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self) -> None:
        # Windows speech (SAPI) runs over COM; a background thread must initialize COM itself,
        # otherwise pyttsx3 can fail silently and nothing is heard.
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            try:
                import comtypes
                comtypes.CoInitialize()
            except Exception:
                pass
        try:
            import pyttsx3
        except Exception as e:                        # pragma: no cover
            print(f"[BantayAso] voice disabled: {e}")
            return
        while True:
            text = self.q.get()
            self.speaking = True
            try:
                engine = pyttsx3.init()               # fresh engine per message: avoids SAPI hangs
                engine.setProperty("rate", self.rate)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:                    # pragma: no cover
                print(f"[BantayAso] voice error: {e!r}")
            finally:
                time.sleep(0.3)                       # let the room go quiet before listening again
                self.speaking = self.q.qsize() > 0


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
        return f"Danger! {who.capitalize()} {r}. Check now!"
    return TEMPLATES.get(level, "{who} is {reason}.").format(who=who.capitalize(), reason=r)
