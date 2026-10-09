"""Local vision-language model (Ollama) that describes an alert in one sentence. Async, one at a time."""
from __future__ import annotations

import base64
import threading
import time

import cv2

PROMPT = ("You are watching a home pet camera. Look at the dog in this picture. "
          "In ONE short sentence, say what the dog is doing and name any object in or near its "
          "mouth (for example a battery, cable, sock, paper, food, toy). If nothing is in its "
          "mouth, say so.")


class VLMWorker:
    def __init__(self, url: str, model: str, timeout: float = 25.0):
        self.url = url.rstrip("/") + "/api/generate"
        self.model = model
        self.timeout = timeout
        self.busy = False
        self.last_ms = 0.0
        self.available = None            # None unknown, True/False after warm-up
        threading.Thread(target=self._warmup, daemon=True).start()

    def _post(self, payload: dict) -> str:
        import requests
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def _warmup(self) -> None:
        try:                              # loads the model into memory so the first alert is fast
            self._post({"model": self.model, "prompt": "Say OK.", "stream": False,
                        "keep_alive": "30m", "options": {"num_predict": 3}})
            self.available = True
        except Exception as e:
            self.available = False
            print(f"[BantayAso] local VLM not available ({e.__class__.__name__}); alerts still work")

    def describe(self, frame, box, callback) -> bool:
        """Start describing the dog in `box`. Calls callback(sentence) later. False if busy."""
        if self.busy or self.available is False:
            return False
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        pad = int(0.25 * max(x2 - x1, y2 - y1))
        crop = frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if crop.size == 0:
            crop = frame
        scale = 512 / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return False
        img = base64.b64encode(jpg.tobytes()).decode()
        self.busy = True

        def _run():
            t0 = time.perf_counter()
            try:
                text = self._post({"model": self.model, "prompt": PROMPT, "images": [img],
                                   "stream": False, "keep_alive": "30m",
                                   "options": {"temperature": 0.1, "num_predict": 60}})
                text = text.split("\n")[0].strip().strip('"')
                if len(text) > 160:
                    text = text[:157].rsplit(" ", 1)[0] + "..."
                callback(text)
            except Exception as e:
                print(f"[BantayAso] VLM error: {e.__class__.__name__}")
            finally:
                self.last_ms = (time.perf_counter() - t0) * 1000
                self.busy = False
        threading.Thread(target=_run, name="VLM", daemon=True).start()
        return True
