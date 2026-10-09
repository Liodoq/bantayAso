"""Drawing on frames. Colors follow the brown/black palette (OpenCV uses BGR)."""
from __future__ import annotations

import cv2


def hex_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


C = {
    "bg": hex_bgr("0E0B09"), "caramel": hex_bgr("C8894E"), "cream": hex_bgr("F2E6D8"),
    "safe": hex_bgr("7FB98A"), "watch": hex_bgr("E3B54F"), "warning": hex_bgr("E3803F"),
    "danger": hex_bgr("E0533F"), "rec": hex_bgr("E0533F"),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, x, y, color, text_color=None):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - 5), FONT, 0.55, text_color or C["bg"], 1, cv2.LINE_AA)


def draw_dogs(img, dogs, color_key: str = "caramel"):
    color = C[color_key]
    for d in dogs:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f"dog {d.conf:.2f}"
        label(img, tag, x1, y1, color)


def draw_status(img, text: str, color_key: str = "safe"):
    """Big status pill top-left (dog-related info only)."""
    label(img, f"  {text}  ", 12, 40, C[color_key])


def draw_recording(img, on: bool):
    if on:
        h, w = img.shape[:2]
        cv2.circle(img, (w - 28, 28), 9, C["rec"], -1)
        cv2.putText(img, "REC", (w - 80, 34), FONT, 0.6, C["cream"], 2, cv2.LINE_AA)


def draw_debug(img, lines):
    """Hidden by default (press D). FPS/device etc. never shown unless asked."""
    h = img.shape[0]
    for i, t in enumerate(reversed(lines)):
        cv2.putText(img, t, (12, h - 14 - i * 22), FONT, 0.55, C["cream"], 1, cv2.LINE_AA)


def draw_toast(img, text: str):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    x = (w - tw) // 2
    cv2.rectangle(img, (x - 12, h - 60 - th), (x + tw + 12, h - 48), C["bg"], -1)
    cv2.putText(img, text, (x, h - 54), FONT, 0.6, C["caramel"], 1, cv2.LINE_AA)
