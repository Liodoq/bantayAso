"""Drawing on frames. Colors follow the brown/black palette (OpenCV uses BGR)."""
from __future__ import annotations

import cv2
import numpy as np


def hex_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


C = {
    "bg": hex_bgr("0E0B09"), "caramel": hex_bgr("C8894E"), "cream": hex_bgr("F2E6D8"),
    "safe": hex_bgr("7FB98A"), "watch": hex_bgr("E3B54F"), "warning": hex_bgr("E3803F"),
    "danger": hex_bgr("E0533F"), "rec": hex_bgr("E0533F"),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


_used: list = []


def reset_labels():
    _used.clear()
    _used.append((0, 0, 620, 52))          # keep the status pill area free


def label(img, text, x, y, color, text_color=None):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    y = max(y, th + 8)
    for _ in range(6):                       # push down until it does not cover another label
        r = (x, y - th - 8, x + tw + 10, y)
        if not any(r[0] < u[2] and u[0] < r[2] and r[1] < u[3] and u[1] < r[3] for u in _used):
            break
        y += th + 10
    _used.append((x, y - th - 8, x + tw + 10, y))
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - 5), FONT, 0.55, text_color or C["bg"], 1, cv2.LINE_AA)


LEVEL_KEY = ["caramel", "watch", "warning", "danger"]
ZONE_KEY = {"trash": "warning", "danger": "danger", "nogo": "watch", "bed": "safe", "food": "caramel", "play": "caramel"}


def draw_dogs(img, dogs, levels: dict | None = None, names: dict | None = None):
    """levels: track_id -> Assessment (optional) to color each dog by its risk."""
    for d in dogs:
        a = (levels or {}).get(d.track_id)
        color = C[LEVEL_KEY[a.level]] if a else C["caramel"]
        x1, y1, x2, y2 = d.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if a and a.level >= 2 else 2)
        who = (names or {}).get(d.track_id) or "dog"
        if a is None:
            tag = f"{who} {d.conf:.2f}"
        elif a.level == 0:
            tag = f"{who} - {a.reason}" if a.reason not in ("all calm", "resting") else f"{who} - calm"
        else:
            tag = f"{who} - {a.reason}"
        label(img, tag, x1, y1, color)


def draw_hazards(img, hazards):
    for hz in hazards:
        color = C[{3: "danger", 2: "warning"}.get(hz.tier, "watch")]
        x1, y1, x2, y2 = hz.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label(img, f"{hz.name} {hz.conf:.2f}", x1, y2 + 22, color)


def draw_zones(img, zones, editor=None, help_bar: bool = True):
    h, w = img.shape[:2]
    layer = img.copy()
    for z in zones:
        if len(z.points) >= 3:
            poly = z.poly(w, h)
            cv2.fillPoly(layer, [poly], C[ZONE_KEY.get(z.type, "watch")])
    cv2.addWeighted(layer, 0.18, img, 0.82, 0, img)
    sel = getattr(editor, "selected", None) if editor is not None else None
    editing = bool(getattr(editor, "editing", False)) if editor is not None else False
    hover = getattr(editor, "hover", None) if editor is not None else None
    for i, z in enumerate(zones):
        if len(z.points) >= 3:
            poly = z.poly(w, h)
            color = C["caramel"] if i == sel else C[ZONE_KEY.get(z.type, "watch")]
            cv2.polylines(img, [poly], True, color, 3 if i == sel else 2, cv2.LINE_AA)
            # numbered tag (matches the numbered zone buttons under the video)
            tag = f" {i + 1}  {z.name} "
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.5, 1)
            top = poly[int(np.argmin(poly[:, 1]))]          # tag sits just above the zone's top corner
            x0 = int(min(max(top[0] - tw // 2, 0), w - tw - 4))
            y0 = int(min(max(top[1] - 10, th + 6), h - 4))
            cv2.rectangle(img, (x0, y0 - th - 6), (x0 + tw + 4, y0 + 4), color, -1)
            cv2.putText(img, tag, (x0 + 2, y0 - 1), FONT, 0.5, C["bg"], 1, cv2.LINE_AA)
            if editing:                                  # corner handles: drag these to reshape
                for j, p in enumerate(poly):
                    hot = hover == (i, j)
                    r = 11 if hot else 8 if i == sel else 6
                    pt = tuple(int(v) for v in p)
                    cv2.circle(img, pt, r, C["caramel"] if hot else (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, pt, r, color, 2, cv2.LINE_AA)
    if editor is not None and editor.active:
        pts = [(int(x * w), int(y * h)) for x, y in editor.current]
        for k, p in enumerate(pts):
            cv2.circle(img, p, 8 if k == 0 else 5, C["caramel"], -1)    # the first corner is bigger: click it to close
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, C["caramel"], 2, cv2.LINE_AA)
    if editor is not None and editor.active and help_bar:
        h, w = img.shape[:2]
        help_ = (f"ZONE EDIT - type: {editor.type.upper()}  |  click: add point  right-click/ENTER: finish  "
                 "1 trash 2 danger 3 no-go 4 bed 5 food 6 play  BACKSPACE: undo  X: delete last  Z: save & exit")
        cv2.rectangle(img, (0, h - 34), (w, h), C["bg"], -1)
        cv2.putText(img, help_, (10, h - 12), FONT, 0.5, C["caramel"], 1, cv2.LINE_AA)


def draw_status(img, text: str, color_key: str = "safe"):
    """Big status pill top-left (dog-related info only)."""
    text = f"  {text}  "
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 1)
    cv2.rectangle(img, (12, 40 - th - 8), (12 + tw + 10, 40), C[color_key], -1)
    cv2.putText(img, text, (17, 35), FONT, 0.55, C["bg"], 1, cv2.LINE_AA)


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
