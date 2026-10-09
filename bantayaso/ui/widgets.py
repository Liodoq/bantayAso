"""Small custom widgets: video view, risk meter, hour chart, stat card."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from . import theme as T


class VideoView(QWidget):
    """Shows frames scaled with aspect ratio; reports clicks in FRAME coordinates."""
    clicked = Signal(int, int, int)            # x, y, button (1 left, 2 right)

    def __init__(self, placeholder: str = "Starting camera...", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(480, 270)
        self._img: QImage | None = None
        self.placeholder = placeholder
        # dog-picking mode: highlight the dog boxes and the one under the mouse
        self.selecting = False
        self.targets: list = []                  # [(track_id, (x1, y1, x2, y2)), ...] in frame pixels
        self.hover = None
        self.setMouseTracking(True)

    def set_image(self, img: QImage) -> None:
        self._img = img
        self.update()

    def _target(self) -> QRect:
        if self._img is None:
            return self.rect()
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = int(iw * s), int(ih * s)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(T.INSET))
        if self._img is None:
            p.setPen(QColor(T.MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        r = self._target()
        if r.width() < self.width() - 2 or r.height() < self.height() - 2:
            # letterbox: fill the empty bars with a soft, dimmed blur of the same frame (no hard empty strip)
            small = self._img.scaled(48, 27, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            iw, ih = self._img.width(), self._img.height()
            sc = max(self.width() / iw, self.height() / ih)
            cw, ch = int(iw * sc), int(ih * sc)
            p.drawImage(QRect((self.width() - cw) // 2, (self.height() - ch) // 2, cw, ch), small)
            p.fillRect(self.rect(), QColor(0, 0, 0, 110))
        path = QPainterPath()
        path.addRoundedRect(QRectF(r), 10, 10)
        p.save()
        p.setClipPath(path)
        p.drawImage(r, self._img)
        p.restore()
        if self.selecting:
            p.fillRect(r, QColor(0, 0, 0, 70))                     # dim the picture
            sx, sy = r.width() / self._img.width(), r.height() / self._img.height()
            from PySide6.QtGui import QPen
            for tid, (x1, y1, x2, y2) in self.targets:
                br = QRect(int(r.x() + x1 * sx), int(r.y() + y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy))
                hot = tid == self.hover
                p.setPen(QPen(QColor(T.CARAMEL), 4 if hot else 2, Qt.SolidLine if hot else Qt.DashLine))
                p.setBrush(QColor(216, 180, 138, 70) if hot else Qt.NoBrush)
                p.drawRoundedRect(br, 8, 8)
                tag = "Click to choose" if hot else "Click me"
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(T.CARAMEL))
                f = QFont(self.font())
                f.setBold(True)
                p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(tag) + 16
                p.drawRoundedRect(QRect(br.x(), max(r.y(), br.y() - 26), tw, 22), 6, 6)
                p.setPen(QColor(T.ON_ACCENT))
                p.drawText(QRect(br.x(), max(r.y(), br.y() - 26), tw, 22), Qt.AlignCenter, tag)

    def _frame_pos(self, pos):
        r = self._target()
        if self._img is None or not r.contains(pos):
            return None
        return (int((pos.x() - r.x()) * self._img.width() / r.width()),
                int((pos.y() - r.y()) * self._img.height() / r.height()))

    def mouseMoveEvent(self, e):
        if not self.selecting:
            return
        fp = self._frame_pos(e.position().toPoint())
        hov = None
        if fp:
            hits = [(tid, b) for tid, b in self.targets if b[0] <= fp[0] <= b[2] and b[1] <= fp[1] <= b[3]]
            if hits:
                hov = min(hits, key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))[0]
        if hov != self.hover:
            self.hover = hov
            self.setCursor(Qt.PointingHandCursor if hov is not None else Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, e):
        if self._img is None:
            return
        r = self._target()
        pos = e.position().toPoint()
        if not r.contains(pos):
            return
        x = int((pos.x() - r.x()) * self._img.width() / r.width())
        y = int((pos.y() - r.y()) * self._img.height() / r.height())
        self.clicked.emit(x, y, 2 if e.button() == Qt.RightButton else 1)


class RiskMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0
        self.setFixedHeight(30)

    def set_level(self, level: int) -> None:
        if level != self.level:
            self.level = level
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        gap, w = 6, (self.width() - 18) / 4
        f = QFont(self.font())
        f.setPointSize(8)
        p.setFont(f)
        for i in range(4):
            x = int(i * (w + gap))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(T.LEVEL_COLORS[i] if i <= self.level else T.RAISED))
            p.drawRoundedRect(x, 0, int(w), 8, 4, 4)
            p.setPen(QColor(T.CREAM if i == self.level else T.FAINT))
            p.drawText(QRect(x, 12, int(w), 16), Qt.AlignLeft, ["Safe", "Watch", "Warning", "Danger"][i])


class HourChart(QWidget):
    """Alerts by hour (bars), simple and themed."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data: dict[int, int] = {}
        self.setMinimumHeight(120)

    def set_data(self, by_hour: dict) -> None:
        self.data = {int(k): int(v) for k, v in by_hour.items()}
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        hours = list(range(6, 24))
        top = max(self.data.values(), default=0) or 1
        bw = (self.width() - 4) / len(hours)
        hot = max(self.data, key=self.data.get) if self.data else None
        f = QFont(self.font())
        f.setPointSize(7)
        p.setFont(f)
        for i, hr in enumerate(hours):
            v = self.data.get(hr, 0)
            bh = int((self.height() - 22) * v / top)
            x = int(i * bw) + 2
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(T.CARAMEL if hr == hot else T.BROWN if v else T.RAISED))
            p.drawRoundedRect(x, self.height() - 18 - max(bh, 3), int(bw) - 4, max(bh, 3), 3, 3)
            if hr % 3 == 0:
                p.setPen(QColor(T.FAINT))
                lab = f"{hr % 12 or 12}{'a' if hr < 12 else 'p'}"
                p.drawText(QRect(x - 6, self.height() - 16, int(bw) + 12, 14), Qt.AlignCenter, lab)


def card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setObjectName("card")
    return f


def stat_card(value: str, label: str, color: str | None = None):
    f = card()
    lay = QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 12)
    v = QLabel(value)
    v.setStyleSheet(f"font-size: 18pt; font-weight: 600; color: {color or T.CREAM};")
    l = QLabel(label)
    l.setObjectName("muted")
    lay.addWidget(v)
    lay.addWidget(l)
    f.value_label = v
    return f


def paw_pixmap(size: int = 64, color: str = T.CARAMEL) -> QPixmap:
    """App/tray icon: a simple paw drawn in code (no image files needed)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    s = size / 64
    p.drawEllipse(QPoint(int(32 * s), int(42 * s)), int(15 * s), int(13 * s))
    for cx, cy in ((14, 26), (25, 14), (39, 14), (50, 26)):
        p.drawEllipse(QPoint(int(cx * s), int(cy * s)), int(7 * s), int(8 * s))
    p.end()
    return pm
