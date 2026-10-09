"""BantayAso main window: Monitor / Events / Zones / Settings + system tray (runs in background)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QFileDialog, QFormLayout,
                               QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QPushButton, QSizePolicy,
                               QStackedWidget, QSystemTrayIcon, QVBoxLayout, QWidget)

from .. import config
from ..zones import ZONE_TYPES
from . import theme as T
from .widgets import HourChart, RiskMeter, VideoView, card, paw_pixmap, stat_card
from .worker import Worker

LEVEL_KEY = ["safe", "watch", "warning", "danger"]


def lbl(text="", name=None, wrap=False):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("BantayAso")
        self.setWindowIcon(QIcon(paw_pixmap(64)))
        self.resize(1280, 780)
        self.worker = Worker(args, self)
        self.frame_size = (1280, 720)
        self.quitting = False
        self.ack_alert_id = None
        self.events_dirty = True
        self.selected_event = None

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._sidebar())
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.stack.addWidget(self._monitor_page())
        self.stack.addWidget(self._events_page())
        self.stack.addWidget(self._zones_page())
        self.stack.addWidget(self._settings_page())
        self._tray()

        self.worker.frame_ready.connect(self.on_frame)
        self.worker.message.connect(self.on_message)
        self.worker.event.connect(self.on_event)
        self.worker.ready.connect(self.on_ready)
        QShortcut(QKeySequence("F12"), self, activated=self.toggle_debug)
        self.snooze_timer = QTimer(self, singleShot=True, timeout=self.end_snooze)
        self.worker.start()

    # ------------------------------------------------------------------ layout
    def _sidebar(self):
        side = QWidget(objectName="side")
        side.setFixedWidth(210)
        v = QVBoxLayout(side)
        v.setContentsMargins(12, 18, 12, 18)
        top = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(paw_pixmap(34))
        top.addWidget(logo)
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(lbl("BantayAso", "brand"))
        names.addWidget(lbl("Local AI dog watcher", "brandSub"))
        top.addLayout(names)
        top.addStretch()
        v.addLayout(top)
        v.addSpacing(16)
        self.nav = QButtonGroup(self)
        for i, (icon, text) in enumerate([("◉", "Monitor"), ("☰", "Events"), ("▢", "Zones"), ("⚙", "Settings")]):
            b = QPushButton(f"  {icon}   {text}", objectName="nav", checkable=True)
            b.setCursor(Qt.PointingHandCursor)
            self.nav.addButton(b, i)
            v.addWidget(b)
        self.nav.button(0).setChecked(True)
        self.nav.idClicked.connect(self.go)
        v.addStretch()
        return side

    def _monitor_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(14)
        top = QHBoxLayout()
        tl = QVBoxLayout()
        tl.setSpacing(0)
        self.title = lbl("Watching your dogs", "h1")
        self.subtitle = lbl("Starting...", "muted")
        tl.addWidget(self.title)
        tl.addWidget(self.subtitle)
        top.addLayout(tl)
        top.addStretch()
        self.btn_voice = QPushButton("🔊 Voice on", objectName="pill", checkable=True, checked=True)
        self.btn_dnd = QPushButton("🌙 Do not disturb", objectName="pill", checkable=True)
        self.btn_rec = QPushButton("● Record", objectName="pill", checkable=True)
        btn_tray = QPushButton("⤓ Minimize to tray", objectName="pill")
        for b in (self.btn_voice, self.btn_dnd, self.btn_rec, btn_tray):
            b.setCursor(Qt.PointingHandCursor)
            top.addWidget(b)
        self.btn_voice.toggled.connect(self.set_voice)
        self.btn_dnd.toggled.connect(self.set_dnd)
        self.btn_rec.clicked.connect(lambda: self.worker.request("record"))
        btn_tray.clicked.connect(self.hide_to_tray)
        v.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(16)
        vid_card = card()
        vc = QVBoxLayout(vid_card)
        vc.setContentsMargins(14, 14, 14, 10)
        self.video = VideoView("Loading the local AI models...")
        vc.addWidget(self.video, 1)
        self.meter = RiskMeter()
        vc.addWidget(self.meter)
        row.addWidget(vid_card, 1)

        right = QVBoxLayout()
        right.setSpacing(16)
        self.status_card = card()
        self.status_card.setFixedWidth(320)
        sc = QVBoxLayout(self.status_card)
        sc.setContentsMargins(16, 16, 16, 16)
        self.level_pill = lbl("● SAFE")
        self.level_pill.setAlignment(Qt.AlignLeft)
        self.status_title = lbl("Starting...", wrap=True)
        self.status_title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        self.status_sub = lbl("", "muted", wrap=True)
        self.spoken = lbl("", wrap=True)
        self.spoken.setStyleSheet(f"background: #0e0b0999; border: 1px solid {T.LINE}; border-radius: 10px; padding: 10px;")
        self.spoken.hide()
        btns = QHBoxLayout()
        self.btn_ack = QPushButton("I've got it", objectName="primary")
        self.btn_snooze = QPushButton("Snooze 5 min")
        self.btn_ack.clicked.connect(self.acknowledge)
        self.btn_snooze.clicked.connect(self.snooze)
        btns.addWidget(self.btn_ack)
        btns.addWidget(self.btn_snooze)
        for w in (self.level_pill, self.status_title, self.status_sub, self.spoken):
            sc.addWidget(w)
        sc.addLayout(btns)
        right.addWidget(self.status_card)

        rec_card = card()
        rec_card.setFixedWidth(320)
        rc = QVBoxLayout(rec_card)
        rc.setContentsMargins(14, 14, 14, 14)
        rc.addWidget(lbl("RECENT", "h3"))
        self.recent = QListWidget()
        self.recent.setIconSize(QSize(46, 34))
        rc.addWidget(self.recent, 1)
        right.addWidget(rec_card, 1)
        row.addLayout(right)
        v.addLayout(row, 1)
        self._set_status_card(0, "Starting...", "", None)
        return page

    def _events_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(14)
        top = QHBoxLayout()
        tl = QVBoxLayout()
        tl.setSpacing(0)
        tl.addWidget(lbl(time.strftime("Today, %b %d").replace(" 0", " "), "h1"))
        self.ev_sub = lbl("", "muted")
        tl.addWidget(self.ev_sub)
        top.addLayout(tl)
        top.addStretch()
        self.ev_filter = QButtonGroup(self)
        for i, name in enumerate(["All", "Danger", "Warning"]):
            b = QPushButton(name, objectName="pill", checkable=True)
            self.ev_filter.addButton(b, i)
            top.addWidget(b)
        self.ev_filter.button(0).setChecked(True)
        self.ev_filter.idClicked.connect(lambda _: self.refresh_events())
        v.addLayout(top)

        stats = QHBoxLayout()
        self.st_watch = stat_card("0m", "Watched")
        self.st_danger = stat_card("0", "Danger", T.DANGER)
        self.st_warn = stat_card("0", "Warnings", T.WARN)
        self.st_hot = stat_card("-", "Hotspot")
        for s in (self.st_watch, self.st_danger, self.st_warn, self.st_hot):
            stats.addWidget(s)
        v.addLayout(stats)

        row = QHBoxLayout()
        row.setSpacing(16)
        left = card()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(14, 14, 14, 14)
        lv.addWidget(lbl("TIMELINE", "h3"))
        self.timeline = QListWidget()
        self.timeline.setIconSize(QSize(96, 56))
        self.timeline.currentRowChanged.connect(self.select_event)
        lv.addWidget(self.timeline, 1)
        lv.addWidget(lbl("ALERTS BY HOUR", "h3"))
        self.chart = HourChart()
        lv.addWidget(self.chart)
        row.addWidget(left, 1)

        det = card()
        det.setFixedWidth(360)
        dv = QVBoxLayout(det)
        dv.setContentsMargins(14, 14, 14, 14)
        dv.addWidget(lbl("SELECTED EVENT", "h3"))
        self.ev_img = QLabel()
        self.ev_img.setFixedHeight(240)
        self.ev_img.setAlignment(Qt.AlignCenter)
        self.ev_img.setStyleSheet("background: #140f0b; border-radius: 10px;")
        dv.addWidget(self.ev_img)
        self.ev_kv = QFormLayout()
        self.kv = {}
        for k in ("Level", "Activity", "Near", "Zone", "Time"):
            self.kv[k] = lbl("-")
            self.ev_kv.addRow(lbl(k, "muted"), self.kv[k])
        dv.addLayout(self.ev_kv)
        self.ev_ai = lbl("", wrap=True)
        self.ev_ai.setStyleSheet(f"background: #0e0b0999; border: 1px solid {T.LINE}; border-radius: 10px; padding: 10px;")
        dv.addWidget(self.ev_ai)
        b = QHBoxLayout()
        self.btn_open = QPushButton("Open snapshot")
        self.btn_false = QPushButton("Mark false alarm")
        self.btn_open.clicked.connect(self.open_snapshot)
        self.btn_false.clicked.connect(self.mark_false)
        b.addWidget(self.btn_open)
        b.addWidget(self.btn_false)
        dv.addLayout(b)
        dv.addStretch()
        row.addWidget(det)
        v.addLayout(row, 1)
        return page

    def _zones_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)
        v.addWidget(lbl("Zones", "h1"))
        v.addWidget(lbl("Click points around an area on the camera view, pick its type, then Finish zone. "
                        "Right-click also finishes. Save when done.", "muted", wrap=True))
        bar = QHBoxLayout()
        self.zone_type = QButtonGroup(self)
        for i, (t, name) in enumerate(zip(ZONE_TYPES, ["Trash", "Danger", "No-go", "Bed"])):
            b = QPushButton(name, objectName="pill", checkable=True)
            self.zone_type.addButton(b, i)
            bar.addWidget(b)
        self.zone_type.button(0).setChecked(True)
        self.zone_type.idClicked.connect(self.set_zone_type)
        bar.addStretch()
        for text, fn in (("Undo point", self.zone_undo), ("Finish zone", self.zone_finish),
                         ("Delete last zone", self.zone_delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            bar.addWidget(b)
        save = QPushButton("Save zones", objectName="primary")
        save.clicked.connect(self.zone_save)
        bar.addWidget(save)
        v.addLayout(bar)
        c = card()
        cv = QVBoxLayout(c)
        cv.setContentsMargins(14, 14, 14, 14)
        self.zone_video = VideoView("Waiting for the camera...")
        self.zone_video.clicked.connect(self.zone_click)
        cv.addWidget(self.zone_video)
        v.addWidget(c, 1)
        self.zone_info = lbl("", "muted")
        v.addWidget(self.zone_info)
        return page

    def _settings_page(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)
        v.addWidget(lbl("Settings", "h1"))
        c = card()
        f = QFormLayout(c)
        f.setContentsMargins(18, 18, 18, 18)
        f.setVerticalSpacing(12)
        cfg = config.load()
        al = cfg.get("alerts", {})
        self.set_name = QLineEdit(cfg.get("dog_name", ""))
        self.set_name.setPlaceholderText("your dog")
        self.set_voice_cb = QCheckBox("Speak alerts out loud")
        self.set_voice_cb.setChecked(al.get("voice", True))
        self.set_vlm_cb = QCheckBox("Also speak the local AI's description")
        self.set_vlm_cb.setChecked(al.get("speak_vlm", True))
        self.set_toast_cb = QCheckBox("Show Windows notifications")
        self.set_toast_cb.setChecked(al.get("toast", True))
        self.set_hz_cb = QCheckBox("Show object boxes on the video")
        self.set_hz_cb.setChecked(True)
        self.set_clip = QLineEdit(al.get("owner_voice_clip", ""))
        self.set_clip.setPlaceholderText("optional .wav of you saying \"No!\"")
        pick = QPushButton("Browse...")
        pick.clicked.connect(self.pick_clip)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.set_clip, 1)
        clip_row.addWidget(pick)
        f.addRow(lbl("Dog name", "muted"), self.set_name)
        f.addRow(lbl("Voice", "muted"), self.set_voice_cb)
        f.addRow(lbl("", "muted"), self.set_vlm_cb)
        f.addRow(lbl("Notifications", "muted"), self.set_toast_cb)
        f.addRow(lbl("Video", "muted"), self.set_hz_cb)
        f.addRow(lbl("Owner voice", "muted"), clip_row)
        save = QPushButton("Save settings", objectName="primary")
        save.clicked.connect(self.save_settings)
        f.addRow("", save)
        v.addWidget(c)
        v.addWidget(lbl("Press F12 anytime to show or hide technical info on the video.", "faint"))
        v.addStretch()
        return page

    def _tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(QIcon(paw_pixmap(64)), self)
        self.tray.setToolTip("BantayAso - watching your dogs")
        m = QMenu()
        for text, fn in (("Show BantayAso", self.show_window), ("Mute / unmute voice", lambda: self.btn_voice.toggle()),
                         ("Do not disturb", lambda: self.btn_dnd.toggle()), ("Quit", self.quit_app)):
            a = QAction(text, m)
            a.triggered.connect(fn)
            m.addAction(a)
        self.tray.setContextMenu(m)
        self.tray.activated.connect(lambda r: self.show_window() if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()
        self._tray_level = -1

    # ------------------------------------------------------------------ live updates
    def on_ready(self):
        self.subtitle.setText(f"Live since {time.strftime('%I:%M %p').lstrip('0')}")
        self.set_voice(self.btn_voice.isChecked())
        self.refresh_recent()

    def on_message(self, msg: str):
        if self.worker.pipe is None:
            self.video.placeholder = msg
            self.video.update()
        if "Recording" in msg or "Saved" in msg or "Snapshot" in msg or "disconnected" in msg:
            self.subtitle.setText(msg)

    def on_frame(self, img, st: dict):
        self.frame_size = st["frame_size"]
        page = self.stack.currentIndex()
        if page == 0:
            self.video.set_image(img)
        elif page == 2:
            self.zone_video.set_image(img)
        self.meter.set_level(st["level"])
        la = st.get("last_alert")
        show_alert = la and la["id"] != self.ack_alert_id and time.time() - la["ts"] < 60 and st["level"] >= 2
        if show_alert:
            sub = "Alert at " + time.strftime("%I:%M:%S %p", time.localtime(la["ts"])).lstrip("0")
            self._set_status_card(la["level"], st["status"].split(": ", 1)[-1].split(" - ")[0].capitalize(), sub, la)
        else:
            dogs = st["dogs"]
            sub = f"{dogs} dog{'s' if dogs != 1 else ''} in view" if dogs else "Waiting for a dog to appear"
            self._set_status_card(st["level"], st["status"], sub, None)
        self.btn_rec.setChecked(self.worker.recording)
        if self.tray and st["level"] != self._tray_level:
            self._tray_level = st["level"]
            self.tray.setIcon(QIcon(paw_pixmap(64, T.LEVEL_COLORS[st["level"]] if st["level"] else T.CARAMEL)))

    def _set_status_card(self, level, title, sub, alert):
        color = T.LEVEL_COLORS[level]
        self.level_pill.setText(f"  ●  {T.LEVEL_NAMES[level]}  ")
        self.level_pill.setStyleSheet(f"background: {color}; color: #1a0f08; font-weight: 700; border-radius: 11px; padding: 3px 4px;")
        self.level_pill.setFixedHeight(24)
        self.level_pill.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.status_title.setText(title)
        self.status_sub.setText(sub)
        hot = level >= 2
        self.status_card.setStyleSheet(
            f"QFrame#card {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #3a2216, stop:1 {T.PANEL});"
            f" border: 1px solid #6b3a22; border-radius: 12px; }}" if hot else "")
        if alert:
            txt = f"<b style='color:{T.CARAMEL}'>Spoken:</b> “{alert['text']}”"
            if alert.get("vlm"):
                txt += f"<br><b style='color:{T.CARAMEL}'>Local AI:</b> {alert['vlm']}"
            self.spoken.setText(txt)
            self.spoken.show()
        else:
            self.spoken.hide()
        self.btn_ack.setVisible(hot)
        self.btn_snooze.setVisible(hot)

    def on_event(self, ev: dict):
        self.events_dirty = True
        self.refresh_recent()
        if self.stack.currentIndex() == 1:
            self.refresh_events()

    # ------------------------------------------------------------------ actions
    def go(self, i: int):
        self.stack.setCurrentIndex(i)
        pipe = self.worker.pipe
        if pipe:
            pipe.editor.active = i == 2
        if i == 1:
            self.refresh_events()
        if i == 2:
            self.update_zone_info()

    def set_voice(self, on: bool):
        self.btn_voice.setText("🔊 Voice on" if on else "🔇 Muted")
        if self.worker.pipe:
            self.worker.pipe.voice = on

    def set_dnd(self, on: bool):
        if self.worker.pipe:
            self.worker.pipe.dnd = on
        if not on:
            self.snooze_timer.stop()
            self.btn_dnd.setText("🌙 Do not disturb")

    def acknowledge(self):
        la = self.worker.pipe.state.last_alert if self.worker.pipe else None
        if la:
            self.ack_alert_id = la["id"]

    def snooze(self):
        self.acknowledge()
        self.btn_dnd.setChecked(True)
        self.btn_dnd.setText("🌙 Snoozed 5 min")
        self.snooze_timer.start(5 * 60 * 1000)

    def end_snooze(self):
        self.btn_dnd.setChecked(False)

    def toggle_debug(self):
        self.worker.debug = not self.worker.debug

    def hide_to_tray(self):
        if self.tray:
            self.hide()
            self.tray.showMessage("BantayAso", "Still watching your dogs in the background.",
                                  QIcon(paw_pixmap(64)), 3000)
        else:
            self.showMinimized()

    def show_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def quit_app(self):
        self.quitting = True
        self.close()

    def closeEvent(self, e):
        if self.tray and not self.quitting:
            e.ignore()
            self.hide_to_tray()
            return
        self.worker.stop()
        if self.tray:
            self.tray.hide()
        e.accept()
        QApplication.quit()

    # ------------------------------------------------------------------ events page
    def _events(self):
        if not self.worker.pipe:
            return []
        return self.worker.pipe.events.today()

    @staticmethod
    def _thumb(path, w, h):
        pm = QPixmap(path) if path and Path(path).exists() else QPixmap()
        if pm.isNull():
            pm = QPixmap(w, h)
            pm.fill(QColor(T.RAISED))
        return pm.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

    def refresh_recent(self):
        self.recent.clear()
        for e in self._events()[:6]:
            t = time.strftime("%I:%M %p", time.localtime(e["ts"])).lstrip("0")
            it = QListWidgetItem(QIcon(self._thumb(e["snapshot"], 46, 34)),
                                 f"{e['reason'].split(' - ')[0].capitalize()}\n{t} · {T.LEVEL_NAMES[e['level']].capitalize()}")
            it.setForeground(QColor(T.CREAM))
            self.recent.addItem(it)
        if self.recent.count() == 0:
            self.recent.addItem(QListWidgetItem("No alerts yet - all calm."))

    def refresh_events(self):
        events = self._events()
        f = self.ev_filter.checkedId()
        shown = [e for e in events if f == 0 or (f == 1 and e["level"] == 3) or (f == 2 and e["level"] == 2)]
        self._shown = shown
        self.timeline.blockSignals(True)
        self.timeline.clear()
        for e in shown:
            t = time.strftime("%I:%M %p", time.localtime(e["ts"])).lstrip("0")
            tag = T.LEVEL_NAMES[e["level"]] + ("  (false alarm)" if e["false_alarm"] else "")
            it = QListWidgetItem(QIcon(self._thumb(e["snapshot"], 96, 56)),
                                 f"{t}   {e['reason'].split(' - ')[0].capitalize()}\n{tag}"
                                 + (f" · {e['zone']}" if e["zone"] else ""))
            it.setForeground(QColor(T.LEVEL_COLORS[e["level"]] if not e["false_alarm"] else T.FAINT))
            self.timeline.addItem(it)
        self.timeline.blockSignals(False)
        s = self.worker.pipe.events.summary_today() if self.worker.pipe else {}
        mins = int((time.time() - self.worker.started_at) / 60)
        self.st_watch.value_label.setText(f"{mins // 60}h {mins % 60}m" if mins >= 60 else f"{mins}m")
        self.st_danger.value_label.setText(str(s.get("danger", 0)))
        self.st_warn.value_label.setText(str(s.get("warning", 0)))
        self.st_hot.value_label.setText((s.get("hotspot") or "-").capitalize())
        self.ev_sub.setText(f"{s.get('total', 0)} alerts so far" if s.get("total") else "No alerts yet today")
        self.chart.set_data(s.get("by_hour", {}))
        if shown:
            self.timeline.setCurrentRow(0)
            self.select_event(0)
        else:
            self.select_event(-1)

    def select_event(self, row: int):
        e = self._shown[row] if 0 <= row < len(getattr(self, "_shown", [])) else None
        self.selected_event = e
        if not e:
            self.ev_img.clear()
            for k in self.kv:
                self.kv[k].setText("-")
            self.ev_ai.setText("Select an alert to see its snapshot.")
            return
        self.ev_img.setPixmap(self._thumb(e["snapshot"], 330, 240).copy(0, 0, 330, 240))
        self.kv["Level"].setText(f"<b style='color:{T.LEVEL_COLORS[e['level']]}'>{T.LEVEL_NAMES[e['level']].capitalize()}</b>")
        self.kv["Activity"].setText(e["reason"].split(" - ")[0].capitalize())
        self.kv["Near"].setText(e["near"] or "-")
        self.kv["Zone"].setText(e["zone"] or "-")
        self.kv["Time"].setText(time.strftime("%I:%M:%S %p", time.localtime(e["ts"])).lstrip("0"))
        self.ev_ai.setText(f"<b style='color:{T.CARAMEL}'>Local AI says:</b> {e['vlm']}" if e["vlm"]
                           else "<span style='color:%s'>No local AI description for this alert.</span>" % T.FAINT)
        self.btn_false.setText("Undo false alarm" if e["false_alarm"] else "Mark false alarm")

    def open_snapshot(self):
        e = self.selected_event
        if e and e["snapshot"] and Path(e["snapshot"]).exists() and hasattr(os, "startfile"):
            os.startfile(e["snapshot"])

    def mark_false(self):
        e = self.selected_event
        if e and self.worker.pipe:
            self.worker.pipe.events.mark_false_alarm(e["id"], not e["false_alarm"])
            self.refresh_events()

    # ------------------------------------------------------------------ zones page
    def set_zone_type(self, i: int):
        if self.worker.pipe:
            self.worker.pipe.editor.type = ZONE_TYPES[i]

    def zone_click(self, x: int, y: int, button: int):
        pipe = self.worker.pipe
        if not pipe:
            return
        if button == 2:
            self.zone_finish()
            return
        w, h = self.frame_size
        pipe.editor.current.append([x / w, y / h])
        self.update_zone_info()

    def zone_undo(self):
        if self.worker.pipe and self.worker.pipe.editor.current:
            self.worker.pipe.editor.current.pop()
            self.update_zone_info()

    def zone_finish(self):
        if self.worker.pipe:
            self.worker.pipe.editor.finish()
            self.update_zone_info()

    def zone_delete(self):
        if self.worker.pipe and self.worker.pipe.zones:
            self.worker.pipe.zones.pop()
            self.update_zone_info()

    def zone_save(self):
        if self.worker.pipe:
            self.worker.pipe.editor.finish()
            self.worker.pipe.save_zones()
            self.update_zone_info("Zones saved.")

    def update_zone_info(self, extra: str = ""):
        pipe = self.worker.pipe
        if not pipe:
            return
        names = ", ".join(z.name for z in pipe.zones) or "none yet"
        self.zone_info.setText(f"Zones: {names}   ·   points in the current zone: {len(pipe.editor.current)}   {extra}")

    # ------------------------------------------------------------------ settings
    def pick_clip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a .wav file", "", "WAV audio (*.wav)")
        if path:
            self.set_clip.setText(path)

    def save_settings(self):
        cfg = config.load()
        cfg["dog_name"] = self.set_name.text().strip()
        al = cfg.setdefault("alerts", {})
        al["voice"] = self.set_voice_cb.isChecked()
        al["speak_vlm"] = self.set_vlm_cb.isChecked()
        al["toast"] = self.set_toast_cb.isChecked()
        al["owner_voice_clip"] = self.set_clip.text().strip()
        config.save(cfg)
        pipe = self.worker.pipe
        if pipe:
            pipe.dog_name = cfg["dog_name"] or "your dog"
            pipe.speak_vlm, pipe.toasts, pipe.owner_clip = al["speak_vlm"], al["toast"], al["owner_voice_clip"]
            pipe.show_hazards = self.set_hz_cb.isChecked()
        self.btn_voice.setChecked(al["voice"])
        self.title.setText(f"Watching {cfg['dog_name']}" if cfg["dog_name"] else "Watching your dogs")


def run_ui(args) -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BantayAso")
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(T.QSS)
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())
