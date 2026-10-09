"""BantayAso main window.

Pages: Monitor · Events · Dogs · Zones · Things · Settings, plus a system tray (runs in background).
Top bar: Record clip, Bantay (ask / hands-free / voice / do-not-disturb) and Menu (theme, tray, quit).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cv2
from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPropertyAnimation, QRect, QSize, Qt,
                            QTimer, Signal)
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QFormLayout, QFrame, QGraphicsColorizeEffect, QGraphicsOpacityEffect,
                               QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListView, QListWidget,
                               QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox, QStackedWidget, QSystemTrayIcon,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import config
from ..zones import ZONE_LABELS, ZONE_TYPES
from . import theme as T
from .widgets import HourChart, RiskMeter, VideoView, card, paw_pixmap, stat_card
from .worker import Worker

TIERS = [(3, "High danger"), (2, "Medium"), (1, "Low"), (0, "Ignore (look-alike)")]
TIER_NAME = dict(TIERS)
PAGES = ["Monitor", "Events", "Dogs", "Zones", "Things", "Behaviours", "Settings"]
P_MONITOR, P_EVENTS, P_DOGS, P_ZONES, P_THINGS, P_BEHAVIOURS, P_SETTINGS = range(7)
LEVEL_CHOICES = ["Safe", "Watch", "Warning", "Danger"]


def lbl(text="", name=None, wrap=False):
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


class SmoothButtons(QObject):
    """Smooth colour feedback for every button: a soft tint fades in on hover, deepens on press
    and fades back out (QSS alone can't animate colours)."""

    def __init__(self, parent):
        super().__init__(parent)
        self._anims = {}

    def attach(self, b: QPushButton):
        eff = QGraphicsColorizeEffect(b)
        eff.setColor(QColor(T.CARAMEL))
        eff.setStrength(0.0)
        b.setGraphicsEffect(eff)
        b.installEventFilter(self)

    def _go(self, b, value, ms):
        eff = b.graphicsEffect()
        if not isinstance(eff, QGraphicsColorizeEffect):
            return
        eff.setColor(QColor(T.CARAMEL))
        old = self._anims.get(b)
        if old:
            old.stop()
        a = QPropertyAnimation(eff, b"strength", self)
        a.setDuration(ms)
        a.setEndValue(value)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.start()
        self._anims[b] = a

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.Enter:
            self._go(obj, 0.18, 160)
        elif t == QEvent.Leave:
            self._go(obj, 0.0, 220)
        elif t == QEvent.MouseButtonPress:
            self._go(obj, 0.45, 90)
        elif t == QEvent.MouseButtonRelease:
            self._go(obj, 0.18 if obj.underMouse() else 0.0, 260)
        return False


class MainWindow(QMainWindow):
    heard = Signal(str, bool)
    answered = Signal(str, str)
    mic_state = Signal(str)
    cams_found = Signal(list)

    CONFIRM = {"Save settings": "Saved ✓", "Save zones": "Zones saved ✓", "Finish zone": "Zone added ✓",
               "Undo point": "Undone ✓", "Delete last zone": "Deleted ✓", "Reset taught actions": "Reset ✓",
               "Forget dog": "Forgotten ✓", "I've got it": "Got it ✓", "Snooze 5 min": "Snoozed ✓",
               "Mark false alarm": "Marked ✓", "Undo false alarm": "Restored ✓", "Open snapshot": "Opening…",
               "Save && apply": "Applied ✓", "Add / update": "Updated ✓", "Remove": "Removed ✓",
               "Use this camera": "Switching…", "Test voice": "Speaking…", "Refresh list": "Searching…"}

    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("BantayAso")
        self.setWindowIcon(QIcon(paw_pixmap(64)))
        self.resize(1320, 800)
        self.worker = Worker(args, self)
        self.frame_size = (1280, 720)
        self.last_img: QImage | None = None
        self.quitting = False
        self.ack_alert_id = None
        self.selected_event = None
        self.pending_dog_name = ""
        self.camera_lost = False
        self.smooth = SmoothButtons(self)

        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._sidebar())
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        for build in (self._monitor_page, self._events_page, self._dogs_page, self._zones_page,
                      self._things_page, self._behaviours_page, self._settings_page):
            self.stack.addWidget(build())
        self._tray()

        self.worker.frame_ready.connect(self.on_frame)
        self.worker.message.connect(self.on_message)
        self.worker.event.connect(self.on_event)
        self.worker.ready.connect(self.on_ready)
        self.heard.connect(self.on_heard)
        self.answered.connect(self.on_answered)
        self.mic_state.connect(self.on_mic_state)
        self.cams_found.connect(self.on_cams_found)
        QShortcut(QKeySequence("F12"), self, activated=self.toggle_debug)
        QShortcut(QKeySequence("F2"), self, activated=self.ask_voice)
        self.snooze_timer = QTimer(self, singleShot=True, timeout=self.end_snooze)
        for b in self.findChildren(QPushButton):
            b.setCursor(Qt.PointingHandCursor)
            self.smooth.attach(b)
            b.clicked.connect(lambda _=False, b=b: self._confirm(b))
        self.worker.start()

    # ================================================================== layout
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
        for i, text in enumerate(PAGES):
            b = QPushButton(text, objectName="nav", checkable=True)
            self.nav.addButton(b, i)
            v.addWidget(b)
        self.nav.button(0).setChecked(True)
        self.nav.idClicked.connect(self.go)
        v.addStretch()
        return side

    @staticmethod
    def _round_menu(m: QMenu) -> QMenu:
        """Rounded menu corners: drop the square native frame/shadow and let the stylesheet draw it."""
        m.setWindowFlags(m.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        m.setAttribute(Qt.WA_TranslucentBackground)
        return m

    @staticmethod
    def _style_combo(cb: QComboBox) -> None:
        """Dropdown popup that matches the theme (no native white frame/shadow, rounded, drops below)."""
        cb.setView(QListView())
        cb.view().setTextElideMode(Qt.ElideRight)
        pop = cb.view().window()
        pop.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        pop.setAttribute(Qt.WA_TranslucentBackground)

    def _page(self, title: str, subtitle: str = ""):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)
        v.addWidget(lbl(title, "h1"))
        if subtitle:
            v.addWidget(lbl(subtitle, "muted", wrap=True))
        return page, v

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

        self.btn_rec = QPushButton("Record clip", objectName="pill", checkable=True)
        self.btn_rec.setToolTip("Record a video of the camera to data\\clips (no boxes drawn).\n"
                                "Use it to keep proof of what happened or to make test clips.")
        self.btn_rec.clicked.connect(lambda: self.worker.request("record"))
        top.addWidget(self.btn_rec)

        self.btn_bantay = QPushButton("Bantay", objectName="pill")
        self.btn_bantay.setProperty("menu", True)
        m = self._round_menu(QMenu(self))
        self.act_ask = m.addAction("Ask now (push to talk)\tF2", self.ask_voice)
        m.addSeparator()
        self.act_hf = m.addAction("Hands-free: listen for \"Bantay\"")
        self.act_hf.setCheckable(True)
        self.act_hf.toggled.connect(self.set_hands_free_from_menu)
        self.act_voice = m.addAction("Voice on")
        self.act_voice.setCheckable(True)
        self.act_voice.setChecked(True)
        self.act_voice.toggled.connect(self.set_voice)
        self.act_dnd = m.addAction("Do not disturb (still logs alerts)")
        self.act_dnd.setCheckable(True)
        self.act_dnd.toggled.connect(self.set_dnd)
        m.addAction("Snooze alerts 5 min", self.snooze)
        self.btn_bantay.setMenu(m)
        top.addWidget(self.btn_bantay)

        self.btn_more = QPushButton("Menu", objectName="pill")
        self.btn_more.setProperty("menu", True)
        mm = self._round_menu(QMenu(self))
        self.act_theme = mm.addAction("Light mode" if T.NAME == "dark" else "Dark mode", self.toggle_theme)
        mm.addAction("Minimize to tray", self.hide_to_tray)
        mm.addAction("Show technical info\tF12", self.toggle_debug)
        mm.addSeparator()
        mm.addAction("Quit BantayAso", self.quit_app)
        self.btn_more.setMenu(mm)
        top.addWidget(self.btn_more)
        v.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(16)
        vid_card = card()
        vc = QVBoxLayout(vid_card)
        vc.setContentsMargins(14, 14, 14, 10)
        self.pick_banner = card()
        pb = QHBoxLayout(self.pick_banner)
        pb.setContentsMargins(14, 8, 8, 8)
        self.pick_label = lbl("")
        self.pick_label.setStyleSheet("font-weight: 600;")
        pb.addWidget(self.pick_label, 1)
        b_cancel = QPushButton("Cancel")
        b_cancel.clicked.connect(self.cancel_pick)
        pb.addWidget(b_cancel)
        self.pick_banner.hide()
        vc.addWidget(self.pick_banner)
        self.video = VideoView("Loading the local AI models...")
        self.video.clicked.connect(self.video_click)
        vc.addWidget(self.video, 1)
        self.meter = RiskMeter()
        vc.addWidget(self.meter)
        row.addWidget(vid_card, 1)

        right = QVBoxLayout()
        right.setSpacing(14)
        self.status_card = card()
        self.status_card.setFixedWidth(330)
        sc = QVBoxLayout(self.status_card)
        sc.setContentsMargins(16, 16, 16, 16)
        self.level_pill = lbl("● SAFE")
        self.status_title = lbl("Starting...", wrap=True)
        self.status_title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        self.status_sub = lbl("", "muted", wrap=True)
        self.spoken = lbl("", wrap=True)
        self.spoken.hide()
        btns = QHBoxLayout()
        self.btn_ack = QPushButton("I've got it", objectName="primary")
        self.btn_snooze = QPushButton("Snooze 5 min")
        self.btn_cam = QPushButton("Choose another camera", objectName="primary")
        self.btn_ack.clicked.connect(self.acknowledge)
        self.btn_snooze.clicked.connect(self.snooze)
        self.btn_cam.clicked.connect(lambda: self.go(P_SETTINGS))
        for b in (self.btn_ack, self.btn_snooze, self.btn_cam):
            btns.addWidget(b)
        for w in (self.level_pill, self.status_title, self.status_sub, self.spoken):
            sc.addWidget(w)
        sc.addLayout(btns)
        right.addWidget(self.status_card)

        ask_card = card()
        ask_card.setFixedWidth(330)
        ac = QVBoxLayout(ask_card)
        ac.setContentsMargins(14, 12, 14, 12)
        ac.addWidget(lbl("ASK BANTAY", "h3"))
        self.ask_box = QLineEdit()
        self.ask_box.setPlaceholderText("Is Oreo sleeping? · What happened today?")
        self.ask_box.returnPressed.connect(lambda: self.ask_text(self.ask_box.text()))
        ac.addWidget(self.ask_box)
        self.ask_answer = lbl("Type a question, press F2 to talk, or turn on hands-free in the Bantay menu.",
                              "muted", wrap=True)
        ac.addWidget(self.ask_answer)
        self.heard_label = lbl("", "faint", wrap=True)
        ac.addWidget(self.heard_label)
        right.addWidget(ask_card)

        rec_card = card()
        rec_card.setFixedWidth(330)
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
        dv.addWidget(self.ev_img)
        f = QFormLayout()
        self.kv = {}
        for k in ("Level", "Dog", "Activity", "Near", "Zone", "Time"):
            self.kv[k] = lbl("-")
            f.addRow(lbl(k, "muted"), self.kv[k])
        dv.addLayout(f)
        self.ev_ai = lbl("", wrap=True)
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

    def _dogs_page(self):
        page, v = self._page("My dogs", "Give each dog a name so alerts and answers say \"Oreo is chewing…\" "
                                        "instead of \"your dog\". Bantay learns each dog's look on this laptop.")
        c = card()
        c.setMaximumWidth(900)
        cv = QVBoxLayout(c)
        cv.setContentsMargins(18, 16, 18, 16)
        add = QHBoxLayout()
        self.add_dog_name = QLineEdit()
        self.add_dog_name.setPlaceholderText("New dog's name, e.g. Oreo")
        self.add_dog_name.returnPressed.connect(self.add_dog)
        b_add = QPushButton("Add dog → click it on the camera", objectName="primary")
        b_add.clicked.connect(self.add_dog)
        add.addWidget(self.add_dog_name, 1)
        add.addWidget(b_add)
        cv.addLayout(add)
        cv.addWidget(lbl("ADDED DOGS", "h3"))
        self.dog_list = QListWidget()
        self.dog_list.setIconSize(QSize(64, 64))
        self.dog_list.setMinimumHeight(260)
        cv.addWidget(self.dog_list, 1)
        row = QHBoxLayout()
        self.dog_btns = []
        for text, fn in (("Add more photos", self.reteach_dog), ("Rename", self.rename_dog),
                         ("Forget dog", self.forget_dog)):
            bb = QPushButton(text)
            bb.clicked.connect(fn)
            bb.setEnabled(False)
            row.addWidget(bb)
            self.dog_btns.append(bb)
        row.addStretch()
        cv.addLayout(row)
        cv.addWidget(lbl("Select a dog first. \"Add more photos\" takes you to the camera: click that dog again "
                         "and Bantay learns another look at it (another angle, lighting or pose). Do it when Bantay "
                         "mixes up your dogs or calls a named dog \"your dog\".", "faint", wrap=True))
        self.dog_list.currentItemChanged.connect(
            lambda cur, _p: [b.setEnabled(bool(cur and cur.data(Qt.UserRole))) for b in self.dog_btns])
        v.addWidget(c, 1)
        c2 = card()
        c2.setMaximumWidth(900)
        c2v = QVBoxLayout(c2)
        c2v.setContentsMargins(18, 14, 18, 14)
        c2v.addWidget(lbl("TEACH WHAT THEY'RE DOING", "h3"))
        c2v.addWidget(lbl("If Bantay shows the wrong activity (e.g. \"scratching\" when your dog is sitting), "
                          "click the dog on the Monitor video → This dog is actually… → pick the right one. "
                          "Your examples are saved on this laptop and used from then on.", "muted", wrap=True))
        rr = QHBoxLayout()
        b_reset = QPushButton("Reset taught actions")
        b_reset.clicked.connect(self.reset_examples)
        rr.addWidget(b_reset)
        rr.addStretch()
        c2v.addLayout(rr)
        v.addWidget(c2)
        return page

    def _zones_page(self):
        page, v = self._page("Zones", "Mark areas of the room so Bantay knows what is safe and what is not. "
                                      "Food bowl and Play area make eating or chewing toys there normal.")
        # ---- step 1: what kind of area
        tb = QFrame(objectName="toolbar")
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(14, 10, 10, 10)
        tl.setSpacing(8)
        tl.addWidget(lbl("AREA", "h3"))
        self.zone_type = QButtonGroup(self)
        for i, t in enumerate(ZONE_TYPES):
            b = QPushButton(ZONE_LABELS[t], objectName="pill", checkable=True)
            self.zone_type.addButton(b, i)
            tl.addWidget(b)
        self.zone_type.button(0).setChecked(True)
        self.zone_type.idClicked.connect(self.set_zone_type)
        tl.addSpacing(6)
        self.zone_name = QLineEdit()
        self.zone_name.setPlaceholderText("Name (optional), e.g. Sofa")
        self.zone_name.setMinimumWidth(170)
        self.zone_name.returnPressed.connect(self.zone_finish)
        tl.addWidget(self.zone_name, 1)
        v.addWidget(tb)

        # ---- step 2: draw on the video; actions as icons (hover shows what each does)
        c = card()
        cv = QVBoxLayout(c)
        cv.setContentsMargins(14, 12, 14, 14)
        cv.setSpacing(10)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.zone_step = lbl("", wrap=True)
        self.zone_step.setStyleSheet("font-weight: 600;")
        head.addWidget(self.zone_step, 1)
        self.zone_icons = []
        for key, tip, fn, primary in (("undo", "Undo last point  (Backspace)", self.zone_undo, False),
                                      ("check", "Finish zone: close the shape you drew  (Enter)", self.zone_finish, False),
                                      ("backspace", "Delete the last zone", self.zone_delete, False),
                                      ("trash", "Clear all zones", self.zone_clear, False),
                                      ("save", "Save zones", self.zone_save, True)):
            b = QPushButton(objectName="iconPrimary" if primary else "icon")
            b.setToolTip(tip)
            b.setIconSize(QSize(18, 18))
            b.clicked.connect(fn)
            if primary:
                head.addSpacing(6)
            head.addWidget(b)
            self.zone_icons.append((b, key, primary))
        cv.addLayout(head)
        self.zone_video = VideoView("Waiting for the camera...")
        self.zone_video.clicked.connect(self.zone_click)
        cv.addWidget(self.zone_video, 1)
        self.zone_info = lbl("", "muted")
        cv.addWidget(self.zone_info)
        v.addWidget(c, 1)
        for key, fn in ((Qt.Key_Return, self.zone_finish), (Qt.Key_Enter, self.zone_finish),
                        (Qt.Key_Backspace, self.zone_undo)):
            QShortcut(QKeySequence(key), page, activated=fn, context=Qt.WidgetWithChildrenShortcut)
        self.zone_step.setText("Pick an area type, then click its corners on the video.")
        self._zone_icons_refresh()
        return page

    def _zone_icons_refresh(self):
        for b, key, primary in getattr(self, "zone_icons", []):
            b.setIcon(QIcon(T.icon_path(key, T.ON_ACCENT if primary else T.CREAM)))

    def _things_page(self):
        page, v = self._page("Things to watch", "Objects Bantay looks for near your dogs, and how dangerous each one "
                                                "is. \"Ignore\" words are look-alikes (e.g. dog collar) that stop "
                                                "false alarms. Changes apply live after Save and apply.")
        c = card()
        c.setMaximumWidth(900)
        cv = QVBoxLayout(c)
        cv.setContentsMargins(18, 16, 18, 16)
        cv.setSpacing(12)
        cv.addWidget(lbl("ADD OR CHANGE", "h3"))
        ed = QHBoxLayout()
        self.thing_name = QLineEdit()
        self.thing_name.setPlaceholderText("Object, e.g. rubber band")
        self.thing_name.returnPressed.connect(self.thing_add)
        self.thing_tier = QComboBox()
        self._style_combo(self.thing_tier)
        self.thing_tier.setMinimumWidth(190)
        for _, name in TIERS:
            self.thing_tier.addItem(name)
        b_up = QPushButton("Add / update")
        b_up.clicked.connect(self.thing_add)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self.thing_remove)
        ed.addWidget(self.thing_name, 1)
        ed.addWidget(self.thing_tier)
        ed.addWidget(b_up)
        ed.addWidget(b_rm)
        cv.addLayout(ed)

        fl = QHBoxLayout()
        self.thing_search = QLineEdit()
        self.thing_search.setPlaceholderText("Search objects...")
        self.thing_search.setClearButtonEnabled(True)
        self.thing_search.textChanged.connect(lambda _t: self.refresh_things())
        fl.addWidget(self.thing_search, 1)
        self.thing_filter = None
        self.thing_chips = {}
        grp = QButtonGroup(c)
        grp.setExclusive(True)
        for key, text in ((None, "All"), (3, "High"), (2, "Medium"), (1, "Low"), (0, "Ignore")):
            b = QPushButton(text, objectName="pill")
            b.setCheckable(True)
            b.setChecked(key is None)
            b.clicked.connect(lambda _c=False, k=key: self._thing_filter(k))
            grp.addButton(b)
            fl.addWidget(b)
            self.thing_chips[key] = b
        cv.addLayout(fl)

        self.thing_table = QTableWidget(0, 2, objectName="things")
        self.thing_table.setHorizontalHeaderLabels(["OBJECT", "DANGER LEVEL"])
        hh = self.thing_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.resizeSection(1, 220)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.thing_table.verticalHeader().setVisible(False)
        self.thing_table.verticalHeader().setDefaultSectionSize(46)
        self.thing_table.setAlternatingRowColors(True)
        self.thing_table.setShowGrid(False)
        self.thing_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.thing_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.thing_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thing_table.setFocusPolicy(Qt.NoFocus)
        self.thing_table.currentCellChanged.connect(lambda r, *_: self.thing_select(r))
        cv.addWidget(self.thing_table, 1)
        sv = QHBoxLayout()
        self.thing_count = lbl("", "muted")
        sv.addWidget(self.thing_count)
        sv.addStretch()
        b_save = QPushButton("Save && apply", objectName="primary")
        b_save.clicked.connect(self.thing_save)
        sv.addWidget(b_save)
        cv.addLayout(sv)
        v.addWidget(c, 1)
        self.things = dict(config.load().get("hazards") or {})
        self.refresh_things()
        return page

    def _behaviours_page(self):
        from ..risk import BEHAVIORS
        page, v = self._page("Behaviours", "Decide how much each thing your dog does matters, and how long it must "
                                           "go on before Bantay alerts. Warning and Danger speak and notify; Watch "
                                           "only shows on screen. Changes apply live after Save and apply.")
        c = card()
        c.setMaximumWidth(900)
        cv = QVBoxLayout(c)
        cv.setContentsMargins(18, 16, 18, 16)
        cv.setSpacing(12)
        tbl = QTableWidget(len(BEHAVIORS), 3, objectName="things")
        tbl.setHorizontalHeaderLabels(["BEHAVIOUR", "LEVEL", "ALERT AFTER"])
        hh = tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for col, wdt in ((1, 200), (2, 170)):
            hh.setSectionResizeMode(col, QHeaderView.Fixed)
            hh.resizeSection(col, wdt)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        tbl.verticalHeader().setVisible(False)
        tbl.verticalHeader().setDefaultSectionSize(52)
        tbl.setAlternatingRowColors(True)
        tbl.setShowGrid(False)
        tbl.setSelectionMode(QAbstractItemView.NoSelection)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setFocusPolicy(Qt.NoFocus)
        self.beh_rows = {}
        for r, (key, (name, _lvl, sec)) in enumerate(BEHAVIORS.items()):
            tbl.setItem(r, 0, QTableWidgetItem(name))
            cb = QComboBox()
            self._style_combo(cb)
            cb.addItems(LEVEL_CHOICES)
            tbl.setCellWidget(r, 1, self._cell(cb))
            sp = None
            if sec is not None:
                sp = QSpinBox()
                sp.setRange(1, 300)
                sp.setSuffix(" s")
                tbl.setCellWidget(r, 2, self._cell(sp))
            else:
                it = QTableWidgetItem("right away")
                it.setForeground(QColor(T.MUTED))
                tbl.setItem(r, 2, it)
            self.beh_rows[key] = (cb, sp)
        cv.addWidget(tbl, 1)
        sv = QHBoxLayout()
        sv.addWidget(lbl("Wrong activity on screen? Click the dog on the Monitor video and choose "
                         "\"This dog is actually...\" to teach Bantay.", "faint", wrap=True), 1)
        b_def = QPushButton("Reset to defaults")
        b_def.clicked.connect(lambda: self.load_behaviours(defaults=True))
        b_save = QPushButton("Save && apply", objectName="primary")
        b_save.clicked.connect(self.save_behaviours)
        sv.addWidget(b_def)
        sv.addWidget(b_save)
        cv.addLayout(sv)
        v.addWidget(c, 1)
        self.load_behaviours()
        return page

    @staticmethod
    def _cell(w: QWidget) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(8, 0, 12, 0)
        h.addWidget(w)
        return box

    def load_behaviours(self, defaults: bool = False):
        from ..risk import BEHAVIORS
        saved = {} if defaults else (config.load().get("behaviors") or {})
        for key, (_n, lvl, sec) in BEHAVIORS.items():
            cb, sp = self.beh_rows[key]
            o = saved.get(key) or {}
            cb.setCurrentIndex(int(o.get("level", lvl)))
            if sp is not None:
                sp.setValue(int(round(float(o.get("seconds", sec)))))

    def save_behaviours(self):
        beh = {k: ({"level": cb.currentIndex()} | ({"seconds": sp.value()} if sp is not None else {}))
               for k, (cb, sp) in self.beh_rows.items()}
        cfg = config.load()
        cfg["behaviors"] = beh
        config.save(cfg)
        if self.worker.pipe:
            self.worker.pipe.engine.set_behaviors(beh)

    def _settings_page(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(20, 18, 20, 18)
        v.setSpacing(12)
        v.addWidget(lbl("Settings", "h1"))
        cfg = config.load()
        al = cfg.get("alerts", {})

        # --- camera
        cam = card()
        cam.setMaximumWidth(860)
        cf = QFormLayout(cam)
        cf.setContentsMargins(18, 16, 18, 16)
        cf.setVerticalSpacing(10)
        cf.addRow(lbl("CAMERA", "h3"))
        self.cam_combo = QComboBox()
        self._style_combo(self.cam_combo)
        self.cam_combo.setMinimumWidth(280)
        b_ref = QPushButton("Refresh list")
        b_ref.clicked.connect(lambda: self.scan_cameras(probe=True))
        r1 = QHBoxLayout()
        r1.addWidget(self.cam_combo, 1)
        r1.addWidget(b_ref)
        cf.addRow(lbl("USB camera", "muted"), r1)
        self.cam_url = QLineEdit()
        self.cam_url.setPlaceholderText("or a phone/Wi-Fi stream: http://192.168.1.5:8080/video  ·  or a video file")
        b_file = QPushButton("Video file...")
        b_file.clicked.connect(self.pick_video)
        b_file.hide()                      # "Other source" removed from the UI (USB camera only)
        b_use = QPushButton("Use this camera", objectName="primary")
        b_use.clicked.connect(self.use_camera)
        cf.addRow("", b_use)
        cf.addRow("", lbl("If the camera disconnects, Bantay keeps retrying it; pick another here to switch "
                          "right away.", "faint", wrap=True))
        v.addWidget(cam)

        # --- voice + alerts
        c = card()
        c.setMaximumWidth(860)
        f = QFormLayout(c)
        f.setContentsMargins(18, 16, 18, 16)
        f.setVerticalSpacing(10)
        f.addRow(lbl("VOICE & ALERTS", "h3"))
        self.set_voice_cb = QCheckBox("Speak alerts and answers out loud")
        self.set_voice_cb.setChecked(al.get("voice", True))
        f.addRow(lbl("Voice", "muted"), self.set_voice_cb)
        self.voice_combo = QComboBox()
        self._style_combo(self.voice_combo)
        self.voice_combo.addItem("(default Windows voice)")
        b_test = QPushButton("Test voice")
        b_test.clicked.connect(self.test_voice)
        vr = QHBoxLayout()
        vr.addWidget(self.voice_combo, 1)
        vr.addWidget(b_test)
        f.addRow(lbl("Which voice", "muted"), vr)
        self.voice_speed = QSlider(Qt.Horizontal)
        self.voice_speed.setRange(-5, 6)
        self.voice_speed.setValue(int(al.get("voice_rate", 1)))
        f.addRow(lbl("Speed", "muted"), self.voice_speed)
        f.addRow("", lbl("More voices: Windows Settings → Time & language → Speech → Add voices.", "faint", wrap=True))
        self.set_vlm_cb = QCheckBox("Also speak the local AI's description of each alert")
        self.set_vlm_cb.setChecked(al.get("speak_vlm", True))
        f.addRow("", self.set_vlm_cb)
        self.set_toast_cb = QCheckBox("Show Windows notifications")
        self.set_toast_cb.setChecked(al.get("toast", True))
        f.addRow(lbl("Notifications", "muted"), self.set_toast_cb)
        self.set_clip = QLineEdit(al.get("owner_voice_clip", ""))
        self.set_clip.setPlaceholderText("optional .wav of you saying \"No!\" (played on Danger)")
        pick = QPushButton("Browse...")
        pick.clicked.connect(self.pick_clip)
        cr = QHBoxLayout()
        cr.addWidget(self.set_clip, 1)
        cr.addWidget(pick)
        f.addRow(lbl("Owner voice", "muted"), cr)
        self.set_hf_cb = QCheckBox("Hands-free: answer when I say \"Bantay, …\"")
        self.set_hf_cb.setChecked(cfg.get("qa", {}).get("hands_free", False))
        f.addRow(lbl("Ask Bantay", "muted"), self.set_hf_cb)
        v.addWidget(c)

        # --- look
        c3 = card()
        c3.setMaximumWidth(860)
        f3 = QFormLayout(c3)
        f3.setContentsMargins(18, 16, 18, 16)
        f3.addRow(lbl("LOOK", "h3"))
        self.set_theme = QComboBox()
        self._style_combo(self.set_theme)
        self.set_theme.addItems(["Dark (espresso + light brown)", "Light (warm white + brown)"])
        self.set_theme.setCurrentIndex(1 if T.NAME == "light" else 0)
        self.set_theme.currentIndexChanged.connect(lambda i: self.apply_theme("light" if i == 1 else "dark"))
        f3.addRow(lbl("Theme", "muted"), self.set_theme)
        self.set_hz_cb = QCheckBox("Show object boxes on the video")
        self.set_hz_cb.setChecked(True)
        f3.addRow(lbl("Video", "muted"), self.set_hz_cb)
        v.addWidget(c3)

        save = QPushButton("Save settings", objectName="primary")
        save.setMaximumWidth(860)
        save.clicked.connect(self.save_settings)
        v.addWidget(save)
        v.addWidget(lbl("Press F12 anytime to show or hide technical info on the video.", "faint"))
        v.addStretch()
        for hl in inner.findChildren(QHBoxLayout):
            hl.setSpacing(10)
        for fl in inner.findChildren(QFormLayout):          # same label column in every card -> fields line up
            fl.setHorizontalSpacing(14)
            fl.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            for r in range(fl.rowCount()):
                it = fl.itemAt(r, QFormLayout.LabelRole)
                if it and it.widget():
                    lw = it.widget()
                    lw.setFixedWidth(104)
                    if isinstance(lw, QLabel):                    # centre the label on its field
                        fi = fl.itemAt(r, QFormLayout.FieldRole)
                        fh = fi.sizeHint().height() if fi else 0
                        lw.setMinimumHeight(min(max(fh, 20), 40))
                        lw.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        return page

    def _tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(QIcon(paw_pixmap(64)), self)
        self.tray.setToolTip("BantayAso - watching your dogs")
        m = self._round_menu(QMenu())
        m.addAction("Show BantayAso", self.show_window)
        m.addAction("Mute / unmute voice", lambda: self.act_voice.toggle())
        m.addAction("Do not disturb", lambda: self.act_dnd.toggle())
        m.addAction("Quit", self.quit_app)
        self.tray.setContextMenu(m)
        self.tray.activated.connect(lambda r: self.show_window() if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()
        self._tray_level = -1

    # ================================================================== feedback + theme
    def _confirm(self, b: QPushButton):
        text = b.text()
        if text in self.CONFIRM and not b.property("confirming"):
            b.setProperty("confirming", True)
            b.setText(self.CONFIRM[text])

            def restore(b=b, text=text):
                if b.text() == self.CONFIRM.get(text):
                    b.setText(text)
                b.setProperty("confirming", False)
            QTimer.singleShot(1400, restore)

    def toggle_theme(self):
        self.apply_theme("light" if T.NAME == "dark" else "dark", save=True)

    def apply_theme(self, name: str, save: bool = False):
        if name == T.NAME and not save:
            return
        # smooth: freeze a picture of the old look on top, switch underneath, fade the picture out
        cover = QLabel(self)
        cover.setPixmap(self.grab())
        cover.setGeometry(self.rect())
        cover.show()
        T.apply(name)
        QApplication.instance().setStyleSheet(T.QSS)
        self._restyle()
        eff = QGraphicsOpacityEffect(cover)
        cover.setGraphicsEffect(eff)
        anim = QPropertyAnimation(eff, b"opacity", self)
        anim.setDuration(420)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.InOutCubic)
        anim.finished.connect(cover.deleteLater)
        anim.start(QPropertyAnimation.DeleteWhenStopped)
        if save:
            cfg = config.load()
            cfg.setdefault("ui", {})["theme"] = T.NAME
            config.save(cfg)

    def _restyle(self):
        inset = f"background: {T.INSET}; border: 1px solid {T.LINE}; border-radius: 10px; padding: 10px;"
        self.spoken.setStyleSheet(inset)
        self.ev_ai.setStyleSheet(inset)
        self.ev_img.setStyleSheet(f"background: {T.INSET}; border-radius: 10px;")
        self.st_danger.value_label.setStyleSheet(f"font-size: 18pt; font-weight: 600; color: {T.DANGER};")
        self.st_warn.value_label.setStyleSheet(f"font-size: 18pt; font-weight: 600; color: {T.WARN};")
        for w in (self.st_watch, self.st_hot):
            w.value_label.setStyleSheet(f"font-size: 18pt; font-weight: 600; color: {T.CREAM};")
        self.act_theme.setText("Light mode" if T.NAME == "dark" else "Dark mode")
        self.set_theme.blockSignals(True)
        self.set_theme.setCurrentIndex(1 if T.NAME == "light" else 0)
        self.set_theme.blockSignals(False)
        self._set_status_card(self.meter.level, self.status_title.text(), self.status_sub.text(), None)
        self.refresh_things()
        self._zone_icons_refresh()
        for w in (self.video, self.zone_video, self.meter, self.chart):
            w.update()
        self.refresh_recent()

    # ================================================================== live updates
    def on_ready(self):
        self.subtitle.setText(f"Live since {time.strftime('%I:%M %p').lstrip('0')}")
        pipe = self.worker.pipe
        self.set_voice(self.act_voice.isChecked())
        lst = pipe.listener
        if lst:
            lst.on_text = lambda text, woke: self.heard.emit(text, woke)
            lst.on_state = lambda s: self.mic_state.emit(s)
            lst.on_heard = lambda t: self.mic_state.emit("heard:" + t)
            self.act_hf.setChecked(self.set_hf_cb.isChecked())
        else:
            self.act_ask.setEnabled(False)
            self.act_hf.setEnabled(False)
        QTimer.singleShot(1500, self.fill_voices)
        self.refresh_recent()
        self.refresh_dogs()
        self.scan_cameras()

    def on_message(self, msg: str):
        if msg == "CAMERA_LOST":
            self.camera_lost = True
            return
        if msg == "CAMERA_OK":
            self.camera_lost = False
            return
        if self.worker.pipe is None:
            self.video.placeholder = msg
            self.video.update()
        if any(k in msg for k in ("Recording", "Saved", "Snapshot", "Switched")):
            self.subtitle.setText(msg)

    def on_frame(self, img, st: dict):
        self.frame_size = st["frame_size"]
        self.last_img = img
        page = self.stack.currentIndex()
        if page == P_MONITOR:
            if self.video.selecting and self.worker.pipe:
                self.video.targets = [(tid, box) for tid, box, _ in self.worker.pipe.state.boxes]
            self.video.set_image(img)
        elif page == P_ZONES:
            self.zone_video.set_image(img)
        self.meter.set_level(st["level"])
        la = st.get("last_alert")
        if self.camera_lost:
            self._set_status_card(1, "Camera disconnected", "Check the cable, or switch to another camera.",
                                  None, camera=True)
        elif la and la["id"] != self.ack_alert_id and time.time() - la["ts"] < 60 and st["level"] >= 2:
            sub = "Alert at " + time.strftime("%I:%M:%S %p", time.localtime(la["ts"])).lstrip("0")
            self._set_status_card(la["level"], st["status"].split(": ", 1)[-1].split(" - ")[0].capitalize(),
                                  sub, la)
        else:
            dogs = st["dogs"]
            sub = f"{dogs} dog{'s' if dogs != 1 else ''} in view" if dogs else "Waiting for a dog to appear"
            self._set_status_card(st["level"], st["status"], sub, None)
        self.btn_rec.setChecked(self.worker.recording)
        self.btn_rec.setText("Stop recording" if self.worker.recording else "Record clip")
        if self.tray and st["level"] != self._tray_level:
            self._tray_level = st["level"]
            self.tray.setIcon(QIcon(paw_pixmap(64, T.LEVEL_COLORS[st["level"]] if st["level"] else T.CARAMEL)))

    def _set_status_card(self, level, title, sub, alert, camera=False):
        color = T.LEVEL_COLORS[level]
        self.level_pill.setText(f"  ●  {'CAMERA' if camera else T.LEVEL_NAMES[level]}  ")
        self.level_pill.setStyleSheet(f"background: {color}; color: {'#FFFFFF' if T.NAME == 'light' else '#1a0f08'};"
                                      " font-weight: 700; border-radius: 11px; padding: 3px 4px;")
        self.level_pill.setFixedHeight(24)
        self.level_pill.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.status_title.setText(title)
        self.status_sub.setText(sub)
        hot = level >= 2
        self.status_card.setStyleSheet(
            f"QFrame#card {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 {T.HOT}, stop:1 {T.PANEL});"
            f" border: 1px solid {T.HOT_LINE}; border-radius: 14px; }}" if hot else "")
        if alert:
            txt = f"<b style='color:{T.CARAMEL}'>Spoken:</b> “{alert['text']}”"
            if alert.get("vlm"):
                txt += f"<br><b style='color:{T.CARAMEL}'>Local AI:</b> {alert['vlm']}"
            self.spoken.setText(txt)
            self.spoken.show()
        else:
            self.spoken.hide()
        self.btn_ack.setVisible(hot and not camera)
        self.btn_snooze.setVisible(hot and not camera)
        self.btn_cam.setVisible(camera)

    def on_event(self, ev: dict):
        self.refresh_recent()
        if self.stack.currentIndex() == P_EVENTS:
            self.refresh_events()

    # ================================================================== top bar actions
    def go(self, i: int):
        self.stack.setCurrentIndex(i)
        self.nav.button(i).setChecked(True)
        pipe = self.worker.pipe
        if pipe:
            pipe.editor.active = i == P_ZONES
        if i == P_EVENTS:
            self.refresh_events()
        elif i == P_ZONES:
            self.update_zone_info()
        elif i == P_DOGS:
            self.refresh_dogs()

    def set_voice(self, on: bool):
        self.act_voice.setText("Voice on" if on else "Voice muted")
        if self.worker.pipe:
            self.worker.pipe.voice = on

    def set_dnd(self, on: bool):
        if self.worker.pipe:
            self.worker.pipe.dnd = on
        if not on:
            self.snooze_timer.stop()
        self.subtitle.setText("Do not disturb: alerts are logged silently" if on else
                              f"Live since {time.strftime('%I:%M %p').lstrip('0')}")

    def acknowledge(self):
        la = self.worker.pipe.state.last_alert if self.worker.pipe else None
        if la:
            self.ack_alert_id = la["id"]

    def snooze(self):
        self.acknowledge()
        self.act_dnd.setChecked(True)
        self.subtitle.setText("Snoozed for 5 minutes")
        self.snooze_timer.start(5 * 60 * 1000)

    def end_snooze(self):
        self.act_dnd.setChecked(False)

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

    # ================================================================== Ask Bantay
    def ask_voice(self):
        lst = self.worker.pipe.listener if self.worker.pipe else None
        if lst and lst.push_to_talk():
            self.ask_answer.setText("Listening... ask your question.")

    def set_hands_free_from_menu(self, on: bool):
        lst = self.worker.pipe.listener if self.worker.pipe else None
        if lst:
            lst.set_hands_free(on)
            self.on_mic_state("hands_free" if on else "idle")
            self.heard_label.setText("Hands-free is on: say \"Bantay\", pause, then your question." if on else "")
        self.set_hf_cb.setChecked(on)

    def on_mic_state(self, s: str):
        if s.startswith("heard:"):
            self.heard_label.setText(f"Heard: “{s[6:]}”")
            return
        hf = self.act_hf.isChecked()
        self.btn_bantay.setText({"listening": "Listening…", "thinking": "Thinking…",
                                 "hf_error": "Mic problem"}.get(s, "Bantay (listening)" if hf else "Bantay"))

    def on_heard(self, text: str, _woke: bool):
        if not text:
            self.ask_answer.setText("I didn't catch that. Try again a bit closer to the mic.")
            return
        self.ask_text(text)

    def ask_text(self, q: str):
        q = q.strip()
        pipe = self.worker.pipe
        if not q or not pipe:
            return
        self.ask_box.clear()
        self.ask_answer.setText(f"<span style='color:{T.MUTED}'>You: {q}</span><br>Thinking...")
        threading.Thread(target=lambda: self.answered.emit(q, pipe.ask(q)), daemon=True).start()

    def on_answered(self, q: str, a: str):
        self.ask_answer.setText(f"<span style='color:{T.MUTED}'>You: {q}</span><br>"
                                f"<b style='color:{T.CARAMEL}'>Bantay:</b> {a}")

    # ================================================================== dogs
    def _crop(self, box) -> QPixmap | None:
        if self.last_img is None:
            return None
        x1, y1, x2, y2 = box
        return QPixmap.fromImage(self.last_img.copy(QRect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))))

    def video_click(self, x: int, y: int, button: int):
        pipe = self.worker.pipe
        if not pipe:
            return
        hit = [(tid, box, name) for tid, box, name in pipe.state.boxes
               if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
        if not hit:
            return
        tid, box, name = min(hit, key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))
        if self.pending_dog_name and pipe.registry:
            self.confirm_dog(tid, box, self.pending_dog_name)
            return
        menu = self._round_menu(QMenu(self))
        if pipe.registry:
            menu.addAction(f"Name this dog{f' ({name})' if name else ''}...").setData(("name", None))
        if pipe.classifier:
            sub = self._round_menu(menu.addMenu("This dog is actually..."))
            for lab in pipe.classifier.labels:
                sub.addAction(lab).setData(("teach", lab))
        act = menu.exec(QCursor.pos())
        if not act or not act.data():
            return
        kind, lab = act.data()
        if kind == "name":
            new, ok = QInputDialog.getText(self, "Name this dog", "What's this dog's name?", text=name or "")
            if ok and new.strip():
                self.confirm_dog(tid, box, new.strip())
        else:
            pipe.classifier.teach(tid, lab)
            self.ask_answer.setText(f"Thanks! Learning what \"{lab}\" looks like for your dog. "
                                    "Keep it in view for a couple of seconds.")

    def confirm_dog(self, tid, box, name):
        pm = self._crop(box)
        box_msg = QMessageBox(self)
        box_msg.setWindowTitle("Is this the right dog?")
        box_msg.setText(f"Is this {name}?")
        box_msg.setInformativeText("Bantay will learn this dog's look over the next few seconds.")
        if pm is not None:
            box_msg.setIconPixmap(pm.scaled(180, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        yes = box_msg.addButton(f"Yes, this is {name}", QMessageBox.AcceptRole)
        box_msg.addButton("No, pick again", QMessageBox.RejectRole)
        self.video.selecting = False
        box_msg.exec()
        if box_msg.clickedButton() is not yes:
            self.start_pick(name)                  # stay in picking mode
            return
        self.cancel_pick()
        self.worker.pipe.registry.start_enroll(tid, name)
        if pm is not None:
            pm.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation).save(
                str(config.DATA_DIR / "dogs" / f"{name.replace(' ', '_')}.jpg"))
        self.subtitle.setText(f"Learning what {name} looks like... keep {name} in view for a few seconds.")
        QTimer.singleShot(4000, self.refresh_dogs)

    def start_pick(self, name: str):
        """Dog-picking mode: dim the video, highlight every dog box, show a banner with Cancel."""
        self.pending_dog_name = name
        self.pick_label.setText(f"Choose {name}: click {name}'s highlighted box on the video")
        self.pick_banner.show()
        self.video.selecting = True
        self.video.update()

    def cancel_pick(self):
        self.pending_dog_name = ""
        self.pick_banner.hide()
        self.video.selecting = False
        self.video.hover = None
        self.video.setCursor(Qt.ArrowCursor)
        self.video.update()

    def add_dog(self):
        name = self.add_dog_name.text().strip()
        if not name:
            self.add_dog_name.setFocus()
            return
        self.add_dog_name.clear()
        self.go(P_MONITOR)
        self.start_pick(name)

    def refresh_dogs(self):
        self.dog_list.clear()
        reg = self.worker.pipe.registry if self.worker.pipe else None
        dogs = (reg.dogs if reg else {})
        for n, arr in sorted(dogs.items()):
            thumb = config.DATA_DIR / "dogs" / f"{n.replace(' ', '_')}.jpg"
            icon = QIcon(QPixmap(str(thumb))) if thumb.exists() else QIcon(paw_pixmap(64))
            it = QListWidgetItem(icon, f"{n}\n{len(arr)} photos learned")
            it.setData(Qt.UserRole, n)
            self.dog_list.addItem(it)
        if not dogs:
            it = QListWidgetItem("No dogs yet - type a name above and click Add dog.")
            it.setFlags(Qt.NoItemFlags)                     # just a message, not a dog
            self.dog_list.addItem(it)
        for b in getattr(self, "dog_btns", []):
            b.setEnabled(False)

    def _selected_dog(self):
        it = self.dog_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def forget_dog(self):
        n, reg = self._selected_dog(), self.worker.pipe.registry if self.worker.pipe else None
        if n and reg:
            reg.delete(n)
            thumb = config.DATA_DIR / "dogs" / f"{n.replace(' ', '_')}.jpg"
            if thumb.exists():
                thumb.unlink()
            self.refresh_dogs()

    def rename_dog(self):
        n, reg = self._selected_dog(), self.worker.pipe.registry if self.worker.pipe else None
        if not (n and reg):
            return
        new, ok = QInputDialog.getText(self, "Rename dog", "New name:", text=n)
        if ok and new.strip() and new.strip() != n:
            reg.add_samples(new.strip(), reg.dogs[n])
            old_thumb = config.DATA_DIR / "dogs" / f"{n.replace(' ', '_')}.jpg"
            reg.delete(n)
            if old_thumb.exists():
                old_thumb.rename(config.DATA_DIR / "dogs" / f"{new.strip().replace(' ', '_')}.jpg")
            self.refresh_dogs()

    def reteach_dog(self):
        n = self._selected_dog()
        if n:
            self.go(P_MONITOR)
            self.start_pick(n)

    def reset_examples(self):
        if self.worker.pipe and self.worker.pipe.classifier:
            self.worker.pipe.classifier.forget_examples()

    # ================================================================== events page
    def _events(self):
        return self.worker.pipe.events.today() if self.worker.pipe else []

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
            who = e["dog"] if e["dog"] and not str(e["dog"]).lstrip("-").isdigit() else ""
            it = QListWidgetItem(QIcon(self._thumb(e["snapshot"], 46, 34)),
                                 f"{(who + ': ') if who else ''}{e['reason'].split(' - ')[0].capitalize()}\n"
                                 f"{t} · {T.LEVEL_NAMES[e['level']].capitalize()}")
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
        self.st_hot.value_label.setText((s.get("hotspot") or "-"))
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
        who = e["dog"] if e["dog"] and not str(e["dog"]).lstrip("-").isdigit() else "-"
        self.kv["Dog"].setText(who)
        self.kv["Activity"].setText(e["reason"].split(" - ")[0].capitalize())
        self.kv["Near"].setText(e["near"] or "-")
        self.kv["Zone"].setText(e["zone"] or "-")
        self.kv["Time"].setText(time.strftime("%I:%M:%S %p", time.localtime(e["ts"])).lstrip("0"))
        self.ev_ai.setText(f"<b style='color:{T.CARAMEL}'>Local AI says:</b> {e['vlm']}" if e["vlm"]
                           else f"<span style='color:{T.FAINT}'>No local AI description for this alert.</span>")
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

    # ================================================================== zones
    def set_zone_type(self, i: int):
        if self.worker.pipe:
            self.worker.pipe.editor.type = ZONE_TYPES[i]
            self.update_zone_info()

    def zone_click(self, x: int, y: int, button: int):
        pipe = self.worker.pipe
        if not pipe:
            return
        if button == 2:
            self.zone_finish()
            return
        w, h = self.frame_size
        cur = pipe.editor.current
        if len(cur) >= 3 and abs(x - cur[0][0] * w) < w * 0.025 and abs(y - cur[0][1] * h) < h * 0.04:
            self.zone_finish()                         # clicked the first point again: close the shape
            return
        cur.append([x / w, y / h])
        self.update_zone_info()

    def zone_undo(self):
        if self.worker.pipe and self.worker.pipe.editor.current:
            self.worker.pipe.editor.current.pop()
            self.update_zone_info()

    def zone_finish(self):
        if self.worker.pipe:
            ed = self.worker.pipe.editor
            ed.pending_name = self.zone_name.text().strip()
            ed.finish()
            ed.pending_name = ""
            self.zone_name.clear()
            self.update_zone_info()

    def zone_delete(self):
        if self.worker.pipe and self.worker.pipe.zones:
            self.worker.pipe.zones.pop()
            self.update_zone_info()

    def zone_clear(self):
        pipe = self.worker.pipe
        if not pipe or not (pipe.zones or pipe.editor.current):
            return
        if QMessageBox.question(self, "Clear all zones", "Remove every zone? (Press Save afterwards to keep it "
                                "that way.)") != QMessageBox.Yes:
            return
        pipe.zones.clear()
        pipe.editor.current = []
        self.update_zone_info("All zones cleared - press Save to keep it.")

    def zone_save(self):
        if self.worker.pipe:
            if self.worker.pipe.editor.current:
                self.zone_finish()
            self.worker.pipe.save_zones()
            self.update_zone_info("Zones saved.")

    def update_zone_info(self, extra: str = ""):
        pipe = self.worker.pipe
        if not pipe:
            return
        names = ", ".join(z.name for z in pipe.zones) or "none yet"
        n = len(pipe.editor.current)
        kind = ZONE_LABELS.get(pipe.editor.type, pipe.editor.type)
        if n == 0:
            step = f"Click the corners of the {kind.lower()} area on the video."
        elif n < 3:
            step = f"{n} point{'s' if n > 1 else ''} - keep clicking around the area (at least 3)."
        else:
            step = f"{n} points - click the first point again or press \u2713 to finish the zone."
        self.zone_step.setText(step)
        self.zone_info.setText(f"Zones: {names}" + (f"   ·   {extra}" if extra else ""))

    # ================================================================== things
    def _thing_filter(self, key):
        self.thing_filter = key
        self.refresh_things()

    def _level_pill(self, tier: int) -> QWidget:
        col = {3: T.DANGER, 2: T.WARN, 1: T.WATCH}.get(tier, T.FAINT)
        short = {3: "High danger", 2: "Medium", 1: "Low", 0: "Ignore"}.get(tier, str(tier))
        q = QColor(col)
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 0, 12, 0)
        p = QLabel(f"\u25CF  {short}")
        p.setStyleSheet(f"color: {col}; background: rgba({q.red()},{q.green()},{q.blue()},40);"
                        f"border: 1px solid rgba({q.red()},{q.green()},{q.blue()},120); border-radius: 11px;"
                        f"padding: 0 12px; font-size: 9pt; font-weight: 600;")
        p.setFixedHeight(24)
        h.addWidget(p, 0, Qt.AlignVCenter)
        h.addStretch()
        w.setAttribute(Qt.WA_TransparentForMouseEvents)
        return w

    def refresh_things(self):
        q = self.thing_search.text().strip().lower() if hasattr(self, "thing_search") else ""
        f = getattr(self, "thing_filter", None)
        rows = [(n, int(t)) for n, t in sorted(self.things.items(), key=lambda kv: (-int(kv[1]), kv[0]))
                if (not q or q in n) and (f is None or int(t) == f)]
        tbl = self.thing_table
        tbl.blockSignals(True)
        tbl.clearContents()
        tbl.setRowCount(len(rows))
        for r, (name, tier) in enumerate(rows):
            it = QTableWidgetItem(name)
            it.setData(Qt.UserRole, name)
            if tier == 0:
                it.setForeground(QColor(T.MUTED))
            tbl.setItem(r, 0, it)
            tbl.setItem(r, 1, QTableWidgetItem(""))
            tbl.setCellWidget(r, 1, self._level_pill(tier))
        tbl.blockSignals(False)
        counts = {k: sum(1 for t in self.things.values() if int(t) == k) for k in (3, 2, 1, 0)}
        for k, b in self.thing_chips.items():
            base = {None: "All", 3: "High", 2: "Medium", 1: "Low", 0: "Ignore"}[k]
            b.setText(f"{base}  {len(self.things) if k is None else counts[k]}")
        self.thing_count.setText(f"Showing {len(rows)} of {len(self.things)}" if len(rows) != len(self.things)
                                 else f"{len(self.things)} objects")

    def thing_select(self, row, *_):
        it = self.thing_table.item(row, 0) if row is not None and row >= 0 else None
        if it and it.data(Qt.UserRole):
            n = it.data(Qt.UserRole)
            self.thing_name.setText(n)
            self.thing_tier.setCurrentIndex([t for t, _ in TIERS].index(int(self.things[n])))

    def thing_add(self):
        n = self.thing_name.text().strip().lower()
        if n:
            self.things[n] = TIERS[self.thing_tier.currentIndex()][0]
            self.refresh_things()

    def thing_remove(self):
        n = self.thing_name.text().strip().lower()
        if n in self.things:
            del self.things[n]
            self.thing_name.clear()
            self.refresh_things()

    def thing_save(self):
        cfg = config.load()
        cfg["hazards"] = dict(self.things)
        config.save(cfg)
        if self.worker.pipe:
            self.worker.pipe.request_vocab(self.things)

    # ================================================================== settings
    def scan_cameras(self, probe: bool = False):
        """List cameras by name (from Windows, in a child process). probe=True also opens each index
        to check it gives frames (only when the user presses Refresh list)."""
        current = self.worker.source

        def run():
            from ..camnames import camera_names
            names = camera_names()
            self._cam_names = names
            found = []
            for i, _n in enumerate(names.get("msmf") or []):
                found.append((i, "msmf", "in use now" if str(i) == str(current) else ""))
            if probe or not found:
                for be_name, be in (("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW)):
                    for i in range(5):
                        if any(f[0] == i for f in found):
                            continue
                        if str(i) == str(current):
                            found.append((i, be_name, "in use now"))
                            continue
                        cap = cv2.VideoCapture(i, be)
                        ok = cap.isOpened() and cap.read()[0]
                        cap.release()
                        if ok:
                            found.append((i, be_name, ""))
            found.sort(key=lambda f: f[0])
            self.cams_found.emit(found)
        threading.Thread(target=run, daemon=True).start()

    def on_cams_found(self, found: list):
        self.cam_combo.clear()
        seen = set()
        for i, be, note in found:
            if i in seen:
                continue
            seen.add(i)
            names = getattr(self, "_cam_names", {})
            lst = names.get(be) or names.get("msmf") or []
            name = lst[i] if i < len(lst) else f"Camera {i}"
            self.cam_combo.addItem(name + ("  (in use now)" if note else ""), (i, be))
            if note:
                self.cam_combo.setCurrentIndex(self.cam_combo.count() - 1)
            self.cam_combo.setItemData(self.cam_combo.count() - 1, f"Camera {i} · {be}", Qt.ToolTipRole)
        if not found:
            self.cam_combo.addItem("No USB camera found", None)

    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", str(config.DATA_DIR / "clips"),
                                              "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.cam_url.setText(path)

    def use_camera(self):
        other = self.cam_url.text().strip()
        cfg = config.load()
        if other:
            self.worker.switch_source(other)
        else:
            data = self.cam_combo.currentData()
            if not data:
                return
            i, be = data
            cfg.setdefault("camera", {}).update({"index": i, "backend": be})
            config.save(cfg)
            self.worker.switch_source(i)
        self.camera_lost = False
        self.go(P_MONITOR)

    def fill_voices(self):
        sp = self.worker.pipe.speaker if self.worker.pipe else None
        if not sp:
            return
        if not sp.voices:
            QTimer.singleShot(1500, self.fill_voices)
            return
        cur = config.load().get("alerts", {}).get("voice_name", "")
        self.voice_combo.clear()
        self.voice_combo.addItem("(default Windows voice)", "")
        for vname in sp.voices:
            self.voice_combo.addItem(vname, vname)
            if cur and cur.lower() in vname.lower():
                self.voice_combo.setCurrentIndex(self.voice_combo.count() - 1)

    def test_voice(self):
        sp = self.worker.pipe.speaker if self.worker.pipe else None
        if sp:
            sp.set_voice(self.voice_combo.currentData() or "", self.voice_speed.value())
            sp.say("Hi! I'm Bantay. I'll watch your dogs for you.", urgent=True)

    def pick_clip(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a .wav file", "", "WAV audio (*.wav)")
        if path:
            self.set_clip.setText(path)

    def save_settings(self):
        cfg = config.load()
        cfg.pop("dog_name", None)
        al = cfg.setdefault("alerts", {})
        al["voice"] = self.set_voice_cb.isChecked()
        al["speak_vlm"] = self.set_vlm_cb.isChecked()
        al["toast"] = self.set_toast_cb.isChecked()
        al["owner_voice_clip"] = self.set_clip.text().strip()
        al["voice_name"] = self.voice_combo.currentData() or ""
        al["voice_rate"] = self.voice_speed.value()
        cfg.setdefault("qa", {})["hands_free"] = self.set_hf_cb.isChecked()
        cfg.setdefault("ui", {})["theme"] = T.NAME
        config.save(cfg)
        pipe = self.worker.pipe
        if pipe:
            pipe.speak_vlm, pipe.toasts, pipe.owner_clip = al["speak_vlm"], al["toast"], al["owner_voice_clip"]
            pipe.show_hazards = self.set_hz_cb.isChecked()
            pipe.speaker.set_voice(al["voice_name"], al["voice_rate"])
        self.act_voice.setChecked(al["voice"])
        self.act_hf.setChecked(self.set_hf_cb.isChecked())


def _install_crash_log() -> None:
    """Write any crash (Python error or a native crash inside a driver/DLL) to data/crash.log."""
    import faulthandler
    import traceback
    path = config.DATA_DIR / "crash.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a", encoding="utf-8", buffering=1)
    f.write(f"\n=== BantayAso started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    faulthandler.enable(file=f, all_threads=True)

    def hook(kind, exc, tb):
        msg = "".join(traceback.format_exception(type(exc), exc, tb)) if exc else ""
        f.write(f"\n--- {time.strftime('%H:%M:%S')} {kind}\n{msg}")
        print(msg, file=sys.stderr)

    sys.excepthook = lambda t, e, tb: hook("error", e, tb)
    threading.excepthook = lambda a: hook(f"error in thread {a.thread.name if a.thread else ''}",
                                          a.exc_value, a.exc_traceback)
    run_ui._crash_file = f                       # keep it open for faulthandler


def run_ui(args) -> None:
    _install_crash_log()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")             # native Windows style ignores parts of the stylesheet (dropdowns)
    app.setApplicationName("BantayAso")
    app.setQuitOnLastWindowClosed(False)
    T.apply(config.load().get("ui", {}).get("theme", "dark"))
    app.setStyleSheet(T.QSS)
    win = MainWindow(args)
    win._restyle()
    win.show()
    sys.exit(app.exec())
