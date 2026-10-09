"""Two themes: "dark" (black + dark/light brown) and "light" (white/cream + dark/light brown).

Call apply(name) BEFORE building the window; widgets read the module-level colors.
"""
PALETTES = {
    # Dark: espresso + light-brown (latte) accents, cream text. No pure black.
    "dark": dict(
        BG="#1F1510", SIDE="#1A110C", SURFACE="#241811", PANEL="#2F2017", RAISED="#3D2A1E",
        LINE="#4E3626", BROWN="#7A5234", CARAMEL="#D8B48A", CREAM="#F6EBDD", MUTED="#C9B096",
        FAINT="#95795F", INSET="#22160F", HOT="#4A2A1A", HOT_LINE="#9A5A34", ON_ACCENT="#2A1B12",
        SAFE="#8CC59A", WATCH="#E8C15E", WARN="#E8904F", DANGER="#E5624E"),
    # Light: warm white + cream cards, dark-brown sidebar, light-brown accents.
    "light": dict(
        BG="#FFFFFF", SIDE="#3E2A1E", SURFACE="#FAF6F1", PANEL="#FFFFFF", RAISED="#F2E7DA",
        LINE="#E7D9C8", BROWN="#6B4A33", CARAMEL="#C08A5B", CREAM="#2E2018", MUTED="#8A7360",
        FAINT="#B3A08D", INSET="#FAF4EC", HOT="#FDECE2", HOT_LINE="#E8B08E", ON_ACCENT="#FFFFFF",
        SAFE="#4E9A5E", WATCH="#C99A2E", WARN="#D9732F", DANGER="#C9412F"),
}
NAME = "dark"
LEVEL_NAMES = ["SAFE", "WATCH", "WARNING", "DANGER"]


def apply(name: str = "dark") -> None:
    global NAME, LEVEL_COLORS, QSS
    NAME = name if name in PALETTES else "dark"
    globals().update(PALETTES[NAME])
    LEVEL_COLORS = [SAFE, WATCH, WARN, DANGER]  # noqa: F821
    QSS = build_qss()


def _svg(name: str, body: str) -> str:
    """QSS can only show images from files: write tiny SVG icons once to the temp folder."""
    import os
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "bantayaso_ui")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".svg")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(body)
    return path.replace("\\", "/")


def _arrow(color: str) -> str:
    return _svg(f"chev_{color.strip('#')}",
                f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8" viewBox="0 0 12 8">'
                f'<path d="M1 1.5l5 5 5-5" fill="none" stroke="{color}" stroke-width="1.8" '
                f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


def _check(color: str) -> str:
    return _svg(f"check_{color.strip('#')}",
                f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="10" viewBox="0 0 12 10">'
                f'<path d="M1.5 5.2l3.2 3.1L10.5 1.8" fill="none" stroke="{color}" stroke-width="2" '
                f'stroke-linecap="round" stroke-linejoin="round"/></svg>')


ICONS = {   # 24x24 stroke icons (lucide-style), drawn in the theme colour
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-4"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "backspace": '<path d="M21 5H9l-6 7 6 7h12a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1z"/><path d="M17.5 9.5l-5 5"/>'
                 '<path d="M12.5 9.5l5 5"/>',
    "trash": '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v5"/>'
             '<path d="M14 11v5"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8"/>'
            '<path d="M7 3v5h8"/>',
}


def icon_path(name: str, color: str) -> str:
    return _svg(f"i_{name}_{color.strip('#')}",
                f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
                f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                f'{ICONS[name]}</svg>')


def build_qss() -> str:
    g = globals()
    chev = _arrow(g["CARAMEL"])
    chev_m = _arrow(g["MUTED"])
    check = _check(g["ON_ACCENT"])
    side_text = "#EADBC8" if NAME == "light" else g["MUTED"]       # light theme: dark-brown sidebar
    side_active = "#5A3D2A" if NAME == "light" else g["RAISED"]
    return f"""
* {{ font-family: "Segoe UI Variable", "Segoe UI", sans-serif; font-size: 10pt; color: {g['CREAM']}; }}
QMainWindow, QWidget#root {{ background: {g['SURFACE']}; }}
QWidget#side {{ background: {g['SIDE']}; border-right: 1px solid {g['LINE']}; }}
QWidget#side QLabel {{ color: {"#F5EADC" if NAME == "light" else g['CREAM']}; }}
QLabel#brand {{ font-size: 13pt; font-weight: 600; }}
QLabel#brandSub {{ color: {side_text}; font-size: 9pt; }}
QLabel#muted {{ color: {g['MUTED']}; font-size: 9pt; }}
QLabel#faint {{ color: {g['FAINT']}; font-size: 9pt; }}
QLabel#h1 {{ font-size: 15pt; font-weight: 600; }}
QLabel#h3 {{ color: {g['MUTED']}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
QPushButton#nav {{ text-align: left; padding: 10px 16px; font-size: 10.5pt; border: none; border-radius: 8px;
                   color: {side_text}; background: transparent; }}
QPushButton#nav:hover {{ color: #FFFFFF; background: {side_active}; }}
QPushButton#nav:checked {{ background: {side_active}; color: #FFFFFF; border-left: 3px solid {g['CARAMEL']}; }}
QPushButton#nav:pressed {{ background: {g['BROWN']}; }}
QFrame#card {{ background: {g['PANEL']}; border: 1px solid {g['LINE']}; border-radius: 14px; }}
QPushButton {{ background: {g['RAISED']}; border: 1px solid {g['LINE']}; border-radius: 9px; padding: 8px 14px; min-height: 20px; }}
QPushButton:hover {{ border-color: {g['CARAMEL']}; }}
QPushButton:pressed {{ background: {g['LINE']}; border-color: {g['CARAMEL']}; padding-top: 10px; padding-bottom: 6px; }}
QPushButton:disabled {{ color: {g['FAINT']}; border-color: {g['LINE']}; }}
QPushButton#primary {{ background: {g['CARAMEL']}; color: {g['ON_ACCENT']}; border-color: {g['CARAMEL']}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {g['BROWN']}; border-color: {g['BROWN']}; color: #FFFFFF; }}
QPushButton#primary:pressed {{ background: {g['BROWN']}; padding-top: 10px; padding-bottom: 6px; }}
QPushButton#icon, QPushButton#iconPrimary {{ padding: 0; min-width: 38px; max-width: 38px; min-height: 38px;
                                             max-height: 38px; border-radius: 10px; }}
QPushButton#iconPrimary {{ background: {g['CARAMEL']}; border-color: {g['CARAMEL']}; }}
QPushButton#iconPrimary:hover {{ background: {g['BROWN']}; border-color: {g['BROWN']}; }}
QPushButton#icon:pressed, QPushButton#iconPrimary:pressed {{ padding-top: 2px; }}
QFrame#toolbar {{ background: {g['PANEL']}; border: 1px solid {g['LINE']}; border-radius: 14px; }}
QPushButton#pill {{ border-radius: 15px; padding: 5px 14px; min-height: 18px; color: {g['MUTED']}; background: {g['PANEL']}; }}
QPushButton#pill[menu="true"] {{ padding-right: 30px; }}
QPushButton::menu-indicator {{ image: url("{chev_m}"); width: 10px; height: 7px; subcontrol-origin: padding;
                               subcontrol-position: center right; right: 12px; }}
QPushButton#pill:hover {{ color: {g['CREAM']}; border-color: {g['CARAMEL']}; }}
QPushButton#pill:pressed {{ background: {g['LINE']}; padding-top: 7px; padding-bottom: 3px; }}
QPushButton#pill:checked {{ color: {g['ON_ACCENT']}; background: {g['CARAMEL']}; border-color: {g['CARAMEL']}; font-weight: 600; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 10px; padding: 6px; margin: 2px 0; }}
QListWidget::item:hover {{ background: {g['INSET']}; }}
QListWidget::item:disabled {{ background: transparent; border: none; color: {g['FAINT']}; }}
QListWidget::item:selected {{ background: {g['RAISED']}; border: 1px solid {g['CARAMEL']}; color: {g['CREAM']}; }}
QLineEdit, QSpinBox, QComboBox {{ background: {g['INSET']}; border: 1px solid {g['LINE']}; border-radius: 9px;
                                   padding: 7px 12px; min-height: 22px; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {g['CARAMEL']}; }}
QComboBox {{ padding: 7px 34px 7px 12px; min-height: 22px; combobox-popup: 0; }}
QComboBoxPrivateContainer {{ background: transparent; border: none; padding: 0; margin: 0; }}
QComboBox:hover {{ border-color: {g['CARAMEL']}; }}
QComboBox:on {{ border-color: {g['CARAMEL']}; background: {g['RAISED']}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 30px; border: none; }}
QComboBox::down-arrow {{ image: url("{chev}"); width: 12px; height: 8px; }}
QComboBox QAbstractItemView {{ background: {g['PANEL']}; color: {g['CREAM']}; border: 1px solid {g['CARAMEL']};
                               border-radius: 8px; padding: 4px; outline: none; }}
QComboBox QAbstractItemView::item {{ min-height: 30px; padding: 4px 10px; border-radius: 6px; color: {g['CREAM']}; }}
QComboBox QAbstractItemView::item:hover {{ background: {g['RAISED']}; }}
QComboBox QAbstractItemView::item:selected {{ background: {g['CARAMEL']}; color: {g['ON_ACCENT']}; }}
QTableWidget#things {{ background: {g['PANEL']}; alternate-background-color: {g['INSET']}; border: 1px solid {g['LINE']};
                       border-radius: 10px; gridline-color: transparent; outline: none; font-size: 11pt; }}
QTableWidget#things::item {{ padding: 0 12px; border: none; }}
QTableWidget#things::item:hover {{ background: {g['RAISED']}; }}
QTableWidget#things::item:selected {{ background: {g['RAISED']}; color: {g['CREAM']}; }}
QHeaderView::section {{ background: {g['SURFACE']}; color: {g['MUTED']}; border: none; border-bottom: 1px solid {g['LINE']};
                        padding: 8px 12px; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
QTableCornerButton::section {{ background: {g['SURFACE']}; border: none; }}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 6px; border: 1.5px solid {g['LINE']}; background: {g['INSET']}; }}
QCheckBox::indicator:hover {{ border-color: {g['CARAMEL']}; }}
QCheckBox::indicator:checked {{ background: {g['CARAMEL']}; border-color: {g['CARAMEL']}; image: url("{check}"); }}
QMenu {{ background: {g['PANEL']}; border: 1px solid {g['LINE']}; border-radius: 12px; padding: 6px; }}
QMenu::item {{ padding: 8px 22px 8px 14px; border-radius: 8px; color: {g['CREAM']}; }}
QMenu::item:selected {{ background: {g['RAISED']}; }}
QMenu::item:disabled {{ color: {g['FAINT']}; }}
QMenu::separator {{ height: 1px; background: {g['LINE']}; margin: 5px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 6px; }}
QMenu::indicator:checked {{ image: url("{_check(g['CARAMEL'])}"); }}
QScrollArea {{ border: none; background: {g['SURFACE']}; }}
QScrollArea > QWidget > QWidget {{ background: {g['SURFACE']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {g['LINE']}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {g['LINE']}; border-radius: 5px; min-width: 30px; }}
QSlider::groove:horizontal {{ height: 6px; background: {g['RAISED']}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {g['CARAMEL']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {g['BROWN']}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QMessageBox {{ background: {g['PANEL']}; }}
QToolTip {{ background: {g['PANEL']}; color: {g['CREAM']}; border: 1px solid {g['LINE']}; }}
"""


apply("dark")
