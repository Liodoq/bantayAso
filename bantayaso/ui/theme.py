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


def build_qss() -> str:
    g = globals()
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
QPushButton#nav {{ text-align: left; padding: 9px 12px; border: none; border-radius: 8px;
                   color: {side_text}; background: transparent; }}
QPushButton#nav:hover {{ color: #FFFFFF; background: {side_active}; }}
QPushButton#nav:checked {{ background: {side_active}; color: #FFFFFF; border-left: 3px solid {g['CARAMEL']}; }}
QPushButton#nav:pressed {{ background: {g['BROWN']}; }}
QFrame#card {{ background: {g['PANEL']}; border: 1px solid {g['LINE']}; border-radius: 14px; }}
QPushButton {{ background: {g['RAISED']}; border: 1px solid {g['LINE']}; border-radius: 9px; padding: 8px 12px; }}
QPushButton:hover {{ border-color: {g['CARAMEL']}; }}
QPushButton:pressed {{ background: {g['LINE']}; border-color: {g['CARAMEL']}; padding-top: 10px; padding-bottom: 6px; }}
QPushButton:disabled {{ color: {g['FAINT']}; border-color: {g['LINE']}; }}
QPushButton#primary {{ background: {g['CARAMEL']}; color: {g['ON_ACCENT']}; border-color: {g['CARAMEL']}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {g['BROWN']}; border-color: {g['BROWN']}; color: #FFFFFF; }}
QPushButton#primary:pressed {{ background: {g['BROWN']}; padding-top: 10px; padding-bottom: 6px; }}
QPushButton#pill {{ border-radius: 14px; padding: 5px 12px; color: {g['MUTED']}; background: {g['PANEL']}; }}
QPushButton#pill:hover {{ color: {g['CREAM']}; border-color: {g['CARAMEL']}; }}
QPushButton#pill:pressed {{ background: {g['LINE']}; padding-top: 7px; padding-bottom: 3px; }}
QPushButton#pill:checked {{ color: {g['ON_ACCENT']}; background: {g['CARAMEL']}; border-color: {g['CARAMEL']}; font-weight: 600; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 10px; padding: 6px; margin: 2px 0; }}
QListWidget::item:hover {{ background: {g['INSET']}; }}
QListWidget::item:selected {{ background: {g['RAISED']}; border: 1px solid {g['CARAMEL']}; color: {g['CREAM']}; }}
QLineEdit, QSpinBox, QComboBox {{ background: {g['INSET']}; border: 1px solid {g['LINE']}; border-radius: 8px; padding: 6px; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {g['CARAMEL']}; }}
QComboBox QAbstractItemView {{ background: {g['PANEL']}; selection-background-color: {g['RAISED']}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {g['LINE']}; background: {g['INSET']}; }}
QCheckBox::indicator:hover {{ border-color: {g['CARAMEL']}; }}
QCheckBox::indicator:checked {{ background: {g['CARAMEL']}; border-color: {g['CARAMEL']}; }}
QMenu {{ background: {g['PANEL']}; border: 1px solid {g['LINE']}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 6px; }}
QMenu::item:selected {{ background: {g['RAISED']}; }}
QScrollArea {{ border: none; background: transparent; }}
QToolTip {{ background: {g['PANEL']}; color: {g['CREAM']}; border: 1px solid {g['LINE']}; }}
"""


apply("dark")
