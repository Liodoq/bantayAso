"""Colors + Qt stylesheet matching docs/ui-mockup.html."""
BG, SURFACE, PANEL, RAISED, LINE = "#0E0B09", "#17110D", "#211812", "#2B2019", "#3A2C22"
BROWN, CARAMEL, CREAM, MUTED, FAINT = "#8B5E3C", "#C8894E", "#F2E6D8", "#A8957F", "#6E5D4E"
SAFE, WATCH, WARN, DANGER = "#7FB98A", "#E3B54F", "#E3803F", "#E0533F"
LEVEL_COLORS = [SAFE, WATCH, WARN, DANGER]
LEVEL_NAMES = ["SAFE", "WATCH", "WARNING", "DANGER"]

QSS = f"""
* {{ font-family: "Segoe UI Variable", "Segoe UI", sans-serif; font-size: 10pt; color: {CREAM}; }}
QMainWindow, QWidget#root {{ background: {SURFACE}; }}
QWidget#side {{ background: #120D0A; border-right: 1px solid {LINE}; }}
QLabel#brand {{ font-size: 13pt; font-weight: 600; }}
QLabel#brandSub, QLabel#muted {{ color: {MUTED}; font-size: 9pt; }}
QLabel#faint {{ color: {FAINT}; font-size: 9pt; }}
QLabel#h1 {{ font-size: 15pt; font-weight: 600; }}
QLabel#h3 {{ color: {MUTED}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
QPushButton#nav {{ text-align: left; padding: 9px 12px; border: none; border-radius: 8px;
                   color: {MUTED}; background: transparent; }}
QPushButton#nav:checked {{ background: {RAISED}; color: {CREAM}; border-left: 3px solid {CARAMEL}; }}
QPushButton#nav:hover {{ color: {CREAM}; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 12px; }}
QPushButton {{ background: {RAISED}; border: 1px solid {LINE}; border-radius: 9px; padding: 8px 12px; }}
QPushButton:hover {{ border-color: {BROWN}; }}
QPushButton#primary {{ background: {CARAMEL}; color: #1A120C; border-color: {CARAMEL}; font-weight: 600; }}
QPushButton#pill {{ border-radius: 14px; padding: 5px 12px; color: {MUTED}; background: {PANEL}; }}
QPushButton#pill:checked {{ color: {CREAM}; border-color: {BROWN}; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border-radius: 10px; padding: 6px; margin: 2px 0; }}
QListWidget::item:selected {{ background: {RAISED}; border: 1px solid {BROWN}; }}
QLineEdit, QSpinBox, QComboBox {{ background: {RAISED}; border: 1px solid {LINE}; border-radius: 8px; padding: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid {LINE}; background: {RAISED}; }}
QCheckBox::indicator:checked {{ background: {CARAMEL}; border-color: {CARAMEL}; }}
QScrollArea {{ border: none; background: transparent; }}
QToolTip {{ background: {PANEL}; color: {CREAM}; border: 1px solid {LINE}; }}
"""
