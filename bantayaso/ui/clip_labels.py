"""Small post-recording form, also used to label older recordings."""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QLabel, QPlainTextEdit, QVBoxLayout)

from ..clip_labels import load_labels, save_labels
from . import theme as T


class ClipLabelDialog(QDialog):
    def __init__(self, clip, dog_names=(), actions=(), parent=None):
        super().__init__(parent)
        self.clip = Path(clip)
        data = load_labels(self.clip)
        self.setWindowTitle('Label this recording')
        self.setObjectName('clipLabels')
        self.setStyleSheet(f'QDialog#clipLabels {{ background: {T.SURFACE}; }} '
                          f'QPlainTextEdit {{ background: {T.INSET}; color: {T.CREAM}; '
                          f'border: 1px solid {T.LINE}; border-radius: 8px; padding: 8px; }}')
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        title = QLabel('What did you record?')
        title.setObjectName('h2')
        layout.addWidget(title)
        filename = QLabel(self.clip.name)
        filename.setTextFormat(Qt.PlainText)
        filename.setWordWrap(True)
        filename.setObjectName('muted')
        layout.addWidget(filename)
        form = QFormLayout()
        form.setSpacing(12)
        self.dogs = QComboBox()
        self.dogs.setEditable(True)
        self.dogs.addItems(['', *sorted(set(dog_names)), 'Unknown dog', 'Multiple dogs'])
        self.dogs.setCurrentText(data.get('dogs', ''))
        self.dogs.lineEdit().setPlaceholderText('Choose or type, e.g. Oreo, Choco')
        self.behavior = QComboBox()
        self.behavior.setEditable(True)
        self.behavior.addItems(['', *dict.fromkeys(actions), 'Mixed / unclear', 'No dog in view'])
        self.behavior.setCurrentText(data.get('behavior', ''))
        self.behavior.lineEdit().setPlaceholderText('Choose or describe, e.g. lying down and chewing')
        for combo in (self.dogs, self.behavior):
            combo.setMinimumHeight(38)
            if parent is not None and hasattr(parent, '_style_combo'):
                parent._style_combo(combo)
        form.addRow('Dog name(s)', self.dogs)
        form.addRow('Behavior', self.behavior)
        self.notes = QPlainTextEdit()
        self.notes.setPlainText(data.get('notes', ''))
        self.notes.setPlaceholderText('Optional: sitting 2–8 s, walking 8–12 s; head partly hidden.')
        self.notes.setMinimumHeight(100)
        form.addRow('Notes / times', self.notes)
        layout.addLayout(form)
        hint = QLabel('These labels describe the video. They do not train Bantay yet.\n'
                      'You can skip now and use Menu → Label a recorded clip later.')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        layout.addWidget(hint)
        self.error = QLabel('')
        self.error.setTextFormat(Qt.PlainText)
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox()
        self.save_button = buttons.addButton('Save labels', QDialogButtonBox.AcceptRole)
        self.save_button.setObjectName('primary')
        buttons.addButton('Skip for now', QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save(self):
        try:
            save_labels(self.clip, self.dogs.currentText(), self.behavior.currentText(),
                        self.notes.toPlainText())
        except (OSError, ValueError) as exc:
            self.error.setText(str(exc))
            return
        self.accept()
