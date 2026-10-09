"""Recording-label tests without a camera or loaded models."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PySide6.QtWidgets import QApplication, QDialog
from bantayaso.clip_labels import labels_path, load_labels, save_labels
from bantayaso.ui.clip_labels import ClipLabelDialog
from bantayaso.ui.worker import Worker
from bantayaso.capture import ClipRecorder


class ClipLabelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / 'data')
        self.addCleanup(self.temp.cleanup)
        self.clip = Path(self.temp.name)/'rec_test.mp4'
        self.clip.write_bytes(b'video fixture unchanged')

    def test_save_and_edit_labels_preserves_video(self):
        before = self.clip.read_bytes()
        path = save_labels(self.clip, 'Oreo, Choco', 'lying down and chewing', '2–8 s: Oreo; Choco hidden')
        data = load_labels(self.clip)
        self.assertEqual(data['clip'], self.clip.name)
        self.assertEqual(data['source'], 'owner')
        self.assertEqual(data['annotation_scope'], 'clip_summary')
        self.assertIn('2–8', data['notes'])
        save_labels(self.clip, 'Oreo', 'sitting', 'corrected')
        self.assertEqual(load_labels(self.clip)['notes'], 'corrected')
        self.assertEqual(self.clip.read_bytes(), before)
        self.assertEqual(path, self.clip.with_name('rec_test.mp4.labels.json'))

    def test_blank_fields_and_missing_video_do_not_save(self):
        for dog, behavior in [('', 'sitting'), ('Oreo', '')]:
            with self.assertRaises(ValueError): save_labels(self.clip, dog, behavior, '')
        self.assertFalse(labels_path(self.clip).exists())
        with self.assertRaises(ValueError): save_labels(self.clip.with_name('gone.mp4'), 'Oreo', 'sitting', '')
        save_labels(self.clip, '', 'No dog in view', '')
        self.assertEqual(load_labels(self.clip)['dogs'], '')

    def test_unrecognized_file_is_not_overwritten(self):
        path = labels_path(self.clip)
        path.write_text('{"keep":true}', encoding='utf-8')
        with self.assertRaises(ValueError): save_labels(self.clip, 'Oreo', 'sitting', '')
        self.assertEqual(json.loads(path.read_text()), {'keep': True})

    def test_write_failure_retains_previous_labels(self):
        path = save_labels(self.clip, 'Oreo', 'sitting', 'before')
        before = path.read_bytes()
        with patch('bantayaso.clip_labels.os.replace', side_effect=OSError('disk error')):
            with self.assertRaises(OSError): save_labels(self.clip, 'Oreo', 'walking', 'after')
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(Path(self.temp.name).glob('*.tmp')))

    def test_dialog_saves_and_reopens(self):
        dialog = ClipLabelDialog(self.clip, ['Oreo'], ['sitting', 'walking'])
        dialog.save()
        self.assertTrue(dialog.error.text())
        self.assertEqual(dialog.result(), QDialog.Rejected)
        dialog.dogs.setCurrentText('Oreo')
        dialog.behavior.setCurrentText('sitting')
        dialog.notes.setPlainText('sitting 2–8 seconds')
        dialog.save()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        edit = ClipLabelDialog(self.clip)
        self.assertEqual(edit.dogs.currentText(), 'Oreo')
        self.assertEqual(edit.notes.toPlainText(), 'sitting 2–8 seconds')
        dialog.close(); edit.close()

    def test_skip_never_deletes_video_or_writes_labels(self):
        dialog = ClipLabelDialog(self.clip)
        dialog.reject()
        self.assertTrue(self.clip.exists())
        self.assertFalse(labels_path(self.clip).exists())
        dialog.close()

    def test_stop_notifies_only_after_writer_is_closed_even_without_frames(self):
        worker = Worker(SimpleNamespace())
        rec = SimpleNamespace(is_recording=True)
        def toggle(size):
            self.assertEqual(size, (0, 0))
            rec.is_recording = False
            return self.clip
        rec.toggle = toggle
        saved = []
        worker.clip_saved.connect(lambda path: saved.append((path, rec.is_recording)))
        worker.request('record')
        worker._handle_requests(rec, None)
        self.assertEqual(saved, [(str(self.clip), False)])
        self.assertFalse(worker.recording)

    def test_start_waits_for_a_frame(self):
        worker = Worker(SimpleNamespace())
        worker.request('record')
        worker._handle_requests(SimpleNamespace(is_recording=False), None)
        self.assertIn('record', worker._req)

    def test_buffer_save_failure_reports_error_without_label_prompt(self):
        worker = Worker(SimpleNamespace())
        messages, saved = [], []
        worker.message.connect(messages.append)
        worker.clip_saved.connect(saved.append)
        def fail():
            raise OSError('disk error')
        worker.request('last')
        worker._handle_requests(SimpleNamespace(save_last=fail), None)
        self.assertEqual(saved, [])
        self.assertIn('disk error', messages[0])

    def test_new_recording_does_not_reuse_existing_path(self):
        folder = Path(self.temp.name)
        old = folder/'rec_fixed.mp4'
        old.write_bytes(b'old clip')
        writer = SimpleNamespace(isOpened=lambda: True)
        with patch('bantayaso.capture.time.strftime', return_value='fixed'), \
             patch('bantayaso.capture.cv2.VideoWriter', return_value=writer):
            path, _ = ClipRecorder(folder)._new_writer((32, 32), 'rec')
        self.assertEqual(path.name, 'rec_fixed_1.mp4')
        self.assertEqual(old.read_bytes(), b'old clip')


if __name__ == '__main__':
    unittest.main()
