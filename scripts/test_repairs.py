"""Regression checks for the Oct 9 review. No camera, downloads or real training writes."""
import sys
import unittest
import tempfile
import time
import queue
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso.actions import ActionResult, ActionClassifier
from bantayaso.alerts import phrase
from bantayaso.detect_dog import Dog
from bantayaso.detect_hazards import Hazard
from bantayaso.risk import RiskEngine
from bantayaso.zones import Zone
from bantayaso.names import DogRegistry
from bantayaso.examples import blend, reference_gates, save_array
from bantayaso.pipeline import Pipeline
from bantayaso.qa import _teach_action
from bantayaso import config


class TeachingRepairs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / 'data')
        self.addCleanup(self.temp.cleanup)
        self.events = []
        self.c = ActionClassifier.__new__(ActionClassifier)
        self.c.labels = ['sitting', 'lying down', 'licking itself']
        self.c.examples_dir = Path(self.temp.name)
        self.c.examples = {}
        self.c._example_gates = {}
        self.c._teach = {}
        self.c.on_teaching = self.events.append
        self.feat = np.array([1., 0., 0.])
        self.pipe = Pipeline.__new__(Pipeline)
        self.pipe.classifier = self.c
        self.pipe._teaching_requests = queue.Queue(maxsize=16)
        self.pipe._teaching_frame_at = time.monotonic()
        self.pipe._teaching_targets = ((1, 'Oreo'),)
        self.pipe._lesson_subjects = {}
        self.pipe.registry = SimpleNamespace(names_by_tid={1: 'Oreo'}, dogs={'Oreo': []})
        self.pipe.state = SimpleNamespace(boxes=[(1, (0, 0, 100, 100), 'Oreo')])
        self.pipe.on_teaching = self.events.append
        self.pipe.log = lambda _: None
        self.pipe.voice = False
        self.pipe.dnd = False
        self.c.on_teaching = self.pipe._teaching_event
        self.dog = Dog(1, (0, 0, 100, 100), .9)

    def parse(self, text):
        return _teach_action(text, text, self.pipe, list(self.pipe.registry.dogs),
                             dict(self.pipe.registry.names_by_tid))

    def test_voice_request_collects_only_after_processing_and_saves_six(self):
        self.assertIn('remember', self.parse('Oreo is sitting right now'))
        self.assertFalse(self.c._teach)  # QA thread only queues
        self.pipe._prepare_teaching([self.dog], time.monotonic())
        self.assertIn(1, self.c._teach)
        for _ in range(5): self.c._collect_example(1, self.feat)
        self.assertEqual(list(self.c.examples_dir.glob('*.npy')), [])
        self.c._collect_example(1, self.feat)
        rows = np.load(self.c.examples_dir / 'sitting.npy', allow_pickle=False)
        self.assertEqual(rows.shape, (6, 3))
        self.assertEqual(self.events[-1]['status'], 'saved')
        self.assertIn('another action', self.events[-1]['message'])

    def test_interruption_keeps_old_examples_and_commits_nothing_partial(self):
        self.c.examples['sitting'] = np.array([self.feat] * 3)
        path = self.c.examples_dir / 'sitting.npy'
        save_array(path, self.c.examples['sitting'])
        before = path.read_bytes()
        self.c.teach(1, 'sitting')
        for _ in range(3): self.c._collect_example(1, self.feat)
        self.c.check_teaching(set())
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.events[-1]['status'], 'cancelled')
        self.c._collect_example(1, self.feat)
        self.assertEqual(path.read_bytes(), before)

    def test_write_failure_never_claims_saved(self):
        self.c.teach(1, 'sitting')
        with patch('bantayaso.actions.save_array', side_effect=OSError('disk full')):
            for _ in range(6): self.c._collect_example(1, self.feat)
        self.assertEqual(self.c.examples, {})
        self.assertEqual(self.events[-1]['status'], 'cancelled')

    def test_held_detection_and_timeout_cancel(self):
        self.c.teach(1, 'sitting')
        self.pipe._prepare_teaching([Dog(1, self.dog.box, .9, observed=False)], time.monotonic())
        self.assertFalse(self.c._teach)
        self.c.teach(1, 'sitting')
        self.c.check_teaching({1}, self.c._teach[1].started + 11)
        self.assertFalse(self.c._teach)

    def test_queued_request_is_revalidated(self):
        self.parse('Oreo is sitting now')
        self.pipe.registry.names_by_tid.clear()
        self.pipe._prepare_teaching([self.dog], time.monotonic())
        self.assertFalse(self.c._teach)
        self.assertEqual(self.events[-1]['status'], 'cancelled')

    def test_reset_and_camera_change_cancel_lesson(self):
        self.c.teach(1, 'sitting')
        self.pipe.request_reset_examples()
        self.pipe._prepare_teaching([self.dog], time.monotonic())
        self.assertFalse(self.c._teach)
        self.parse('Oreo is sitting now')
        self.pipe.cancel_teaching('Camera changed.')
        self.assertTrue(self.pipe._teaching_requests.empty())
        self.assertIn('current camera', self.parse('Oreo is sitting now'))

    def test_questions_plain_statements_and_unsupported_labels_do_not_teach(self):
        for text in ('Is Oreo sitting now?', 'Oreo is sitting now?', 'Oreo is sitting',
                     'What if Oreo is sitting now', 'Can you remember Oreo is sitting'):
            self.assertIsNone(self.parse(text))
        self.assertIn('cannot teach', self.parse('remember Oreo is dancing'))
        self.assertIn("don't know", self.parse('Max is sitting now'))
        self.assertTrue(self.pipe._teaching_requests.empty())

    def test_pronouns_and_visibility(self):
        self.parse('Oreo is sitting now')
        self.assertIn('remember', self.parse('remember she is lying down'))
        self.pipe._last_subject_at -= 61
        self.assertIn('name the dog', self.parse('remember she is lying down'))
        self.pipe.state.boxes.append((2, self.dog.box, None))
        self.assertIn('name or click', self.parse("that's licking"))
        self.pipe.state.boxes = []
        self.assertIn('dog in view', self.parse("that's licking"))
        self.assertIn("can't clearly see", self.parse('Oreo is sitting now'))

    def test_capture_reload_and_recognition(self):
        for label, vec in [('sitting', self.feat), ('lying down', np.array([0., 1., 0.]))]:
            self.c.teach(1, label)
            for _ in range(6): self.c._collect_example(1, vec)
        loaded = {p.stem.replace('_', ' '): np.load(p, allow_pickle=False)
                  for p in self.c.examples_dir.glob('*.npy')}
        pred = blend(self.c.labels, loaded, reference_gates(loaded), np.array([.1, .6, .3]), self.feat)
        self.assertEqual(self.c.labels[int(pred.argmax())], 'sitting')

    def test_real_pipeline_orders_identity_validation_before_save(self):
        outer = self
        class FakeClassifier(ActionClassifier):
            def __init__(self, *args, **kwargs):
                self.__dict__.update(outer.c.__dict__)
                self.embeddings = {}
            def __call__(self, frame, dogs, collect=True):
                self.embeddings = {d.track_id: outer.feat.copy() for d in dogs}
                return {d.track_id: ActionResult('sitting', .9, .1, .1) for d in dogs}
        cfg = {'models': {'device': 'cpu', 'dog_detector': 'unused'}, 'actions': self.c.labels,
               'qa': {'enabled': False}, 'detect': {'action_every_seconds': 0},
               'alerts': {'voice': False, 'toast': False}}
        with patch.object(config, 'DATA_DIR', Path(self.temp.name)), \
             patch.object(config, 'MODELS_DIR', Path(self.temp.name)), \
             patch('bantayaso.detect_dog.DogDetector', return_value=lambda frame: [self.dog]), \
             patch('bantayaso.actions.ActionClassifier', FakeClassifier), \
             patch('bantayaso.pipeline.Speaker', return_value=SimpleNamespace()), \
             patch('bantayaso.pipeline.EventLog', return_value=SimpleNamespace()), \
             patch('bantayaso.pipeline.overlay.draw_dogs'), patch('bantayaso.pipeline.overlay.draw_hazards'):
            pipe = Pipeline(cfg, use_hazards=False, use_vlm=False, log=lambda _: None)
            pipe.registry.add_samples('Oreo', [self.feat]*3)
            frame = np.zeros((120, 120, 3), np.uint8)
            base, tick = time.monotonic(), 0
            def process():
                nonlocal tick
                tick += 1
                with patch('bantayaso.pipeline.time.monotonic', return_value=base + tick * .1):
                    pipe.process(frame)
            for _ in range(3): process()
            pipe.on_teaching = self.events.append
            self.assertIn('remember', pipe.request_teach(1, 'sitting', subject='Oreo'))
            for _ in range(5): process()
            self.assertFalse((Path(self.temp.name)/'sitting.npy').exists())
            pipe.registry.names_by_tid.clear()
            process()
            self.assertEqual(self.events[-1]['status'], 'cancelled')
            self.assertFalse((Path(self.temp.name)/'sitting.npy').exists())
            pipe.request_teach(1, 'sitting', subject='Oreo')
            for _ in range(6): process()
            self.assertEqual(self.events[-1]['status'], 'saved')


class RecognitionRepairs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / 'data')
        self.addCleanup(self.temp.cleanup)
        self.registry = DogRegistry(Path(self.temp.name))
        self.oreo = np.array([1., 0., 0.])
        self.other = np.array([0., 1., 0.])
        self.registry.add_samples('Oreo', [self.oreo] * 3)

    def observe(self, emb, n, tid=1):
        for _ in range(n): self.registry.update({tid: emb}, {tid})

    def test_identity_expires_and_recovers(self):
        self.observe(self.oreo, 7)
        self.observe(self.other, 2)
        self.assertEqual(self.registry.names_by_tid, {1: 'Oreo'})
        self.observe(self.other, 5)
        self.assertEqual(self.registry.names_by_tid, {})
        self.observe(self.oreo, 3)
        self.assertEqual(self.registry.names_by_tid, {1: 'Oreo'})

    def test_handover_does_not_renew_identity_evidence(self):
        self.observe(self.oreo, 7)
        self.observe(self.other, 4)
        self.registry.handover(1, 2)
        self.observe(self.other, 1, 2)
        self.assertNotIn(2, self.registry.names_by_tid)

    def test_deleted_name_cannot_reappear_from_votes(self):
        self.observe(self.oreo, 7)
        self.registry.start_enroll(2, 'Oreo')
        self.registry.delete('Oreo')
        self.registry.update({1: self.oreo, 2: self.oreo}, {1, 2})
        self.assertEqual(self.registry.names_by_tid, {})
        self.assertNotIn('Oreo', self.registry.dogs)

    def test_two_dogs_and_one_name_per_dog(self):
        self.registry.add_samples('Choco', [self.other] * 3)
        for _ in range(7):
            self.registry.update({1: self.oreo, 2: self.other}, {1, 2})
        self.assertEqual(self.registry.names_by_tid, {1: 'Oreo', 2: 'Choco'})
        for _ in range(7):
            self.registry.update({1: self.other, 2: self.other}, {1, 2})
        self.assertEqual(list(self.registry.names_by_tid.values()), ['Choco'])

    def test_weak_examples_never_force_the_review_counterexample(self):
        examples = {'sitting': np.array([[np.sqrt(.99), 0., .1]] * 6),
                    'lying down': np.array([[0., 1., 0.]] * 6)}
        text = np.array([.05, .05, .90])
        result = blend(['sitting', 'lying down', 'chewing'], examples, reference_gates(examples),
                       text, np.array([0., 0., 1.]))
        np.testing.assert_array_equal(result, text)

    def test_supported_examples_work_and_ambiguous_examples_abstain(self):
        labels = ['sitting', 'lying down', 'chewing']
        examples = {'sitting': np.array([self.oreo] * 6), 'lying down': np.array([self.other] * 6)}
        text = np.array([.1, .6, .3])
        result = blend(labels, examples, reference_gates(examples), text, self.oreo)
        self.assertEqual(int(result.argmax()), 0)
        examples['lying down'] = np.array([self.oreo] * 6)
        np.testing.assert_array_equal(blend(labels, examples, reference_gates(examples), text, self.oreo), text)
        self.assertEqual(reference_gates({'sitting': examples['sitting']}), {})
        self.assertEqual(reference_gates({}), {})

    def test_examples_reload_without_changed_predictions(self):
        examples = {'sitting': np.array([self.oreo] * 6), 'lying down': np.array([self.other] * 6)}
        for label, arr in examples.items(): save_array(Path(self.temp.name)/f'{label}.npy', arr)
        loaded = {k: np.load(Path(self.temp.name)/f'{k}.npy', allow_pickle=False) for k in examples}
        self.assertEqual(reference_gates(examples), reference_gates(loaded))


class AlertRepairs(unittest.TestCase):
    def setUp(self):
        self.dog = Dog(1, (100, 100, 200, 200), .9)
        self.battery = Hazard('battery', 3, (200, 130, 210, 150), .9)
        self.cfg = {'hazards': {'battery': 3, 'toy': 1, 'slipper': 2}}
        self.sleep = {1: ActionResult('sleeping', .9, .1, .1)}

    def step(self, e, i, hazards, dt=.1, zones=(), acts=None):
        return e.update([self.dog], hazards, zones, (400, 400), now=10+i*dt,
                        hazards_fresh=i % 5 == 0, actions=acts or self.sleep)[0]

    def test_missing_hazard_survives_cached_frames_and_expires(self):
        for dt in (.04, .1, .2):
            with self.subTest(dt=dt):
                e = RiskEngine(self.cfg)
                for i in range(20): self.step(e, i, [self.battery], dt)
                states = [self.step(e, i, [], dt) for i in range(20, 20+int(3/dt))]
                self.assertTrue(any(a.level == 3 and a.in_mouth == 'battery' for a in states))
                for i in range(20+int(3/dt), 20+int(9/dt)): a = self.step(e, i, [], dt)
                self.assertEqual(a.level, 0)
                self.assertIsNone(a.in_mouth)

    def test_reappearance_and_unconfirmed_sighting(self):
        e = RiskEngine(self.cfg)
        for i in range(20): self.step(e, i, [self.battery])
        for i in range(20, 30): a = self.step(e, i, [])
        self.assertEqual(a.in_mouth, 'battery')
        a = self.step(e, 30, [self.battery])
        self.assertIsNone(a.in_mouth)
        e = RiskEngine(self.cfg)
        for i in range(5): self.step(e, i, [self.battery])
        for i in range(5, 30):
            self.assertIsNone(self.step(e, i, []).in_mouth)

    def test_memory_stays_with_its_dog(self):
        e = RiskEngine(self.cfg)
        other = Dog(2, (0, 0, 40, 40), .9)
        for i in range(35):
            aa = e.update([self.dog, other], [self.battery] if i < 20 else [], [], (400, 400),
                          now=10+i*.1, hazards_fresh=i % 5 == 0,
                          actions={**self.sleep, 2: ActionResult('sleeping', .9, .1, .1)})
        self.assertEqual(aa[0].in_mouth, 'battery')
        self.assertIsNone(aa[1].in_mouth)

    def test_toys_and_real_hazards_in_calm_zones(self):
        acts = {1: ActionResult('chewing something', .9, .6, .6)}
        for kind in ('food', 'play'):
            zone = Zone(kind, kind, [(0, 0), (1, 0), (1, 1), (0, 1)])
            for name, tier, expected in [('toy', 1, 0), ('slipper', 2, 2), ('battery', 3, 3)]:
                with self.subTest(zone=kind, hazard=name):
                    e = RiskEngine(self.cfg)
                    hz = Hazard(name, tier, self.battery.box, .9)
                    for i in range(90): a = self.step(e, i, [hz], zones=[zone], acts=acts)
                    self.assertEqual(a.level, expected)
            e = RiskEngine(self.cfg)
            for i in range(90): a = self.step(e, i, [], zones=[zone], acts=acts)
            self.assertEqual(a.level, 0)
            for i in range(90, 120): a = self.step(e, i, [], acts=acts)
            self.assertEqual(a.level, 2)

    def test_missing_battery_is_not_suppressed_in_play_zone(self):
        e = RiskEngine(self.cfg)
        zone = Zone('play', 'play', [(0, 0), (1, 0), (1, 1), (0, 1)])
        for i in range(40):
            a = self.step(e, i, [self.battery] if i < 20 else [], zones=[zone])
        self.assertEqual(a.level, 3)

    def test_spoken_severity(self):
        for level, prefix in [(2, 'Warning!'), (3, 'Danger!'), (0, 'Oreo'), (1, 'Oreo')]:
            text = phrase(level, 'has been chewing for 16 s', 'Oreo')
            self.assertTrue(text.startswith(prefix), text)
            self.assertIn('16 seconds', text)


if __name__ == '__main__':
    unittest.main()
