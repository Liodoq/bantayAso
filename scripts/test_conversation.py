"""Offline regression checks for spoken clocks, check-ins and conversation controls."""
import sys
import time
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso.speech_text import spoken_text
from bantayaso.companion import Companion
from bantayaso.alerts import Speaker
from bantayaso.voice_in import Listener
from bantayaso.persona import Persona
from bantayaso.pipeline import Pipeline
from bantayaso.history import ActivityHistory
from bantayaso import qa

class ConversationTests(unittest.TestCase):
    def speaker(self):
        with patch('bantayaso.alerts.threading.Thread'):
            return Speaker()

    def test_sapi_completion_and_cancellation(self):
        for cancel in (False, True):
            s=self.speaker(); completed=[]; spoken=[]
            s.say('at 8:15', on_done=lambda: completed.append(True))
            item=s.q.get_nowait()
            def wait(ms):
                if cancel:
                    s.stop()
                    return False
                return True
            sapi=SimpleNamespace(GetVoices=lambda: SimpleNamespace(Count=0),
                                 Speak=lambda *args: spoken.append(args), WaitUntilDone=wait)
            client=SimpleNamespace(Dispatch=lambda name: sapi)
            modules={'pythoncom': SimpleNamespace(CoInitialize=lambda: None),
                     'win32com': SimpleNamespace(client=client), 'win32com.client': client}
            with patch.dict(sys.modules,modules), patch.object(s.q,'get',side_effect=[item,SystemExit]), \
                 patch('bantayaso.alerts.time.sleep'):
                with self.assertRaises(SystemExit): s._run()
            self.assertEqual(spoken[0],('at eight fifteen',1))
            self.assertEqual(bool(completed),not cancel)
            if cancel: self.assertEqual(spoken[-1],('',3))

    def test_spoken_clocks(self):
        self.assertEqual(spoken_text('Oreo was sleeping from 8:15–9:00.'),
                         "Oreo was sleeping from eight fifteen to nine o'clock.")
        self.assertEqual(spoken_text('since 8:15 - 9:0'), "from eight fifteen to nine o'clock")
        self.assertEqual(spoken_text('At 00:05 and 13:30'), 'At twelve oh five A M and one thirty P M')
        self.assertEqual(spoken_text('8:15 AM to 9:00 PM'), "eight fifteen A M to nine o'clock P M")
        self.assertEqual(spoken_text('2 dogs for 15 minutes'), '2 dogs for 15 minutes')

    def test_timeline_is_sentence(self):
        h=ActivityHistory()
        with patch.object(h, 'segments', return_value=[(1000,1060,'Oreo','sleeping','Bed 1')]):
            text=h.timeline(5)
        self.assertTrue(text.startswith('Oreo was sleeping in Bed 1 from '))
        self.assertNotIn('–', text)
        self.assertNotIn(':', spoken_text(text))

    def test_priority_and_cancel(self):
        s=self.speaker()
        s.say('Reply')
        s.say('Danger', urgent=True)
        s.say('Another reply')
        s.stop()
        first=s.q.get_nowait()
        self.assertEqual(first[4], 'Danger')
        self.assertTrue(s._valid(first))
        self.assertFalse(s._valid(s.q.get_nowait()))

    def test_stale_and_muted_speech(self):
        s=self.speaker()
        s.say('Reply', valid=lambda: False)
        self.assertFalse(s._valid(s.q.get_nowait()))
        with patch('bantayaso.alerts.time.monotonic',return_value=0): s.say('old')
        self.assertFalse(s._valid(s.q.get_nowait()))
        s.enabled=False
        self.assertFalse(s.say('muted'))

    def test_casual_does_not_queue_behind_reply(self):
        s=self.speaker(); s.say('reply')
        self.assertFalse(s.say('resting', casual=True))

    def test_full_queue_still_accepts_danger(self):
        s=self.speaker()
        for i in range(8): s.say(f'reply {i}')
        self.assertTrue(s.say('Danger',urgent=True))
        self.assertEqual(s.q.get_nowait()[4],'Danger')

    def test_checkin_dwell_and_repeat(self):
        c=Companion(dwell=3,cooldown=10); obs=[(1,'Oreo','sleeping',True)]
        for t in range(3): self.assertIsNone(c.update(t,obs))
        self.assertEqual(c.update(3,obs), 'Oreo appears to be resting.')
        c.mark_spoken(3)
        for t in range(4,25): self.assertIsNone(c.update(t,obs))

    def test_checkin_camera_gap_and_held_reset_dwell(self):
        c=Companion(dwell=3); obs=[(1,'Oreo','lying down',True)]
        c.update(0,obs); c.update(1,obs)
        self.assertIsNone(c.update(10,obs))
        c.update(11,[(1,'Oreo','lying down',False)])
        self.assertIsNone(c.update(12,obs))

    def test_checkin_blocked_and_unknown_names(self):
        c=Companion(dwell=1)
        obs=[(1,None,'sleeping',True),(2,'Oreo','lying down',True)]
        c.update(0,obs)
        self.assertIsNone(c.update(1,obs,blocked=True))
        self.assertEqual(c.update(2,obs), 'The dogs in view appear to be resting.')
        self.assertIsNone(c.update(3,[(1,None,'chewing something',True)]))

    def test_followup_window(self):
        l=Listener(Path('.'))
        l.open_followup(); self.assertEqual(l._expect_until,0)
        l.hands_free=True
        with patch('bantayaso.voice_in.time.monotonic',return_value=100): l.open_followup()
        self.assertEqual(l._expect_until,108)
        l.set_hands_free(False); self.assertEqual(l._expect_until,0)

    def pipe(self):
        p=Pipeline.__new__(Pipeline)
        p._ask_lock=threading.Lock();p._asking=False;p._reply_generation=0
        p.persona=Persona(None);p.listener=None;p.registry=None
        p.speaker=self.speaker();p.voice=True;p.dnd=False;p.log=lambda x: None
        return p

    def test_reply_is_not_urgent_and_dnd_suppresses(self):
        p=self.pipe()
        with patch('bantayaso.qa.answer',return_value='Oreo is resting.'):
            self.assertEqual(p.ask('Where is Oreo?'),'Oreo is resting.')
            self.assertEqual(p.speaker.q.get_nowait()[0],1)
            p.dnd=True;p.ask('Where is Oreo?')
            self.assertTrue(p.speaker.q.empty())

    def test_repeated_fallback_not_spoken_twice(self):
        p=self.pipe()
        with patch('bantayaso.qa.answer',return_value=qa.OUT_OF_SCOPE):
            p.ask('unsupported question')
            p.ask('unsupported question again')
        self.assertEqual(p.speaker.q.qsize(),1)

    def test_cancel_inflight_answer(self):
        p=self.pipe()
        def cancelled(*args):
            p.stop_reply();return 'old answer'
        with patch('bantayaso.qa.answer',side_effect=cancelled):
            self.assertEqual(p.ask('Where is Oreo?'),'')
        self.assertTrue(p.speaker.q.empty())
        self.assertFalse(p._asking)

    def test_no_double_translation(self):
        p=Persona(None,llm_call=lambda *args: '{"status":"answer","answer":"Nakahiga si Oreo."}')
        p.style.lang='taglish'
        _,answer=p.ask_llm('Ano ginagawa ni Oreo?', 'Oreo lying down')
        p.llm_call=lambda *args: self.fail('Second model request')
        self.assertEqual(p.finish('question',answer),answer)

    def test_checkin_revalidation_while_speaking(self):
        p=self.pipe();p.companion_enabled=True;p.state=SimpleNamespace(level=0)
        p.speaker.speaking=True
        self.assertFalse(p._can_check_in())
        self.assertTrue(p._can_check_in(ignore_speaker=True))
        p.dnd=True;self.assertFalse(p._can_check_in(ignore_speaker=True))

    def test_ambiguous_followup(self):
        p=SimpleNamespace(registry=None,dog_name='your dog',state=SimpleNamespace(boxes=[1,2]))
        with patch('bantayaso.qa._teach_action',return_value=None):
            self.assertEqual(qa.answer('How long?',p),'Which dog do you mean?')

if __name__=='__main__': unittest.main()
