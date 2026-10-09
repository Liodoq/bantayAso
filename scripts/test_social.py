"""Social responses and per-dog recap evidence checks, without a model."""
import sys,time,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bantayaso.social import appreciation,recap,protect
from bantayaso.persona import Persona
from bantayaso.history import ActivityHistory
from bantayaso.events import EventLog

class SocialTests(unittest.TestCase):
    def pipe(self):
        h=ActivityHistory(); now=time.monotonic()
        h.samples.extend([(now-60,'Oreo','sleeping',0,'Bed'),(now-1,'Oreo','sleeping',0,'Bed')])
        return NS(history=h,events=NS(today=lambda:[]),persona=Persona(None))

    def test_appreciation_varies(self):
        p=self.pipe()
        answers=[appreciation('Okay good, thanks!',p) for _ in range(4)]
        self.assertEqual(len(set(answers)),4)
        self.assertIsNotNone(appreciation('Good job',p))
        self.assertIsNotNone(appreciation('Thank you so much',p))

    def test_mixed_question_not_swallowed(self):
        self.assertIsNone(appreciation('Thanks, but did Oreo chew something today?',self.pipe()))
        self.assertIsNone(appreciation('Is Oreo good today?',self.pipe()))

    def test_rest_recap_qualifies_day(self):
        result=recap('Did Oreo do something bad today?',self.pipe(),['Oreo','Brownie'])
        self.assertIn('appearing to sleep',result)
        self.assertIn('not a full-day',result)
        self.assertIn('rest schedule',result)
        self.assertNotIn('good all day',result)

    def test_other_dogs_and_false_alarms_excluded(self):
        p=self.pipe();p.events.today=lambda:[{'dog':'Brownie','false_alarm':False,'ts':time.time(),'level':3,'reason':'battery'},
                                            {'dog':'Oreo','false_alarm':True,'ts':time.time(),'level':3,'reason':'battery'}]
        result=recap('Was Oreo good today?',p,['Oreo','Brownie'])
        self.assertNotIn('battery',result)
        self.assertIn('rest schedule',result)

    def test_danger_is_factual(self):
        p=self.pipe();p.events.today=lambda:[{'dog':'Oreo','false_alarm':False,'ts':time.time(),'level':3,'reason':'near a battery'}]
        result=recap('Was Oreo good today?',p,['Oreo'])
        self.assertIn('danger',result);self.assertIn('near a battery',result)
        self.assertNotIn('rest schedule',result)
        self.assertNotIn('swallowed',result)

    def test_no_history_is_unknown(self):
        p=self.pipe();p.history.samples.clear()
        self.assertIn("can't judge",recap('Did Oreo do anything bad today?',p,['Oreo']))

    def test_ordinary_chewing_not_moralized(self):
        p=self.pipe();p.history.samples.clear();p.history.samples.append((time.monotonic()-1,'Oreo','chewing something',0,''))
        result=recap('What did Oreo do earlier?',p,['Oreo'])
        self.assertIn('chewing something',result);self.assertNotIn('bad',result)
        self.assertNotIn('rest schedule',result)

    def test_coverage_survives_short_style(self):
        p=self.pipe();p.persona.style.length='short'
        text=protect(p,recap('Was Oreo good today?',p,['Oreo']))
        self.assertEqual(p.persona.finish('question',text),text)

    def test_sqlite_named_query(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data') as temp:
            e=EventLog(Path(temp)/'test.db',Path(temp)/'snaps')
            a=NS(level=2,reason='chewing something',action='chewing',near=[],zone='',track_id=1)
            e.add(a,dog_label='Brownie')
            bad=e.add(a,dog_label='Oreo');e.mark_false_alarm(bad)
            e.add(a,dog_label='Oreo')
            rows=e.for_dog_today('oreo')
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['dog'],'Oreo')

if __name__=='__main__': unittest.main()
