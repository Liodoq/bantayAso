import sys,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bantayaso.app_qa import app_answer
from bantayaso.voice_in import prepare_audio,Listener,RATE

class AppQuestionsTests(unittest.TestCase):
    def pipe(self):
        events=[dict(ts=time.time()-i*60,level=2,dog='Oreo',reason='chewing something') for i in range(5)]
        return NS(events=NS(recent=lambda **kw:events),registry=NS(dogs={'Oreo':None}),
                  hazard_det=NS(names=['pillow'],vocab={'pillow':0}),zones=[],voice=True,dnd=False,
                  listener=None,scene={'at':time.monotonic(),'width':300,'dogs':[],
                  'objects_enabled':True,'objects_at':time.monotonic(),'objects':[],
                  'zones':[],'zones_reliable':True})
    def test_events_paginate(self):
        p=self.pipe();a=app_answer('Read the events',p)
        self.assertEqual(a.count('Oreo was'),3)
        self.assertIn('read more events',a)
        self.assertEqual(app_answer('Read more events',p).count('Oreo was'),2)
        self.assertIn('no more',app_answer('Read more events',p))
    def test_screen_question(self):
        a=app_answer('What is in your screen?',self.pipe())
        self.assertIn('camera view',a);self.assertIn("don't see any dog",a)
    def test_configured_not_visible(self):
        a=app_answer('List all things',self.pipe())
        self.assertIn('not necessarily visible',a);self.assertIn('pillow (harmless)',a)
    def test_stale_screen(self):
        p=self.pipe();p.scene['at']=0
        self.assertIn('fresh',app_answer('What is on your screen?',p))
    def test_settings_without_camera(self):
        p=self.pipe();p.scene=None
        self.assertIn('Voice is on',app_answer('Read my settings',p))
    def test_application_access(self):
        self.assertIn('saved events',app_answer('What can you access?',self.pipe()))
    def test_audio_silence_and_gain(self):
        self.assertTrue(np.allclose(prepare_audio(np.ones(100)*.2),0,atol=1e-6))
        signal=np.tile([-.01,.01],1000).astype(np.float32)
        result=prepare_audio(signal)
        self.assertAlmostEqual(float(np.max(result)),.03,places=5)
        self.assertTrue(np.isfinite(prepare_audio(np.array([np.nan,np.inf,0]))).all())
    def test_silence_never_invokes_whisper(self):
        l=Listener(Path('.'))
        with patch.object(l,'_load',side_effect=AssertionError('Should not load')):
            self.assertEqual(l.transcribe(np.zeros(RATE)), '')

if __name__=='__main__':unittest.main()
