"""Scene questions and entry detection without camera/model loading."""
import sys,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bantayaso.scene import scene_answer,Presence
from bantayaso.detect_hazards import Hazard,HazardDetector
from bantayaso.qa import answer

class SceneTests(unittest.TestCase):
    def pipe(self):
        zones=[{'name':'Bed 1','type':'bed'}]
        dogs=[{'id':1,'name':'Oreo','box':(10,10,80,80),'zones':zones},
              {'id':2,'name':'Brownie','box':(180,10,250,80),'zones':[]}]
        return NS(scene={'at':time.monotonic(),'dogs':dogs,'width':300,'zones':zones,'zones_reliable':True,
                         'objects_enabled':True,'objects_at':time.monotonic(),
                         'objects':(Hazard('pillow',0,(82,10,100,30),.8),)},
                  hazard_det=NS(names=['pillow','blanket','battery']))

    def test_location_named(self):
        self.assertEqual(scene_answer('Where is Oreo?',self.pipe(),'Oreo'),'Oreo is in Bed 1.')
        self.assertIn('right',scene_answer('Where is Brownie?',self.pipe(),'Brownie'))

    def test_all_on_bed_partial(self):
        self.assertIn('Not all',scene_answer('is all you are on the bed.',self.pipe()))

    def test_all_on_bed_complete(self):
        p=self.pipe();p.scene['dogs'][1]['zones']=p.scene['zones']
        self.assertIn('all 2 dogs in view',scene_answer('Are all dogs on the bed?',p))

    def test_bed_unknown(self):
        p=self.pipe();p.scene['zones_reliable']=False
        self.assertIn('check the area',scene_answer('Are all dogs on the bed?',p))
        p.scene['zones_reliable']=True;p.scene['zones']=[]
        self.assertIn('drawn',scene_answer('Are all dogs on the bed?',p))

    def test_pillow_scope_and_grounding(self):
        p=self.pipe()
        self.assertIn('see a pillow',scene_answer('But can you detect if there is a pillow?',p))
        self.assertIn('hidden or missed',scene_answer('Is there a blanket?',p))

    def test_nearby_is_subject_specific(self):
        p=self.pipe()
        self.assertIn('pillow',scene_answer('Are there things beside Oreo?',p,'Oreo'))
        self.assertIn('not detected nearby',scene_answer('Are there things beside Brownie?',p,'Brownie'))

    def test_stale_and_disabled(self):
        p=self.pipe();p.scene['objects_at']=0
        self.assertIn('fresh object',scene_answer('Is there a pillow?',p))
        p.scene['objects_enabled']=False
        self.assertIn('off',scene_answer('Is there a pillow?',p))
        p.scene['at']=0
        self.assertIn('fresh camera',scene_answer('Where is the dog?',p))

    def test_entry_dwell_and_no_repeats(self):
        e=Presence()
        self.assertEqual(e.update(0,{'camera':1}),[])
        self.assertEqual(e.update(1,{'camera':1}),[])
        self.assertIn('I can see 1 dog',e.update(2,{'camera':1})[0][1])
        for t in range(3,10): self.assertEqual(e.update(t,{'camera':1}),[])
        e.update(10,{'camera':2})
        self.assertIn('entered',e.update(12,{'camera':2})[0][1])

    def test_short_loss_and_reconnect(self):
        e=Presence();e.update(0,{'camera':1});e.update(2,{'camera':1})
        e.update(3,{'camera':0});self.assertEqual(e.update(4,{'camera':1}),[])
        self.assertEqual(e.update(30,{'camera':2}),[])
        self.assertEqual(e.update(32,{'camera':2}),[])

    def test_zone_entry_and_exit(self):
        e=Presence();e.stable={'camera':1,'Bed 1':0};e.initial=False
        e.update(0,{'camera':1,'Bed 1':1})
        self.assertEqual(e.update(2,{'camera':1,'Bed 1':1}),[('Bed 1','A dog entered Bed 1.')])
        for t in range(3,10): e.update(t,{'camera':1,'Bed 1':0})
        e.update(10,{'camera':1,'Bed 1':1})
        self.assertTrue(e.update(12,{'camera':1,'Bed 1':1}))

    def test_zero_tier_is_scene_only(self):
        def tensor(arr): return NS(cpu=lambda:NS(numpy=lambda:np.array(arr)))
        result=NS(names={0:'pillow',1:'battery'},boxes=NS(
            xyxy=tensor([[0,0,20,20],[50,50,60,60]]),conf=tensor([.8,.9]),cls=tensor([0,1])))
        d=HazardDetector.__new__(HazardDetector)
        d.model=NS(predict=lambda *a,**k:[result]);d.vocab={'pillow':0,'battery':3}
        d.names=list(d.vocab);d.conf=.25;d.imgsz=640;d.device='cpu';d.half=False;d.confirm=True;d._prev=[]
        self.assertEqual(d(np.zeros((4,4,3))),[])
        risks=d(np.zeros((4,4,3)))
        self.assertEqual([h.name for h in risks],['battery'])
        self.assertEqual([h.name for h in d.scene_objects],['pillow','battery'])

    def test_questions_route_before_rejection(self):
        p=self.pipe();p.registry=None;p.state=NS();p.dog_name='your dog'
        with patch('bantayaso.qa._teach_action',return_value=None):
            self.assertIn('see a pillow',answer('But can you detect if there is a pillow?',p))
            self.assertIn('Not all',answer('is all you are on the bed.',p))

if __name__=='__main__': unittest.main()
