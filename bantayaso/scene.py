"""Fresh scene questions and debounced presence notices, independent of risk rules."""
import re
import time


def scene_answer(question, pipe, subject=None):
    q = question.lower()
    snap = getattr(pipe, 'scene', None)
    vocab = getattr(getattr(pipe, 'hazard_det', None), 'names', [])
    requested = next((n for n in sorted(vocab, key=len, reverse=True)
                      if re.search(r'\b' + re.escape(n.lower()) + r's?\b', q)), None)
    object_question = bool(requested or re.search(r'\b(things?|objects?|pillow|blanket)\b|beside|next to|near them|near him|near her', q))
    location_question = bool(re.search(r'\bwhere\b|nasaan|asan|\b(on|in|at) (the |a )?(bed|sofa|play|food|trash|area|zone)', q))
    if location_question and re.search(r"\b(on|in|at) (the |a )?(bed|sofa|play|food|trash|area|zone)",q) and not re.search(r"things?|objects?|beside|next to",q):
        object_question=False
    if not (object_question or location_question):
        return None
    if snap is None:  # compatibility for offline fixtures; live pipeline always publishes a snapshot
        return None
    if time.monotonic() - snap['at'] > 3:
        return "I don't have a fresh camera view, so I can't check right now."
    if subject is None and re.search(r'\b(him|her|he|she)\b',q):
        if time.monotonic()-getattr(pipe,'_last_subject_at',0)<=60:
            subject=getattr(pipe,'_last_subject',None)
        if subject is None and len(snap['dogs'])>1:
            return 'Which dog do you mean?'
    dogs = [d for d in snap['dogs'] if not subject or d['name'] == subject]
    if subject and not dogs:
        return f"I don't see {subject} right now."
    if object_question:
        if not snap['objects_enabled']:
            return 'Object detection is off, so I cannot check nearby things right now.'
        if time.monotonic() - snap['objects_at'] > 3:
            return 'I do not have a fresh object detection yet. Please ask again in a moment.'
        nearby = bool(re.search(r'beside|next to|near|around', q))
        objects = list(snap['objects'])
        if nearby:
            if not dogs:
                return "I don't see a dog to check nearby objects against."
            objects = [o for o in objects if any(box_gap(o.box, d['box']) <= 0.08 * snap['width'] for d in dogs)]
        labels = sorted(set(o.name for o in objects))
        if requested:
            if requested in labels:
                return f'I can see a {requested}' + (' near the dog in view.' if len(dogs)==1 and nearby else ' near the dogs in view.' if nearby else ' in the camera view.')
            return f'I have not detected a {requested}' + (' near the dogs' if nearby else ' in this view') + '. It may still be hidden or missed.'
        if re.search(r'\bpillow\b',q) and 'pillow' not in vocab:
            return 'Pillow is not in my object list yet. Add it on the Things page to let me look for it.'
        if labels:
            return 'I can see ' + ', '.join(labels) + (' near the dogs in view.' if nearby else ' in the camera view.')
        return 'I have not detected nearby objects. Small or hidden things may be missed.' if nearby else 'I have not detected any configured objects in this view.'
    if not dogs:
        return "I don't see any dog in the camera view."
    place = re.search(r'\b(bed|sofa|play|food|trash)\b',q)
    if place and not re.search(r'\bwhere\b|nasaan|asan',q):
        if not snap['zones_reliable']:
            return 'The camera position changed, so please check the area outlines before I confirm their location.'
        target = place[1]
        if not any(target in (z['name']+' '+z['type']).lower() for z in snap['zones']):
            return f'I need a {target} area drawn on the Zones page to confirm that location.'
        hits = [d for d in dogs if any(target in (z['name']+' '+z['type']).lower() for z in d['zones'])]
        if len(hits)==len(dogs):
            return f'Yes, ' + (f'{subject} is' if subject else f'all {len(dogs)} dogs in view are' if len(dogs)>1 else 'the dog in view is') + f' in the {target} area.'
        if hits:
            return f'Not all of them. {len(hits)} of the {len(dogs)} dogs in view are in the {target} area.'
        return f'No, none of the dogs I can see are in the {target} area.'
    parts=[]
    for d in dogs:
        name=d['name'] or ('The dog' if len(dogs)==1 else 'One dog')
        places=', '.join(z['name'] for z in d['zones']) if snap['zones_reliable'] else ''
        x=(d['box'][0]+d['box'][2])/2/snap['width']
        side='left' if x<0.33 else 'middle' if x<0.66 else 'right'
        parts.append(f'{name} is in {places}' if places else f'{name} is on the {side} of the camera view')
    return '. '.join(parts)+'.'


def box_gap(a,b):
    dx=max(a[0]-b[2],b[0]-a[2],0)
    dy=max(a[1]-b[3],b[1]-a[3],0)
    return (dx*dx+dy*dy)**0.5


class Presence:
    """Count-based entry notices avoid repeated announcements from tracker-ID swaps.

    New counts must persist 1.5s, decreases 5s. Reconnect establishes a fresh baseline.
    Counts cannot detect a same-count replacement; do not pretend otherwise.
    """
    def __init__(self):
        self.last=None
        self.stable={}
        self.pending={}
        self.initial=True

    def update(self, now, counts):
        if self.last is not None and now-self.last>3:
            self.stable=dict(counts);self.pending.clear();self.last=now
            return []
        self.last=now
        notices=[]
        for area in set(self.stable)|set(counts):
            n=counts.get(area,0); old=self.stable.get(area,0)
            if n==old:
                self.pending.pop(area,None);continue
            candidate,since=self.pending.get(area,(None,now))
            if candidate!=n:
                self.pending[area]=(n,now);continue
            if now-since < (1.5 if n>old else 5): continue
            self.stable[area]=n;self.pending.pop(area,None)
            if n>old:
                if area=='camera':
                    text=(f'I can see {n} dog' + ('s' if n!=1 else '') + ' in the camera view.') if self.initial else ('A dog entered the camera view.' if n-old==1 else f'{n-old} dogs entered the camera view.')
                    self.initial=False
                else:
                    text=('A dog entered ' if n-old==1 else f'{n-old} dogs entered ')+area+'.'
                notices.append((area,text))
        # One scene entry suffices if it also lands inside a named zone.
        camera=[x for x in notices if x[0]=='camera']
        return camera or notices
