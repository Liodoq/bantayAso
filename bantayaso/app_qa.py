"""Read-only application questions; responses come from app state, never invented tools."""
import re
import time
from .scene import scene_answer


def event_page(pipe,rows,start=0):
    page=rows[start:start+3]
    pipe._event_readout=(rows,start+len(page))
    if not page: return 'There are no more recorded events in this list.'
    lines=[]
    for e in page:
        clock=time.strftime('%b %d at %I:%M %p',time.localtime(e['ts'])).replace(' 0',' ')
        who=e.get('dog') or ''
        who=who if not who.lstrip('-').isdigit() else 'an unidentified dog'
        level='Danger' if e['level']>=3 else 'Warning'
        lines.append(f"{clock}. {level} for {who or 'an unidentified dog'}: {e.get('reason') or e.get('action') or 'activity needing attention'}.")
    if start+len(page)<len(rows): lines.append('Say read more events to continue.')
    return ' '.join(lines)


def app_answer(question,pipe):
    q=question.lower()
    if re.search(r"(last|latest) (alert|warning|danger)\b",q):
        return None
    if re.search(r'(read|tell|dictate|list|show|what|any|have).*(events?|alerts?|warnings?)|recent events|event log',q):
        if re.search(r'more|next|continue',q):
            saved=getattr(pipe,'_event_readout',None)
            return event_page(pipe,*saved) if saved else 'Ask me to read the recent events first.'
        dog=next((name for name in getattr(getattr(pipe,'registry',None),'dogs',{}) if re.search(r'\b'+re.escape(name.lower())+r'\b',q)),None)
        today=bool(re.search(r'today|ngayong araw',q))
        if hasattr(pipe.events,'recent'):
            rows=pipe.events.recent(dog=dog,today=today)
        else:
            rows=[e for e in pipe.events.today() if not e.get('false_alarm') and (not dog or e.get('dog')==dog)]
        if not rows:return 'No matching events are recorded' + (' today.' if today else '.')
        return ('Here are the most recent recorded events. ' if not today else "Here are today's most recent recorded events. ")+event_page(pipe,rows)
    if re.search(r'what.*(screen|camera view|can you see|do you see)|describe.*(screen|scene|view)|what is in your screen',q):
        snap=getattr(pipe,'scene',None)
        if not snap or time.monotonic()-snap['at']>3:return 'I do not have a fresh camera view right now.'
        locations=scene_answer('Where are the dogs?',pipe)
        things=scene_answer('What objects are visible?',pipe)
        return f'In the camera view: {locations} {things}'
    if re.search(r'(list|all|configured|watching for|looking for|things page|object list).*(things|objects)|things page|object list|what.*(watching for|looking for)',q):
        vocab=getattr(getattr(pipe,'hazard_det',None),'vocab',None)
        if vocab is None:vocab=getattr(pipe,'cfg',{}).get('hazards',{})
        entries=[f"{name} ({ {0:'harmless',1:'low risk',2:'medium risk',3:'high risk'}.get(tier,'configured')})" for name,tier in vocab.items()]
        if not entries:return 'The Things list is empty.'
        pipe._thing_readout=(entries,8)
        return 'Configured things, not necessarily visible: '+', '.join(entries[:8])+('. Say more things to continue.' if len(entries)>8 else '.')
    if re.fullmatch(r'(read )?(more|next) things[.!? ]*',q):
        saved=getattr(pipe,'_thing_readout',None)
        if not saved:return 'Ask me to list the Things page first.'
        rows,start=saved;pipe._thing_readout=(rows,start+8)
        return ', '.join(rows[start:start+8])+('. Say more things to continue.' if start+8<len(rows) else '.') if start<len(rows) else 'That is the full Things list.'
    if re.search(r'\b(settings|configuration|microphone|listening mode|voice settings)\b|is.*(muted|do not disturb)',q):
        lst=getattr(pipe,'listener',None)
        return (f"Voice is {'on' if getattr(pipe,'voice',False) else 'off'}. Do not disturb is {'on' if getattr(pipe,'dnd',False) else 'off'}. "
                f"Hands-free is {'on' if lst and lst.hands_free else 'off'}. "
                f"Calm check-ins are {'on' if getattr(pipe,'companion_enabled',False) else 'off'}. "
                f"Entry announcements are {'on' if getattr(pipe,'entry_alerts',False) else 'off'}.")
    if re.search(r'(list|what|which|show).*(zones|areas)|zones page',q):
        zones=getattr(pipe,'zones',[])
        return 'Configured areas: '+', '.join(z.name for z in zones)+'.' if zones else 'No areas have been drawn yet.'
    if re.search(r'(list|what|which|show).*(dogs|names)|dogs page',q) and not re.search(r'doing|did|where',q):
        names=list(getattr(getattr(pipe,'registry',None),'dogs',{}))
        return 'Registered dogs: '+', '.join(names)+'.' if names else 'No dogs have been named yet.'
    if re.search(r'how many.*(clips|recordings|snapshots)|saved (clips|recordings|snapshots)',q):
        from .config import DATA_DIR
        clips=sum(1 for p in (DATA_DIR/'clips').glob('*') if p.suffix.lower() in ('.mp4','.avi','.mov','.mkv'))
        snapshots=sum(1 for p in (DATA_DIR/'snapshots').glob('*.jpg'))
        return f'The app has {clips} saved video clips and {snapshots} snapshots. Recording labels do not automatically train the model.'
    if re.search(r'whole app|application|app status|what can you (access|read)',q):
        return 'I can read the camera observations, dog locations, recent activities, saved events, registered dogs, drawn areas, Things list and voice settings. What would you like?'
    return None
