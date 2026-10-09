"""Read-only application questions; responses come from app state, never invented tools."""
import re
import time
from .scene import scene_answer


WORDNUM={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}


def wanted_count(q):
    """'the recent event' -> 1, 'last 2 events' / 'two events' -> 2, plain 'events' -> 3 (a page)."""
    m=re.search(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+)?(events?|alerts?|warnings?)',q)
    if m:
        n=int(m.group(1)) if m.group(1).isdigit() else WORDNUM[m.group(1)]
        return max(1,min(n,10))
    if re.search(r'\b(event|alert|warning)\b(?!s)',q) and not re.search(r'\b(all|every|list)\b',q):
        return 1
    return 3


def one_event(e):
    today=time.strftime('%Y%m%d')==time.strftime('%Y%m%d',time.localtime(e['ts']))
    clock=time.strftime('%I:%M %p',time.localtime(e['ts'])).lstrip('0')
    when=f"at {clock}" if today else time.strftime('on %b %d at ',time.localtime(e['ts'])).replace(' 0',' ')+clock
    who=e.get('dog') or ''
    who=who if who and not str(who).lstrip('-').isdigit() else 'an unidentified dog'
    what=(e.get('reason') or e.get('action') or 'activity needing attention').split(' - ')[0]
    return f"{'Danger' if e['level']>=3 else 'Warning'} {when}: {who} was {what}."


def event_page(pipe,rows,start=0,size=3):
    page=rows[start:start+size]
    pipe._event_readout=(rows,start+len(page))
    if not page: return 'There are no more recorded events in this list.'
    lines=[one_event(e) for e in page]
    if start+len(page)<len(rows): lines.append('Say read more events to continue.')
    return ' '.join(lines)


def events_reply(pipe,rows,n,today,q=''):
    if re.match(r'(any|are there|were there|did .* have|have there been)\b',q):     # yes/no: say so, then the latest
        pipe._event_readout=(rows,1)
        return (f"Yes, {len(rows)} recorded{' today' if today else ''}. The latest: "+one_event(rows[0])
                +(' Say read more events for the others.' if len(rows)>1 else ''))
    if n==1:
        pipe._event_readout=(rows,1)
        more=' Say read more events for older ones.' if len(rows)>1 else ''
        return 'The most recent event: '+one_event(rows[0])+more
    if n!=3:
        pipe._event_readout=(rows,min(n,len(rows)))
        head=f"The last {min(n,len(rows))} events: "
        return head+' '.join(one_event(e) for e in rows[:n])+(' Say read more events to continue.' if len(rows)>n else '')
    return ('Here are the most recent recorded events. ' if not today else "Here are today's most recent recorded events. ")+event_page(pipe,rows)


LEVEL_WORD = {0: 'Safe', 1: 'Watch', 2: 'Warning', 3: 'Danger'}


def dogs_roster(pipe, only=None):
    """Who the dogs are: registered names, who is on camera now, what each is doing and where."""
    reg = getattr(pipe, 'registry', None)
    known = list(getattr(reg, 'dogs', {}) or {})
    names = dict(getattr(reg, 'names_by_tid', {}) or {})
    st = pipe.state
    seen, unnamed = [], 0
    for a in getattr(st, 'assessments', []) or []:
        who = names.get(a.track_id)
        doing = a.reason.split(' - ')[0]
        where = f' on {a.zone}' if getattr(a, 'zone', None) else ''
        if who:
            seen.append((who, f'{who} is {doing}{where}'))
        else:
            unnamed += 1
            seen.append((None, f'an unnamed dog is {doing}{where}'))
    if only:
        line = next((s for w, s in seen if w == only), None)
        if line:
            return line[0].upper() + line[1:] + '.'
        return f"{only} is one of your registered dogs, but I don't see {only} on camera right now." \
            if only in known else f"I don't have a dog named {only}. Registered dogs: {', '.join(known) or 'none yet'}."
    parts = []
    if known:
        parts.append(f"You have {len(known)} registered dog{'s' if len(known) != 1 else ''}: {', '.join(known)}.")
    else:
        parts.append('No dogs are named yet. Add them on the Dogs page.')
    if seen:
        parts.append('On camera now: ' + '; '.join(s for _, s in seen) + '.')
        away = [k for k in known if k not in {w for w, _ in seen}]
        if away:
            parts.append(f"Not in view: {', '.join(away)}.")
    else:
        parts.append("I don't see any dog on camera right now.")
    if unnamed and known:
        parts.append('An unnamed dog may be one I have not recognised yet; adding more photos on the Dogs page helps.')
    return ' '.join(parts)


def behaviours_answer(pipe, q):
    from .risk import BEHAVIORS
    eng = getattr(pipe, 'engine', None)
    beh = getattr(eng, 'beh', None) or {k: (v[1], v[2]) for k, v in BEHAVIORS.items()}
    hit = next((k for k, (name, *_r) in BEHAVIORS.items()
                if re.search(r'\b' + re.escape(name.lower().split()[0]) + r'\w*', q)), None)
    def line(k):
        lvl, sec = beh.get(k, (BEHAVIORS[k][1], BEHAVIORS[k][2]))
        return f"{BEHAVIORS[k][0]}: {LEVEL_WORD.get(lvl, lvl)}" + (f' after {int(sec)} seconds' if sec else '')
    if hit and not re.search(r'all|list|settings|page', q):
        return line(hit) + '. You can change it on the Behaviours page.'
    alerting = [line(k) for k in BEHAVIORS if beh.get(k, (0,))[0] >= 2]
    watch = [BEHAVIORS[k][0] for k in BEHAVIORS if beh.get(k, (0,))[0] == 1]
    return ('Alerts: ' + '; '.join(alerting) + '.' + (f" Watch only: {', '.join(watch)}." if watch else '')
            + ' Everything else counts as safe. Change these on the Behaviours page.')


def app_answer(question,pipe):
    q=question.lower()
    reg_names=list(getattr(getattr(pipe,'registry',None),'dogs',{}) or {})
    # who the dogs are ("Who are the dogs?", "Who do you see?", "Do you know my dogs?", "Sino sila?")
    if re.search(r"who (are|r) (the|my|your|these|those)?\s*dogs|who('s| is) (there|here|on camera|in view)|who do you see|"
                 r"(names? of|name) (the|my) dogs|what are (the|my) dogs'? names|do you know (my|the) dogs|"
                 r"how many dogs do i have|which dogs (do i have|are registered)|\bsino (sila|ang mga aso)|kilala mo|"
                 r"(which|what) dogs (are|do you|can you)|dogs (are )?(on camera|in view|here|there)|dogs do you see",q) \
            and not (re.search(r"how many", q) and not re.search(r"do i have", q)):
        return dogs_roster(pipe)
    m=re.search(r"(who is|tell me about|do you know)\s+(\w+)",q)
    if m:
        name=next((n for n in reg_names if n.lower()==m.group(2)),None)
        if name: return dogs_roster(pipe, only=name)
    # Behaviours page ("What are the behaviour settings?", "How dangerous is digging?")
    if re.search(r"behaviou?rs?( page| settings| levels)?\b|how (dangerous|serious|bad) is (digging|chewing|fighting|jumping|scratching|sniffing|licking|rough)|"
                 r"what level is (digging|chewing|fighting|jumping|scratching|sniffing|licking)",q):
        return behaviours_answer(pipe, q)
    # one object's danger level ("Is a battery dangerous?" / "How dangerous is a slipper?")
    vocab=getattr(getattr(pipe,'hazard_det',None),'vocab',None) or getattr(pipe,'cfg',{}).get('hazards',{}) or {}
    obj=next((w for w in sorted(vocab,key=len,reverse=True) if re.search(r'\b'+re.escape(w.lower())+r's?\b',q)),None)
    if obj and re.search(r'dangerous|danger level|how (risky|bad)|is .* (safe|harmful|risky)|tier',q):
        t=int(vocab[obj]); word={0:'ignored (a look-alike)',1:'low risk',2:'medium risk',3:'high danger'}.get(t,'configured')
        lead=({0:'No. ',1:'Only a little. ',2:'Yes. ',3:'Yes. '}.get(t,'') if re.match(r'(is|are)\b',q) else '')
        return f"{lead}{obj[0].upper()+obj[1:]} is set as {word} on the Things page."
    # camera source
    if re.search(r'which camera|what camera|camera (source|name)',q):
        cam=getattr(pipe,'cfg',{}).get('camera',{})
        return f"I'm using USB camera {cam.get('index',0)} at {cam.get('width',1280)} by {cam.get('height',720)}. Change it in Settings."
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
        if not rows:return ('No. ' if re.match(r'(any|are there|were there)\b',q) else '')+'No matching events are recorded' + (' today.' if today else '.')
        return events_reply(pipe,rows,wanted_count(q),today,q)
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
        return ('I can read the camera view, who the dogs are and what each is doing, recent activity, saved events, '
                'drawn areas, the Things list, the Behaviours levels, the camera and voice settings, and saved clips. What would you like?')
    return None
