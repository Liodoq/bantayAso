"""Varied social replies and evidence-based, lightly playful dog recaps."""
import re
import time
from .history import fmt_dur

APPRECIATION = (
    "You're welcome. Happy to help keep an eye on them.",
    "Anytime. That's what I'm here for.",
    "Glad to help. Your dogs have quite the support team.",
    "You're welcome. You handle the cuddles; I'll help with the watching.",
)

def appreciation(question, pipe):
    # Full utterance only: never swallow 'Thanks, but did Oreo chew something?'.
    words=re.sub(r'[^a-z ]',' ',question.lower())
    words=re.sub(r'\s+',' ',words).strip()
    if not re.fullmatch(r'(?:(?:okay|ok|good|great|nice|perfect|awesome|alright|well done|good job|thank you|thanks|thank you so much|thanks a lot|salamat|maraming salamat|bantay|very good|appreciate it|i appreciate it|that s great|you re the best)\s*)+',words):
        return None
    n=getattr(pipe,'_appreciation_index',0)
    pipe._appreciation_index=n+1
    return APPRECIATION[n%len(APPRECIATION)]


def recap(question,pipe,known):
    q=question.lower()
    day=bool(re.search(r'\btoday\b|whole day|ngayong araw',q))
    retrospective=bool(re.search(r'\b(good|bad|naughty|trouble|previous|earlier|before)\b|behav|what.*\bdid\b|\bdid\b.*\bdo\b',q))
    subject=next((n for n in known if re.search(r'\b'+re.escape(n.lower())+r'\b',q)),None)
    # Generic event-count questions keep their existing route.
    if not (retrospective or (day and subject and re.search(r'chew|sleep|eat|walk|lick|scratch|do',q))):
        return None
    if subject is None and re.search(r'\b(he|she|him|her)\b',q):
        if time.monotonic()-getattr(pipe,'_last_subject_at',0)<60:
            subject=getattr(pipe,'_last_subject',None)
    if subject is None:
        if len(known)==1: subject=known[0]
        else: return 'Which dog would you like a recap for?'
    pipe._last_subject,pipe._last_subject_at=subject,time.monotonic()
    events=[]
    if day:
        if hasattr(pipe.events,'for_dog_today'):
            events=pipe.events.for_dog_today(subject)
        else:
            events=[e for e in pipe.events.today() if (e.get('dog') or '').casefold()==subject.casefold() and not e.get('false_alarm')]
    events=sorted(events,key=lambda e:e['ts'],reverse=True)
    if events:
        e=events[0]
        when=time.strftime('%I:%M %p',time.localtime(e['ts'])).lstrip('0')
        reason=e.get('reason') or e.get('action') or 'activity needing attention'
        prefix='danger' if e.get('level',0)>=3 else 'warning'
        # No joke about hazards and no transformation of an alert into a confirmed ingestion.
        return f"Today's log has a {prefix} for {subject} at {when}: {reason}. That is a recorded alert, not a complete account of the day."
    now=time.monotonic()
    rows=[s for s in list(pipe.history.samples) if 0<=now-s[0]<=600 and s[1].casefold()==subject.casefold()]
    if not rows:
        return (f"I have no recent activity for {subject}" + (" or named alerts in today's log" if day else '') + ". I can't judge what happened while unobserved.")
    # Aggregate observed samples; do not turn frequencies into an invented chronological sequence.
    counts={}
    for _,_,action,level,_ in rows:
        counts[action]=counts.get(action,0)+1
    ordered=sorted(counts,key=counts.get,reverse=True)
    span=fmt_dur(max(1,rows[-1][0]-rows[0][0]))
    risky=any(r[3]>=2 for r in rows)
    if risky:
        acts=list(dict.fromkeys(r[2] for r in rows if r[3]>=2))
        return f"In the recent observations spanning {span}, {subject} showed activity needing attention: {', '.join(acts)}. I can't speak for the unobserved time."
    descriptions=[('appearing to sleep' if a=='sleeping' else a) for a in ordered[:3]]
    sentence=f"In the recent observations spanning {span}, {subject} was mostly {descriptions[0]}"
    if len(descriptions)>1: sentence+=', with some '+', '.join(descriptions[1:])
    sentence+='.'
    if day: sentence+=" I only have recent activity and today's alert log, not a full-day activity record."
    else: sentence+=' This covers the observations I have, including possible gaps.'
    if len(ordered)==1 and ordered[0] in ('sleeping','lying down','resting'):
        sentence+=' Quite the demanding rest schedule.'
    # Ordinary chewing is an action, not automatically bad behavior.
    return sentence


def protect(pipe,text):
    persona=getattr(pipe,'persona',None)
    if persona is not None:
        persona.protected_text=text
    return text
