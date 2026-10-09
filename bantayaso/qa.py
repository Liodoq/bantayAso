"""Ask Bantay: answers questions about the dogs from the live state, 10-min history and event log.

Facts are computed in code first; a small local LLM (Ollama) may phrase open questions, but only
from those facts, so it can't invent things.
"""
from __future__ import annotations

import re
import time

from .history import fmt_dur

NUM = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
       "seven": 7, "eight": 8, "nine": 9, "ten": 10, "isang": 1, "dalawang": 2, "tatlong": 3,
       "limang": 5, "sampung": 10}
# Whisper hears "Bantay" many ways: "Van Ty", "Ban tai", "Bun tie", "Ban-tay", "Pantay", "Montay"...
# Pattern: (b|v|p|m) + vowel + n + optional space/hyphen + (t|d) + (ay|ai|ie|y|i|e|eye)
WAKE = re.compile(r"^\W*(hey|hi|ok|okay|uy)?[\s,]*"
                  r"\b([bvpm][aouüe]h?n{1,2}[\s\-]?[td]h?(?:ay|ai|aye|ie|eye|y|i|e|ey)|bantayaso|bantay)\b[\s,.!?]*",
                  re.I)
WAKE_ANYWHERE = re.compile(r"\b(bantay|bantai|bontay|buntay|bantayaso)\b[,.!?]*", re.I)


def _fuzzy_first_words(text: str) -> int:
    """If the first one or two words sound like 'Bantay', return how many words to drop (else 0)."""
    import difflib
    words = re.findall(r"[A-Za-z']+", text)[:2]
    if not words:
        return 0
    one = words[0].lower()
    two = (words[0] + words[1]).lower() if len(words) > 1 else one
    if (len(words) > 1 and one[:1] in "bvpm" and len(one) <= 4
            and difflib.SequenceMatcher(None, two, "bantay").ratio() >= 0.72):
        return 2                                 # "Van Ty", "Ban tai", "Bun tie"
    if one[:1] in "bvpm" and difflib.SequenceMatcher(None, one, "bantay").ratio() >= 0.72:
        return 1
    return 0


def strip_wake(text: str, strict: bool = False) -> tuple[bool, str]:
    """Detect the wake word (tolerating mis-hearings) and remove it from the question.
    strict=True (hands-free): only counts when the sentence STARTS with the wake word."""
    t = text.strip()
    woke = False
    for _ in range(3):                       # "Bantay, Bantay, ..." -> strip repeats
        m = WAKE.search(t)
        if m and m.start() == 0:
            t, woke = t[m.end():].strip(" ,.!?"), True
            continue
        n = _fuzzy_first_words(t)
        if n:
            rest = re.split(r"[\s,]+", t, maxsplit=n)
            t, woke = (rest[n] if len(rest) > n else "").strip(" ,.!?"), True
            continue
        break
    if not woke and not strict:
        m = WAKE_ANYWHERE.search(t)
        if m:
            t, woke = (t[:m.start()] + t[m.end():]).strip(" ,.!?"), True
    return woke, t


def parse_minutes(q: str, default: int = 1) -> int:
    m = re.search(r"(\d+|" + "|".join(NUM) + r")\s*(minutes?|mins?|minuto)", q, re.I)
    if m:
        v = m.group(1).lower()
        n = int(v) if v.isdigit() else NUM.get(v, default)
    elif re.search(r"(last|past) minute", q, re.I):
        n = 1
    else:
        n = default
    return max(1, min(10, n))


def _alerts_text(events, minutes: int) -> str:
    since = time.time() - minutes * 60
    ev = [e for e in events if e["ts"] >= since and not e["false_alarm"]]
    if not ev:
        return f"No alerts in the last {fmt_dur(minutes * 60)}."
    ev = sorted(ev, key=lambda e: e["ts"])
    lines = []
    for e in ev[-3:]:
        t = time.strftime("%I:%M %p", time.localtime(e["ts"])).lstrip("0")
        who = e["dog"] if e["dog"] and not e["dog"].lstrip("-").isdigit() else "a dog"
        lines.append(f"{'a danger' if e['level'] == 3 else 'a warning'} at {t}: {who} {e['reason'].split(' - ')[0]}")
    n = len(ev)
    return f"{n} alert{'s' if n > 1 else ''} in the last {fmt_dur(minutes * 60)}: " + "; ".join(lines) + "."


def _where_text(boxes, assessments, names, frame_w, default_name) -> str:
    if not boxes:
        return "I don't see any dog right now."
    by = {a.track_id: a for a in assessments}
    parts = []
    for tid, box, _ in boxes:
        a = by.get(tid)
        who = names.get(tid) or (default_name if len(boxes) == 1 else "one dog")
        cx = (box[0] + box[2]) / 2 / max(1, frame_w)
        side = "on the left" if cx < 0.33 else "in the middle" if cx < 0.66 else "on the right"
        z = (a.zone or "") if a else ""
        zl = z.lower()
        place = ("on the bed" if zl.startswith("bed") else "at the trash" if zl.startswith("trash")
                 else "at the food bowl" if zl.startswith("food") else f"at the {z}" if z else side)
        doing = a.reason.split(" - ")[0] if a else ""
        doing = doing.replace(" on the bed", "")
        parts.append(f"{who} is {place}" + (f", {doing}" if doing and doing not in ("all calm", "resting") else ""))
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def _now_text(st, names, default, subject=None) -> str:
    if not st.assessments:
        return "I don't see any dog right now."
    parts = []
    items = [a for a in st.assessments if not subject or names.get(a.track_id) == subject] or st.assessments
    if subject and not any(names.get(a.track_id) == subject for a in st.assessments):
        return f"I don't see {subject} right now."
    for a in items:
        who = names.get(a.track_id) or (default if len(st.assessments) == 1 else "one dog")
        r = a.reason.split(" - ")[0]
        r = "resting" if r in ("all calm", "resting") else r
        parts.append(f"{who} is {r}")
    txt = ". ".join(p[0].upper() + p[1:] for p in parts) + "."
    return txt + (" That's why I raised an alert." if any(a.level >= 2 for a in items) else "")


HAZARD_WORDS = ("battery", "cable", "cord", "pill", "medicine", "plastic", "chocolate", "sock",
                "shoe", "slipper", "remote", "coin", "earphone", "wrapper", "paper", "comb", "pen")


def guard(text: str | None, facts: str) -> str | None:
    """Reject LLM answers that mention a hazard/object or a number that isn't in the facts."""
    if not text:
        return None
    t, f = text.lower(), facts.lower()
    for w in HAZARD_WORDS:
        if w in t and w not in f:
            return None
    for n in re.findall(r"\b\d+\b", t):
        if n not in f:
            return None
    return text if len(text) < 300 else None


def _today_text(pipe) -> str:
    s = pipe.events.summary_today()
    if not s.get("total"):
        return "Today has been calm so far. No warnings or dangers."
    hot = f" Most alerts were at {s['hotspot']}." if s.get("hotspot") else ""
    hour = max(s["by_hour"], key=s["by_hour"].get) if s.get("by_hour") else None
    when = f" The busiest time was around {hour % 12 or 12} {'AM' if hour < 12 else 'PM'}." if hour is not None else ""
    return (f"Today there {'was' if s['total'] == 1 else 'were'} {s['total']} alert{'s' if s['total'] != 1 else ''}: "
            f"{s['danger']} danger and {s['warning']} warning.{hot}{when}")


SCOPE_WORDS = (r"\bdogs?\b|\bpupp?(y|ies)\b|\baso\b|\baso(ng)?\b|\bpets?\b|\bhe\b|\bshe\b|\bhim\b|\bher\b|\bthey\b|"
               r"\bthem\b|\bit\b|siya|sila|doing|happen|fight|rough|away|eat|chew|bite|bit|sleep|lying|sit|stand|walk|play|"
               r"lick|scratch|dig|sniff|bark|where|nasaan|asan|how many|ilan|alert|warning|danger|safe|okay|"
               r"\bok\b|wrong|problem|today|minute|minuto|second|timeline|when|bed|trash|sofa|bowl|zone|"
               r"camera|battery|cable|toy|food|mouth|swallow|ginagawa|kumain|natutulog|bantay")

OUT_OF_SCOPE = "I can help with the camera view or something in Bantay. What would you like me to check?"
CANT_TELL = "I don't have enough recorded information to answer that yet."


def in_scope(q: str, known_names=()) -> bool:
    ql = q.lower()
    if any(n.lower() in ql for n in known_names):
        return True
    return re.search(SCOPE_WORDS, ql) is not None


ACT_WORDS = {"sleep": "sleeping", "eat": "eating", "chew": "chewing", "sit": "sitting", "lying": "lying",
             "lie": "lying", "lay": "lying", "stand": "standing", "walk": "walking", "lick": "licking",
             "scratch": "scratching", "dig": "digging", "sniff": "sniffing", "play": "chewing", "fight": "fight", "rough": "fight"}


def _act_in(q: str) -> str | None:
    for k, v in ACT_WORDS.items():
        if re.search(rf"\b{k}", q):
            return v
    return None


def _current(st, names, default):
    out = []
    for a in st.assessments:
        who = names.get(a.track_id) or (default if len(st.assessments) == 1 else "one of your dogs")
        out.append((who, a.reason.split(" - ")[0], a))
    return out


def _yes_no(q, st, names, default, subject) -> str | None:
    """'Is Oreo sleeping?' / 'Are my dogs eating?' -> answered from the live state."""
    if not re.match(r"^(is|are|was|were|does|do)\b", q):
        return None
    act = _act_in(q)
    if not act:
        return None
    cur = [c for c in _current(st, names, default) if not subject or c[0] == subject]
    if not cur:
        return f"I don't see {subject or 'any dog'} right now."
    hits = [c for c in cur if act in c[1] or (act == "eating" and "chewing" in c[1])]
    if hits:
        who = ", ".join(c[0] for c in hits)
        return f"Yes, {who} {'is' if len(hits) == 1 else 'are'} {hits[0][1]}."
    who, doing, _ = cur[0]
    return f"No. {who[0].upper() + who[1:]} is {doing if doing not in ('all calm',) else 'calm'} right now."


def _who_is(q, st, names, default) -> str | None:
    """'Which dog is eating?' / 'Who is on the bed?'"""
    if not re.search(r"^(who|which dog|which one|sino)", q):
        return None
    act = _act_in(q)
    place = re.search(r"\b(bed|trash|sofa|bowl|food|danger|play)\b", q)
    cur = _current(st, names, default)
    hits = [c for c in cur if (act and act in c[1]) or (place and place.group(1) in (c[2].zone or "").lower() + c[1])]
    if not hits:
        return "None of them right now." if cur else "I don't see any dog right now."
    who = [c[0] for c in hits]
    return f"{', '.join(who)} {'is' if len(who) == 1 else 'are'} {hits[0][1]}."


def _how_long(q, pipe, subject, default) -> str | None:
    if not re.search(r"how long|gaano katagal", q):
        return None
    segs = pipe.history.segments(10, subject)
    if not segs:
        return f"I haven't seen {subject or 'your dogs'} in the last 10 minutes."
    st_, en, w, act, _z = segs[-1]
    who = default if w == "unnamed" else w
    return f"{who[0].upper() + who[1:]} has been {act} for about {fmt_dur(en - st_)}."


def _last_time(q, pipe, subject, default) -> str | None:
    """'When did Oreo last eat?'"""
    if not re.search(r"\blast\b.*\b(eat|ate|chew|sleep|slept|drink|play)|\b(eat|ate|chew|sleep|slept)\b.*\blast\b", q):
        return None
    act = _act_in(q.replace("ate", "eat").replace("slept", "sleep")) or "eating"
    segs = [x for x in pipe.history.segments(10, subject) if act in x[3] or (act == "eating" and "chewing" in x[3])]
    if not segs:
        return f"I haven't seen {subject or 'your dogs'} {act} in the last 10 minutes."
    st_, en, w, a, _z = segs[-1]
    who = default if w == "unnamed" else w
    ago = max(0, time.time() - en)
    when = "just now" if ago < 20 else f"about {fmt_dur(ago)} ago"
    return f"{who[0].upper() + who[1:]} was {a} {when}, for about {fmt_dur(en - st_)}."


def _last_alert(q, pipe) -> str | None:
    if not re.search(r"(last|latest|recent) (alert|warning|danger)", q):
        return None
    ev = pipe.events.today(1)
    if not ev:
        return "There haven't been any alerts today."
    e = ev[0]
    t = time.strftime("%I:%M %p", time.localtime(e["ts"])).lstrip("0")
    who = e["dog"] if e["dog"] and not str(e["dog"]).lstrip("-").isdigit() else "a dog"
    return f"The last alert was {'a danger' if e['level'] == 3 else 'a warning'} at {t}: {who} {e['reason'].split(' - ')[0]}."


TEACH_ALIASES = {
    'sitting': 'sitting', 'sit': 'sitting', 'nakaupo': 'sitting',
    'lying': 'lying down', 'lying down': 'lying down', 'nakahiga': 'lying down',
    'sleeping': 'sleeping', 'asleep': 'sleeping', 'natutulog': 'sleeping',
    'standing': 'standing', 'nakatayo': 'standing', 'walking': 'walking', 'naglalakad': 'walking',
    'licking': 'licking itself', 'licking itself': 'licking itself',
    'scratching': 'scratching itself', 'scratching itself': 'scratching itself',
    'scratching furniture': 'scratching furniture', 'jumping': 'jumping on furniture',
    'jumping on furniture': 'jumping on furniture', 'digging': 'digging',
    'sniffing': 'sniffing the floor', 'sniffing the floor': 'sniffing the floor',
    'chewing': 'chewing something', 'chewing something': 'chewing something',
    'eating': 'eating', 'kumakain': 'eating',
}


def _teach_action(question, q, pipe, known, names):
    """Conservative, anchored grammar: statements about visible dogs, never questions."""
    text = q.lower().replace('’', "'").strip(' .!')
    if '?' in question or re.match(r'^(is|are|was|were|does|do|did|can|could|would|should|what|why|when|where|how|if)\b', text):
        return None
    cue = bool(re.match(r'^(?:please )?(?:remember|learn)\b', text))
    text = re.sub(r'^(?:please )?(?:remember|learn)(?: that)?\s+', '', text)
    deictic = re.fullmatch(r"(?:that's|that is|this is)\s+(.+)", text)
    match = re.fullmatch(r"(.+?)(?:\s+is\s+|'s\s+)(.+)", text)
    if deictic:
        target, action, cue = '', deictic[1], True
    elif match:
        target, action = match.groups()
    else:
        return 'Say which dog and action, for example: remember Oreo is sitting.' if cue else None
    explicit_now = bool(re.search(r'\s+(?:right now|now|ngayon)$', action))
    if not (cue or explicit_now):
        return None
    action = re.sub(r'\s+(?:right now|now|ngayon)$', '', action)
    action = re.sub(r'^actually\s+', '', action)
    label = TEACH_ALIASES.get(action)
    classifier = getattr(pipe, 'classifier', None)
    if classifier is None:
        return 'Action teaching is unavailable while the action model is off.'
    if label not in classifier.labels:
        return 'I cannot teach that action. Choose an action from the teaching menu.'
    boxes = list(pipe.state.boxes)
    visible = {tid for tid, _box, _name in boxes if tid >= 0}
    subject = next((n for n in known if n.lower() == target), None)
    if target in ('he', 'she', 'it'):
        if time.monotonic() - getattr(pipe, '_last_subject_at', float('-inf')) <= 60:
            subject = getattr(pipe, '_last_subject', None)
        if not subject:
            return "Please name the dog first; I don't have a recent subject to remember."
    elif target and not subject and target not in ('my dog', 'the dog', 'this dog'):
        return f"I don't know a dog named {target}. Name it on the Dogs page first."
    if subject:
        ids = [tid for tid, name in names.items() if name == subject and tid in visible]
        if len(ids) != 1:
            return f"I can't clearly see {subject} right now. No examples saved."
        tid = ids[0]
    else:
        if len(visible) != 1:
            return 'Please name or click the dog to teach.' if visible else 'I need a dog in view before learning.'
        tid = next(iter(visible))
        subject = names.get(tid)
    pipe._last_subject = subject
    pipe._last_subject_at = time.monotonic()
    return pipe.request_teach(tid, label, subject=subject)


def answer(question: str, pipe) -> str:
    """Return a short spoken answer. `pipe` is the running Pipeline."""
    _, q = strip_wake(question)
    ql = q.lower()
    names = pipe.registry.names_by_tid if pipe.registry else {}
    known = sorted((pipe.registry.dogs if pipe.registry else {}).keys(), key=len, reverse=True)
    st = pipe.state
    default = pipe.dog_name
    taught = _teach_action(question, q, pipe, known, dict(names))
    if taught is not None:
        return taught
    if not re.sub(r"[\W_]+", "", q):                    # just "Bantay" / "Bantay?" -> short reply, then listen
        return "Yes?"
    # small talk the assistant should handle itself
    if re.search(r"can you hear me|are you (there|listening|awake)|naririnig mo", ql):
        return "Yes, I can hear you."
    if re.fullmatch(r"(hi|hello|hey|good (morning|afternoon|evening)|kumusta)[\s!.]*", ql):
        return "Hi! I'm watching them."
    from .social import appreciation, recap, protect
    social = appreciation(q, pipe)
    if social:
        return protect(pipe, social)
    from .app_qa import app_answer
    application_reply = app_answer(q, pipe)
    if application_reply is not None:
        return protect(pipe, application_reply)
    previous = recap(q, pipe, known)
    if previous:
        return protect(pipe, previous)
    if re.search(r"who are you|what can you do|^help\b|how do i use you", ql):
        return ("I'm Bantay, your offline dog watcher. I can tell you what your dogs did in the last 1 to 10 "
                "minutes, where they are, how many I see, and what alerts happened today.")
    if re.fullmatch(r"(how long|gaano katagal)[?!. ]*", ql):
        ql = 'how long has the dog been doing that'
    # A disconnected camera must not answer a present-tense question from stale boxes.
    frame_at = getattr(pipe, '_teaching_frame_at', None)
    historical = re.search(r'past|last|today|earlier|kanina|timeline|when', ql)
    if frame_at is not None and time.monotonic() - frame_at > 3 and not historical:
        return "I don't have a fresh camera view, so I can't check the dogs right now."
    from .scene import scene_answer
    scene_subject = next((n for n in known if re.search(rf"\b{re.escape(n.lower())}\b",ql)),None)
    scene_reply = scene_answer(ql, pipe, scene_subject)
    if scene_reply is not None:
        if scene_subject:
            pipe._last_subject = scene_subject
            pipe._last_subject_at = time.monotonic()
        return scene_reply
    # guardrail 1: off-topic questions (homework, code, news...) are politely declined
    if not in_scope(ql, known):
        return OUT_OF_SCOPE
    # which dog? explicit name, or a follow-up pronoun ("and him?", "what about her")
    subject = next((n for n in known if re.search(rf"\b{re.escape(n.lower())}\b", ql)), None)
    followup = re.search(r"\b(he|him|his|she|her|it|siya|niya)\b|how long|gaano katagal", ql)
    if subject is None and followup:
        if time.monotonic() - getattr(pipe, '_last_subject_at', 0) <= 60:
            subject = getattr(pipe, '_last_subject', None)
        if subject is None and len(st.boxes) > 1:
            return 'Which dog do you mean?'

    pipe._last_subject = subject or getattr(pipe, "_last_subject", None)
    if subject:
        pipe._last_subject_at = time.monotonic()
    for fn in (lambda: _yes_no(ql, st, names, default, subject), lambda: _who_is(ql, st, names, default),
               lambda: _how_long(ql, pipe, subject, default), lambda: _last_time(ql, pipe, subject, default),
               lambda: _last_alert(ql, pipe)):
        r = fn()
        if r:
            return r
    if re.search(r"\btoday\b|ngayong araw|since (this )?morning|whole day", ql):
        return _today_text(pipe)
    if re.search(r"\bwhen\b|timeline|what time|step by step|sequence|anong oras", ql):
        return pipe.history.timeline(parse_minutes(ql, default=5), subject, default)
    if re.search(r"\bwhere\b|nasaan|asan", ql):
        return _where_text(st.boxes, st.assessments, names, st.frame_w, default)
    if re.search(r"how many|ilan", ql):
        n = st.dogs
        return f"I can see {n} dog{'s' if n != 1 else ''} right now." if n else "I don't see any dog right now."
    if re.search(r"right now|\bnow\b|currently|ngayon", ql) and not re.search(r"\d|minute|past|last", ql):
        return _now_text(st, names, default, subject)
    minutes = parse_minutes(ql, default=1)
    if re.search(r"danger|alert|warning|wrong|happen|okay|ok\b|safe|problem|anything", ql):
        mins = parse_minutes(ql, default=10)
        now_txt = st.status if st.level >= 2 else "Right now everything is calm."
        return f"{_alerts_text(pipe.events.today(), mins)} {now_txt}"
    if re.search(r"doing|happen|up to|ginagawa|activity|been|past|last|minute", ql) or len(ql) < 4 \
            or (subject and len(ql.split()) <= 4):
        return pipe.history.describe(minutes, default, only=subject)
    # open question -> local LLM with facts only; rejected if it invents things
    facts = (f"Current status: {st.status}. Now: {_now_text(st, names, default)} "
             f"Last 10 minutes: {pipe.history.describe(10, default, only=subject)} "
             f"Timeline: {pipe.history.timeline(10, subject, default)} "
             f"{_alerts_text(pipe.events.today(), 10)} {_today_text(pipe)}")
    # guardrail 2: the LLM may only use the facts; it says OUT_OF_SCOPE / UNKNOWN when it can't
    reply = pipe.ask_llm(q, facts)
    if reply and "OUT_OF_SCOPE" in reply.upper():
        return OUT_OF_SCOPE
    if reply and "UNKNOWN" in reply.upper():
        return CANT_TELL
    # guardrail 3: reject answers that mention objects/numbers that aren't in the facts
    return guard(reply, facts) or CANT_TELL
