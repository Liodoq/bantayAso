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
    if difflib.SequenceMatcher(None, two, "bantay").ratio() >= 0.72 and len(words) > 1:
        return 2
    if difflib.SequenceMatcher(None, one, "bantay").ratio() >= 0.72:
        return 1
    return 0


def strip_wake(text: str) -> tuple[bool, str]:
    """Detect the wake word (tolerating mis-hearings) and remove it from the question."""
    t = text.strip()
    m = WAKE.search(t)
    if m:
        return True, t[m.end():].strip(" ,.!?") if m.start() == 0 else (t[:m.start()] + t[m.end():]).strip(" ,.!?")
    m = WAKE_ANYWHERE.search(t)
    if m:
        return True, (t[:m.start()] + t[m.end():]).strip(" ,.!?")
    n = _fuzzy_first_words(t)
    if n:
        rest = re.split(r"[\s,]+", t, maxsplit=n)
        return True, (rest[n] if len(rest) > n else "").strip(" ,.!?")
    return False, t


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
               r"\bthem\b|\bit\b|siya|sila|doing|happen|eat|chew|bite|bit|sleep|lying|sit|stand|walk|play|"
               r"lick|scratch|dig|sniff|bark|where|nasaan|asan|how many|ilan|alert|warning|danger|safe|okay|"
               r"\bok\b|wrong|problem|today|minute|minuto|second|timeline|when|bed|trash|sofa|bowl|zone|"
               r"camera|battery|cable|toy|food|mouth|swallow|ginagawa|kumain|natutulog|bantay")

OUT_OF_SCOPE = ("Sorry, that's not in my scope. I only watch your dogs. You can ask me: "
                "\"What were my dogs doing for the past 2 minutes?\", \"Where are my dogs?\", "
                "\"How many dogs do you see?\", or \"What happened today?\"")
CANT_TELL = ("I can't tell that from what I've seen. Try asking what your dogs were doing in the last "
             "1 to 10 minutes, where they are, or what happened today.")


def in_scope(q: str, known_names=()) -> bool:
    ql = q.lower()
    if any(n.lower() in ql for n in known_names):
        return True
    return re.search(SCOPE_WORDS, ql) is not None


def answer(question: str, pipe) -> str:
    """Return a short spoken answer. `pipe` is the running Pipeline."""
    _, q = strip_wake(question)
    ql = q.lower()
    names = pipe.registry.names_by_tid if pipe.registry else {}
    known = sorted((pipe.registry.dogs if pipe.registry else {}).keys(), key=len, reverse=True)
    st = pipe.state
    default = pipe.dog_name
    if not q:
        return "Yes? Ask me what your dogs are doing, where they are, or if anything happened."
    # small talk the assistant should handle itself
    if re.search(r"can you hear me|are you (there|listening|awake)|naririnig mo", ql):
        return "Yes, I can hear you! Ask me what your dogs are doing, where they are, or what happened today."
    if re.fullmatch(r"(hi|hello|hey|good (morning|afternoon|evening)|kumusta)[\s!.]*", ql):
        return "Hi! I'm watching your dogs. Ask me what they're doing or if anything happened."
    if re.search(r"\b(thank you|thanks|salamat)\b", ql):
        return "You're welcome! I'll keep watching them."
    if re.search(r"who are you|what can you do|^help\b|how do i use you", ql):
        return ("I'm Bantay, your offline dog watcher. I can tell you what your dogs did in the last 1 to 10 "
                "minutes, where they are, how many I see, and what alerts happened today.")
    # guardrail 1: off-topic questions (homework, code, news...) are politely declined
    if not in_scope(q, known):
        return OUT_OF_SCOPE
    # which dog? explicit name, or a follow-up pronoun ("and him?", "what about her")
    subject = next((n for n in known if re.search(rf"\b{re.escape(n.lower())}\b", ql)), None)
    if subject is None and re.search(r"\b(he|him|his|she|her|it|siya|niya)\b", ql):
        subject = getattr(pipe, "_last_subject", None)
    pipe._last_subject = subject or getattr(pipe, "_last_subject", None)
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
