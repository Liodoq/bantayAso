"""Yes/no questions get "Yes" or "No" first, then the reason, from what the camera actually saw.

    "Is Oreo sleeping?"            -> "Yes. Oreo is sleeping on Bed 1, for about 3 minutes."
    "Is Pachuchay eating?"         -> "No. Pachuchay is just lying down, not eating. She was eating about 4 minutes ago."
    "Did Brownie eat anything?"    -> "Yes. Brownie was eating about 2 minutes ago, for about 40 seconds."
    "Is everything okay?"          -> "No. Oreo is chewing an unknown object right now."
    "Should I check on them?"      -> "No. All calm, and no alerts in the last 10 minutes."
    "Is Oreo on the bed?"          -> "Yes. Oreo is on Bed 1."
    "Natutulog ba si Oreo?"        -> (same as 'Is Oreo sleeping?'; Tagalog is restyled by persona)
If Bantay can't see the dog it says "Not sure." plus why, instead of guessing.
Returns None when the question isn't a yes/no question it can answer, so the normal path continues.
"""
from __future__ import annotations

import re
import time

from .history import fmt_dur

YN_START = r"^(is|are|was|were|does|do|did|has|have|can|could|should|will|would|any|anything)\b"
ACTS = [  # spoken words -> activity text as Bantay writes it (first match wins)
    (r"sleep|asleep|natutulog|tulog", "sleeping"),
    (r"\beat|\bate\b|kumakain|kumain|food", "eating"),
    (r"chew|ngumunguya|nginunguya", "chewing"),
    (r"lick|dila", "licking"),
    (r"scratch|kamot", "scratching"),
    (r"\bsit|sitting|nakaupo|upo", "sitting"),
    (r"\blie|lying|laying|nakahiga|higa", "lying"),
    (r"stand|nakatayo|tayo\b", "standing"),
    (r"walk|naglalakad|lakad", "walking"),
    (r"dig|hukay", "digging"),
    (r"jump|talon", "jumping"),
    (r"sniff|amoy", "sniffing"),
    (r"fight|away|rough|bite each", "fight"),
    (r"play|laro", "play"),
]
OK_WORDS = r"\b(ok|okay|safe|fine|alright|all right|good|behaving|ayos|okay lang|mabuti)\b"
BAD_WORDS = r"(should i (worry|check|go)|do i need to (check|go)|something wrong|anything wrong|any (danger|problem|alert|warning)s?|in danger|is there (a )?(danger|problem)|may problema)"
YES_RE = r"^(yes|no|not sure|oo|hindi)\b"


def is_yes_no(q: str) -> bool:
    q = q.lower().strip()
    return bool(re.match(YN_START, q) or re.search(r"\bba\b", q))


def _act(q: str):
    for pat, act in ACTS:
        if re.search(pat, q):
            return act
    return None


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _matches(act: str, reason: str) -> bool:
    r = reason.lower()
    if act == "eating":
        return "eating" in r or "chewing" in r
    if act == "fight":
        return "fight" in r or "rough" in r
    if act == "lying":
        return "lying" in r or "resting" in r
    return act in r


def _people(pipe, names, default, subject, q):
    """[(who, assessment)] for the dogs the question is about."""
    st = pipe.state
    cur = []
    for a in st.assessments:
        who = names.get(a.track_id) or (default if len(st.assessments) == 1 else "one of your dogs")
        cur.append((who, a))
    if subject:
        return [c for c in cur if c[0] == subject], subject
    if re.search(r"\b(they|them|dogs|both|all|sila)\b", q):
        return cur, "your dogs"
    if len(cur) == 1:
        return cur, cur[0][0]
    return cur, "your dogs"


def _where(a) -> str:
    return f" on {a.zone}" if getattr(a, "zone", None) else ""


def _last_seg(pipe, subject, act, minutes=10):
    segs = [s for s in pipe.history.segments(minutes, subject) if _matches(act, s[3])]
    return segs[-1] if segs else None


def _ago(end_wall: float) -> str:
    ago = max(0.0, time.time() - end_wall)
    return "just now" if ago < 20 else f"about {fmt_dur(ago)} ago"


def answer_yes_no(q: str, pipe, names: dict, default: str, subject: str | None) -> str | None:
    q = q.lower().strip().rstrip("?!. ")
    if not is_yes_no(q):
        return None
    cur, who = _people(pipe, names, default, subject, q)
    st = pipe.state

    # ---- "Is everything okay?" / "Should I check on them?" / "Any danger?"
    bad_q = re.search(BAD_WORDS, q)
    ok_q = re.search(OK_WORDS, q) and not _act(q)
    if bad_q or ok_q:
        risky = sorted([c for c in cur if c[1].level >= 2], key=lambda c: -c[1].level)
        recent = [e for e in pipe.events.today(20) if time.time() - e["ts"] <= 600]
        if risky:
            w, a = risky[0]
            why = f"{_cap(w)} is {a.reason.split(' - ')[0]} right now."
            return ("Yes. " if bad_q else "No. ") + why
        if recent:
            e = recent[0]
            t = time.strftime("%I:%M %p", time.localtime(e["ts"])).lstrip("0")
            why = (f"It's calm now, but there was a {'danger' if e['level'] == 3 else 'warning'} at {t}: "
                   f"{e['reason'].split(' - ')[0]}.")
            return ("Not right now. " if bad_q else "Yes, for now. ") + why
        if not cur:
            return "Not sure. I don't see any dog on camera right now."
        return ("No. " if bad_q else "Yes. ") + "All calm, and no alerts in the last 10 minutes."

    act = _act(q)
    past = re.match(r"^(did|has|have|was|were)\b", q) or re.search(r"\b(kanina|earlier|today|before|already|na ba)\b", q)

    # ---- "Did Brownie eat?" / "Has Oreo been sleeping?"
    if act and past:
        seg = _last_seg(pipe, subject, act)
        if seg:
            w = default if seg[2] == "unnamed" else seg[2]
            return f"Yes. {_cap(w)} was {seg[3]} {_ago(seg[1])}, for about {fmt_dur(seg[1] - seg[0])}."
        segs = pipe.history.segments(10, subject)
        if not segs:
            return f"Not sure. I haven't seen {who} in the last 10 minutes."
        totals = {}
        for s in segs:
            totals[s[3]] = totals.get(s[3], 0) + (s[1] - s[0])
        top = max(totals, key=totals.get)
        return f"No. I didn't see {who} {act} in the last 10 minutes; mostly {top}."

    # ---- "Is Oreo on the bed?" / "Is she in the trash zone?"
    zone_words = [z.name.lower() for z in getattr(pipe, "zones", [])] + ["bed", "sofa", "trash", "bowl", "food", "play", "danger"]
    zw = next((z for z in zone_words if re.search(rf"\b(on|in|at|near|sa)\s+(the\s+)?{re.escape(z)}\b", q)), None)
    if zw and not act:
        if not cur:
            return f"Not sure. I don't see {who} on camera right now."
        inside = [c for c in cur if zw in (c[1].zone or "").lower()]
        if inside:
            return f"Yes. {', '.join(_cap(c[0]) for c in inside)} {'is' if len(inside) == 1 else 'are'} on {inside[0][1].zone}."
        w, a = cur[0]
        return f"No. {_cap(w)} is {a.reason.split(' - ')[0]}{_where(a) or ', outside that area'}."

    # ---- "Is Oreo there?" / "Can you see Pachuchay?"
    if re.search(r"\b(there|here|on camera|visible|see|nakikita)\b", q) and not act:
        if cur:
            w, a = cur[0]
            return f"Yes. I can see {w}{_where(a)}; {'they are' if w == 'your dogs' else 'it is'} {a.reason.split(' - ')[0]}."
        return f"No. I don't see {who} on camera right now."

    # ---- "Is Oreo sleeping?" (now)
    if act:
        if not cur:
            return f"Not sure. I don't see {who} on camera right now."
        hits = [c for c in cur if _matches(act, c[1].reason) or _matches(act, c[1].action or "")]
        if hits:
            w, a = hits[0]
            seg = _last_seg(pipe, subject or (w if w not in ("your dogs", "one of your dogs") else None), act, 10)
            dur = f", for about {fmt_dur(seg[1] - seg[0])}" if seg and seg[1] - seg[0] >= 20 else ""
            names_ = ", ".join(_cap(c[0]) for c in hits)
            return f"Yes. {names_} {'is' if len(hits) == 1 else 'are'} {a.reason.split(' - ')[0]}{_where(a)}{dur}."
        w, a = cur[0]
        doing = a.reason.split(" - ")[0]
        doing = "calm" if doing in ("all calm",) else doing
        extra = ""
        seg = _last_seg(pipe, subject, act, 10)
        if seg:
            extra = f" {_cap(w)} was {seg[3]} {_ago(seg[1])}."
        return f"No. {_cap(w)} is {doing}{_where(a)}, not {act}.{extra}"
    return None


def ensure_yes_no(q: str, reply: str) -> str:
    """For yes/no questions answered elsewhere: if the reply clearly says no / none / nothing,
    lead with "No."; otherwise leave it (never invent a yes)."""
    if not is_yes_no(q) or re.match(YES_RE, reply.strip().lower()):
        return reply
    r = reply.strip().lower()
    if re.match(r"^(none|nothing|there (haven't|hasn't) been|no alerts|i (haven't|didn't) see|i don't see)", r):
        return "No. " + reply
    return reply
