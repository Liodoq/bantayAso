"""How Bantay talks: grounded, objective answers shaped to the owner's own way of talking.

Three layers (all local, nothing leaves the laptop):
1. StyleProfile  - learns the owner's language (English / Tagalog / Taglish), how long their own
                   sentences are and how formal they are ("po/opo"), and obeys explicit requests
                   ("Bantay, keep it short", "speak Tagalog", "more details", "be formal").
                   Saved in data/style.json.
2. Grounded LLM  - open questions go to the local LLM with: a fixed objective-speaking system
                   prompt, the camera facts as a bullet list, the last few turns (for follow-ups),
                   a style directive, few-shot examples, and a JSON schema reply
                   {"status": answer|unknown|out_of_scope, "answer": "..."} (Ollama structured
                   outputs), temperature 0.1. qa.guard() still rejects invented numbers/objects.
3. Finishing     - template answers are trimmed to the preferred length; for Tagalog/Taglish
                   they are rewritten by the LLM *only if* the rewrite keeps every number and adds
                   no object (same guard), otherwise the original English stays.
"""
from __future__ import annotations

import json
import re
import threading
from collections import deque
from pathlib import Path

TL_WORDS = set("""ang ng mga si ni sina kay ba na pa po opo ko mo niya siya sila kami tayo kayo ano
saan nasaan asan bakit paano sino ilan kailan ngayon kanina mamaya lang naman talaga aso wala
meron mayroon oo hindi huwag diba yung ito iyan iyon dito diyan doon kumain kumakain ginagawa
natutulog nakahiga nakaupo nakatayo naglalakad tulog gising salamat sige pakisabi pakiulit
mabuti lang ba't anong nga rin din pala muna eh kasi tapos""".split())
FORMAL_WORDS = re.compile(r"\b(po|opo|please|kindly|sir|ma'am|maam)\b")
LENGTHS = ("auto", "short", "detailed")
LANGS = ("auto", "en", "tl", "taglish")
TONES = ("auto", "casual", "formal")
LANG_NAME = {"en": "English", "tl": "Tagalog", "taglish": "Taglish (natural mix of Tagalog and English)"}

SYSTEM = """You are Bantay, the voice of a home pet camera. You answer the owner's questions about their dogs.
Speak objectively, like a careful observer reading a log:
- Use ONLY the camera facts given. Never invent objects, times, numbers, names or places.
- Lead with the direct answer, then at most one supporting detail. No greetings, no filler, no exclamation marks.
- Describe what the camera saw ("Oreo was lying on the bed for 3 minutes"), not feelings or intentions ("Oreo is happy/bored").
- If a fact says something is risky (warning/danger), mention it first.
- If the facts don't answer the question, status is "unknown". If the question is not about the dogs, camera or supplied application facts, status is "out_of_scope".
- No medical advice; for health worries say to contact a vet.
Return JSON only."""

FEW_SHOT = [
    ("Facts:\n- Now: Oreo is just lying down on bed 1.\n- Last 10 minutes: Oreo was lying down for about 8 minutes, then walking around for about 1 minute.\n- Alerts today: none.\n"
     "Style: English, short, casual.\nQuestion: is oreo being good?",
     {"status": "answer", "answer": "Yes, Oreo has mostly been lying on bed 1, with no alerts today."}),
    ("Facts:\n- Now: your dog is chewing an unknown object.\n- Last 10 minutes: chewing something for about 40 seconds.\n- Alerts today: 1 warning (chewing an unknown object).\n"
     "Style: Taglish, short, casual.\nQuestion: ano ginagawa ng aso ko?",
     {"status": "answer", "answer": "May nginunguya siya ngayon, mga 40 seconds na. Check mo kung ano yun."}),
    ("Facts:\n- Now: Choco is sleeping.\n- Alerts today: none.\nStyle: English, short, formal.\nQuestion: did choco drink water?",
     {"status": "unknown", "answer": ""}),
    ("Facts:\n- Now: Choco is sleeping.\nStyle: English, short, casual.\nQuestion: write me a poem about cats",
     {"status": "out_of_scope", "answer": ""}),
]
SCHEMA = {"type": "object",
          "properties": {"status": {"type": "string", "enum": ["answer", "unknown", "out_of_scope"]},
                         "answer": {"type": "string"}},
          "required": ["status", "answer"]}


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in ("bantay", "bantayaso")]
    if not words:
        return "en"
    r = sum(w in TL_WORDS for w in words) / len(words)
    return "tl" if r >= 0.45 else "taglish" if r >= 0.15 else "en"


class StyleProfile:
    def __init__(self, path: Path | None):
        self.path = path
        self.lang, self.length, self.tone = "auto", "auto", "auto"     # explicit owner choices
        self.lang_votes = {"en": 1.0, "tl": 0.0, "taglish": 0.0}        # learned from how they talk
        self.avg_words = 8.0
        self.formal = 0.0
        self._lock = threading.Lock()
        self.load()

    # ---------------- persistence
    def load(self) -> None:
        try:
            d = json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.exists() else {}
        except Exception:
            d = {}
        self.lang = d.get("lang", "auto") if d.get("lang") in LANGS else "auto"
        self.length = d.get("length", "auto") if d.get("length") in LENGTHS else "auto"
        self.tone = d.get("tone", "auto") if d.get("tone") in TONES else "auto"
        self.lang_votes.update({k: float(v) for k, v in (d.get("lang_votes") or {}).items() if k in self.lang_votes})
        self.avg_words = float(d.get("avg_words", self.avg_words))
        self.formal = float(d.get("formal", self.formal))

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"lang": self.lang, "length": self.length, "tone": self.tone,
                                       "lang_votes": self.lang_votes, "avg_words": round(self.avg_words, 2),
                                       "formal": round(self.formal, 3)}, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    # ---------------- learning from each question
    def observe(self, text: str) -> None:
        t = re.sub(r"^\W*bantay\w*\W*", "", text.strip(), flags=re.I)
        if not t:
            return
        with self._lock:
            lang = detect_language(t)
            for k in self.lang_votes:                                   # recent talk matters most
                self.lang_votes[k] = 0.8 * self.lang_votes[k] + (0.2 if k == lang else 0.0)
            self.avg_words = 0.8 * self.avg_words + 0.2 * len(t.split())
            self.formal = 0.8 * self.formal + (0.2 if FORMAL_WORDS.search(t.lower()) else 0.0)
            self.save()

    # ---------------- explicit requests ("Bantay, keep it short")
    def command(self, text: str) -> str | None:
        q = re.sub(r"^\W*bantay\w*\W*", "", text.strip().lower(), flags=re.I)
        if re.search(r"\?$", q) or re.match(r"(what|is|are|did|does|how|where|why|when|ano|bakit|nasaan)\b", q):
            return None
        rules = [
            (r"(keep it|be|answer|make it|mas)\s+(short|shorter|brief|briefer|quick|maikli|maiksi)|^(shorter|short answers?)\b|\bikli\b",
             ("length", "short"), "Okay, I'll keep answers short."),
            (r"(more|full|longer)\s+(detail|details|explanation)|explain more|be detailed|^(longer|detailed)\b|detalye",
             ("length", "detailed"), "Okay, I'll give more detail."),
            (r"(speak|talk|answer|reply|respond)\s+(in\s+)?(tagalog|filipino)|mag[- ]?tagalog|tagalog (na|ka|please|lang)",
             ("lang", "tl"), "Sige, magtatagalog na ako."),
            (r"(speak|talk|answer|reply|respond)\s+(in\s+)?taglish|taglish (na|please|lang)",
             ("lang", "taglish"), "Sige, Taglish na lang."),
            (r"(speak|talk|answer|reply|respond)\s+(in\s+)?english|english (na|please|lang)|mag[- ]?english",
             ("lang", "en"), "Okay, I'll answer in English."),
            (r"be (more )?(formal|polite|respectful)|use po\b", ("tone", "formal"), "Noted. I'll answer more formally."),
            (r"be (more )?(casual|relaxed|chill)|no need (for|to say) po", ("tone", "casual"), "Okay, I'll keep it casual."),
            (r"(reset|forget|clear) (my )?(style|preferences|settings for talking)|talk normally",
             ("reset", None), "Okay, back to normal. I'll adapt to how you talk again."),
        ]
        for pat, (field, value), reply in rules:
            if re.search(pat, q):
                with self._lock:
                    if field == "reset":
                        self.lang = self.length = self.tone = "auto"
                    else:
                        setattr(self, field, value)
                    self.save()
                return reply
        return None

    # ---------------- what to use right now
    def effective(self) -> tuple[str, str, str]:
        lang = self.lang if self.lang != "auto" else max(self.lang_votes, key=self.lang_votes.get)
        length = self.length if self.length != "auto" else ("short" if self.avg_words <= 10 else "normal")
        tone = self.tone if self.tone != "auto" else ("formal" if self.formal >= 0.3 else "casual")
        return lang, length, tone

    def directive(self) -> str:
        lang, length, tone = self.effective()
        size = {"short": "short (one sentence, under 20 words)", "normal": "one or two sentences",
                "detailed": "up to four sentences with the useful details"}[length]
        polite = ", add 'po' naturally" if tone == "formal" and lang != "en" else ""
        return f"{LANG_NAME[lang]}, {size}, {tone}{polite}"

    def describe(self) -> str:
        lang, length, tone = self.effective()
        auto = lambda v: " (learned)" if v == "auto" else ""             # noqa: E731
        return (f"{LANG_NAME[lang].split(' (')[0]}{auto(self.lang)} · {length}{auto(self.length)} · "
                f"{tone}{auto(self.tone)}")


def shape(text: str, length: str) -> str:
    """Trim a finished answer to the preferred length without cutting a sentence in half."""
    if length != "short" or len(text.split()) <= 22:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = parts[0]
    for p in parts[1:]:
        if len((out + " " + p).split()) > 22:
            break
        out += " " + p
    return out


class Persona:
    """Glue used by Pipeline.ask(): commands, learning, grounded LLM call, finishing."""

    def __init__(self, data_dir: Path | None, llm_call=None):
        self.style = StyleProfile(Path(data_dir) / "style.json" if data_dir else None)
        self.turns: deque = deque(maxlen=4)            # (question, answer) for follow-ups
        self.llm_call = llm_call                       # fn(messages, schema, max_tokens) -> str | None

    def messages(self, question: str, facts: str) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM}]
        for q, a in FEW_SHOT:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
        convo = "".join(f"- Owner: {q}\n- Bantay: {a}\n" for q, a in self.turns)
        bullets = "\n".join(f"- {s.strip()}" for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", facts) if s.strip())
        msgs.append({"role": "user", "content": (f"Facts:\n{bullets}\n"
                                                 + (f"Recent conversation:\n{convo}" if convo else "")
                                                 + f"Style: {self.style.directive()}.\nQuestion: {question}")})
        return msgs

    def ask_llm(self, question: str, facts: str) -> tuple[str, str] | None:
        """-> (status, answer) or None when the LLM isn't available."""
        if not self.llm_call:
            return None
        _, length, _ = self.style.effective()
        raw = self.llm_call(self.messages(question, facts), SCHEMA, {"short": 60, "normal": 110, "detailed": 180}[length])
        if not raw:
            return None
        try:
            d = json.loads(raw)
            status = str(d.get("status", "")).lower()
            ans = str(d.get("answer", "")).strip()
        except (ValueError, AttributeError):          # model ignored the schema: treat as plain text
            status, ans = "answer", raw.strip().split("\n")[0]
        if status not in ("answer", "unknown", "out_of_scope"):
            status = "answer" if ans else "unknown"
        self.already_styled = status == "answer" and bool(ans)
        return status, ans

    def finish(self, question: str, text: str, guard=None) -> str:
        """Shape a template/LLM answer to the owner's preferences and remember the turn."""
        if text == getattr(self, 'protected_text', None):
            # Keep coverage/uncertainty and danger details intact; no LLM embellishment.
            self.turns.append((question, text))
            self.protected_text = None
            return text
        lang, length, tone = self.style.effective()
        out = text
        if lang != "en" and self.llm_call and not getattr(self, "already_styled", False) and len(text) > 12 and text != "Yes?":
            msgs = [{"role": "system", "content": "Rewrite the message for a Filipino dog owner. Keep every fact, "
                                                  "name and number exactly. Add nothing. Return JSON only."},
                    {"role": "user", "content": f"Language: {LANG_NAME[lang]}. Tone: {tone}. Message: {text}"}]
            raw = self.llm_call(msgs, {"type": "object", "properties": {"text": {"type": "string"}},
                                       "required": ["text"]}, 120)
            try:
                cand = json.loads(raw or "{}").get("text", "").strip()
            except ValueError:
                cand = ""
            nums_ok = cand and set(re.findall(r"\d+", text)) == set(re.findall(r"\d+", cand))
            if nums_ok and (guard is None or guard(cand, text)):
                out = cand
        out = shape(out, length)
        self.turns.append((question, out))
        return out
