"""Offline tests for Ask Bantay answers (no camera, GPU, mic or Ollama).  python scripts/test_qa.py"""
import sys
import tempfile
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso.events import EventLog            # noqa: E402
from bantayaso.history import ActivityHistory    # noqa: E402
from bantayaso.qa import answer, guard           # noqa: E402
from bantayaso.risk import Assessment            # noqa: E402

fails = 0


def check(name, cond, info=""):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  -> {info}" if info else ""))
    fails += 0 if cond else 1


d = Path(tempfile.mkdtemp())
hist = ActivityHistory()
now = time.monotonic()
for i in range(150):                           # 2.5 minutes of fake history
    hist.record(now - 150 + i, [Assessment(1, 0, 0, "just lying down on the bed" if i < 100 else
                                           "chewing an unknown object - check what it is", zone="Bed 1"),
                                Assessment(2, 0, 0, "sleeping")], {1: "Choco", 2: "Bantay"})
log = EventLog(d / "e.db", d / "s")
log.add(Assessment(1, 2, 2, "chewing an unknown object - check what it is", "Bed 1"), None, "Choco")
st = types.SimpleNamespace(status="WARNING: Choco is chewing an unknown object", level=2, dogs=2,
                           assessments=[Assessment(1, 2, 2, "chewing an unknown object - check what it is", "Bed 1"),
                                        Assessment(2, 0, 0, "sleeping")],
                           boxes=[(1, (100, 300, 300, 500), "Choco"), (2, (900, 300, 1100, 500), "Bantay")], frame_w=1280)
reg = types.SimpleNamespace(names_by_tid={1: "Choco", 2: "Bantay"}, dogs={"Choco": [], "Bantay": []})
llm_reply = {"text": None}
pipe = types.SimpleNamespace(registry=reg, state=st, dog_name="your dog", history=hist, events=log,
                             ask_llm=lambda q, facts: llm_reply["text"])

a = answer("Bantay, what were my dogs doing for the past 2 minutes?", pipe)
check("past N minutes summary names both dogs", "Choco" in a and "Bantay" in a and "2 minutes" in a, a)
a = answer("Bantay, what was Choco doing?", pipe)
check("per-dog question only talks about Choco", "Choco" in a and "Bantay was" not in a, a)
a = answer("and what about him in the last 2 minutes?", pipe)
check("follow-up pronoun keeps the last dog", "Choco" in a, a)
a = answer("Bantay, where are my dogs?", pipe)
check("where -> positions", "Choco" in a and ("left" in a or "bed" in a.lower()), a)
a = answer("how many dogs do you see", pipe)
check("how many", "2 dogs" in a, a)
a = answer("Bantay, what is Bantay doing right now?", pipe)
check("right now -> current state", "sleeping" in a, a)
a = answer("Bantay, what happened today?", pipe)
check("today -> event log summary", "1 alert" in a, a)
a = answer("Bantay, did anything dangerous happen in the last 5 minutes?", pipe)
check("alerts in window", "alert" in a and "Choco" in a, a)
a = answer("Bantay, when did Choco start chewing?", pipe)
check("when -> timeline with clock times", ":" in a and "chewing" in a, a)
llm_reply["text"] = "Choco chewed a battery for 40 seconds."
a = answer("Bantay, is Choco okay to leave alone?", pipe)
check("LLM invents 'battery' -> rejected, safe fallback used", "battery" not in a, a)
check("guard keeps a faithful LLM answer", guard("Choco is chewing something; check on him.", "Choco chewing") is not None)
a = answer("Alright, can you generate me a html code?", pipe)
check("off-topic (code) is declined politely", a.startswith("Sorry, that's not in my scope"), a)
a = answer("Bantay, what's the capital of France?", pipe)
check("off-topic (trivia) is declined politely", a.startswith("Sorry"), a)
llm_reply["text"] = "OUT_OF_SCOPE"
a = answer("Bantay, can the dog help me with my thesis?", pipe)
check("LLM says OUT_OF_SCOPE -> polite decline", a.startswith("Sorry"), a)
llm_reply["text"] = "UNKNOWN"
a = answer("Bantay, did Choco bark at the neighbor?", pipe)
check("LLM says UNKNOWN -> can't-tell + suggestions", a.startswith("I can't tell"), a)
llm_reply["text"] = None
a = answer("Bantay, is the dog hungry?", pipe)
check("in-scope but no LLM -> can't-tell (no random summary)", a.startswith("I can't tell") or "Choco" in a, a)
from bantayaso.qa import strip_wake  # noqa: E402
for heard in ["Van Ty, what are my dogs doing?", "Ban tai what are my dogs doing", "bun tie, where are my dogs",
              "Ban-tay where are my dogs", "Pantay where are my dogs", "Hey Bantay, where are my dogs",
              "Bantai, where are my dogs", "Vantay where are my dogs"]:
    woke, rest = strip_wake(heard)
    check(f"wake word heard as {heard.split()[0]!r} / {heard[:12]!r}", woke and rest.lower().startswith(("what", "where")), rest)
woke, rest = strip_wake("Van Ty can you hear me?")
check("'Van Ty can you hear me?' -> wake + small talk", woke and answer("Van Ty can you hear me?", pipe).startswith("Yes, I can hear you"))
check("normal sentence is not a false wake", not strip_wake("What are my dogs doing?")[0])
a = answer("Bantay, is Bantay sleeping?", pipe)
check("yes/no: is Bantay sleeping -> yes", a.startswith("Yes"), a)
a = answer("Bantay, is Choco sleeping?", pipe)
check("yes/no: is Choco sleeping -> no + what instead", a.startswith("No") and "chewing" in a, a)
a = answer("Bantay, which dog is chewing?", pipe)
check("which dog is chewing -> Choco", "Choco" in a and "Bantay" not in a, a)
a = answer("Bantay, how long has Bantay been sleeping?", pipe)
check("how long -> duration", "sleeping for about" in a, a)
a = answer("Bantay, when did Choco last chew?", pipe)
check("when did X last chew -> time ago", "Choco was chewing" in a, a)
a = answer("Bantay, what was the last alert?", pipe)
check("last alert -> latest event", "last alert was a warning" in a, a)
check("'Bantay, Bantay, BantayAso.' -> wake with empty question", strip_wake("Bantay, Bantay, BantayAso.") == (True, ""))
check("hands-free strict: wake word mid-sentence is ignored", not strip_wake("I told Bantay to sit", strict=True)[0])
from bantayaso.voice_in import looks_hallucinated  # noqa: E402
check("repetition loop is filtered", looks_hallucinated("One, two, three, four, five, six, six, six, six, six, six"))
check("prompt echo is filtered", looks_hallucinated("My dogs are Oreo.", "My dogs are Oreo."))
check("normal question is kept", not looks_hallucinated("Bantay, what is Oreo doing?"))
a = answer("Bantay", pipe)
check("wake word alone -> prompt", a.startswith("Yes?"), a)

# teaching by voice
taught = []
pipe.classifier = types.SimpleNamespace(labels=["sitting", "lying down", "licking itself", "sleeping"],
                                        teach=lambda tid, lab: taught.append((tid, lab)))
pipe._actions = {1: None, 2: None}
a = answer("Bantay, Choco is sitting right now", pipe)
check("'Choco is sitting right now' teaches Choco=sitting", taught == [(1, "sitting")] and "remember" in a, a)
a = answer("Bantay, remember she is lying down", pipe)
check("'remember she is lying down' with 2 dogs and a known subject", "lying down" in a or "name" in a, a)
taught.clear()
a = answer("Bantay, is Choco sitting?", pipe)
check("a question does not teach", not taught, a)
a = answer("Bantay, Choco is sitting", pipe)
check("plain statement without 'now'/cue does not teach", not taught, a)
pipe._actions = {1: None}
a = answer("Bantay, that's licking", pipe)
check("one dog on camera: 'that's licking' teaches it", taught and taught[-1] == (1, "licking itself"), a)

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
