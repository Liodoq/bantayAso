"""Offline checks for how Bantay talks (no Ollama needed: the LLM is faked).

    python scripts\\test_persona.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso.persona import Persona, detect_language, shape   # noqa: E402
from bantayaso.qa import guard                                  # noqa: E402

fails = 0


def check(name, cond, info=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -> {info}" if info else ""))
    fails += 0 if cond else 1


d = Path(tempfile.mkdtemp())
check("English detected", detect_language("Bantay, what is Oreo doing?") == "en")
check("Tagalog detected", detect_language("Bantay, ano ang ginagawa ng aso ko ngayon?") == "tl")
check("Taglish detected", detect_language("Bantay, is Oreo okay lang ba?") == "taglish",
      detect_language("Bantay, is Oreo okay lang ba?"))

p = Persona(d)
for q in ["ano ginagawa ni Oreo?", "nasaan si Choco ngayon?", "kumain na ba sila?", "okay lang ba sila?"]:
    p.style.observe("Bantay, " + q)
lang, length, tone = p.style.effective()
check("learns Tagalog/Taglish from how the owner talks", lang in ("tl", "taglish"), lang)
check("short questions -> short answers", length == "short", length)
for _ in range(4):
    p.style.observe("Bantay, ano po ginagawa ng aso ko?")
check("'po' -> formal tone", p.style.effective()[2] == "formal")

check("command: keep it short", p.style.command("Bantay, keep it short") and p.style.length == "short")
check("command: more details", p.style.command("Bantay, more details please") and p.style.length == "detailed")
check("command: speak English", p.style.command("Bantay, speak English") and p.style.lang == "en")
check("a question is not a command", p.style.command("Bantay, what is Oreo doing?") is None)
check("preferences are saved", json.loads((d / "style.json").read_text())["lang"] == "en")
p2 = Persona(d)
check("...and reloaded next start", p2.style.lang == "en" and p2.style.length == "detailed")
check("reset", p2.style.command("Bantay, reset my preferences") and p2.style.lang == "auto")

long = "Oreo was lying down for about 8 minutes. Then he walked around for 1 minute. No alerts today."
check("shape short keeps whole sentences", shape(long + " " + long, "short").endswith("."), shape(long + long, "short"))
check("shape detailed keeps everything", shape(long, "detailed") == long)

# grounded LLM call: structured JSON reply
seen = {}


def fake(msgs, schema, n):
    seen["msgs"], seen["schema"] = msgs, schema
    return json.dumps({"status": "answer", "answer": "Oreo has been lying on bed 1 for about 8 minutes."})


p3 = Persona(tempfile.mkdtemp(), llm_call=fake)
st, ans = p3.ask_llm("is oreo behaving?", "Now: Oreo is just lying down on bed 1. Last 10 minutes: lying down for about 8 minutes.")
check("LLM answer parsed from JSON", st == "answer" and "Oreo" in ans, ans)
check("facts sent as bullets + style directive", "- Now: Oreo" in seen["msgs"][-1]["content"]
      and "Style:" in seen["msgs"][-1]["content"])
check("schema requested", seen["schema"]["required"] == ["status", "answer"])
p4 = Persona(tempfile.mkdtemp(), llm_call=lambda m, s, n: '{"status": "out_of_scope", "answer": ""}')
check("out_of_scope status", p4.ask_llm("write a poem", "Now: calm.")[0] == "out_of_scope")
p5 = Persona(tempfile.mkdtemp(), llm_call=lambda m, s, n: "Oreo is sleeping.")
check("plain-text reply still works", p5.ask_llm("q", "Now: Oreo is sleeping.") == ("answer", "Oreo is sleeping."))

# follow-up memory
p3.finish("is oreo behaving?", ans)
p3.ask_llm("and how long?", "Now: calm.")
check("recent turns go to the LLM for follow-ups", "Owner: is oreo behaving?" in seen["msgs"][-1]["content"])

# Tagalog finishing keeps facts, rejects invented numbers/objects
tl = Persona(tempfile.mkdtemp(), llm_call=lambda m, s, n: json.dumps({"text": "Nakahiga si Oreo sa kama ng mga 8 minuto."}))
tl.style.lang = "tl"
out = tl.finish("q", "Oreo was lying on the bed for about 8 minutes.", guard)
check("Tagalog rewrite used when facts are kept", out.startswith("Nakahiga"), out)
bad = Persona(tempfile.mkdtemp(), llm_call=lambda m, s, n: json.dumps({"text": "Nakahiga si Oreo ng 20 minuto malapit sa battery."}))
bad.style.lang = "tl"
out = bad.finish("q", "Oreo was lying on the bed for about 8 minutes.", guard)
check("rewrite that changes numbers/adds objects is rejected", out.startswith("Oreo was"), out)
none = Persona(tempfile.mkdtemp(), llm_call=lambda m, s, n: None)
none.style.lang = "tl"
check("LLM offline -> English template stays", none.finish("q", "Oreo is sleeping.", guard) == "Oreo is sleeping.")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
