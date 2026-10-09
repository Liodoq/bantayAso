# Bantay companion training program

Updated October 10, 2026 (Asia/Manila). Owner goal: a calm, helpful companion that knows what the dogs are doing, answers naturally and speaks at useful moments. This extends TRAINING_PLAN.md. It is a training and implementation specification, not a deployed companion feature or a completed model fine-tune.

## Current starting point

The latest repository includes Claude's persona/style preferences, four-turn conversation context, grounded Ollama chat, video-based teaching, low-light detector changes and camera-following zones. The newest handoff reports zone tests; those tests were not rerun for this document. Live low-light detection remains unverified.

Read-only inspection found 81 saved action embeddings: chewing 12, eating 10, lying 6, scratching 6, sitting 11, sleeping 36. All six pass the current reference gates. ACTIVE means eligible for matching; it is not measured accuracy, and embeddings from one recording are not independent lessons.

Five owner-labeled clips exist: Brownie lying; two Oreo sleeping; Oreo chewing; and Oreo lying/scratching/licking. All have empty timing notes. These labels have not been visually verified in this session. Do not ingest the mixed clip as one action or assume a label applies to every frame. Current examples have no reliable source provenance, so collect entirely new sessions for final testing.

The persona suite passes 24 mocked checks with temporary files inside the workspace. It does not establish real Ollama quality, microphone recognition or voice latency. Default speech configuration is still English (`base.en`, `en`) because YAML supplies no override. Current chat waits for a complete response and may make a second translation request. The listener pauses during speech output. There is no established proactive companion scheduler or occasion/reminder store.

## 1. Teach reliable observations

First fix the motion-measurement limitations documented in TRAINING_PLAN.md. Do not make uncertain perception sound more confident by improving the voice alone.

Owner exercise, about 20 minutes per collection session:

1. Record 10–20 seconds of a naturally occurring behavior using the monitoring camera. Start with sitting, standing, lying awake, walking, licking, scratching and safe chewing/eating. Include empty scenes and partially hidden dogs.
2. Stop and label dog, behavior, session and exact interval. Example notes: `session morning-02; lying 2–7 s; scratching 8–11 s; face hidden`. For the existing mixed clip, review it and identify separate intervals before teaching.
3. Collect 3–5 independent takes per behavior as a pilot, over several sessions and views. Repeated frames from one take do not count as extra sessions. Expand according to observed errors, with the earlier 20–30-episode target as a starting collection goal.
4. Hold back fresh sessions before extracting features. Do not teach on them later and continue calling them test footage. Keep original videos and sidecar labels backed up outside Git.
5. With Bantay closed, use the existing `scripts/teach_from_video.py` only on verified intervals and the correct visible dog. Biggest/left/right selection can switch subjects in multi-dog footage: inspect selection or use single-dog clips. Back up existing action arrays before adding examples. Restart and compare on withheld footage.

Immediate method: improve saved examples. Next measured experiment: frozen CLIP classifier, then a small causal temporal head, using the splits and release checks in TRAINING_PLAN.md. No model weight training should be claimed for storing embeddings.

Sleep policy: lying still does not establish sleep. Default to “resting” when eyes/head are hidden or evidence is weak. “They appear to be sleeping” needs sustained supporting observations. “All the dogs” additionally requires reliable coverage of every enrolled dog; otherwise say “the dogs in view.” No alerts is not proof that everything is safe.

## 2. Teach when to speak

Implement a deterministic event policy before asking an LLM to phrase updates. Proposed initial settings below are adjustable targets, not current behavior:

- A new quiet/resting episode must persist for 60 seconds with fresh, supported observations. Speak once: “Oreo and Brownie are resting. No new alerts.” Only use those names and that alert claim when supported by the current state and log.
- Suppress repeats within the same episode. Allow another calm update only after a meaningful change and at least 15 minutes since the last unsolicited calm update. Do not schedule chatter every 15 minutes.
- If a tracked dog gets up after a sustained rest, an optional update is “Oreo is up and walking now.” Require a stable transition, not one changing frame label.
- Suppress casual updates while the owner speaks, Bantay speaks, an alert is active, DND/quiet hours are active, or camera/identity observations are stale. Expire queued casual messages; recheck facts immediately before speaking.
- Keep existing danger rules independent. One speech arbiter orders danger alerts, direct replies, due reminders and casual updates. Never let an ordinary reply marked urgent clear a pending danger alert. Preserve the owner's DND policy; do not silently add an override.
- Store why each candidate update was spoken or suppressed for evaluation. Use monotonic time for cooldowns and freshness; use timezone-aware wall time for schedules.

A camera does not establish that the room is acoustically quiet. Say “the dogs are resting,” not “everything is quiet,” unless a separately tested audio signal supports it.

## 3. Teach natural conversation

Use the existing persona and local model first. Fine-tuning is a later option if prompting, facts and turn-taking still fail on measured cases.

The conversation contract:

- Warm, short, direct: normally one or two sentences. Follow the owner's explicit language/tone preference. Keep greetings available for social exchanges without adding them to every status answer.
- Feed structured facts with observation timestamps, source, visibility, identity uncertainty, activity duration and event IDs. Recent conversation helps resolve “he”; it must not turn an old observation into a current fact.
- Follow-ups such as “How long?” retain the dog/topic only while unambiguous. Ask “Oreo or Brownie?” when necessary. Corrections to conversation must not silently save training examples.
- A proposed eight-second follow-up listening window after a wake-word exchange permits a second question without repeating “Bantay.” Show listening state; close on timeout, stop command or mute. Test background TV and other speakers before enabling by default.
- First implement click/push-to-talk interruption that cancels old playback and generation. True hands-free interruption needs echo handling; the current listener pauses during TTS and cannot provide this as-is.
- Instrument end of owner speech, transcription complete, answer ready and first audible output. Initial targets: median simple-status response under 2 seconds, p95 under 4 seconds on the actual laptop. Report actual values; fail gracefully with a factual template if Ollama is slow/unavailable.
- Avoid an unnecessary second LLM translation call. Prefer a single validated answer in the requested language. Streaming can improve responsiveness, but never speak partial unvalidated JSON or a sentence whose factual check has not finished.

Multilingual Whisper is needed for spoken Tagalog/Taglish; benchmark the already-cached `base` with automatic language detection before changing defaults. Verify dog-name transcription in noise. English-only `.en` and multilingual variants are distinguished in the [official Whisper model documentation](https://github.com/openai/whisper#available-models-and-languages).

Ollama supports [streaming](https://docs.ollama.com/capabilities/streaming) and [structured outputs](https://docs.ollama.com/capabilities/structured-outputs). A valid JSON schema constrains structure, not truth. Validate names, times, activities, negations and provenance; the current numbers/objects guard alone is insufficient.

## 4. Removed from scope

Owner explicitly excluded routines and occasions. No scheduler, birthdays or reminders will be built in this batch.

## 5. Training examples and evaluation

`companion_scenarios.jsonl` contains 19 authored development scenarios: supported rest/sleep, absent dogs, ambiguity, hazards, DND, repeated updates, speech overlap, language, missing history and conversation control. They are synthetic specifications, not real observations, not a benchmark already passed and not runtime-loaded training data. Use them as prompt-development cases and manual acceptance scripts.

For each scenario, replay the supplied facts and owner turn into the candidate implementation. Record the actual answer, speech/silence decision, tool/persistence result, latency and reviewer verdict. Match meaning rather than exact wording. Do not score desired answers against themselves. Cases requiring a new scheduler must remain “not implemented” until that integration exists.

After development, collect 50 fresh owner conversations over at least three sessions, including new phrasings. Reserve them for evaluation; keep corrections used for prompting in a separate development set. These counts are pilot targets, not proof of reliability. Assess factual grounding, correct dog/topic, appropriate speech timing, false wake-ups, reminder persistence/cancellation, interruption and language comprehension. Critical failures include fabricated sleep, wrong dog, repeated calm chatter, speaking during DND, lost danger alerts and claiming an unsaved reminder exists. All must pass the scripted suite; also report observed failure rates on fresh sessions.

If fine-tuning later becomes worthwhile, curate owner-approved facts/question/answer examples with source IDs and explicit unknowns; split by conversation session. Train a small adapter offline, compare against the unchanged prompt baseline, and keep rollback. Do not fine-tune on unreviewed model guesses or synthetic cases alone and call that dog recognition training.

## Implementation order and release gates

1. **Trustworthy state:** fresh observation snapshot, uncertainty and corrected motion; structured interval labels/provenance. Exit: jitter, lost camera and occlusion cannot create confident calm/sleep claims.
2. **Smooth replies:** one bounded answer path, measured latency, explicit subject handling and cancellation. Exit: fresh owner exchanges succeed with camera/voice active; no old answer spoken after cancellation.
3. **Companion timing:** event policy and speech arbiter; default quiet setting with owner controls. Exit: one calm update per episode, no DND speech, danger priority survives concurrent questions.
4. Routines and occasions removed by owner.
5. **Model comparison and release:** held-out video/conversation evaluation, 30-minute live soak, measured CPU/GPU memory and response latency, then a versioned release with rollback. Keep the hackathon code-freeze requirements in BRIEF.md separate from this longer-term roadmap.

Original training-pack session changed documentation only; see the implemented conversation batch below for subsequent runtime changes. Next implementation should start with trustworthy state and speech priority, then add calm check-ins; do not enable unsolicited sleep announcements on today's unvalidated action labels.

## Implemented conversation batch — October 10

Owner prioritized smooth replies and timely interaction, explicitly excluding routines/occasions. Added natural speech clock formatting and sentence-based history ranges; hands-free eight-second follow-up window after a completed reply; menu Stop speaking and F2 cancellation of ordinary speech; non-overlapping answer generation; expired/ambiguous subject handling; stale camera reply guard; single translation pass for already-styled LLM answers; bounded model requests; and speech priority that preserves danger alerts.

Calm check-ins are enabled by default (Bantay menu toggle, persisted on change), require 60 seconds of fresh visible still/lying-or-sleeping action observations with Safe assessments, and say “appears to be resting.” No acoustic silence, certain sleep or all-dogs claim is inferred. Once per episode, with 15-minute minimum separation between different episodes; no periodic timer chatter. DND/voice off, listening, reply generation, fresh camera gaps and active alerts suppress check-ins; queued check-ins expire after two seconds and revalidate before/during speech. This remains dependent on imperfect visual predictions, not newly trained accuracy.

Sixteen focused regression tests pass, including mocked SAPI completion/purge. Existing 43 QA, 23 repair, 24 persona, 26 Zones UI and 10 recording-label checks pass. Real microphone, SAPI audio quality and Ollama latency are still unverified. SAPI supports immediate ordinary-speech cancellation; the pyttsx3 fallback remains blocking and may finish the current sentence. Voice interruption while TTS is playing still requires F2 or the menu (no acoustic echo-cancellation implementation). No weights were trained. UI and config defaults do not change the existing microphone language setting.
