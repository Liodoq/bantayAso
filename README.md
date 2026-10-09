# BantayAso — Local AI Dog Watcher

A desktop app that watches your dog through a USB camera and **warns you out loud** when he goes from safe (sleeping, sitting) to risky (chewing, raiding the trash) to dangerous (mouth on a battery, cable or medicine). **All AI runs on your laptop. No video ever leaves your device, and it works offline.**

Built for the AppBuildersPH Hackathon 2026 (Local AI). Work in progress — setup: see [SETUP.md](SETUP.md). Full disclosures will be added before submission.

## Run
```powershell
.\.venv\Scripts\Activate.ps1
python -m bantayaso                         # desktop app (Monitor / Events / Zones / Settings, tray)
python -m bantayaso --source data\clips\rec_YYYYMMDD_HHMMSS.mp4   # test on a recorded clip
python -m bantayaso --cv                    # simple OpenCV developer window
python -m bantayaso --source http://<phone-ip>:8080/video   # phone as a wireless camera (IP Webcam app)
```
Desktop app pages: Monitor · Events · Dogs · Zones · Things · Settings. **F2** = ask Bantay by voice, **F12** = technical info on the video. Top bar: ● Record clip, 🎙 Bantay ▾ (ask, hands-free, voice, do not disturb, snooze), ☰ (light/dark mode, tray, quit). Closing the window keeps it running in the tray (right-click the paw icon → Quit).

**Label recordings (desktop):** Record clip → Stop recording → enter dog name(s), behavior and optional notes/timestamps → Save labels. Skip for now keeps the video. Use Menu → Label a recorded clip to label or edit older videos, including phone videos copied locally. Labels are saved beside the video as `.mp4.labels.json`; no upload or automatic training occurs.

OpenCV window keys: **R** record · **C** save last 10 s · **S** snapshot · **Z** zones · **H** object boxes · **M** mute · **N** do not disturb · **D** debug · **Q** quit

Tests (no camera/GPU): `python scripts\test_risk.py` · `python scripts\test_qa.py` · Measure a clip: `python scripts\calibrate_actions.py <clip.mp4>`

## Teaching actions

With the dog visible, click it → **This dog is actually…**, or ask **“Bantay, Oreo is sitting right now.”** Use the dog's enrolled name when several dogs are visible. Wait for **Saved 6 examples**; the first reply only acknowledges the lesson. Capture cancels if the dog disappears or its identity becomes uncertain. Questions such as “Is Oreo sitting?” do not teach.

Examples are local visual references, not retrained model weights. Teach at least two distinct actions; examples influence predictions only when they match clearly. Mouth/hazard detection stays independent of a corrected action label.

Repair regressions: `python scripts\test_repairs.py`. Bounded model check on an existing recording: `python scripts\check_inference.py data\clips\<clip>.mp4 --device cuda` (not an accuracy/FPS benchmark).

See [the training plan](docs/TRAINING_PLAN.md) for correcting motion measurements, collecting labeled video and evaluating temporal models, and [BRIEF.md](BRIEF.md) for the current cross-agent handoff and validation limits.

### Talking with Bantay

Use F2 to ask, or enable Bantay → Hands-free. After a completed answer, hands-free accepts a follow-up for eight seconds without repeating the wake word. F2 or Bantay → Stop speaking cancels ordinary speech (SAPI); danger alerts retain priority. Say “Bantay, keep it short” to shorten answers. Clock ranges are spoken as words instead of punctuation.

Bantay → Calm check-ins controls occasional resting updates, enabled by default. A check-in needs one minute of fresh, safe resting observations; it says “appears to be resting,” once per episode, with at least 15 minutes between episodes. Voice off and Do not disturb suppress speech. These changes do not train model weights or add reminders/occasions. Regression checks: `python scripts/test_conversation.py`.

### Scene questions and entry announcements

Ask “Where is Oreo?”, “Are all the dogs on the bed?”, “Is there a pillow?” or “Are there things beside them?”. Locations use the dog-box center inside drawn zones, otherwise left/middle/right. Object answers use fresh confirmed detections from the Things vocabulary; harmless items such as pillows stay excluded from hazard alerts. A missed detection is not proof that an object is absent. “Nearby” means close in the camera image, not measured physical distance.

Bantay → Announce dogs entering is on by default and can be disabled. New camera/zone dog counts must persist 1.5 seconds; losses must persist five seconds before another entry. Startup describes visible dogs; reconnect resets the baseline. Announcements appear in the status line and use voice/toasts when enabled, respecting DND and danger priority. Count-based detection avoids tracker-ID flicker but cannot detect a same-count dog replacement. These notices are not Warning/Danger events in the Events database.

### Friendly replies and behavior recaps

Try “Okay good, thanks”, “Good job”, or “I appreciate it” for varied acknowledgements. Ask “Did Oreo do something bad today?” or “What did Oreo do earlier?” for a named-dog recap. Bantay separates today's recorded alerts from the rolling ten-minute activity history and states coverage limits. Calm recaps may add gentle dry humor; hazards stay factual. These bounded replies currently use English templates and preserve coverage details even with short-answer style. Ordinary chewing is not automatically called bad behavior. No full-day activity memory or new model training is implied.

### Ask about the application

Try “What is in your screen?” (camera observations), “Read the events”, “Read more events”, “List all things”, “More things”, “List the dogs”, “What areas are configured?”, “Read my settings”, or “How many saved clips?”. This is read-only access to Bantay's structured state and stored events, not unrestricted desktop access. Events are read three at a time from the most recent 100 matching non-false-alarm records, optionally filtered by named dog or today. Things are read eight at a time and explicitly separated from currently visible objects.

Unknown questions get a short helpful redirect. Identical fallback speech is suppressed for 30 seconds. Audio capture now allows a 0.85-second pause and up to 12 seconds per utterance, uses a lower onset floor and bounded gain/DC removal, and reports audio that is too quiet. Real microphone accuracy still needs testing in the room; no speech model or language configuration was changed.
