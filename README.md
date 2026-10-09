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
