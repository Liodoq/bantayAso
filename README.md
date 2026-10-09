# BantayAso — Local AI Dog Watcher

A desktop app that watches your dog through a USB camera and **warns you out loud** when he goes from safe (sleeping, sitting) to risky (chewing, raiding the trash) to dangerous (mouth on a battery, cable or medicine). **All AI runs on your laptop. No video ever leaves your device, and it works offline.**

Built for the AppBuildersPH Hackathon 2026 (Local AI). Work in progress — setup: see [SETUP.md](SETUP.md). Full disclosures will be added before submission.

## Run
```powershell
.\.venv\Scripts\Activate.ps1
python -m bantayaso                         # desktop app (Monitor / Events / Zones / Settings, tray)
python -m bantayaso --source data\clips\rec_YYYYMMDD_HHMMSS.mp4   # test on a recorded clip
python -m bantayaso --cv                    # simple OpenCV developer window
```
Desktop app: **F12** shows technical info on the video. Closing the window keeps it running in the tray (right-click the paw icon → Quit).

OpenCV window keys: **R** record · **C** save last 10 s · **S** snapshot · **Z** zones · **H** object boxes · **M** mute · **N** do not disturb · **D** debug · **Q** quit

Tests (no camera/GPU): `python scripts\test_risk.py` · Measure a clip: `python scripts\calibrate_actions.py <clip.mp4>`
