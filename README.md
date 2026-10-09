# BantayAso — Local AI Dog Watcher

A desktop app that watches your dog through a USB camera and **warns you out loud** when he goes from safe (sleeping, sitting) to risky (chewing, raiding the trash) to dangerous (mouth on a battery, cable or medicine). **All AI runs on your laptop. No video ever leaves your device, and it works offline.**

Built for the AppBuildersPH Hackathon 2026 (Local AI). Work in progress — setup: see [SETUP.md](SETUP.md). Full disclosures will be added before submission.

## Run (current build)
```powershell
.\.venv\Scripts\Activate.ps1
python -m bantayaso                         # live from the action camera
python -m bantayaso --source data\clips\rec_YYYYMMDD_HHMMSS.mp4   # test on a recorded clip
```
Keys (debug view **D** also shows each dog's action, mouth score and motion): **R** record clip · **C** save last 10 s · **S** snapshot · **Z** draw zones · **H** hazard boxes · **D** debug info · **Q** quit

Tests (no camera/GPU): `python scripts\test_risk.py`
