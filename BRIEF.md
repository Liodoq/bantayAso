# BantayAso — Local AI Dog Watcher

> **Handoff file.** This one document is the spec, the build plan and the live build status.
> Any AI agent or chat continuing this project: **read the whole file first, then jump to "Build Status & Handoff" at the bottom.**
> After every batch, update the bottom sections (status table, decisions, known issues, file map, next step) before stopping.
>
> - Hackathon: AppBuildersPH Hackathon 2026, theme **Local AI**. Solo builder: John Leorick Salalila.
> - Build day: Oct 9, 2026, 2:30 PM. **Submission deadline: Oct 10, 10:00 AM (no extensions, code freezes, one submission only).**
> - Demo Day: Oct 10, 1:00 PM–7:00 PM, Cyberzone, SM Makati. 5-minute live pitch and demo, plus 3 minutes of Q&A. **The real dog will be there.**
> - Code location: `C:\Users\johnl\Desktop\projectayiks\bantayaso\` (this file is `bantayaso/BRIEF.md`). A copy is also in the claude.ai Project as `claude/bantayaso.md`.
> - Working name "BantayAso" ("bantay" = guard, and a classic Filipino dog name). It can be renamed.

---

## Work Description

Develop a **desktop Local AI pet-safety monitor** for dog owners who work at a computer and can't watch their dog. It uses Python, OpenCV, Ultralytics YOLO (dog and hazard-object detection), a zero-shot CLIP/SigLIP action classifier, a local vision-language model through Ollama, offline text-to-speech, and a PySide6 desktop UI. An **action camera** acts as the home CCTV and feeds the laptop. Every frame is analyzed **on the laptop's GPU and never leaves the device**. The owner gets a spoken alert and a popup when the dog moves from safe behavior (sleeping, sitting) to risky behavior (chewing, digging, raiding the trash) or dangerous behavior (mouth near a battery, cable, medicine or plastic bag). The app runs in the background, and opening the live view is optional.

---

## Requirements

### Technology

- The app must be **Python 3.11** with **PyTorch (GPU)**, **OpenCV**, **Ultralytics** (YOLO11 + YOLOE/YOLO-World), **open_clip or transformers SigLIP**, **Ollama** (local VLM), **pyttsx3** (Windows SAPI offline TTS, with Piper as an optional upgrade), **PySide6** (UI + system tray) and **SQLite** (event log).
- **All AI inference must run locally.** After the one-time model download during setup, the app must work with the internet unplugged. This must be demonstrated in the demo video.
- The GPU must be used when present (`cuda` for NVIDIA, otherwise DirectML/CPU fallback). The device in use must be logged at startup and visible in the debug overlay.
- Settings (camera source, zones, hazard vocabulary, thresholds, voice on/off, cooldowns) must live in `config.yaml` and be editable from the UI.
- The code must be a Python package `bantayaso/` with one module per stage (see the file map) so batches can be built and tested separately.

### Camera Input

- The app must accept three source types through one `FrameSource` interface: **(a)** a USB/UVC device index (action camera in webcam mode, or a laptop webcam), **(b)** a network stream URL (RTSP/HTTP/MJPEG from the action camera over Wi-Fi), and **(c)** a **video file** (for testing and as a demo backup).
- Frames must be read on a background thread with a queue of size 1 (always the latest frame) so slow inference never builds lag.
- The app must reconnect automatically if the stream drops, and show "Camera disconnected" without crashing.
- A "Record clip" button must save the last N seconds to `data/clips/` for building test videos.

### Perception Pipeline (the Local AI)

Layered so that cheap models run on every frame and expensive models run only when needed:

1. **Dog detector — every frame.** YOLO11 (COCO class `dog`) gives a bounding box and track ID per dog (Ultralytics tracking). It must reach at least **15 FPS** on the laptop GPU.
2. **Hazard object detector — every 3rd frame (configurable).** Open-vocabulary **YOLOE or YOLO-World** with a text vocabulary from `config.yaml`. Defaults: `battery, phone charger cable, electric cord, medicine pill, pill bottle, plastic bag, slipper, shoe, sock, remote control, trash can, chocolate, toy`. Each item maps to a hazard tier.
3. **Action classifier — on the dog crop, about 4 times per second.** Zero-shot CLIP/SigLIP compares the crop with text prompts: `sleeping, lying down, sitting, standing, walking, sniffing the floor, chewing something, eating, digging, scratching itself, scratching furniture, jumping on furniture`. The output is the top label plus a confidence, smoothed over a 2-second rolling window.
4. **Motion energy — every frame.** Frame differencing inside the dog box gives an activity level (still / active / frantic). It is used to support "destroying/digging/scratching" and to cut false alarms on a sleeping dog.
5. **Optional dog-pose (stretch).** A YOLO11-pose model fine-tuned on the Ultralytics **dog-pose** dataset gives head/muzzle keypoints for more precise "mouth near object". Without it, the muzzle is estimated as the head end of the dog box (the edge nearest the hazard).
6. **VLM confirmation — only on Warning/Danger.** A local Ollama vision model (default `moondream` (fits 4 GB VRAM alongside YOLO/CLIP); upgrade to `qwen2.5vl:3b` if VRAM allows) receives the frame plus a prompt. It must return strict JSON: `{"dog_activity": str, "object_in_mouth_or_near": str|null, "risk": "safe|watch|warning|danger", "say": str}`. It runs **asynchronously** so the video never freezes, and its sentence becomes the spoken alert.

### Risk Engine

- Risk must be computed by a **transparent rule engine** (`risk.py`) from the detector, zones, action and motion outputs. The VLM may upgrade or downgrade one level but never decides alone.
- The four levels and their default rules:

| Level | Meaning | Example triggers |
|---|---|---|
| 0 **Safe** (green) | Resting | sleeping, lying, sitting, low motion, outside all zones |
| 1 **Watch** (yellow) | Curious | sniffing the floor; entering a zone; near a low-tier object |
| 2 **Warning** (orange) | Doing something wrong | chewing/eating near a shoe, slipper or sock; inside the trash zone; digging; scratching furniture; frantic motion inside a zone for 3 s or more |
| 3 **Danger** (red) | Possibly harmful | muzzle within X px of a **high-tier hazard** (battery, cable, pill, plastic bag, chocolate); "chewing/eating" while a high-tier hazard was last seen at that spot; inside a "danger zone" while chewing |

- A level must persist for **at least 1.0 s** (configurable) before it fires, to stop flicker. Each level has its own alert cooldown (defaults: Warning 20 s, Danger 5 s).
- **Zones:** the user draws polygons on the live view (types: `trash`, `danger`, `no-go`, `bed`). They are saved to `config.yaml`. A dog inside a `bed` zone biases toward Safe.
- **"Last seen" memory:** when a hazard object disappears while the dog's muzzle is on it, the engine must treat it as "possibly in mouth" for 5 s. This is what covers "the battery is too small or hidden to see."

### Alerts and Voice

- On Warning or Danger, the app must: speak a short alert (the VLM sentence if it's ready within 3 s, otherwise a template such as "Warning! Your dog is chewing something near the trash can."), show a Windows toast notification, flash the tray icon in the level's color, and save an event.
- An optional **"talk to the dog"** mode plays a recorded owner voice clip ("No!", "Leave it!") through the speakers on Danger.
- Voice can be muted. A "Do not disturb" mode keeps logging without speaking.
- TTS must be fully offline (pyttsx3/SAPI by default). Filipino/Taglish phrasing in templates is a nice touch.

### Event Log

- Every Warning or Danger event must be saved to SQLite `data/events.db` with: timestamp, level, action label, objects nearby, zone, VLM sentence, and a snapshot JPEG path (`data/snapshots/`).
- The UI must show a timeline of today's events with thumbnails. Clicking one shows the snapshot and its details.
- **Stretch:** an end-of-day summary written by a local text LLM via Ollama (e.g. "3 warnings today, mostly the trash can around 3 PM").

### Desktop UI (PySide6)

- **Look (approved mockup: `docs/ui-mockup.html`):** a dark brown-and-black theme via a Qt stylesheet. Tokens: bg `#0E0B09`, surface `#17110D`, panel `#211812`, raised `#2B2019`, line `#3A2C22`, brown `#8B5E3C`, caramel accent `#C8894E`, cream text `#F2E6D8`, muted `#A8957F`. Risk colors are used **only** for levels: safe `#7FB98A`, watch `#E3B54F`, warning `#E3803F`, danger `#E0533F`. Font: Segoe UI. Left sidebar (Monitor, Events, Zones, Settings). **No spec-narrating text on screen** (no "Offline/on-device", GPU name, resolution, FPS or "saved locally" labels): only content about the dog.
- **Monitor screen:** live video with zone and box overlays, a 4-step risk meter, a status card (level pill, activity, zone, duration, spoken sentence, "I've got it" / "Snooze 5 min"), a recent-events list, and top toggles (Voice, Do not disturb, Minimize to tray).
- **Events screen:** stat cards (time watched, danger count, warning count, hotspot zone), level filter chips, a timeline with thumbnails, an alerts-by-hour bar chart, and a selected-event panel (snapshot, level, activity, nearby object, zone, "Local AI says" sentence, Open snapshot / Mark false alarm).
- **Main window:** a live view with overlay boxes (dog, hazard objects, zones, current level badge, action label, FPS, GPU device), a risk-level bar, an event timeline, and Start/Stop.
- **Background mode:** closing the window minimizes to the **system tray** and monitoring continues. The tray menu has Show live view, Mute, Do not disturb and Quit.
- **Settings page:** camera source picker and test, zone editor, hazard vocabulary editor with tiers, thresholds, voice options.
- Technical info (FPS, GPU device, VLM state) appears only in a **debug overlay that is off by default** (Settings toggle or F12), used to measure and demo performance honestly when asked. The privacy and offline story is told in the pitch and README, not by labels in the UI.
- **Stretch:** a LAN-only phone view (a FastAPI MJPEG stream and event list on `http://<laptop-ip>:8000`) so a phone on the same Wi-Fi can watch without any cloud.

### Performance

- At least 15 FPS live overlay with the dog detector on the target laptop's GPU. Measure this honestly and show it in the debug overlay (**no fake benchmarks**, which can disqualify).
- Under 1.5 s from behavior start to Warning/Danger alert without the VLM. The VLM sentence follows asynchronously.
- No memory leak over a 30-minute run. VLM calls are capped at one in flight at a time.

### Demo Day Setup (with the real dog)

- The action camera on a small tripod or mount, powered by USB (no battery runs out), framing a 2×2 m "demo corner": a dog bed (bed zone), a small trash bin (trash zone), and a "danger tray".
- **Dog safety rule: never let the dog actually reach a real battery, medicine or live cable.** Use props: a large **dummy or dead battery taped inside a clear sealed bag**, an **unplugged** charger cable, an empty pill bottle with the cap glued, a slipper. Hold the leash and intervene. Judges will respect this.
- A recorded **backup video** of the dog doing each behavior, loaded through the video-file source, in case the dog won't cooperate or the venue Wi-Fi interferes. A USB cable to the camera is preferred over Wi-Fi at the venue.
- Show the **internet unplugged / Wi-Fi off** during the demo.

### Out of Scope

- Cloud inference of any kind, user accounts, or cloud video storage.
- A native mobile app with on-device models (pitched as a roadmap item; a LAN phone view is the stretch substitute).
- Training a custom action model from scratch (only zero-shot plus an optional small pose fine-tune).
- Multi-camera support, multi-pet identity recognition, automatic treat dispensers or other hardware actuation.
- Medical or poisoning advice beyond "contact your vet."

---

## Provided Material

- The user's own dog (live at Demo Day) and an **action camera** used as CCTV (model to be confirmed; see open questions).
- The user's laptop **with a GPU** (vendor/VRAM to be confirmed).
- Hackathon briefing: AppBuildersPH Hackathon 2026 Participant Briefing (rules, judging criteria, submission checklist summarized below).
- Pretrained open-source models (downloaded once at setup): Ultralytics YOLO11n/s, YOLOE or YOLO-World v2, OpenCLIP ViT-B/32 or SigLIP base, Ollama `qwen2.5vl:3b` (or `gemma3:4b` / `moondream`), optionally the Ultralytics dog-pose dataset.

---

## Deliverables

1. A **public GitHub repository** (public before 10:00 AM Oct 10) containing the `bantayaso/` package, `config.yaml`, `requirements.txt`, `scripts/` (setup, model download, self-test), and **no** model weights, clips or `data/` contents (gitignored).
2. A **README.md** with: a one-paragraph description, setup steps a judge can follow (Python, CUDA PyTorch, `ollama pull`, `python -m bantayaso`), how to run on a sample video, what runs locally and what needs internet, and the full disclosures list.
3. A **demo video of about 1 minute**: the problem, the dog safe, then chewing near the trash (Warning), then nosing the prop battery (Danger and voice alert), the timeline, and Wi-Fi off the whole time.
4. An **X or LinkedIn video post** tagging Devin / Cognition with **#AppBuildersPH**.
5. Submission form answers (draft in "Submission Answers" below), submitted **once** on the Cerebral Valley event page.

---

## Client Request

The project is for a dog owner who spends hours at a computer with his back to his dog. The moment he turns around, the dog is already chewing or eating something, and sometimes it's something dangerous like a battery, a cable or medicine. He wants a camera that watches the dog for him and tells him right away, by voice, how serious it is, without paying for a cloud subscription and without home video leaving his computer.

Specifically, the request was:

- Use an action camera as the home CCTV and run everything on the laptop with a GPU.
- Detect what the dog is doing and how dangerous it is, from just sitting, to eating something, to destroying things, to eating batteries.
- Detect the objects around the dog, not just the dog itself.
- Talk: speak an alert out loud while running in the background.
- Make watching the camera optional while working on the PC.
- Keep the data on the PC only; the AI must work offline.
- It must work live at Demo Day with the real dog, and be resumable across AI chats batch by batch.

---

## Reasons

- **Local AI is fundamental, not a feature.** Continuous video analysis at 15+ FPS all day would be expensive and slow through a cloud vision API, and it means streaming the inside of your home to a server. Running locally makes it free, private, real-time and offline-capable. This maps directly to the 25% Local AI criterion.
- **It solves a real, relatable problem** for a clear user (pet owners who work at a computer: WFH, students, freelancers). Ingesting batteries, cords or medicine is a genuine emergency risk for dogs.
- **Layered perception is a solid engineering design:** a fast detector on every frame, open-vocabulary hazard detection, zero-shot action recognition, motion cues, and an expensive VLM only on escalation. This shows judges a deliberate trade-off between speed and intelligence on edge hardware.
- **It is honest about model limits.** Tiny objects like batteries are handled through "last seen near muzzle" memory, zones and VLM confirmation instead of pretending to detect the battery perfectly. Rules are transparent and tunable.
- **No custom training is needed to start** (open-vocabulary detection plus zero-shot CLIP), so it is buildable in 20 hours, with an optional dog-pose fine-tune as a stretch.
- **A live demo with the real dog** plus a recorded backup and Wi-Fi visibly off makes a convincing Product & Demo score.
- **It fits the organizer's own philosophy** (Bryl Lim's Tarsi): privacy-first, offline-first, no account, does one thing well, and has personality (the voice).

---

## Submission Answers (draft, finalize in Batch 7)

- **Why does this product benefit from running AI locally?** A pet camera has to watch continuously and react in about a second. Sending every frame to a cloud AI would cost money every minute, add network delay exactly when the dog is about to swallow something, stop working when the internet drops, and stream the inside of your home to someone else's server. BantayAso runs detection, action recognition, the vision-language model and the voice entirely on the laptop's GPU, so it is free to run 24/7, real-time, offline, and private: video never leaves the device.
- **What runs locally:** camera capture, YOLO11 dog detection and tracking, YOLOE/YOLO-World hazard detection, CLIP/SigLIP action classification, the risk engine, the Ollama VLM explanation, TTS voice, the event log and the UI.
- **What requires internet:** only the one-time install (pip packages, model weights, `ollama pull`). Nothing at runtime.
- **Models:** (fill exact names/versions) YOLO11n, YOLOE-11s or yolov8s-worldv2, OpenCLIP ViT-B-32 (laion2b) or SigLIP base, Ollama qwen2.5vl:3b, Windows SAPI voices / Piper.
- **Technologies:** Python, PyTorch, OpenCV, Ultralytics, open_clip/transformers, Ollama, PySide6, SQLite, pyttsx3.
- **APIs/cloud services:** none at runtime.
- **Existing code/assets:** none (built during the hackathon); open-source libraries and models as listed.
- **AI development tools:** Claude (planning and code generation), plus any others actually used.

---

## Build Plan (batches)

Times are targets. Every batch must end with: the code runs, the batch's self-test passes, and this file is updated.

| # | Batch | Target | Done when (acceptance) |
|---|---|---|---|
| 0 | **Setup & skeleton** | 2:30–3:30 PM | venv; GPU torch confirmed (`scripts/check_env.py` prints the GPU name); Ollama installed and `moondream` pulled (try `qwen2.5vl:3b` too); repo skeleton + `config.yaml` + `.gitignore`; the action camera is readable from OpenCV (index or URL found); public GitHub repo created |
| 1 | **Camera + dog detection + live view** | 3:30–5:30 PM | `FrameSource` (device/URL/file), threaded latest-frame reader, YOLO11 dog boxes + tracking, FPS/device overlay in a simple OpenCV window, clip recorder. **Record test clips of the dog tonight** |
| 2 | **Hazard objects + zones + risk engine v1** | 5:30–8:30 PM | YOLOE/YOLO-World vocabulary from config with tiers; zone polygons (editor in the OpenCV window first); muzzle-near-object logic; "last seen" memory; levels 0–3 with persistence and cooldown; levels shown on the overlay; tested on recorded clips |
| 3 | **Action classifier + motion** | 8:30–10:30 PM | CLIP/SigLIP zero-shot on the dog crop at about 4 Hz with a smoothed label; motion energy; both fed into the risk engine; a small labeled-clip check reports how often the label is right (honest numbers) |
| 4 | **Alerts + event log + background** | 10:30 PM–12:00 AM | pyttsx3 voice (non-blocking), Windows toast, cooldowns, SQLite events + snapshots, mute/DND |
| 5 | **VLM confirmation** | 12:00–1:30 AM | async Ollama worker, strict-JSON prompt + parser + fallback, adjusts risk by ±1, its sentence used as the voice; video never stalls |
| 6 | **PySide6 UI + tray** | 1:30–4:00 AM | main window (live view, badge, timeline), settings (source, zone editor, vocabulary, thresholds), tray background mode, offline indicator. Stretch: LAN phone view |
| 7 | **Hardening + submission** | 6:00–9:30 AM | 30-minute soak run; README + disclosures; 1-minute demo video (Wi-Fi off); X/LinkedIn post; final push and repo public; **submit by 9:30 AM** (buffer before 10:00) |

Sleep: aim for about 2 hours between Batch 6 and Batch 7 (around 4–6 AM). Cut stretch items first if behind. **Priority if time runs short: 0 → 1 → 2 → 4 → 7.** 3, 5 and 6 make it impressive; without them it still works (OpenCV window + rules + voice).

---

## Planned File Map

```
bantayaso/                     # repo root = C:\Users\johnl\Desktop\projectayiks\bantayaso
  BRIEF.md                     # THIS FILE (spec + plan + status)
  README.md                    # judge-facing setup + disclosures (Batch 7)
  config.yaml                  # source, zones, vocabulary+tiers, thresholds, voice
  requirements.txt
  .gitignore                   # data/, models/, *.pt, .venv/
  scripts/
    check_env.py               # GPU/torch/ollama/camera check
    find_camera.py             # probe device indexes / test a URL
    download_models.py         # one-time weight download
  bantayaso/
    __main__.py                # entry: python -m bantayaso [--source ...] [--no-ui]
    config.py                  # load/save config.yaml
    capture.py                 # FrameSource + threaded reader + clip recorder
    detect_dog.py              # YOLO11 dog detection + tracking
    detect_hazards.py          # YOLOE/YOLO-World open-vocab hazards
    actions.py                 # CLIP/SigLIP zero-shot action + smoothing
    motion.py                  # motion energy in dog box
    zones.py                   # polygons, point-in-zone, editor
    risk.py                    # rule engine, persistence, cooldown, last-seen memory
    vlm.py                     # async Ollama VLM worker + JSON parsing
    alerts.py                  # TTS, toast, owner voice clip
    events.py                  # SQLite log + snapshots
    pipeline.py                # orchestrates all stages per frame
    overlay.py                 # draw boxes/zones/badge/FPS
    ui/                        # PySide6 main window, settings, tray (Batch 6)
    web/                       # optional LAN phone view (stretch)
  data/                        # gitignored: clips/, snapshots/, events.db
  models/                      # gitignored: weights
```

---

## Open Questions (answer at the start of Batch 0)

1. ~~Action camera~~ **ANSWERED: USB webcam mode. Build USB/device-index input only for now**; URL and file sources stay in the interface, but URL is deferred. A video file is still needed for testing and backup.
2. ~~GPU~~ **ANSWERED: ASUS TUF A15 with NVIDIA RTX 3050 Laptop GPU (likely 4 GB VRAM; confirm with `nvidia-smi`), AMD Ryzen CPU.** Use CUDA. VRAM is tight: YOLO "n" models, CLIP ViT-B/32, and a small VLM (`moondream` default, `qwen2.5vl:3b` only if VRAM allows; Ollama may partly offload to CPU, which is fine because the VLM is async). Load the VLM lazily, keep one model of each type. The Ryzen CPU + local AI is a possible AMD Award angle.
3. ~~RAM~~ **ANSWERED: 16 GB.** Windows version still to confirm.
4. Final project name (still open).

---

## Build Status & Handoff

**Current batch:** Batch 0 ✅ done (Oct 9, 3:22 PM): `check_env.py` reports ALL GOOD. GitHub repo (SETUP.md step 6) still to confirm.
**Next step:** Batch 1: camera + dog detection + live view. Then record test clips of the dog.

| Batch | Status | Notes |
|---|---|---|
| 0 Setup & skeleton | ✅ done | Files: requirements.txt, .gitignore, config.yaml, SETUP.md, README.md (stub), docs/ui-mockup.html, bantayaso/{__init__,__main__,config}.py, scripts/{check_env,find_camera,download_models}.py. All checks pass on the laptop. |
| 1 Camera + dog detection | ⬜ | |
| 2 Hazards + zones + risk v1 | ⬜ | |
| 3 Actions + motion | ⬜ | |
| 4 Alerts + events | ⬜ | |
| 5 VLM confirmation | ⬜ | |
| 6 PySide6 UI + tray | ⬜ | |
| 7 Hardening + submission | ⬜ | |

### Decisions Log
- 2026-10-09: Idea chosen: local AI dog watcher. Desktop-first; phone app is roadmap only; a LAN phone view is a stretch.
- 2026-10-09: The action camera is the CCTV; the real dog is at Demo Day; the laptop has a GPU.
- 2026-10-09: CLIP now loads through the OpenAI `clip` package (`clip_source: openai-clip`, ViT-B/32, weights in `models/clip`, downloaded from OpenAI's CDN) because Hugging Face downloads stalled. open_clip stays as an option. Batch 3 `actions.py` must use `clip.load(name, download_root=models/clip)`.
- 2026-10-09: User rule: no hardcoded spec/status narration on screen (removed offline/GPU/saved-locally cards, FPS chips, resolution subtitle). FPS/device only in an optional debug overlay.
- 2026-10-09: UI direction set: brown/black dark theme, mockup saved at `docs/ui-mockup.html` (Monitor + Events screens). PySide6 + QSS must match it in Batch 6. "Choco" in the mockup is a placeholder dog name (configurable `dog_name`).
- 2026-10-09: Batch 0: OpenCV DirectShow backend default (`camera.backend: dshow`), MSMF fallback; find_camera.py probes both and saves index+backend to config.yaml. Hazard vocabulary and action prompts live in config.yaml.
- 2026-10-09: USB webcam mode only for camera input for now (network URL deferred). VLM default switched to moondream for 4 GB VRAM.
- 2026-10-09: Use a layered pipeline (YOLO11 → YOLOE/YOLO-World → CLIP zero-shot → rules → Ollama VLM). No custom training required; a dog-pose fine-tune is optional.

### Known Issues / Risks
- `config.save()` (yaml.safe_dump) strips comments from config.yaml. This is harmless; keep the device copy as the source of truth.
- Windows PATH gotcha: refreshing `$env:Path` drops the venv, so re-run `.\.venv\Scripts\Activate.ps1`. Ollama lives at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.
- Setup (Oct 9, 3:08 PM): the Hugging Face CLIP download crawled at about 16 kB/s through the `hf_xet` downloader, while GitHub downloads ran at about 7 MB/s. Fix: `$env:HF_HUB_DISABLE_XET="1"` (or `pip uninstall hf_xet -y`) and rerun `download_models.py`. YOLO11n, YOLOE and the MobileCLIP text encoder were already downloaded; `clip` was auto-installed by Ultralytics.
- Tiny hazards (batteries, pills) may not be detected directly. This is mitigated by "last seen near muzzle", zones and the VLM.
- Zero-shot action labels can be noisy. This is mitigated by smoothing, motion cues and persistence thresholds.
- The venue's lighting and background differ from home. Re-check thresholds on site during the 12:15 PM AV check.

### Environment Facts (fill in during Batch 0)
- Laptop: ASUS TUF A15 (AMD Ryzen), 16 GB RAM, Windows 10/11 (Python reports Windows 10) · GPU: NVIDIA GeForce RTX 3050 Laptop, **4.0 GB VRAM**, CUDA 12.4 · Python 3.11.9 in `.venv` · torch 2.6.0+cu124 · opencv 5.0.0 · ultralytics 8.4.174 · open_clip 3.3.0 · PySide6 6.12.0 · Ollama models: `moondream:latest`, `qwen2.5vl:3b` · TTS: 2 SAPI voices (Microsoft David default) · **Camera: index 1, backend msmf**, delivers 1920x1080 by default (config asks for 1280x720) · Weights: models/yolo11n.pt, models/yoloe-11s-seg.pt, models/clip/ViT-B-32.pt · Measured FPS: — (Batch 1)

### How to resume (for any AI agent)
1. Read this whole file. Respect the deadline and the "no fake benchmarks" rule.
2. Inspect the repo at `C:\Users\johnl\Desktop\projectayiks\bantayaso\` and compare it with the file map and status table.
3. Run `python scripts/check_env.py`, then the current batch's acceptance checks.
4. Build the next unfinished batch only. Keep modules small and testable.
5. Before stopping, update: Current batch, Next step, status table, Decisions Log, Known Issues, Environment Facts, and the file map if it changed. Then sync this file to the claude.ai Project (`claude/bantayaso.md`) if available.
