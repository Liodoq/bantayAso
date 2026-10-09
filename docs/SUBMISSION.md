# BantayAso — Batch 7: Submission Kit

Deadline: **Oct 10, 10:00 AM**. Target: submit on the Cerebral Valley page by **9:30 AM**. One submission only.

## 1. Final checklist (in order)

- [ ] **Soak run (start now, ~30 min):** start BantayAso with the real camera, press F12, note FPS at the start. Let it run while you do steps 2–4. At the end, note FPS again, check Task Manager (memory should not keep climbing), and confirm no crash in `data\crash.log`. Write only what you saw.
- [ ] **Download small.en while online:** `python scripts\download_models.py` (otherwise the mic falls back to base.en).
- [ ] **Push:** README + this kit + the actions.py fix are committed; run `git push`.
- [ ] **Demo video (~1 min, Wi-Fi OFF)** — shot list in section 3.
- [ ] **Post on X or LinkedIn** with the video — text in section 4.
- [ ] **Submit** the form with the answers in section 2. Paste the repo link, video link and post link.

## 2. Submission answers

**Project name:** BantayAso — Local AI Dog Watcher

**One-liner:** A desktop app that watches your dogs through a USB camera, knows each one by name, and warns you out loud when they start chewing, raiding the trash or getting near a battery, cable or medicine. All AI runs on your PC.

**Why does this product benefit from running AI locally?**
A dog camera has to watch all day and react within seconds. Sending every frame to a cloud AI would cost money every minute, add network delay right when the dog is about to swallow something, stop working when the internet drops, and stream the inside of your home to someone else's server. BantayAso runs dog detection, object detection, action recognition, dog name recognition, speech recognition, the language model and the voice on the laptop. It is free to run, works offline, and the video never leaves the computer.

**What runs locally:** camera capture; dog detection and tracking (YOLO11s); hazard object detection (YOLOE-11s-seg, open vocabulary); action recognition, dog names and learned "safe chews" (OpenAI CLIP ViT-B/32 embeddings); the rule-based risk engine (Safe/Watch/Warning/Danger); speech recognition (OpenAI Whisper small.en, fallback base.en); answers and phrasing (Ollama qwen2.5:1.5b); alert descriptions (Ollama moondream); text-to-speech (Windows SAPI); the event log (SQLite) and the desktop UI (PySide6).

**What requires internet:** only the one-time install (Python packages, model weights, `ollama pull`). Nothing at runtime.

**Models used:** Ultralytics YOLO11s (`yolo11s.pt`), Ultralytics YOLOE-11s-seg (`yoloe-11s-seg.pt`) with its MobileCLIP text encoder (`mobileclip_blt.ts`), OpenAI CLIP ViT-B/32, OpenAI Whisper small.en / base.en, Ollama `qwen2.5:1.5b`, Ollama `moondream`, Windows SAPI voices.

**Technologies:** Python 3.11, PyTorch (CUDA), OpenCV, Ultralytics, OpenAI CLIP, OpenAI Whisper, sounddevice, Ollama, PySide6 (Qt), SQLite, Windows SAPI (pywin32), winotify, PyInstaller and Inno Setup (Windows installer).

**APIs / cloud services at runtime:** none.

**Existing code or assets:** none from before the hackathon. Open-source libraries and pretrained models as listed. The app logo was made during the hackathon.

**AI development tools used (disclosure):**
- Claude (Anthropic) — planning, code generation, debugging, UI work, packaging, docs.
- Codex (OpenAI) — code review, repairs and tests.
- [VIDEO TOOL NAME] — AI motion graphics for the promo video, made from real app screenshots.

**Honest limits:** accuracy and FPS numbers are only what we measured on the build laptop (29.6 FPS with the dog detector alone, measured on Oct 9; lower with every model running). Action labels come from zero-shot CLIP plus owner teaching, not a trained action model. Very small or hidden objects can't always be seen; Bantay flags the chewing behaviour instead and says "check what it is".

## 3. Demo video shot list (~60 s, Wi-Fi OFF the whole time)

| Time | Show | Say / caption |
|---|---|---|
| 0–5 s | Taskbar: Wi-Fi off / airplane mode. Open BantayAso from the Start menu. | "No internet. Everything runs on this laptop." |
| 5–15 s | Monitor: dogs boxed with names, status Safe. | "BantayAso knows my dogs by name and what they're doing." |
| 15–28 s | A dog chews a slipper or paper (prop). Warning card, voice alert, **It's safe / I've got it**. | "Chewing? It warns me and asks if it's safe." |
| 28–40 s | Prop battery in a sealed clear bag near the dog (hold the leash). Danger alert and voice. | "Near a battery, cable or medicine: Danger." |
| 40–52 s | Press F2: "Bantay, is Oreo sleeping?" → "Yes. …" | "I can just ask." |
| 52–60 s | Events page timeline, then the logo. | "BantayAso. Bantay, para sa aso mo." |

Safety: never let a dog reach a real battery, medicine or live cable. Use sealed props and hold the leash.

## 4. Social post (X or LinkedIn)

> Meet BantayAso 🐾 a Local AI dog watcher I built solo for #AppBuildersPH Hackathon 2026.
>
> It watches my dogs through a USB camera, knows each one by name, and warns me out loud when they chew something risky or get near a battery, cable or medicine. I can ask it "Bantay, is Oreo sleeping?" and it answers.
>
> Everything runs on my laptop: no cloud, no subscription, works with Wi-Fi off. Video never leaves the PC.
>
> Built with YOLO11, YOLOE, CLIP, Whisper and Ollama. @cognition @DevinAI
>
> Repo: https://github.com/Liodoq/bantayAso

(Check the Devin / Cognition handles before posting. On LinkedIn, tag the Cognition page instead.)
