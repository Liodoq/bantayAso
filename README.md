<p align="center"><img src="assets/bantayaso_icon.png" width="120" alt="BantayAso logo"></p>

# BantayAso — Local AI Dog Watcher

**Built for the AppBuildersPH Hackathon 2026 (theme: Local AI).**

BantayAso is a Windows desktop app that watches your dogs through a USB camera while you work. It knows each dog by name, sees what they are doing (sleeping, sitting, eating, chewing, digging, playing rough) and spots risky things near them, like cables, batteries, medicine or plastic bags. When something looks wrong, it warns you out loud and with a notification, rated **Safe, Watch, Warning or Danger**. You can also just ask it: *"Bantay, is Oreo sleeping?"* or *"What happened today?"*

All of the AI runs on your own computer. Your camera video never leaves your PC, there is no account or subscription, and it keeps working with the internet off. Internet is only needed once, to download the app's models.

## Minimum requirements

| | Minimum | What it was built and tested on |
|---|---|---|
| OS | Windows 10 or 11, 64-bit | ASUS TUF A15 laptop, Windows |
| GPU | NVIDIA GPU with 4 GB VRAM (CUDA) recommended; runs on CPU only, but much slower | RTX 3050 Laptop, 4 GB |
| RAM | 16 GB recommended | 16 GB |
| Disk | About 8 GB free (app + models) | |
| Camera | Any USB webcam or action camera in webcam mode | DJI Osmo Action (USB) |
| Audio | Speakers for voice alerts; a microphone to talk to Bantay (optional) | Laptop mic and speakers |
| Optional | [Ollama](https://ollama.com) with `qwen2.5:1.5b` and `moondream` for open questions and alert descriptions | |

Running from source needs Python 3.11 and PyTorch with CUDA. Setup steps are in [SETUP.md](SETUP.md).
