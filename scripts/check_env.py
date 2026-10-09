"""Batch 0 environment check. Run from repo root:  python scripts/check_env.py"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = 0


def report(status: str, msg: str) -> None:
    global problems
    if status == BAD:
        problems += 1
    print(f"{status} {msg}")


def main() -> None:
    print("=== BantayAso environment check ===")
    v = sys.version_info
    report(OK if (3, 10) <= (v.major, v.minor) <= (3, 12) else WARN,
           f"Python {platform.python_version()} ({platform.system()} {platform.release()})")

    try:
        import torch
        report(OK, f"torch {torch.__version__}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report(OK, f"CUDA GPU: {p.name}  VRAM {p.total_memory / 1024**3:.1f} GB  (CUDA {torch.version.cuda})")
        else:
            report(BAD, "CUDA not available -> reinstall torch with the CUDA index URL (see SETUP.md)")
    except ImportError:
        report(BAD, "torch not installed")

    for mod, label in [("cv2", "opencv"), ("ultralytics", "ultralytics"), ("open_clip", "open_clip"),
                       ("yaml", "pyyaml"), ("pyttsx3", "pyttsx3"), ("PySide6", "PySide6"), ("requests", "requests")]:
        try:
            m = __import__(mod)
            report(OK, f"{label} {getattr(m, '__version__', '')}")
        except ImportError:
            report(BAD, f"{label} not installed")

    from bantayaso import config
    cfg = config.load()

    # Ollama
    try:
        import requests
        r = requests.get(cfg["models"]["ollama_url"] + "/api/tags", timeout=3)
        names = [m["name"] for m in r.json().get("models", [])]
        report(OK, f"Ollama running, models: {', '.join(names) or '(none)'}")
        want = cfg["models"]["vlm"]
        if not any(n.split(":")[0] == want.split(":")[0] for n in names):
            report(WARN, f"VLM '{want}' not pulled yet -> run: ollama pull {want}")
    except Exception as e:
        report(BAD, f"Ollama not reachable ({e.__class__.__name__}) -> install/start Ollama")

    # TTS
    try:
        import pyttsx3
        eng = pyttsx3.init()
        voices = eng.getProperty("voices")
        report(OK, f"TTS voices: {len(voices)} ({voices[0].name if voices else 'none'})")
    except Exception as e:
        report(WARN, f"TTS init failed: {e}")

    # Camera
    try:
        import cv2
        cam = cfg["camera"]
        backend = cv2.CAP_DSHOW if cam.get("backend") == "dshow" else cv2.CAP_MSMF
        cap = cv2.VideoCapture(int(cam["index"]), backend)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            report(OK, f"Camera index {cam['index']} -> {frame.shape[1]}x{frame.shape[0]}")
        else:
            report(BAD, f"Camera index {cam['index']} gave no frame -> run scripts/find_camera.py")
    except Exception as e:
        report(BAD, f"Camera check error: {e}")

    # Model weights
    for w in [cfg["models"]["dog_detector"], cfg["models"]["hazard_detector"]]:
        p = config.MODELS_DIR / w
        report(OK if p.exists() else WARN, f"weights {w}: {'found' if p.exists() else 'missing -> run scripts/download_models.py'}")

    print("=== " + ("ALL GOOD" if problems == 0 else f"{problems} problem(s) to fix") + " ===")


if __name__ == "__main__":
    main()
