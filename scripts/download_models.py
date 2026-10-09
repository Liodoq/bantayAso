"""One-time model download (needs internet). After this, BantayAso runs offline.

Run:  python scripts/download_models.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bantayaso import config  # noqa: E402


def get_yolo(name: str) -> bool:
    from ultralytics import YOLO
    dest = config.MODELS_DIR / name
    if dest.exists():
        print(f"  {name}: already present")
        return True
    cwd = os.getcwd()
    os.chdir(config.MODELS_DIR)          # ultralytics downloads into the cwd
    try:
        YOLO(name)
        print(f"  {name}: downloaded")
        return True
    except Exception as e:
        print(f"  {name}: FAILED ({e})")
        return False
    finally:
        os.chdir(cwd)


def main() -> None:
    cfg = config.load()
    config.ensure_dirs()
    m = cfg["models"]

    print("[1/4] Dog detector")
    get_yolo(m["dog_detector"])

    print("[2/4] Hazard detector (open vocabulary)")
    if not get_yolo(m["hazard_detector"]):
        fb = "yolov8s-worldv2.pt"
        print(f"  trying fallback {fb}")
        if get_yolo(fb):
            cfg["models"]["hazard_detector"] = fb
            config.save(cfg)
            print(f"  config.yaml now uses {fb}")

    print("    warming up text vocabulary (downloads the text encoder once)")
    try:
        from ultralytics import YOLO
        hz = cfg["models"]["hazard_detector"]
        names = list(cfg["hazards"].keys())
        model = YOLO(str(config.MODELS_DIR / hz))
        cwd = os.getcwd(); os.chdir(config.MODELS_DIR)
        try:
            if "yoloe" in hz:
                model.set_classes(names, model.get_text_pe(names))
            else:
                model.set_classes(names)
        finally:
            os.chdir(cwd)
        print(f"    vocabulary OK ({len(names)} words)")
    except Exception as e:
        print(f"    vocabulary warm-up FAILED ({e})")

    print("[3/4] CLIP action model")
    src = m.get("clip_source", "openai-clip")
    try:
        if src == "openai-clip":
            # OpenAI CLIP package (auto-installed by Ultralytics). Downloads ~340 MB
            # from OpenAI's CDN instead of Hugging Face (HF was ~16 kB/s for us).
            import clip
            root = config.MODELS_DIR / "clip"
            clip.load(m.get("clip_openai_name", "ViT-B/32"), device="cpu", download_root=str(root))
            print(f"  {m.get('clip_openai_name', 'ViT-B/32')}: cached in models/clip")
        else:
            import open_clip
            open_clip.create_model_and_transforms(m["clip_model"], pretrained=m["clip_pretrained"])
            print(f"  {m['clip_model']} / {m['clip_pretrained']}: cached")
    except Exception as e:
        print(f"  CLIP FAILED ({e})")

    print("[4/5] Ollama models (vision + Ask Bantay text model)")
    qa_model = cfg.get("qa", {}).get("model", "qwen2.5:1.5b")
    exe = shutil.which("ollama") or os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")
    if exe and os.path.exists(exe):
        for name in (m["vlm"], qa_model):
            subprocess.run([exe, "pull", name], check=False)
    else:
        print(f"  ollama not found -> install from ollama.com, then: ollama pull {m['vlm']} ; ollama pull {qa_model}")

    print("[5/5] Whisper speech recognition (Ask Bantay)")
    try:
        import whisper
        whisper.load_model(cfg.get("qa", {}).get("whisper_model", "base"), device="cpu",
                           download_root=str(config.MODELS_DIR / "whisper"))
        print("  whisper model cached in models/whisper")
    except Exception as e:
        print(f"  Whisper FAILED ({e}) -> pip install openai-whisper sounddevice")

    print("Done. Next: python scripts/check_env.py")


if __name__ == "__main__":
    main()
