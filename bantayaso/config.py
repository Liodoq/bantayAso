"""Load and save config.yaml (repo root)."""
from __future__ import annotations

from pathlib import Path
import yaml

import os
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

if getattr(sys, "frozen", False):
    # Installed app (PyInstaller): the program and its AI models live in the install folder; each
    # Windows user's settings, dogs, zones, events and clips live in %LOCALAPPDATA%\BantayAso, so
    # nobody else's data ships with the app and uninstalling the program keeps your data.
    APP_DIR = Path(sys.executable).resolve().parent
    USER_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BantayAso"
    USER_DIR.mkdir(parents=True, exist_ok=True)
    ROOT = APP_DIR
    MODELS_DIR = APP_DIR / "models"
    DATA_DIR = USER_DIR / "data"
    CONFIG_PATH = USER_DIR / "config.yaml"
    if not CONFIG_PATH.exists():                      # first run: start from the clean default settings
        default = APP_DIR / "config.default.yaml"
        if not default.exists():
            default = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "config.default.yaml"
        if default.exists():
            shutil.copyfile(default, CONFIG_PATH)


def load(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save(cfg: dict, path: Path = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def resolve_device(cfg: dict) -> str:
    want = cfg.get("models", {}).get("device", "auto")
    if want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dirs() -> None:
    for d in (DATA_DIR / "clips", DATA_DIR / "snapshots", MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
