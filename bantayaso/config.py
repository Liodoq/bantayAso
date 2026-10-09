"""Load and save config.yaml (repo root)."""
from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


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
