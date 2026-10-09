"""List camera names in OpenCV's numbering, in a separate process so a buggy camera driver
(e.g. NVIDIA Broadcast's virtual camera) can never crash the app.

    python -m bantayaso.camnames   ->  {"msmf": ["OsmoAction", ...], "dshow": [...]}
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _collect() -> dict:
    out = {"msmf": [], "dshow": []}
    try:                                   # Media Foundation order == cv2.CAP_MSMF order
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtMultimedia import QMediaDevices
        _app = QCoreApplication.instance() or QCoreApplication([])
        out["msmf"] = [d.description() for d in QMediaDevices.videoInputs()]
    except Exception:
        pass
    try:                                   # DirectShow order == cv2.CAP_DSHOW order (optional package)
        from pygrabber.dshow_graph import FilterGraph
        out["dshow"] = list(FilterGraph().get_input_devices())
    except Exception:
        pass
    return out


def camera_names(timeout: float = 10.0) -> dict:
    """Run _collect() in a child process; {} lists on any failure or crash."""
    try:
        if getattr(sys, "frozen", False):              # installed app: same exe in "names only" mode,
            import tempfile                            # answering through a file (no console window)
            out = Path(tempfile.gettempdir()) / f"bantayaso_cams_{id(timeout)}.json"
            subprocess.run([sys.executable, "--camnames", str(out)], timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            data = json.loads(out.read_text(encoding="utf-8"))
            out.unlink(missing_ok=True)
            return data
        r = subprocess.run([sys.executable, "-m", "bantayaso.camnames"], capture_output=True, text=True,
                           cwd=str(Path(__file__).resolve().parents[1]),
                           timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for line in reversed(r.stdout.strip().splitlines()):
            if line.startswith("{"):
                return json.loads(line)
    except Exception:
        pass
    return {"msmf": [], "dshow": []}


if __name__ == "__main__":
    print(json.dumps(_collect()), flush=True)
