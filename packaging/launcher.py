"""Entry point of the installed BantayAso.exe (PyInstaller)."""
import json
import sys

if len(sys.argv) >= 3 and sys.argv[1] == "--camnames":
    # helper mode used by Settings -> camera list (runs in a separate process on purpose)
    from bantayaso.camnames import _collect
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(_collect(), f)
    sys.exit(0)

if sys.stdout is None or sys.stderr is None:      # windowed app: keep print() from crashing
    import os
    from pathlib import Path
    log = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BantayAso" / "bantayaso.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    stream = open(log, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or stream
    sys.stderr = sys.stderr or stream

from bantayaso.__main__ import main   # noqa: E402

main()
