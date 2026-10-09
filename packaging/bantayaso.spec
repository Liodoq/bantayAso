# PyInstaller spec for BantayAso (Windows, one-folder app).  Build with:  packaging\build.ps1
# Result: dist\BantayAso\BantayAso.exe (+ _internal\, models\, config.default.yaml)
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = SPECPATH + "\\.."
datas, binaries, hidden = [], [], []
for pkg in ("ultralytics", "whisper", "clip", "sounddevice", "winotify"):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hidden += h
    except Exception:
        pass
for pkg in ("_sounddevice_data", "tiktoken_ext"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
hidden += collect_submodules("bantayaso") + ["tiktoken_ext.openai_public", "win32com.client", "pythoncom",
                                              "pywintypes", "requests", "yaml", "PIL"]
datas += [
    (ROOT + "\\assets", "assets"),
    (ROOT + "\\bantayaso\\trackers", "bantayaso\\trackers"),
    (ROOT + "\\packaging\\config.default.yaml", "."),
]

a = Analysis([ROOT + "\\packaging\\launcher.py"], pathex=[ROOT], binaries=binaries, datas=datas,
             hiddenimports=hidden, excludes=["tkinter", "matplotlib.tests", "IPython", "jupyter"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="BantayAso", console=False,
          icon=ROOT + "\\assets\\bantayaso.ico", upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="BantayAso", upx=False)
