# BantayAso — Setup (Windows, NVIDIA GPU)

Run these in **PowerShell** inside `C:\Users\johnl\Desktop\projectayiks\bantayaso`.

## 1. Python 3.11 + virtual env
```powershell
py -3.11 --version          # if missing: winget install Python.Python.3.11
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
```

## 2. PyTorch with CUDA (GPU) — install this FIRST
```powershell
nvidia-smi                  # shows GPU, VRAM and driver; note the VRAM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `True NVIDIA GeForce RTX 3050 ...`. If `False`, update the NVIDIA driver and reinstall.

## 3. The rest
```powershell
pip install -r requirements.txt
```

## 4. Ollama (local vision model)
Install from https://ollama.com/download (Windows), then:
```powershell
ollama pull moondream
ollama pull qwen2.5vl:3b    # optional, bigger/smarter; test if VRAM allows
```

## 5. Models + camera
```powershell
python scripts\download_models.py   # one-time, needs internet
python scripts\find_camera.py       # plug in the action camera in USB webcam mode
python scripts\check_env.py         # should end with ALL GOOD
python -m bantayaso                 # prints "BantayAso skeleton OK"
```

## 6. GitHub (public repo — required by the deadline)
```powershell
git init
git add .
git commit -m "Batch 0: project skeleton and setup scripts"
gh repo create bantayaso --public --source . --push   # or create it on github.com and push
```

**Send the full output of `check_env.py` back to the AI chat** so the brief's Environment Facts can be filled in.
