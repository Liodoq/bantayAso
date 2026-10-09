# Build the BantayAso Windows app and installer.
#   1. Open PowerShell in the repo folder, with the .venv that already runs BantayAso.
#   2. powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# Output: dist\BantayAso\BantayAso.exe (portable folder), dist\BantayAso-portable.zip,
#         and Output\BantayAso-Setup.exe if Inno Setup 6 is installed (https://jrsoftware.org/isdl.php).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pyinstaller

# the models must already be downloaded (python scripts\download_models.py) - they are shipped with the app
foreach ($m in @("models\yolo11s.pt", "models\yoloe-11s-seg.pt", "models\clip\ViT-B-32.pt")) {
    if (-not (Test-Path $m)) { throw "Missing $m - run: python scripts\download_models.py" }
}

python -m PyInstaller --noconfirm --clean packaging\bantayaso.spec

# ship the AI models next to the exe (read from the install folder, never from user data)
$dst = "dist\BantayAso\models"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item models\*.pt $dst -Force
Copy-Item models\*.ts $dst -Force -ErrorAction SilentlyContinue
Copy-Item models\clip $dst -Recurse -Force
New-Item -ItemType Directory -Force "$dst\whisper" | Out-Null
Copy-Item models\whisper\*.pt "$dst\whisper" -Force
Copy-Item packaging\config.default.yaml dist\BantayAso\ -Force
Write-Host "App folder ready: dist\BantayAso\BantayAso.exe"

Compress-Archive -Path dist\BantayAso -DestinationPath dist\BantayAso-portable.zip -Force
Write-Host "Portable zip: dist\BantayAso-portable.zip"

$iscc = @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
    & $iscc packaging\BantayAso.iss
    Write-Host "Installer: Output\BantayAso-Setup.exe"
} else {
    Write-Host "Inno Setup 6 not found - install it to also build BantayAso-Setup.exe (the zip works without it)."
}
