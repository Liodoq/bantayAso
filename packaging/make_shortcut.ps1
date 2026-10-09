# Quick option (no build): a Desktop + Start-menu shortcut that opens BantayAso with its logo and
# no terminal window, using this folder's .venv.
#   powershell -ExecutionPolicy Bypass -File packaging\make_shortcut.ps1
$repo = Split-Path $PSScriptRoot -Parent
$pyw = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "No .venv found in $repo" }
$ws = New-Object -ComObject WScript.Shell
$targets = @([Environment]::GetFolderPath("Desktop"),
             (Join-Path ([Environment]::GetFolderPath("Programs")) ""))
foreach ($dir in $targets) {
    $lnk = $ws.CreateShortcut((Join-Path $dir "BantayAso.lnk"))
    $lnk.TargetPath = $pyw
    $lnk.Arguments = "-m bantayaso"
    $lnk.WorkingDirectory = $repo
    $lnk.IconLocation = (Join-Path $repo "assets\bantayaso.ico")
    $lnk.Description = "BantayAso - Local AI dog watcher"
    $lnk.Save()
    Write-Host "Created $($lnk.FullName)"
}
