; Inno Setup script: BantayAso-Setup.exe from dist\BantayAso (run packaging\build.ps1 first).
; Installs for the current user only (no admin), so the app folder is writable for model caches.
; Your settings/dogs/zones/events live in %LOCALAPPDATA%\BantayAso and are kept on uninstall.
#define AppName "BantayAso"
#define AppVersion "1.0.0"

[Setup]
AppId={{6C1A3E52-8B7F-4C21-9E0D-B4A7A5D1C0A1}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=John Leorick Salalila
AppComments=Local AI dog watcher - all AI runs on this PC
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
OutputDir=..\Output
OutputBaseFilename=BantayAso-Setup
SetupIconFile=..\assets\bantayaso.ico
UninstallDisplayIcon={app}\BantayAso.exe
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
DiskSliceSize=max
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\BantayAso\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\BantayAso.exe"; IconFilename: "{app}\BantayAso.exe"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\BantayAso.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\BantayAso.exe"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent
