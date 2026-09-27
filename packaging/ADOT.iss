; Inno Setup script for ADOT.
;
; Packages the PORTABLE layout built by packaging/build_portable.ps1:
;
;     portable\runtime\   embeddable CPython 3.10 + torch, PyQt5, scipy, librosa
;     portable\app\       the application, the firmware templates and the icon
;
; There is no frozen .exe. PyInstaller was tried first and abandoned: the
; frozen build reliably failed with WinError 1114 loading torch's c10.dll,
; and none of the usual causes applied (files byte-identical to a working
; install, no duplicate or shadowed DLLs, matching MSVC runtime, not the
; bundled UCRT, not an OpenMP conflict, not the DLL search path). Shipping an
; ordinary Python environment sidesteps the whole class of problem: torch is
; imported by the normal import machinery, exactly as it is when the script
; is run from a checkout, which works.
;
; The Start-menu entry points at pythonw.exe with the script as its argument,
; so it is still one click and shows no console window.
;
;     iscc packaging\ADOT.iss

#define AppName        "ADOT"
#define AppVersion     "1.3.0"
#define AppPublisher   "Canberk Kaplan"
#define AppScript      "edge_ai_train_model_gui_quantization_according_succes_rate.py"
#ifndef SourceDir
  #define SourceDir "..\portable"
#endif

[Setup]
AppId={{7B3C1A54-6E2D-4C91-9F0B-2D7A5E8C1F33}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=ADOT-{#AppVersion}-setup
; Overridable so build_portable.ps1 -Fast can trade size for compile time:
;   ISCC /DCompressionLevel=lzma2/fast /DSolidComp=no packaging\ADOT.iss
; Solid compression is the expensive half -- it makes the whole payload one
; stream, which compresses better and takes far longer.
#ifndef CompressionLevel
  #define CompressionLevel "lzma2/max"
#endif
#ifndef SolidComp
  #define SolidComp "yes"
#endif
Compression={#CompressionLevel}
SolidCompression={#SolidComp}
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-user by default: no administrator prompt, and the install folder is
; writable, so export/ lands next to the application instead of in Documents.
; The first wizard page still offers a machine-wide install.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
SetupIconFile=..\portable\app\ADOT.ico
UninstallDisplayIcon={app}\app\ADOT.ico
UninstallDisplayName={#AppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\app\*"; DestDir: "{app}\app"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; pythonw.exe, not python.exe: no console window flashes up behind the app
Name: "{group}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; \
    Parameters: """{app}\app\{#AppScript}"""; WorkingDir: "{app}\app"; \
    IconFilename: "{app}\app\ADOT.ico"; Comment: "BLDC acoustic classification for the TMS320F28379D"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\runtime\pythonw.exe"; \
    Parameters: """{app}\app\{#AppScript}"""; WorkingDir: "{app}\app"; \
    IconFilename: "{app}\app\ADOT.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\runtime\pythonw.exe"; Parameters: """{app}\app\{#AppScript}"""; \
    WorkingDir: "{app}\app"; \
    Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; results the app wrote next to itself, and bytecode caches the runtime made;
; the installer did not create these, so Inno would leave the folder behind
Type: filesandordirs; Name: "{app}\app\export"
Type: filesandordirs; Name: "{app}\app\__pycache__"
Type: filesandordirs; Name: "{app}\runtime\Lib\site-packages"
