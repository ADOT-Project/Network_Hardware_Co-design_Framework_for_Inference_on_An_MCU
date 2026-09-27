# Builds the ADOT desktop application.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_portable.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build_portable.ps1 -Refresh
#
# Produces:
#   portable\        a self-contained folder that runs anywhere (zip and hand over)
#   dist\ADOT-<version>-setup.exe        the installer, if Inno Setup 6 is present
#
# The application is NOT frozen. See packaging\README.md for why PyInstaller
# was abandoned; in short, the frozen build could not load torch's c10.dll and
# an ordinary Python environment can.
#
# -Refresh rebuilds the runtime from scratch (re-downloads Python, reinstalls
# every package). Without it, an existing portable\runtime is reused and only
# the application files are refreshed, which takes seconds.

param(
    [switch]$Refresh,
    [switch]$SkipInstaller,
    # -Fast stores the payload instead of compressing it: quick to build,
    # about 1 GB of .exe, for local testing only.
    #
    # MEASURED, 2026-09-21, same 990 MB payload:
    #   lzma2/max + solid   592 s -> 187 MB     (the default, and the one to ship)
    #   lzma2/fast, NO solid 685 s -> 263 MB    (slower AND bigger -- do not)
    # Turning solid compression off was the mistake: the payload is ~22,000
    # files, most of them small torch headers, and solid mode is what lets
    # one dictionary cover all of them. Without it the per-file setup cost
    # outweighs everything the cheaper algorithm saves. If -Fast ever needs
    # to become "compressed but quick", lower the algorithm and KEEP solid.
    [switch]$Fast,
    [string]$PythonVersion = "3.10.11"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$portable = Join-Path $root "portable"
$runtime  = Join-Path $portable "runtime"
$appdir   = Join-Path $portable "app"
$script   = "edge_ai_train_model_gui_quantization_according_succes_rate.py"

Write-Host "== ADOT portable build ==" -ForegroundColor Cyan

if ($Refresh -and (Test-Path $runtime)) {
    Write-Host "-- removing the existing runtime (-Refresh) --"
    Remove-Item $runtime -Recurse -Force
}

if (-not (Test-Path (Join-Path $runtime "python.exe"))) {
    Write-Host "`n-- fetching embeddable CPython $PythonVersion --" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force $runtime | Out-Null
    $zip = Join-Path $env:TEMP "python-embed-$PythonVersion.zip"
    $url = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $runtime -Force
    Remove-Item $zip -Force

    # The embeddable build ships with "import site" commented out and no
    # site-packages on the path, which makes every pip-installed package
    # invisible. Undo both.
    $pth = Get-ChildItem $runtime -Filter "python3*._pth" | Select-Object -First 1
    @("python310.zip", ".", "Lib\site-packages", "", "import site") |
        Set-Content -Path $pth.FullName -Encoding ascii
    New-Item -ItemType Directory -Force (Join-Path $runtime "Lib\site-packages") | Out-Null

    Write-Host "-- bootstrapping pip --"
    $getpip = Join-Path $env:TEMP "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/pip/get-pip.py" -OutFile $getpip
    & (Join-Path $runtime "python.exe") $getpip --no-warn-script-location -q
    Remove-Item $getpip -Force

    Write-Host "`n-- installing dependencies (this is the slow part) --" -ForegroundColor Cyan
    # CPU wheels only: this tool trains small CNNs on the machine it runs on,
    # and the CUDA wheels would add gigabytes for nothing.
    & (Join-Path $runtime "python.exe") -m pip install --no-warn-script-location -q `
        torch numpy --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
    & (Join-Path $runtime "python.exe") -m pip install --no-warn-script-location -q `
        PyQt5 scipy librosa
    if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }
} else {
    Write-Host "-- reusing the existing runtime (pass -Refresh to rebuild it) --"
}

# DO NOT copy msvcp140.dll next to python.exe. It is tempting: torch_cpu,
# torch_python and c10 all import it, the embeddable CPython does not ship
# it, and PyQt5 has a copy right there in site-packages. Measured, 2026-09-21:
# doing that breaks torch outright -- 3 of 3 imports die with
#
#   OSError: [WinError 1114] ... Error loading ...torch\lib\c10.dll
#
# and 3 of 3 succeed the moment the file is removed again. PyQt5 ships
# 14.26.28720.3; torch is built against a newer toolset and System32 here has
# 14.50.35719.0. A DLL beside python.exe wins the search, so the old copy
# shadows the good one and torch's initialisers fail.
#
# This is almost certainly the same WinError 1114 that killed the PyInstaller
# build (see packaging/README.md) -- that one bundled DLLs too. The runtime
# therefore takes msvcp140.dll from the target machine, which every Windows
# with any VC++ 2015-2022 redistributable has; see README "What the target
# machine needs".

Write-Host "`n-- staging the application --" -ForegroundColor Cyan
New-Item -ItemType Directory -Force $appdir | Out-Null
Copy-Item (Join-Path $root $script) $appdir -Force
# Only the three files write_embedded_files() patches; the rest of the 22 MB
# firmware tree is never read. A user who wants their own checkout patched
# points EDGE_AI_FIRMWARE_DIR at it or drops it beside the app.
$fwDst = Join-Path $appdir "CNN_Inference_Edge-AI_with_CLA"
New-Item -ItemType Directory -Force $fwDst | Out-Null
foreach ($f in @("definitions.h", "weights.h", "main.c")) {
    Copy-Item (Join-Path $root "CNN_Inference_Edge-AI_with_CLA\$f") $fwDst -Force
}
if (-not (Test-Path (Join-Path $appdir "ADOT.ico"))) {
    & (Join-Path $runtime "python.exe") (Join-Path $portable "make_icon.py")
}
Get-ChildItem $portable -Recurse -Directory -Filter "__pycache__" -EA SilentlyContinue |
    Remove-Item -Recurse -Force -EA SilentlyContinue

Write-Host "`n-- verifying the portable build actually starts --" -ForegroundColor Cyan
$selftest = Join-Path $env:TEMP "edgeai_selftest.py"
@'
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, sys.argv[1])
import edge_ai_train_model_gui_quantization_according_succes_rate as M
from PyQt5.QtWidgets import QApplication
app = QApplication([])
w = M.EdgeAiGui(); w.show(); app.processEvents()
w.dataset_path = "x.npz"; w.refresh(); app.processEvents()
assert w.max_ch and w.canvas.blocks, "the device view did not build"
assert M.find_firmware_templates(), "firmware templates not found"
print("selftest ok: ceiling", w.max_ch, "| templates", M.find_firmware_templates())
'@ | Set-Content $selftest -Encoding utf8
& (Join-Path $runtime "python.exe") $selftest $appdir
if ($LASTEXITCODE -ne 0) { throw "the portable build failed its self-test" }
Remove-Item $selftest -Force

$mb = [math]::Round(((Get-ChildItem $portable -Recurse -File | Measure-Object Length -Sum).Sum/1MB),0)
Write-Host "portable folder ready: $portable ($mb MB)" -ForegroundColor Green

if ($SkipInstaller) { Write-Host "`nskipping the installer (-SkipInstaller)."; exit 0 }

$iscc = (Get-Command iscc -EA SilentlyContinue).Source
if (-not $iscc) {
    foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                     "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $c) { $iscc = $c; break }
    }
}
if (-not $iscc) {
    Write-Warning ("Inno Setup 6 not found - portable\ is ready but no " +
                   "setup.exe was produced. https://jrsoftware.org/isdl.php")
    exit 0
}

Write-Host "`n-- Inno Setup --" -ForegroundColor Cyan
$isccArgs = @("packaging\ADOT.iss")
if ($Fast) {
    Write-Warning ("-Fast: payload STORED, not compressed. Expect about 1 GB " +
                   "of .exe. Local testing only -- ship a build without it.")
    $isccArgs = @("/DCompressionLevel=none", "/DSolidComp=no") + $isccArgs
}
$t0 = Get-Date
& $iscc $isccArgs
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Write-Host ("compile took {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)
Get-ChildItem "dist\ADOT-*-setup.exe" | ForEach-Object {
    Write-Host ("installer: {0} ({1} MB)" -f $_.FullName,
                [math]::Round($_.Length/1MB,0)) -ForegroundColor Green
}
