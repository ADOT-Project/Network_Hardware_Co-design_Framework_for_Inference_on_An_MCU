# Builds the Edge AI NAS desktop application.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -SkipInstaller
#
# Step 1 produces dist\EdgeAiNAS\EdgeAiNAS.exe (a folder you can zip and hand
# over as-is). Step 2 wraps that folder in dist\EdgeAiNAS-<version>-setup.exe
# and needs Inno Setup 6 installed.

param(
    [switch]$SkipInstaller
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "== Edge AI NAS build ==" -ForegroundColor Cyan
Write-Host "repository: $root"

# The metrics replacing scikit-learn are checked against it here, not in the
# shipped app -- if this environment still has scikit-learn, make sure the
# numpy versions still agree before freezing anything.
if (Test-Path "tools\test_metrics.py") {
    Write-Host "`n-- verifying metrics against scikit-learn --" -ForegroundColor Cyan
    python tools\test_metrics.py
    if ($LASTEXITCODE -ne 0) { throw "metric check failed; not building" }
}

Write-Host "`n-- PyInstaller --" -ForegroundColor Cyan
python -m PyInstaller packaging\EdgeAiNAS.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$appDir = Join-Path $root "dist\EdgeAiNAS"
$exe = Join-Path $appDir "EdgeAiNAS.exe"
if (-not (Test-Path $exe)) { throw "expected $exe" }

$sizeMb = [math]::Round(((Get-ChildItem $appDir -Recurse -File |
    Measure-Object Length -Sum).Sum / 1MB), 0)
Write-Host "built $exe ($sizeMb MB folder)" -ForegroundColor Green

if ($SkipInstaller) {
    Write-Host "`nskipping the installer (-SkipInstaller)."
    exit 0
}

$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                     "$env:ProgramFiles\Inno Setup 6\ISCC.exe")) {
        if (Test-Path $c) { $iscc = $c; break }
    }
}
if (-not $iscc) {
    Write-Warning ("Inno Setup 6 not found - the application folder is ready " +
                   "at dist\EdgeAiNAS but no setup.exe was produced. " +
                   "Install it from https://jrsoftware.org/isdl.php, or ship " +
                   "the folder as a zip.")
    exit 0
}

Write-Host "`n-- Inno Setup --" -ForegroundColor Cyan
& $iscc "packaging\EdgeAiNAS.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }

Get-ChildItem "dist\EdgeAiNAS-*-setup.exe" | ForEach-Object {
    $mb = [math]::Round($_.Length / 1MB, 0)
    Write-Host "installer: $($_.FullName) ($mb MB)" -ForegroundColor Green
}
