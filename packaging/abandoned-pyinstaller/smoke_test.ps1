# Smoke-tests a built Edge AI NAS folder.
#
#   powershell -ExecutionPolicy Bypass -File packaging\smoke_test.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\smoke_test.ps1 -AppDir build_out\dist\EdgeAiNAS
#
# Catches the failure mode a frozen GUI build is prone to: it starts from a
# directory that is not the source tree, cannot find something it used to
# find relative to the working directory, and dies before a window appears --
# silently, because there is no console.

param(
    [string]$AppDir = "dist\EdgeAiNAS",
    [int]$StartupSeconds = 25
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$app = Join-Path $root $AppDir
$exe = Join-Path $app "EdgeAiNAS.exe"
$failed = 0
function Check($ok, $msg) {
    if ($ok) { Write-Host "  PASS $msg" -ForegroundColor Green }
    else { Write-Host "  FAIL $msg" -ForegroundColor Red; $script:failed++ }
}

Write-Host "== smoke test: $app ==" -ForegroundColor Cyan
Check (Test-Path $exe) "EdgeAiNAS.exe exists"
if (-not (Test-Path $exe)) { exit 1 }

Write-Host "`n-- bundled resources --"
$templates = @("definitions.h", "weights.h", "main.c")
foreach ($t in $templates) {
    $hit = Get-ChildItem $app -Recurse -Filter $t -ErrorAction SilentlyContinue |
        Where-Object { $_.DirectoryName -like "*firmware_templates*" } |
        Select-Object -First 1
    Check ($null -ne $hit) "firmware template $t is bundled"
}

# Test by package marker, not by directory name: torch ships a mock "scipy"
# stub under _inductor and pyarrow has a "tensorflow" header folder, so a
# name-only scan reports both as present when neither really is.
function Test-Bundled($pkg) {
    $root = Join-Path $app "_internal"
    if (-not (Test-Path $root)) { $root = $app }
    return (Test-Path (Join-Path $root "$pkg\__init__.py")) -or
           (Test-Path (Join-Path $root "$pkg\__init__.pyc")) -or
           ((Get-ChildItem $root -Filter "$pkg*" -Directory -ErrorAction SilentlyContinue |
             Where-Object { $_.Name -eq $pkg -or $_.Name -eq "$pkg.libs" } |
             Measure-Object).Count -gt 0)
}

Write-Host "`n-- nothing we excluded got in --"
foreach ($pkg in @("tensorflow", "keras", "tensorboard", "sklearn")) {
    Check (-not (Test-Bundled $pkg)) "$pkg is absent"
}

# The other half of the same check: these ARE needed, and losing one is
# quiet. build_feature_tables_h() imports librosa and scipy inside a
# try/except, so an excluded scipy does not crash the app -- it just stops
# producing embedded/feature_tables.h, with a warning in the log.
Write-Host "`n-- everything we need is in --"
foreach ($pkg in @("torch", "numpy", "PyQt5", "scipy", "librosa")) {
    Check (Test-Bundled $pkg) "$pkg is bundled"
}

$sizeMb = [math]::Round(((Get-ChildItem $app -Recurse -File |
    Measure-Object Length -Sum).Sum / 1MB), 0)
Write-Host "`n  folder size: $sizeMb MB"

Write-Host "`n-- cold start from an unrelated working directory --"
# The crash handler writes here; make sure a stale one does not fool us.
$crashLogs = @((Join-Path $app "EdgeAiNAS-crash.log"),
               (Join-Path $env:USERPROFILE "Documents\Edge AI NAS\EdgeAiNAS-crash.log"))
foreach ($c in $crashLogs) { if (Test-Path $c) { Remove-Item $c -Force } }

$proc = Start-Process -FilePath $exe -WorkingDirectory ([System.IO.Path]::GetTempPath()) -PassThru
Write-Host "  started pid $($proc.Id), waiting ${StartupSeconds}s ..."
$deadline = (Get-Date).AddSeconds($StartupSeconds)
while ((Get-Date) -lt $deadline -and -not $proc.HasExited) { Start-Sleep -Milliseconds 500 }

Check (-not $proc.HasExited) "still running after ${StartupSeconds}s (the window came up)"
if ($proc.HasExited) { Write-Host "  exit code: $($proc.ExitCode)" -ForegroundColor Red }

foreach ($c in $crashLogs) {
    if (Test-Path $c) {
        Check $false "crash log written at $c"
        Get-Content $c -Tail 25 | ForEach-Object { Write-Host "    $_" }
    }
}

if (-not $proc.HasExited) {
    $proc.CloseMainWindow() | Out-Null
    Start-Sleep -Seconds 2
    if (-not $proc.HasExited) { $proc.Kill() }
    Write-Host "  closed"
}

Write-Host ""
if ($failed -eq 0) { Write-Host "ALL PASS" -ForegroundColor Green; exit 0 }
Write-Host "$failed FAILURE(S)" -ForegroundColor Red
exit 1
