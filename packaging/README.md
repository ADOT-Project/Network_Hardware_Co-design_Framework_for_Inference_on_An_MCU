# Packaging ADOT as a Windows application

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_portable.ps1
```

Produces:

* `portable\` — a self-contained folder. Zip it, unzip it anywhere, run
  `runtime\pythonw.exe app\edge_ai_...py`. No installer needed.
* `dist\ADOT-<version>-setup.exe` — the installer, when Inno Setup 6 is
  installed.

The first run downloads CPython and the dependencies (~10 minutes). After
that the script reuses `portable\runtime` and finishes in seconds; pass
`-Refresh` to rebuild the runtime from scratch.

## How it is shipped: a real Python environment, not a frozen exe

`portable\` is an embeddable CPython 3.10 with `torch`, `PyQt5`, `scipy` and
`librosa` pip-installed into it, plus the application script. The Start-menu
shortcut points at `runtime\pythonw.exe` with the script as its argument —
still one click, and `pythonw` means no console window.

This is deliberate. PyInstaller was tried first and abandoned: the frozen
build reliably failed to load torch's `c10.dll` with WinError 1114, and every
usual cause was ruled out by direct test. See
[`abandoned-pyinstaller/README.md`](abandoned-pyinstaller/README.md) for the
full list of what was eliminated. Shipping an ordinary Python environment
sidesteps the entire class of problem, because torch is then imported by the
normal import machinery — exactly as it is when the script is run from a
checkout, which has always worked.

The trade is size: ~1.1 GB unpacked, 193 MB installer, against PyInstaller's
766 MB / 183 MB. A build that works is worth 30 MB of download.

## Why a desktop build and not a web app

The tool is bound to the local machine at both ends: it reads a `.npz`
dataset and the `feature_config.json` next to it, and it writes C headers by
patching the user's own `CNN_Inference_Edge-AI_with_CLA` checkout, which is
then built in Code Composer Studio and flashed to a board on the same USB
port. A server has access to none of that, so a hosted version would degrade
to "upload dataset, download zip, unpack by hand" — more steps than today,
plus a training backend to pay for.

## What the target machine needs

Nothing. Verified present in the build: the Python interpreter, Qt's
`qwindows.dll` platform plugin (no window without it) and the MSVC runtime
(`vcruntime140`, `vcruntime140_1`, `msvcp140`). No Python install, no Visual
C++ Redistributable. 64-bit Windows and a CPU with AVX, which means anything
from roughly 2013 on.

One correction to that list: `msvcp140.dll` is **not** bundled next to
`python.exe`, and must not be. The embeddable CPython ships the two
`vcruntime140` DLLs but not that one, while `torch_cpu.dll`,
`torch_python.dll` and `c10.dll` all import it — so bundling it looks like
the obvious hardening. Measured on 2026-09-21, it is the opposite: copying
PyQt5's redistributed copy (14.26.28720.3) beside `python.exe` made `import
torch` fail 3 times out of 3 with

```
OSError: [WinError 1114] ... Error loading ...torch\lib\c10.dll
```

and succeed 3 times out of 3 as soon as it was removed. A DLL next to the
interpreter wins the search order, so the old copy shadowed System32's
14.50.35719.0 and torch's initialisers failed against it. **This is very
likely the same WinError 1114 that killed the PyInstaller build** — that one
bundled DLLs as well.

So `msvcp140.dll` comes from the target machine. Every Windows that has had
any VC++ 2015-2022 redistributable installed has it, which in practice is
every machine with Office, a game, a GPU driver or most desktop software. On
a genuinely bare Windows, install
[VC_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) once —
do not put the DLL in the application folder.

The installer defaults to a **per-user** install
(`PrivilegesRequired=lowest`), which lands in `%LOCALAPPDATA%\Programs` — no
administrator prompt, and the folder is writable, so results stay next to the
application. A machine-wide install still works from the first wizard page;
results then go to `Documents\ADOT` — or to `Documents\Edge AI NAS` if that
folder already exists, which is how an upgrade from the pre-1.3.0 name keeps
writing where the previous results are.

## Verifying a build

"The process is still alive" is **not** a passing test — that mistake hid a
broken build for several rounds here. A windowed Python app that dies on
import shows an error dialog and keeps running, so `HasExited` stays false.
Check the main window title instead:

```powershell
$p.Refresh(); $p.MainWindowTitle -like "*ADOT*"
```

`build_portable.ps1` runs a headless self-test (build the window, compute a
ceiling, resolve the firmware templates) before it packages anything.

## Paths: what changes once it is installed

Nothing may assume the current working directory — launched from a shortcut,
the app starts wherever Windows decided. Everything resolves through the
helpers in `# 0z. INSTALL LAYOUT`:

| | source checkout | installed |
|---|---|---|
| `app_dir()` | the repository | `{install}\app` |
| `output_root()` → `export/` | the repository | `{install}\app` when writable, else `Documents\ADOT` |

`find_firmware_templates()` looks for `definitions.h`, `weights.h` and
`main.c` in this order: `EDGE_AI_FIRMWARE_DIR`, then
`CNN_Inference_Edge-AI_with_CLA\` under the current directory, then the same
beside the application, then the copy shipped in the build. The first three
mean an installed build still patches **your** checkout, so hand edits to
`main.c` survive. Only those three files ship; the rest of the 22 MB firmware
tree is never read.

## Dependencies

scikit-learn is not a dependency: the only two things ever used from it —
`f1_score` twice and `confusion_matrix` once — are a few lines of numpy under
`# 0a. METRICS`. `tools/test_metrics.py` checks them against the real
scikit-learn on 15 cases, including the degenerate single-class predictions a
collapsing architecture produces. Run it in a dev environment that still has
scikit-learn before changing them.

scipy and librosa **are** required, despite scipy being a scikit-learn
dependency too. `build_feature_tables_h()` imports `librosa`,
`scipy.fft.dct` and `scipy.signal.get_window` to generate
`embedded/feature_tables.h`. That import sits inside a `try/except`, so
dropping them does not crash the app — it quietly stops emitting the MFCC
tables and logs a warning. An earlier PyInstaller spec excluded scipy for
exactly this reason and shipped a build that silently lost the feature
tables.

## Code signing

The installer and `pythonw.exe` are unsigned, so SmartScreen shows "Windows
protected your PC" on first run until the binary builds reputation: *More
info → Run anyway*. The real fix is an OV/EV code-signing certificate and a
`signtool` step.
