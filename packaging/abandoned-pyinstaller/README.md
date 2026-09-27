# The PyInstaller attempt (abandoned)

Kept for the record, **not** part of the build. `packaging/build_portable.ps1`
is what produces the shipped application.

The frozen build assembled fine but reliably failed at startup:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading "...\_internal\torch\lib\c10.dll" or one
    of its dependencies.

Ruled out, each by direct test:

| hypothesis | result |
|---|---|
| missing or corrupted files | 3616/3616 present, `c10.dll` and every other torch DLL SHA256-identical to the working install |
| spaces in the install path | copied to `C:\EdgeAiProbe\Edge AI NAS\` — same failure |
| duplicate / shadowed torch DLLs | exactly one copy of each, `torch\lib` matches the source install file for file |
| MSVC runtime mismatch | bundled `vcruntime140`/`msvcp140` identical to System32 (14.50.35719.0) |
| bundled UCRT shadowing the system one | moved `ucrtbase.dll` + 43 `api-ms-win-*` out of `_internal` — same failure |
| duplicate OpenMP runtimes | `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1` — same failure |
| DLL search path | `torch\lib` prepended to PATH — same failure |

Two things this cost, worth remembering:

* **A full build is ~35 minutes**, most of it PyInstaller importing torch's
  ~1938 submodules in isolated subprocesses, hundreds of which crash with
  `Windows fatal exception: access violation` on torch's own DLL load. That
  is a very slow debugging loop.
* **"The process is still alive" is not a passing test.** A windowed
  PyInstaller build shows an error *dialog* and keeps running, so an early
  smoke test that only checked `HasExited` reported success for builds that
  were in fact broken. The portable build's checks read the main window
  title instead.

`hook-torch.py` here is still independently useful reading: the stock
`pyinstaller-hooks-contrib` torch hook calls `collect_submodules("torch")`,
which *imports* `torch.utils.tensorboard`, which imports tensorboard →
tensorflow → keras. If those are installed in the build environment, the
analysis grinds for ten minutes and bundles hundreds of megabytes of them.
Listing `tensorflow` in the spec's `excludes` does not help, because
`excludes` filters the resulting graph while the damage is done earlier, by
the hook's own imports.
