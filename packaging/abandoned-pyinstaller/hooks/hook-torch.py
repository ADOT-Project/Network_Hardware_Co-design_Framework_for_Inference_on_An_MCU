"""Replaces the torch hook that ships with pyinstaller-hooks-contrib.

The stock hook ends up collecting TensorFlow. `collect_submodules("torch")`
walks the package with pkgutil, and walking *imports* what it walks, so it
imports `torch.utils.tensorboard`, which imports `tensorboard`, which imports
`tensorflow` and `keras`. If those happen to be installed in the build
environment, the analysis spends ten minutes probing them in isolated
subprocesses, throws access violations, and then wedges -- all to bundle
hundreds of megabytes this app never calls.

Putting `tensorflow` in the spec's `excludes` does not help: `excludes`
filters the resulting module graph, while the damage is done earlier, by the
hook's own imports.

So this hook does what the stock one does, minus the submodules we know are
dead weight, using `collect_submodules`' `filter` argument -- which is applied
before the walk descends, so the offending packages are never imported.

PyInstaller searches user hook directories before the bundled ones, so this
file wins as long as `packaging/hooks` is on the spec's `hookspath`.

NOTE: the stock hook also collects MKL DLLs on Windows. This build's torch
(2.9.1+cpu) does not depend on MKL -- the stock hook logged "this torch build
does not depend on MKL" before this replacement existed -- so that part is
deliberately not reproduced. If you move to a torch build that needs MKL, the
DLLs will have to be collected here.
"""

from PyInstaller.utils.hooks import (PY_DYLIB_PATTERNS, collect_data_files,
                                     collect_dynamic_libs, collect_submodules,
                                     logger)

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

# Subpackages this application never reaches. Each is either a training
# integration we do not use, or a developer tool.
SKIP_PREFIXES = (
    "torch.utils.tensorboard",   # the tensorflow magnet described above
    "torch.utils.bottleneck",
    "torch.utils.benchmark",
    "torch.testing",
)


def _wanted(name):
    return not name.startswith(SKIP_PREFIXES)


datas = collect_data_files(
    "torch",
    excludes=["**/*.h", "**/*.hpp", "**/*.cuh", "**/*.lib", "**/*.cpp",
              "**/*.pyi", "**/*.cmake"],
)

hiddenimports = collect_submodules("torch", filter=_wanted)
logger.info("hook-torch (local): collected %d submodules, skipping %r",
            len(hiddenimports), SKIP_PREFIXES)

binaries = collect_dynamic_libs(
    "torch",
    search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"],
)
