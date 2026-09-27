# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Edge AI NAS.

One-FOLDER build on purpose, not one-file. A one-file build would have to
unpack ~700 MB of torch/Qt into %TEMP% on every launch (10-30 s cold start)
and the self-extracting stub is what most antivirus heuristics flag. The
folder starts in a couple of seconds and is what the Inno Setup script in
this directory packages into an installer.

Build (from the repository root):

    python -m PyInstaller packaging/EdgeAiNAS.spec --noconfirm

Output: dist/EdgeAiNAS/EdgeAiNAS.exe
"""
import os

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = os.path.abspath(os.getcwd())
HOOKS = os.path.join(ROOT, "packaging", "hooks")
ENTRY = os.path.join(
    ROOT, "edge_ai_train_model_gui_quantization_according_succes_rate.py")
FIRMWARE = os.path.join(ROOT, "CNN_Inference_Edge-AI_with_CLA")

# Only the three files write_embedded_files() patches travel with the build.
# The rest of the firmware tree (22 MB, including Debug output) is never read
# by this app; a user who wants their own checkout patched instead points
# EDGE_AI_FIRMWARE_DIR at it or drops it beside the exe.
datas = [(os.path.join(FIRMWARE, name), "firmware_templates")
         for name in ("definitions.h", "weights.h", "main.c")
         if os.path.isfile(os.path.join(FIRMWARE, name))]

# Packages nothing in this app imports. scikit-learn and scipy are the big
# win: the two metrics that needed them are implemented in numpy in the
# application itself, which keeps 152 MB out of the installer.
#
# The TensorFlow block is not optional housekeeping. PyInstaller's torch hook
# walks torch.utils.tensorboard, which imports tensorboard -> tensorflow ->
# keras. If those happen to be installed in the build environment (they are,
# from unrelated work in this one) the analysis spends ten minutes probing
# them in isolated subprocesses, throws access violations, and then hangs --
# all to bundle hundreds of megabytes this app never calls. Nothing here uses
# TensorBoard: training reports through progress_signal, not a writer.
excludes = [
    "tensorflow", "tensorboard", "keras",
    "tensorboard_data_server", "tensorflow_estimator", "tensorflow_io",
    "torch.utils.tensorboard",
    # scikit-learn only: its two metrics are implemented in numpy under
    # "# 0a. METRICS". scipy is NOT excluded -- build_feature_tables_h()
    # imports librosa, scipy.fft.dct and scipy.signal.get_window to generate
    # embedded/feature_tables.h, and that import sits inside a try/except, so
    # dropping scipy would not crash the app, it would quietly stop producing
    # the MFCC tables. librosa drags numba and llvmlite along with it.
    "sklearn", "joblib", "threadpoolctl",
    "matplotlib", "pandas", "PIL", "cv2", "h5py",
    "torchvision", "torchaudio",
    "IPython", "jupyter", "notebook", "nbformat",
    "pytest", "doctest", "unittest", "pydoc",
    "tkinter", "turtle", "curses",
    # Qt modules the UI never touches; the PyQt5 hook is happy to ship all of
    # them otherwise, and QtWebEngine alone is >100 MB
    "PyQt5.QtWebEngineWidgets", "PyQt5.QtWebEngineCore", "PyQt5.QtWebEngine",
    "PyQt5.QtWebKit", "PyQt5.QtWebKitWidgets", "PyQt5.QtWebChannel",
    "PyQt5.QtQml", "PyQt5.QtQuick", "PyQt5.QtQuickWidgets", "PyQt5.Qt3DCore",
    "PyQt5.QtMultimedia", "PyQt5.QtMultimediaWidgets", "PyQt5.QtBluetooth",
    "PyQt5.QtNfc", "PyQt5.QtPositioning", "PyQt5.QtLocation",
    "PyQt5.QtSensors", "PyQt5.QtSerialPort", "PyQt5.QtSql", "PyQt5.QtTest",
    "PyQt5.QtDesigner", "PyQt5.QtHelp", "PyQt5.QtXmlPatterns",
]

a = Analysis(
    [ENTRY],
    pathex=[ROOT],
    binaries=collect_dynamic_libs("torch"),
    datas=datas,
    hiddenimports=[],
    # our hook-torch.py must win over the one in pyinstaller-hooks-contrib;
    # see packaging/hooks/hook-torch.py for why
    hookspath=[HOOKS],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EdgeAiNAS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX is off: it mangles some of torch's DLLs and is another thing that
    # makes antivirus heuristics unhappy, for a few per cent of a 700 MB tree
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EdgeAiNAS",
)
