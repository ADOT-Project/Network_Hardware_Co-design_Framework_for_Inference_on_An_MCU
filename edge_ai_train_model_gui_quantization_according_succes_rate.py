"""
ADOT - BLDC motor acoustic classification (C2000 / F28379D target)

Works with the output of prepare_features.py:
    folds/fold0..4.npz  -> architecture search and accuracy reporting
    folds/full.npz      -> final on-device model (contains no X_test)

Changes vs. the previous version:
  * Dataset picked from the GUI; input shape and class count read from the file
  * Model selection / early stopping use X_val (X_test is never touched)
  * (N,32,12) -> unsqueeze(1);  permute(0,3,1,2) removed
  * Best-epoch weights are kept, seed fixed
  * Quantization scale written as a real C constant
  * Quantize-dequantize accuracy measured before export
  * mu / sigma written into the header
"""

import datetime
import json
import os
import re
import sys

import numpy as np


def _import_torch(attempts=4):
    """`import torch`, with a few retries.

    Loading torch's DLLs can fail with WinError 1114 ("a dynamic library
    initialization routine failed") on the very first launch after an
    install, while the antivirus is still working through several hundred
    megabytes of freshly written binaries. Observed exactly once, seconds
    after setup finished; the same files -- byte-identical, verified by
    hash -- loaded fine on every later attempt and from other directories.

    Retrying costs nothing when the import works first time, and is the
    difference between a new user's first run working and a raw Python
    traceback. Python drops a module from sys.modules when its import
    raises, so each attempt is a genuine retry, not a cached failure.
    """
    import time
    last = None
    for attempt in range(attempts):
        try:
            import torch as _torch
            return _torch
        except OSError as exc:
            last = exc
            time.sleep(1.0 + attempt)
    raise last


torch = _import_torch()
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import (QColor, QDesktopServices, QFont, QPainter,
                         QPainterPath, QPen, QTextDocument)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow, QProgressBar,
                             QMessageBox, QPushButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QTextEdit,
                             QVBoxLayout, QWidget)

# ==========================================
# 0z. INSTALL LAYOUT
# ==========================================
# Nothing here may assume the current working directory. Launched from a
# shortcut, an installed build starts in whatever directory Windows felt
# like, and it may live under Program Files, where it cannot write next to
# itself. Both of those would silently break the export step, so every path
# the app reads or writes is resolved through the helpers below.

def app_dir():
    """The directory the application lives in: the folder holding the .exe
    for a frozen build, the folder holding this file when run from source."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts):
    """Read-only data shipped with the app. PyInstaller unpacks it to
    sys._MEIPASS in one-file mode and leaves it beside the exe otherwise."""
    return os.path.join(getattr(sys, "_MEIPASS", app_dir()), *parts)


def _is_writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w"):
            pass
        os.remove(probe)
        return True
    except OSError:
        return False


def install_crash_handler():
    """A frozen windowed build has nowhere to print a traceback -- there is no
    console, so an unhandled exception just looks like the window vanishing.
    Write it down and say where. PyQt5 routes exceptions raised inside slots
    through sys.excepthook before aborting, so this catches those too."""
    log_path = os.path.join(output_root(), "ADOT-crash.log")

    def hook(exc_type, exc, tb):
        import traceback
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== {stamp} =====\n{text}")
        except OSError:
            log_written = False
        else:
            log_written = True
        sys.__excepthook__(exc_type, exc, tb)
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None, "ADOT",
                f"{exc_type.__name__}: {exc}\n\n"
                + (f"Details were written to:\n{log_path}" if log_written
                   else "The crash log could not be written."))

    sys.excepthook = hook
    return log_path


def output_root():
    """Where export/ is created. Beside the app when that is writable -- a
    source checkout or a portable unzip, where keeping results next to the
    tool is what you want -- otherwise the user's Documents, because an
    installed build under Program Files is not writable."""
    here = app_dir()
    if _is_writable(here):
        return here
    # The app was called "Edge AI NAS" up to 1.2.1 and its results landed in
    # Documents\Edge AI NAS. Renaming to ADOT must not orphan those: an
    # existing folder from the old name keeps being used, and only a fresh
    # installation gets the new one.
    documents = os.path.join(os.path.expanduser("~"), "Documents")
    legacy = os.path.join(documents, "Edge AI NAS")
    if os.path.isdir(legacy):
        return legacy
    docs = os.path.join(documents, "ADOT")
    os.makedirs(docs, exist_ok=True)
    return docs


SEED = 1337
# Phase 2 stops shrinking if the score falls more than this many points below
# the best score seen so far, even when the target is still met. Without it a
# low target lets the search collapse to a degenerate 1-channel model.
DEGRADE_TOL = 1.5

# Robustness check: a single seed + a single (clean) val score can let phase 2
# shrink to a fragile micro-architecture that only *looks* perfect because it
# got lucky and/or the val set has no channel/domain shift to expose it (seen
# empirically: a 1-channel model hit 100% on same-recording val but collapsed
# to ~30% on a held-out physical unit, while 2+ channels held up). Each
# candidate channel count is now trained with N_SEEDS different seeds and
# also scored under a synthetic channel-shift perturbation (gain/tilt/noise
# applied to features at eval time only, never during training); the accept/
# reject decision uses the WORST of all those scores, not a single clean run.
N_SEEDS = 3
PERTURB_TRIALS = 2
PERTURB_GAIN = (0.75, 1.35)      # +/- gain (speaker/mic level mismatch)
PERTURB_TILT = 0.25              # cross-coefficient tilt (frequency-response mismatch)
PERTURB_NOISE = 0.4              # extra noise (features are ~unit-scale)


# ==========================================
# 0a. METRICS
# ==========================================
# Only two things were ever used out of scikit-learn, and both are a few
# lines of numpy. Doing them here drops scikit-learn AND its scipy
# dependency -- 152 MB that would otherwise have to ship inside the
# installer for three function calls. Behaviour matches
# sklearn.metrics.{confusion_matrix, f1_score} for the way they are called
# below (see tools/test_metrics.py, which checks them against sklearn).

def confusion_matrix(y_true, y_pred, labels):
    """counts[i][j] = samples of true class labels[i] predicted as labels[j].
    Values outside `labels` are ignored, as sklearn does."""
    labels = [int(v) for v in labels]
    n = len(labels)
    pos = {v: i for i, v in enumerate(labels)}
    t = np.array([pos.get(int(v), -1) for v in np.asarray(y_true).ravel()],
                 dtype=np.int64)
    p = np.array([pos.get(int(v), -1) for v in np.asarray(y_pred).ravel()],
                 dtype=np.int64)
    keep = (t >= 0) & (p >= 0)
    flat = np.bincount(t[keep] * n + p[keep], minlength=n * n)
    return flat.reshape(n, n).astype(np.int64)


def f1_score(y_true, y_pred, average="macro", zero_division=0):
    """Macro-averaged F1 over the labels appearing in either array -- the
    label set sklearn defaults to, which is what the search thresholds were
    calibrated against."""
    if average != "macro":
        raise ValueError("only average='macro' is implemented")
    labels = np.union1d(np.unique(np.asarray(y_true)),
                        np.unique(np.asarray(y_pred)))
    if labels.size == 0:
        return 0.0
    cm = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)   # column totals
    actual = cm.sum(axis=1).astype(np.float64)      # row totals
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, float(zero_division))
        rec = np.where(actual > 0, tp / actual, float(zero_division))
        denom = prec + rec
        f1 = np.where(denom > 0, 2.0 * prec * rec / denom, 0.0)
    return float(f1.mean())


# ==========================================
# 0b. TARGET DEVICES -- the C2000 parts that carry a CLA
# ==========================================
# Picking a device rewrites the three memory budgets the ceiling search runs
# against and hands the timing model a clock. The budgets stay editable
# afterwards on purpose: on a type-1/type-2 CLA the LSx blocks are switched
# between "CLA program" and "CLA data" in software, so how much CLA data RAM
# a project actually gets is a decision in its .cmd file, not a number in the
# datasheet.
#
# All sizes are STORAGE bytes, not address units. The C28x is word-addressed
# (CHAR_BIT == 16), so TI's "10240 words" of CLA RAM is 20 KB here -- the
# same convention calculate_metrics() and dtype_size() use.
#
# `pool_exact` marks the entries whose CLA-visible RAM pool is a documented
# figure rather than the family's usual arrangement:
#   F28379D / F28377S / F28075  LS0..LS5, 4 KB each          -> 24 KB
#   F280049C                    LS0..LS7, 4 KB each          -> 32 KB
#   F28069                      L0 (4 KB) + L1 + L2 (2 KB)   ->  8 KB data,
#                               L3 as CLA program            ->  8 KB
# The rest are the family's typical layout and are flagged in the dialog as
# a starting point to check against your own linker command file.
#
# flash_kb is per C28x core; ram_kb is the SRAM ONE C28x can reach (for the
# dual-core parts that is its local RAM plus the shared global blocks, not
# the device total, which the note gives).

# What the firmware itself costs in flash before a single weight is stored:
# CNN_Inference_Edge-AI_with_CLA's Debug/dense_layer_cla.map totals 0x2947 +
# 0x857 + 0xd3fa = 66968 words = 131 KB of code, MFCC tables and the three
# golden vectors. 160 KB is that with slack, and is what a device's flash
# budget is offered as: everything left over after the firmware.
FIRMWARE_FLASH_KB = 160

C2000_DEVICES = [
    dict(key="F28379D", name="TMS320F28379D", family="F2837xD", cores=2,
         cpu_mhz=200, cla_mhz=200, cla_type=1,
         flash_kb=1024, ram_kb=164, pool_kb=24, cla_data_kb=20, cla_prog_kb=4,
         pool_exact=True,
         note="This project's board, and the only entry whose CLA split is "
              "measured rather than typical: 2837xD_FLASH_CLA_lnk_cpu1.cmd "
              "gives RAMLS_0_1_2_3_4 = 0x2800 words (20 KB) to data and "
              "RAMLS5_PROG = 0x800 words (4 KB) to program. Device total is "
              "204 KB of RAM; 164 KB of it is reachable from one C28x."),
    dict(key="F28377S", name="TMS320F28377S", family="F2837xS", cores=1,
         cpu_mhz=200, cla_mhz=200, cla_type=1,
         flash_kb=1024, ram_kb=164, pool_kb=24, cla_data_kb=20, cla_prog_kb=4,
         pool_exact=True,
         note="Single-core F2837x: same 200 MHz C28x + CLA and the same "
              "LS0..LS5 pool as the F28379D, with all 164 KB on one core."),
    dict(key="F28075", name="TMS320F28075", family="F2807x", cores=1,
         cpu_mhz=120, cla_mhz=120, cla_type=1,
         flash_kb=512, ram_kb=100, pool_kb=24, cla_data_kb=20, cla_prog_kb=4,
         pool_exact=True,
         note="F2807x: the F2837x LS layout at 120 MHz. The 1.67x slower "
              "clock shows up directly in the timing model."),
    dict(key="F28388D", name="TMS320F28388D", family="F2838x", cores=2,
         cpu_mhz=200, cla_mhz=200, cla_type=1,
         flash_kb=512, ram_kb=172, pool_kb=24, cla_data_kb=20, cla_prog_kb=4,
         pool_exact=False,
         note="Two C28x+CLA cores plus a Connectivity Manager; 338 KB of RAM "
              "on the device, of which one C28x reaches roughly its 44 KB of "
              "local RAM plus the 128 KB global pool. CLA split assumed to "
              "follow the F2837x layout -- confirm in your .cmd file."),
    dict(key="F28P650DK", name="TMS320F28P650DK", family="F28P65x", cores=2,
         cpu_mhz=200, cla_mhz=200, cla_type=2,
         flash_kb=640, ram_kb=160, pool_kb=32, cla_data_kb=24, cla_prog_kb=8,
         pool_exact=False,
         note="1.28 MB of flash and 248 KB of RAM on the device, listed here "
              "per C28x core. CLA pool assumed 32 KB (8 LS blocks) -- "
              "confirm in your .cmd file."),
    dict(key="F28P550SJ", name="TMS320F28P550SJ", family="F28P55x", cores=1,
         cpu_mhz=150, cla_mhz=150, cla_type=2,
         flash_kb=1088, ram_kb=133, pool_kb=32, cla_data_kb=24, cla_prog_kb=8,
         pool_exact=False,
         note="Carries a TinyEngine NPU next to the CLA. This tool only maps "
              "the network onto the C28x and the CLA -- the NPU is not "
              "modelled here. CLA pool assumed 32 KB."),
    dict(key="F280049C", name="TMS320F280049C", family="F28004x", cores=1,
         cpu_mhz=100, cla_mhz=100, cla_type=2,
         flash_kb=256, ram_kb=100, pool_kb=32, cla_data_kb=24, cla_prog_kb=8,
         pool_exact=True,
         note="LS0..LS7, 4 KB each, every one of them switchable to the CLA "
              "as program or as data -- the largest CLA pool in the list "
              "relative to the device. Split shown is 24 KB data + 8 KB "
              "program; move the line in your .cmd file if you need to."),
    dict(key="F280039C", name="TMS320F280039C", family="F28003x", cores=1,
         cpu_mhz=120, cla_mhz=120, cla_type=2,
         flash_kb=384, ram_kb=69, pool_kb=16, cla_data_kb=12, cla_prog_kb=4,
         pool_exact=False,
         note="69 KB of RAM on the device; the CLA-visible pool is assumed "
              "to be 16 KB here -- confirm in your .cmd file."),
    dict(key="F280025C", name="TMS320F280025C", family="F28002x", cores=1,
         cpu_mhz=100, cla_mhz=100, cla_type=2,
         flash_kb=128, ram_kb=24, pool_kb=8, cla_data_kb=6, cla_prog_kb=2,
         pool_exact=False,
         note="Smallest part in the list: 24 KB of RAM total. Check your "
              "exact part number -- the CLA is not fitted on every F28002x "
              "variant. Pool figures assumed."),
    dict(key="F28069", name="TMS320F28069", family="F2806x", cores=1,
         cpu_mhz=90, cla_mhz=90, cla_type=0,
         flash_kb=256, ram_kb=100, pool_kb=None, cla_data_kb=8, cla_prog_kb=8,
         pool_exact=True,
         note="Type-0 CLA with fixed blocks rather than a switchable pool: "
              "L0 (4 KB) + L1 + L2 (2 KB each) as data, L3 (8 KB) as "
              "program. No background task."),
    dict(key="F28035", name="TMS320F28035", family="F2803x", cores=1,
         cpu_mhz=60, cla_mhz=60, cla_type=0,
         flash_kb=128, ram_kb=20, pool_kb=None, cla_data_kb=4, cla_prog_kb=2,
         pool_exact=False,
         note="The original CLA part: 60 MHz, 20 KB of RAM, two 1 Kword CLA "
              "data RAMs. The C28x here has no FPU, so the CLA is the only "
              "float unit on the device."),
]

CUSTOM_DEVICE = "CUSTOM"


def device_by_key(key):
    for d in C2000_DEVICES:
        if d["key"] == key:
            return d
    return None


def device_budgets(dev):
    """The three budget fields and the two clocks a device implies.

    Flash is what is LEFT for weights after the firmware (see
    FIRMWARE_FLASH_KB), never the whole part, and never less than 16 KB --
    on the small parts that floor is a reminder that the firmware is the
    problem, not the model."""
    return {
        "flash_kb": float(max(16, dev["flash_kb"] - FIRMWARE_FLASH_KB)),
        "cpu_kb": float(dev["ram_kb"]),
        "cla_kb": float(dev["cla_data_kb"]),
        "cla_prog_kb": float(dev["cla_prog_kb"]),
        "cpu_mhz": float(dev["cpu_mhz"]),
        "cla_mhz": float(dev["cla_mhz"]),
    }


def device_spec_line(dev):
    pool = (f"{dev['pool_kb']:g} KB CPU/CLA pool"
            if dev["pool_kb"] else "fixed CLA blocks")
    return (f"{dev['family']}  ·  {dev['cores']}x C28x @ {dev['cpu_mhz']} MHz  ·  "
            f"CLA type {dev['cla_type']} @ {dev['cla_mhz']} MHz  ·  "
            f"{dev['flash_kb']} KB flash/core  ·  {dev['ram_kb']} KB RAM  ·  "
            f"{pool} → {dev['cla_data_kb']} KB data + "
            f"{dev['cla_prog_kb']} KB program")


# ==========================================
# 0c. TIMING MODEL -- clock cycles, not a stopwatch
# ==========================================
# Every number the ceiling search already derives is an operation count: the
# conv stages' MACs fall straight out of the tensor shapes, and the MFCC's
# out of the feature configuration. Turning those into microseconds needs
# one more thing -- what an operation costs in cycles on each domain -- and
# that is what this section pins down, so the tool can answer the question
# the memory gauges cannot: is feature extraction or inference the thing
# that actually takes the frame?
#
# The costs below are for the kernels this firmware really runs, not for a
# datasheet peak. Both halves are ANCHORED to measurements taken on the
# board (2026-09-26, 200 MHz F28379D, the Debug build in this repository):
#
#   inference  6 channels, 65x12 input, S1+S2+S3 in one CLA task, the
#              GPIO67 toggle in Cla1Task1 -> 73 ms. 14.6 M cycles for
#              108,072 MACs, i.e. 135 effective cycles a MAC.
#   MFCC       one pass of mfcc_extract(), 65 frames, 512-point FFT,
#              40 mel bands -> 202 ms. 40.4 M cycles for 2.74 M modelled
#              float operations: 14.6 cycles each, which is CYC_FLOP.
#
# They were measured on different cores through different code, and each
# asks the hand count for almost the same correction: 4.48 on the CLA, 4.88
# on the C28x. One unoptimised build, one factor -- which is why CPU_TUNE is
# allowed to borrow the CLA's. Hand-counting both kernels off the same
# primitive table then puts a conv MAC at ~73 effective cycles on the C28x
# against the CLA's measured 135: the CLA is about 1.8x the more expensive
# place to run this kernel.
#
# That is the opposite of what a datasheet suggests (same clock, same float
# width) and it is not a paradox: this kernel is index arithmetic and int16
# conversions wrapped around a multiply-add, and the C28x brings a 32-bit
# integer ALU, RPTB hardware loops and richer addressing to exactly that,
# where the CLA has a narrow integer side, no hardware loop and three delay
# slots on every branch. So on this build the CLA does not make a stage
# faster -- it makes it CONCURRENT, which is a different and often better
# thing, and it is what the frame budget is there to show.
#
# CAVEAT, and it is a big one: both anchors are from the DEBUG build --
# unoptimised code, nothing unrolled, CLA program RAM at 2042 of 2048 words.
# They describe this firmware as it is today, not the silicon's ceiling. A
# release build with unrolled taps and pointer walks instead of recomputed
# uint32 indices would move every number here, so re-measure before quoting
# any of it.
#
# 135 cycles for a multiply and an add wants explaining. The multiply and
# the add are not the cost; everything around them is:
#
#   * the operands are int16 and the accumulator is float, so every MAC is
#     two loads and two MI16TOF32 conversions before the MMPYF32/MADDF32
#     (dense_layer.cla and convolution.c both do
#     `sum += (float)input[..] * (float)filters[..]`);
#   * the indices are uint32 expressions recomputed per iteration
#     (`row_ptr + (c + fj)`, `filter_offset + fi*FILTER_SIZE + fj`), and the
#     CLA is a float machine -- its integer side is narrow, so address
#     arithmetic costs real instructions there;
#   * there is no hardware loop on the CLA and MBCNDD carries three delay
#     slots, so a three-iteration inner loop pays a branch tail per MAC;
#   * and CLA program RAM is at 2042 of 2048 words, which means this is a
#     build optimised for SIZE -- nothing is unrolled.
#
# The same three things are why a stage is priced per ACCUMULATION as well
# as per MAC: each `sum` ends with a bias multiply-add, a ReLU compare, an
# int16 conversion, a pool compare and -- the expensive one -- a float
# DIVISION by the quantization scale, which neither core has an instruction
# for (`sum / CONV1_WEIGHT_SCALE`, and `sum / FC_WEIGHT_SCALE` per class in
# the dense stage).
#
# One effect pulls the other way and is folded into the C28x's price
# already: a CPU stage reads its weights from flash with wait states, where
# a CLA stage reads LS RAM in a single cycle. It is not enough to close a
# 2.5x gap. Both prices stay editable, and typing a measured inference time
# re-anchors them from the GUI without touching this file.
# The two measurements, as DATA. Everything priced per MAC or per float
# operation below is solved from these, so correcting one of them is a
# one-line change and never a hunt for a magic number.
#
# The inference anchor is the build in CNN_Inference_Edge-AI_with_CLA: its
# Cla1Task1 toggles GPIO67 around conv1+conv2+dense, which is what a scope
# sees. Its channel count is not a guess -- the .map pins it down, since
# NUM_FILTERS is baked into every weight array's size:
#     _conv2_weights - _conv1_weights = 0x906d - 0x9037 =  54 words = 9*1*F
#     _weight        - _conv2_weights = 0x91b1 - 0x906d = 324 words = 9*F*F
#     _fc_bias       - _workspace     = 0x9028 - 0x8640 = 2536 words
#                                     = F*POOL1 + F*POOL2
# all three give F = 6, and that .map was built the same day as the .cla
# file it describes.
CLA_ANCHOR = {"channels": 6, "input_h": 65, "input_w": 12, "classes": 3,
              "layers": 2, "pool_w": False, "mhz": 200.0, "ms": 73.0,
              "tasks": 1}
# Held out of the fit on purpose: the same firmware rebuilt at 4 channels
# and measured the same way, 2026-09-27. Nothing is solved from it -- it is
# the only number here that is free to disagree, which is what makes it
# worth quoting.
CLA_VALIDATION = {"channels": 4, "input_h": 65, "input_w": 12, "classes": 3,
                  "layers": 2, "pool_w": False, "mhz": 200.0, "ms": 34.0,
                  "tasks": 1}
MFCC_ANCHOR = {"input_h": 65, "input_w": 12, "sr": 16000, "n_fft": 512,
               "hop": 256, "n_mels": 40, "mhz": 200.0, "ms": 202.0,
               # not the 2*bins approximation the fallback uses: this is
               # FEAT_MEL_NNZ out of the firmware's own feature_tables.h
               "mel_nnz": 490}
# What the 200 ms window contains, since an anchor is only as good as the
# thing it was measured around. main.c toggles GPIO18 over exactly two
# statements: mfcc_extract(), and the 65x12 loop that scales its output by
# INPUT_SCALE, clamps it and stores it as int16. That second loop is 0.06%
# of the work, so the figure is mfcc_extract() and the model is right to
# treat it as such.
#
# Two things it is NOT, both of which would make it an over-estimate if you
# ever compare against a live system:
#   * the input is golden_inputs[], a const array in FLASH, and the
#     function reads it twice -- 33,792 16-bit flash reads. With the
#     prefetch buffer that is ~0.1% of the total, so it does not matter
#     here, but an ADC buffer in RAM would be no slower.
#   * there is no waiting for audio in it. The 1056 ms the window spans in
#     real time is not part of this number, and must not be added to it:
#     acquisition overlaps everything else.
# Which channel count the inference figure belongs to took two rounds to
# settle, and the wrong answer was argued for first. The committed
# Debug/dense_layer_cla.map is a 6-channel build -- conv1_weights 54 words =
# 9*1*F, conv2_weights 324 = 9*F*F, workspace 2536 = F*(31*10+14*8), and
# definitions.h agrees -- but a plausibility argument was preferred over it:
# the hand count below needed 1.8x to reach 70 ms at 10 channels against
# 4.3x at 6, and 4.3x looked like the instruction counting being wrong.
#
# Re-measuring the firmware as it stands settled it: 6 channels, 73 ms. The
# .map was right and the plausibility argument was not. Worth keeping as a
# note to self -- a static fact about the binary beat a chain of reasoning
# about what a number ought to be.
#
# The reading is now also self-consistent, which the other one was not. The
# two measurements are on different cores through different code, and the
# correction each one asks of the hand count comes out at 4.48 (CLA) and
# 4.88 (C28x) -- one build, one optimisation level, one factor, agreeing to
# within 9%. At 10 channels those were 1.81 and 4.88.

# --- primitives ---------------------------------------------------------
# What each core pays for one operation of the kernels in dense_layer.cla
# and convolution.c, counted by hand from the instruction set. These set the
# SHAPE of the estimate -- how conv1 compares to conv2, what another channel
# count costs -- while their absolute level is corrected by the tune factor
# solved from the measurement further down. Keeping the two apart is the
# point: the structure comes from the code, the level from the board.
#
# The CLA and the C28x differ in exactly three of them, which is the whole
# story of why the same C is slower on the CLA: integer adds, integer
# multiplies and the loop tail. The C28x has a 32-bit integer ALU and RPTB
# hardware loops; the CLA has a narrow integer side and MBCNDD's three delay
# slots. The float and memory work is identical on both.
_PRIM_COMMON = {"load": 1.0, "store": 1.0, "i2f": 1.0, "f2i": 2.0,
                "fmul": 2.0, "fadd": 2.0, "div": 30.0}
CLA_PRIMITIVES = dict(_PRIM_COMMON, iadd=3.0, imul=8.0, loop=6.0)
CPU_PRIMITIVES = dict(_PRIM_COMMON, iadd=1.0, imul=2.0, loop=2.0)

CYC_MISC = 1.5        # a compare or a store on its own
CYC_LOG = 75.0        # log10f() from the RTS library
CYC_SINCOS = 60.0     # one cosf + sinf pair (FFT twiddle seed, once per stage)
CYC_DIV = 24.0        # float division
CLA_TASK_OVERHEAD = 60.0   # trigger a Cla1TaskN and collect its result

# Used when no feature_config.json can be found next to the dataset: the
# pipeline prepare_features.py ships with. n_frames and the coefficient
# count are never assumed -- they come from the model's input shape.
FEAT_ASSUMED = {"sr": 16000, "n_fft": 512, "hop": 256, "n_mels": 40}


def feature_plan(input_h, input_w, fcfg=None):
    """What the MFCC front end has to compute for ONE window.

    `fcfg` is a parsed feature_config.json when one was found. Whatever is
    missing falls back to FEAT_ASSUMED and the plan says so, because an
    assumed n_fft changes the FFT cost by a factor of two."""
    src = fcfg or {}
    sr = int(src.get("sr", FEAT_ASSUMED["sr"]))
    n_fft = int(src.get("n_fft", FEAT_ASSUMED["n_fft"]))
    hop = int(src.get("hop", FEAT_ASSUMED["hop"]))
    n_mels = int(src.get("n_mels", FEAT_ASSUMED["n_mels"]))
    n_frames, n_out = int(input_h), int(input_w)
    # center=False, so the window is exactly what the frames span -- the same
    # arithmetic prepare_features.py used (65 frames, hop 256, n_fft 512 ->
    # 16896 samples, which is FEAT_SLICE_LEN in the firmware's header).
    slice_len = int(src.get("slice_len", hop * (n_frames - 1) + n_fft))
    bins = n_fft // 2 + 1
    # Sum of the mel filters' lengths. The triangles overlap by half, so the
    # total is about twice the bin count -- the real header's FEAT_MEL_NNZ is
    # 490 against 2*257 = 514 for this configuration.
    mel_nnz = int(src.get("mel_nnz", 2 * bins))
    return {"sr": sr, "n_fft": n_fft, "hop": hop, "n_mels": n_mels,
            "n_frames": n_frames, "n_out": n_out, "slice_len": slice_len,
            "bins": bins, "mel_nnz": mel_nnz,
            "window_ms": 1000.0 * slice_len / sr,
            "assumed": fcfg is None}


def mfcc_ops(plan):
    """What one pass of mfcc_extract() has to do, counted, before anything
    is priced: float operations, the library calls and divisions that are
    not priced per flop, the FFT's share, and the working set.

    Follows that file statement by statement rather than quoting a headline
    figure -- including the two places it is deliberately naive: the power
    spectrum is recomputed inside the mel loop (once per non-zero weight,
    not once per bin), and the FFT is a textbook radix-2 complex transform
    on a real signal, which the file itself notes is roughly 10x the cost of
    C2000Ware's RFFT_f32.

    Split out from mfcc_cost() so the flop price can be SOLVED from the
    measured anchor instead of being typed in."""
    n_fft, n_mels = plan["n_fft"], plan["n_mels"]
    n_frames, n_out = plan["n_frames"], plan["n_out"]
    stages = max(1, int(round(np.log2(n_fft))))
    butterflies = (n_fft // 2) * stages

    # per frame
    f_window = 2 * n_fft                    # windowed real part + zeroed imag
    f_fft = 16 * butterflies                # 6 for the product, 4 for the
                                            # add/sub pair, 6 for the twiddle
                                            # recurrence
    f_mel = 5 * plan["mel_nnz"]             # |X|^2 (3) + weighted accumulate (2)
    f_dct = 2 * n_out * n_mels
    flops = n_frames * (f_window + f_fft + f_mel + f_dct)
    flops += 3 * plan["slice_len"]          # mean, subtract, scale
    flops += 2 * n_frames * n_mels          # global max pass + floor pass
    fixed = n_frames * (CYC_SINCOS * stages + CYC_LOG * n_mels
                        + n_out * (CYC_DIV + 2 * CYC_MISC))
    # Working set: x[slice_len], log_mel[n_frames][n_mels], fft_re/im[n_fft],
    # all float (4 storage bytes each on this target).
    ram = 4 * (plan["slice_len"] + n_frames * n_mels + 2 * n_fft)
    return {"flops": flops, "fixed_cycles": fixed,
            "fft_flops": n_frames * f_fft, "ram_bytes": ram}


def mfcc_cost(plan):
    """Cycles and RAM for one pass of mfcc_extract(), at CYC_FLOP."""
    o = mfcc_ops(plan)
    cycles = CYC_FLOP * o["flops"] + o["fixed_cycles"]
    return {"cycles": cycles, "ram_bytes": o["ram_bytes"],
            "fft_share": CYC_FLOP * o["fft_flops"] / max(cycles, 1.0)}


def stage_op_counts(ch, layers, ih, iw, ncls, pool_w):
    """What each pipeline stage has to compute, in the order the diagram
    draws them: one entry per fused conv+pool stage, then the dense stage.

    One count per LOOP LEVEL of the kernels in dense_layer.cla, because the
    levels cost different things and do not scale together -- conv2's
    innermost tap is wrapped in an in_c loop that conv1 does not have, so a
    model that only counts MACs gets the ratio between them wrong, and gets
    it wrong differently at every channel count the search tries:

      macs     `sum += (float)a[i] * (float)b[j]`    -- the fj body
      taps     the same thing, named for what it is  -- an alias of macs
      fi       one `row_ptr = (r + fi) * WIDTH`      -- once per 3 taps
      inc      one in_c iteration: in_offset and filter_offset  (conv2 only)
      windows  one completed accumulation: bias, ReLU, the float division,
               the int16 conversion, the pool compare. For 2x1 pooling
               there are two per pooled output -- the `sub` loop runs twice
               before the kernel compares.
      pooled   one stored output, with its own index arithmetic
      rows     one `pr` iteration, channels one `f` iteration

    Fused means the unpooled rows are never stored -- and never computed
    either, so only the rows a pooled output actually needs are counted,
    exactly like conv1_pool_fused_cla()."""
    out = []
    h, w, in_ch = ih, iw, 1
    for _ in range(layers):
        ch_h, ch_w = h - 2, w - 2                      # valid 3x3 conv
        ph, pw = ch_h // 2, (ch_w // 2 if pool_w else ch_w)
        rows = 2 * ph                                  # rows the pool consumes
        cols = 2 * pw if pool_w else ch_w
        windows = ch * rows * cols
        out.append({"macs": windows * in_ch * 9,
                    "taps": windows * in_ch * 9,
                    "fi": windows * in_ch * 3,
                    # conv1 has no in_c loop at all: its fi loop sits
                    # directly inside `sub`
                    "inc": windows * in_ch if in_ch > 1 else 0,
                    "windows": windows,
                    "pooled": ch * ph * pw,
                    "rows": ch * ph,
                    "channels": ch})
        h, w, in_ch = ph, pw, ch
    fc_in = ch * h * w
    # The dense stage is two loops: one accumulation per class, with a
    # division at the end of it, and no pooling or row/channel structure.
    out.append({"macs": fc_in * ncls, "taps": fc_in * ncls, "fi": 0,
                "inc": 0, "windows": ncls, "pooled": 0, "rows": 0,
                "channels": 0})
    return out


# --------------------------------------------------------------------------
# The two constants nothing above could state: solved from the anchors, now
# that there is something to count the anchors' work with.
# --------------------------------------------------------------------------
def _solve_flop_cycles(anchor):
    """Cycles per modelled float operation, from a measured MFCC time."""
    plan = feature_plan(anchor["input_h"], anchor["input_w"], anchor)
    o = mfcc_ops(plan)
    target = anchor["ms"] / 1000.0 * anchor["mhz"] * 1e6
    return max(0.1, (target - o["fixed_cycles"]) / o["flops"])


def level_costs(prim):
    """What one iteration of each loop level costs, assembled from the
    primitives -- i.e. the kernel read as instructions rather than as
    mathematics. Straight off dense_layer.cla:

      tap     `sum += (float)input[row_ptr + (c + fj)] *
                      (float)filters[filter_offset + fi*FILTER_SIZE + fj]`
              two loads, two int-to-float conversions, a multiply, an add,
              two index adds (the multiply by FILTER_SIZE is constant and
              strength-reduced), and the fj loop's own tail.
      fi      `row_ptr = (r + fi) * WIDTH` -- an add, a multiply, a tail.
      inc     in_offset and filter_offset, two multiplies and an add.
      window  `sum = 0`, the bias multiply-add, the ReLU compare, the
              DIVISION by the weight scale, the int16 conversion, the pool
              compare, and the `sub` loop's tail.
      pooled  the store, its index arithmetic, and the `c` loop's tail.
    """
    p = prim
    return {
        "taps": 2 * p["load"] + 2 * p["i2f"] + p["fmul"] + p["fadd"]
                + 2 * p["iadd"] + p["loop"],
        "fi": p["iadd"] + p["imul"] + p["loop"],
        "inc": 2 * p["imul"] + p["iadd"] + p["loop"],
        "windows": (p["load"] + p["i2f"] + p["fmul"] + p["fadd"]   # bias
                    + p["fadd"]                                    # ReLU
                    + p["div"] + p["f2i"] + p["fadd"] + p["loop"]),
        "pooled": p["store"] + 2 * p["iadd"] + p["imul"] + p["loop"],
        "rows": p["loop"],
        "channels": 2 * p["imul"] + p["loop"],
    }


CLA_LEVELS = level_costs(CLA_PRIMITIVES)
CPU_LEVELS = level_costs(CPU_PRIMITIVES)


def _hand_count(counts, levels):
    return sum(sum(c.get(k, 0) * v for k, v in levels.items()) for c in counts)


def _solve_tune(anchor):
    """How much worse the compiled kernel is than the hand count above.

    One number, from one measurement, and it is the only fitted quantity in
    the conv model -- everything else is counted. It absorbs what cannot be
    read off an instruction set: that this is an unoptimised build, that
    locals live in memory rather than registers, and the pipeline stalls
    that go with both. A value near 1 would mean the hand count was already
    right; far from 1 means it is not, and the figure is worth looking at
    rather than hiding, which is why the GUI reports it."""
    counts = stage_op_counts(anchor["channels"], anchor["layers"],
                             anchor["input_h"], anchor["input_w"],
                             anchor["classes"], anchor["pool_w"])
    hand = (_hand_count(counts, CLA_LEVELS)
            + anchor["tasks"] * CLA_TASK_OVERHEAD)
    target = anchor["ms"] / 1000.0 * anchor["mhz"] * 1e6
    return max(0.1, target / hand) if hand > 0 else 1.0


CYC_FLOP = _solve_flop_cycles(MFCC_ANCHOR)
CLA_TUNE = _solve_tune(CLA_ANCHOR)


def _validation_error(v):
    """How far the model lands from a measurement it was NOT fitted to.

    Deliberately not folded into the fit. A constant solved from two points
    reproduces both and proves nothing; one point held back is the only
    thing here that can be wrong, and therefore the only thing that says
    the model works. Re-fitting on both would trade the evidence for a
    smaller residual."""
    counts = stage_op_counts(v["channels"], v["layers"], v["input_h"],
                             v["input_w"], v["classes"], v["pool_w"])
    cycles = (_hand_count(counts, CLA_LEVELS) * CLA_TUNE
              + v["tasks"] * CLA_TASK_OVERHEAD * CLA_TUNE)
    predicted = 1000.0 * cycles / (v["mhz"] * 1e6)
    return {"predicted_ms": predicted, "measured_ms": v["ms"],
            "error_pct": 100.0 * (predicted - v["ms"]) / v["ms"]}


VALIDATION = _validation_error(CLA_VALIDATION)


def validation_note():
    """The provenance line the screen owes the reader: what the cycle costs
    were fitted on, and how well they predicted something they were not."""
    v = CLA_VALIDATION
    return (f"cycle costs from an unoptimised (-O0) build · "
            f"validated on a held-out {v['channels']}-channel measurement: "
            f"{VALIDATION['predicted_ms']:.1f} ms modelled vs "
            f"{VALIDATION['measured_ms']:.0f} ms measured, "
            f"{VALIDATION['error_pct']:+.0f}%")
# The C28x conv path has no measurement of its own, so it is hand-counted
# with the same structure and corrected by the same factor: same compiler,
# same optimisation level, same source. What separates the two cores is
# then the primitive table alone -- integer adds, integer multiplies, loop
# tails -- rather than any fitted difference.
CPU_TUNE = CLA_TUNE
# Reported, not used: cycles per MAC, for the parts of the GUI that quote a
# single number.
CYC_MAC_CLA = CLA_LEVELS["taps"] * CLA_TUNE
CYC_MAC_CPU = CPU_LEVELS["taps"] * CPU_TUNE


def stage_cycles(counts, on_cla):
    levels = CLA_LEVELS if on_cla else CPU_LEVELS
    tune = CLA_TUNE if on_cla else CPU_TUNE
    return (_hand_count([counts], levels) * tune
            + (CLA_TASK_OVERHEAD * CLA_TUNE if on_cla else 0.0))


def pipeline_timing(plan, ch, layers, ih, iw, ncls, pool_w, cla_stage,
                    cpu_mhz, cla_mhz, measured_fe_ms=None,
                    period_ms=None, measured_infer_ms=None):
    """Puts the whole frame on one clock.

    Two numbers come out of it, and they are not the same number:

      latency    MFCC + every stage in order. Nothing overlaps inside one
                 window, because each stage eats the previous one's output.

      period     how often a new window can START. The CLA only buys you
                 anything here, and only for the stages at the END of the
                 chain: once the C28x has handed the last of its own work
                 over, it is free to start the next window's features while the
                 CLA finishes. A CPU stage sitting between two CLA stages
                 throws that away -- the CPU has to wait for the CLA result
                 to carry on -- which is why the advice is 'move the WHOLE
                 tail', not 'move something'.

    measured_fe_ms replaces the estimated feature-extraction time and
    changes NOTHING else. It used to scale the CPU stages by the same ratio,
    on the reasoning that both run on the same core out of the same
    compiler -- but the front end is whatever the user's is, and a 5 ms
    front end means a different algorithm, not a C28x forty times faster
    than the reference. `calib` is still reported, as what it honestly is:
    how far this front end sits from the MFCC pipeline this project models.

    measured_infer_ms is the re-anchor for the network: the time the WHOLE
    chain takes as currently assigned. One factor over every stage, so the
    split between them is still the model's."""
    mf = mfcc_cost(plan)
    model_fe_ms = 1000.0 * mf["cycles"] / (cpu_mhz * 1e6)
    calib = 1.0
    if measured_fe_ms and model_fe_ms > 0:
        calib = measured_fe_ms / model_fe_ms
    fe_ms = measured_fe_ms if measured_fe_ms else model_fe_ms

    flags = list(cla_stage["stages"][:layers]) + [bool(cla_stage["dense"])]
    counts = stage_op_counts(ch, layers, ih, iw, ncls, pool_w)
    stages = []
    for i, (cnt, on_cla) in enumerate(zip(counts, flags)):
        cyc = stage_cycles(cnt, on_cla)
        mhz = cla_mhz if on_cla else cpu_mhz
        stages.append({"index": i, "on_cla": bool(on_cla), "macs": cnt["macs"],
                       "windows": cnt["windows"], "cycles": cyc,
                       "ms": 1000.0 * cyc / (mhz * 1e6),
                       "alt_ms": 1000.0 * stage_cycles(cnt, not on_cla)
                                 / ((cpu_mhz if on_cla else cla_mhz) * 1e6)})

    # A measured inference time re-anchors the per-MAC cost for the whole
    # chain; failing that, a measured MFCC time can still carry a correction
    # over to the stages that run on the same core it was measured on.
    infer_scale = 1.0
    modelled_infer = sum(s["ms"] for s in stages)
    if measured_infer_ms and modelled_infer > 0:
        infer_scale = measured_infer_ms / modelled_infer
    if measured_infer_ms:
        for s in stages:
            s["ms"] *= infer_scale
            s["alt_ms"] *= infer_scale

    infer_ms = sum(s["ms"] for s in stages)
    # What a MAC ended up costing on each side -- EFFECTIVE, i.e. the whole
    # stage divided by its MACs, so the per-window division and the loop
    # tails are in it. The per-tap figure on its own reads lower than the
    # work really is, and this is the number the screen quotes.
    def _per_mac(on_cla, fallback):
        cyc = sum(s["cycles"] for s in stages if s["on_cla"] == on_cla)
        mac = sum(s["macs"] for s in stages if s["on_cla"] == on_cla)
        return (cyc / mac if mac else fallback) * infer_scale

    all_counts = {"macs": sum(c["macs"] for c in counts)}
    nominal = {d: (_hand_count(counts, CLA_LEVELS if d else CPU_LEVELS)
                   * (CLA_TUNE if d else CPU_TUNE)
                   / max(all_counts["macs"], 1)) for d in (True, False)}
    mac_cost = {"cla": _per_mac(True, nominal[True]),
                "cpu": _per_mac(False, nominal[False])}
    # the maximal run of CLA stages at the end of the chain -- the only part
    # that can overlap the next window's feature extraction
    tail = 0.0
    for s in reversed(stages):
        if not s["on_cla"]:
            break
        tail += s["ms"]
    cpu_busy = fe_ms + (infer_ms - tail)
    period = max(cpu_busy, tail)
    latency = fe_ms + infer_ms
    all_cla = all(s["on_cla"] for s in stages)
    fe_share = fe_ms / latency if latency > 0 else 0.0
    # the same network with every stage on the CLA: the chain then sits
    # entirely behind the C28x's own work, so the window period drops to
    # whichever of the two sides is longer
    cla_infer = sum(s["ms"] if s["on_cla"] else s["alt_ms"] for s in stages)
    period_all_cla = max(fe_ms, cla_infer)

    return {"plan": plan, "stages": stages, "fe_ms": fe_ms,
            "model_fe_ms": model_fe_ms, "measured": bool(measured_fe_ms),
            "calib": calib,
            "infer_measured": bool(measured_infer_ms),
            "infer_scale": infer_scale, "model_infer_ms": modelled_infer,
            "mac_cost": mac_cost,
            "total_macs": sum(c["macs"] for c in counts),
            "infer_ms": infer_ms, "latency_ms": latency, "period_ms": period,
            "cpu_busy_ms": cpu_busy, "overlap_ms": tail, "all_cla": all_cla,
            "fe_share": fe_share, "cla_infer_ms": cla_infer,
            "period_all_cla_ms": period_all_cla,
            "mfcc_ram_bytes": mf["ram_bytes"], "fft_share": mf["fft_share"],
            "deadline_ms": period_ms, "window_ms": plan["window_ms"]}


def timing_advice(t):
    """One sentence about where the frame actually goes, and what to do
    about it. Returns (severity, text) with severity in
    'ok' | 'info' | 'warn' | 'error'."""
    pct = 100.0 * t["fe_share"]
    fe, inf = t["fe_ms"], t["infer_ms"]
    deadline = t["deadline_ms"]
    full = t["period_all_cla_ms"]          # window period with the whole
                                           # chain offloaded
    gain = t["period_ms"] / full if full > 0 else 1.0
    cla_c, cpu_c = t["mac_cost"]["cla"], t["mac_cost"]["cpu"]
    # On the measured build a MAC is about 2.5x dearer on the CLA, so
    # "offload it" is a trade, not a free win, and every branch below that
    # recommends offloading has to say so.
    slower = cla_c / cpu_c if cpu_c > 0 else 1.0
    trade = (f" The stages get slower there -- {slower:.1f}x per MAC on this "
             f"build -- so what you are buying is the overlap, not speed."
             if slower >= 1.1 else "")

    if deadline and t["period_ms"] > deadline:
        return ("error",
                f"Over the real-time budget: a window arrives every "
                f"{deadline:.0f} ms but the pipeline needs {t['period_ms']:.1f} ms "
                f"({fe:.1f} ms of features + {inf:.2f} ms of inference). "
                f"Shorten the front end or lengthen the window.")

    # The line the network has to stay under. While inference fits inside the
    # front end's time, putting it on the CLA hides it completely -- the C28x
    # is busy with the next window's features anyway, so the network is free.
    # Cross that line and it stops being free: the frame is now paced by the
    # network, and every extra channel is paid for in full.
    if inf > fe:
        head = (f"Inference is now the longer half: {inf:.2f} ms against "
                f"{fe:.1f} ms of feature extraction. ")
        if t["all_cla"]:
            return ("warn", head +
                    f"Up to {fe:.1f} ms the CLA could hide the whole network "
                    f"behind the C28x's own work and it cost nothing; past "
                    f"that it sets the pace, so a window now takes "
                    f"{t['period_ms']:.1f} ms instead of {fe:.1f}. Fewer "
                    f"channels, or fewer MACs per channel, buys that back.")
        return ("warn", head +
                f"Put every stage on the CLA first — that alone takes a "
                f"window from {t['period_ms']:.1f} ms to {full:.1f} ms. If "
                f"that is still over {fe:.1f} ms, the network itself is what "
                f"has to come down.")

    if pct >= 60.0:
        if t["all_cla"]:
            return ("ok",
                    f"Feature extraction is {pct:.0f}% of the frame "
                    f"({fe:.1f} ms vs {inf:.2f} ms of inference) and every "
                    f"stage is already on the CLA, so the CLA runs this "
                    f"window's network while the C28x starts the next "
                    f"window's features. A new result every "
                    f"{t['period_ms']:.1f} ms "
                    f"instead of {t['latency_ms']:.1f} ms.")
        return ("warn",
                f"Feature extraction dominates: {fe:.1f} ms of it against "
                f"{inf:.2f} ms of inference ({pct:.0f}% of the frame). Put "
                f"EVERY stage on the CLA -- a single CPU stage in the chain "
                f"makes the C28x wait for the CLA and the two stop "
                f"overlapping. Fully offloaded: a window every {full:.1f} ms "
                f"instead of {t['period_ms']:.1f} ms." + trade)

    if pct <= 25.0:
        msg = (f"Inference is the expensive half: {inf:.2f} ms against "
               f"{fe:.1f} ms of feature extraction "
               f"({100.0 - pct:.0f}% of the frame). ")
        if slower >= 1.1:
            # The interesting case, and the one this build is in: the CLA is
            # the SLOWER place to run a MAC, so the two goals part company --
            # the C28x wins on latency, the CLA on results per second.
            msg += (f"A MAC costs {cpu_c:.0f} cycles on the C28x against "
                    f"{cla_c:.0f} on the CLA, so the two goals pull apart: "
                    f"the C28x is the faster place to run a stage, the CLA "
                    f"the one that lets it overlap. Pick by which you need.")
        else:
            msg += (f"At {cla_c:.0f} cycles per MAC, speed has to come from "
                    f"a smaller network or a cheaper kernel -- unroll the "
                    f"3x3 and walk pointers instead of recomputing uint32 "
                    f"indices per tap.")
        if not t["all_cla"]:
            msg += (f" All of it on the CLA: a window every {full:.1f} ms "
                    f"instead of {t['period_ms']:.1f} ms.")
        return ("info", msg)

    if not t["all_cla"] and gain >= 1.15:
        where = ("the chain runs entirely on the C28x, serially with the "
                 "front end"
                 if not any(s["on_cla"] for s in t["stages"]) else
                 "the chain is split across both domains, so nothing overlaps")
        return ("warn",
                f"Feature extraction {fe:.1f} ms and inference {inf:.2f} ms are "
                f"the same "
                f"order and {where}: a window every {t['period_ms']:.1f} ms. "
                f"With every stage on the CLA the two run side by side -- "
                f"{full:.1f} ms per window, {gain:.1f}x the throughput."
                + trade)

    tail = ("." if t["overlap_ms"] <= 0 else
            f", with {t['overlap_ms']:.2f} ms of it overlapped by the CLA.")
    return ("info",
            f"Features {fe:.1f} ms · inference {inf:.2f} ms · latency "
            f"{t['latency_ms']:.1f} ms, a new window every "
            f"{t['period_ms']:.1f} ms" + tail)


# ==========================================
# 0. LOOK AND FEEL (front-end only -- no behaviour depends on this)
# ==========================================
# One palette, used by the stylesheet, the painted pipeline canvas and the
# budget bars alike, so nothing drawn by hand can drift away from the theme
# the widgets get.
C_BG         = "#eef1f6"   # window
C_PANEL      = "#ffffff"   # cards, header
C_PANEL_DEEP = "#f6f8fb"   # inputs, log, tiles -- sunken against a white card
C_LANE       = "#eceff5"   # pipeline lane bed
C_BOX        = "#ffffff"   # pipeline stage box, so it lifts off the lane
C_BORDER     = "#ccd3df"
C_BORDER_HI  = "#9aa5b8"   # hover edge
# Text colours are picked for contrast on white, not for looking soft:
# C_TEXT 15.9:1, C_MUTED 7.6:1, C_FAINT 5.4:1 -- every one of them clears
# WCAG AA for normal text, which the small 9-10px captions need.
C_TEXT       = "#16191f"
C_MUTED      = "#4a5364"
C_FAINT      = "#6b7382"   # notes, placeholders, disabled
C_ACCENT     = "#1d63d8"   # primary action / focus            (6.1:1 on white)
C_OK         = "#15803d"   # secondary action / pass           (4.9:1)
C_WARN       = "#9a5b00"   # over-budget, not-deployable       (5.6:1)
C_ERR        = "#b3261e"   # errors, off-diagonal confusion    (6.4:1)
C_CPU        = "#1d63d8"   # CPU domain
C_CLA        = "#6d28d9"   # CLA domain                        (7.3:1)
C_STAGE      = "#c2410c"   # conv+pool stage marker -- its own colour so the
                           # diagram never borrows the warning amber

APP_QSS = f"""
/* No background on the universal rule: on a light theme a grey window colour
   inherited by every plain QWidget bleeds through cards and tab pages as
   stripes behind checkboxes and panels. Only the surfaces that own a colour
   declare one; everything else stays transparent. */
QWidget {{
    color: {C_TEXT};
    font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    font-size: 12px;
}}
QMainWindow {{ background: {C_BG}; }}
QLabel, QCheckBox {{ background: transparent; }}
QDialog {{ background: {C_BG}; }}

QLabel#Title       {{ font-size: 18px; font-weight: 600; color: {C_TEXT}; }}
QLabel#Subtitle    {{ font-size: 11px; color: {C_MUTED}; }}
QLabel#CardTitle   {{ font-size: 10px; font-weight: 700; color: {C_MUTED};
                      letter-spacing: 1.2px; }}
QLabel#CardSub     {{ font-size: 10px; color: {C_FAINT}; }}
QLabel#FieldLabel  {{ color: {C_MUTED}; }}
QLabel#Hint        {{ font-size: 10px; color: {C_FAINT}; }}
QLabel#TileCap     {{ font-size: 9px; font-weight: 700; color: {C_MUTED};
                      letter-spacing: 1px; }}
QLabel#TileVal     {{ font-size: 19px; font-weight: 600; color: {C_TEXT}; }}
QLabel#TileValSm   {{ font-size: 13px; font-weight: 600; color: {C_TEXT}; }}
QLabel#TileNote    {{ font-size: 9px; color: {C_FAINT}; }}
QLabel#MatrixHead  {{ font-size: 10px; font-weight: 700; color: {C_MUTED};
                      font-family: "Cascadia Mono", "Consolas", monospace; }}
QLabel#MatrixCell  {{ font-size: 11px; color: {C_TEXT};
                      font-family: "Cascadia Mono", "Consolas", monospace; }}
QLabel#ToolCap     {{ font-size: 9px; color: {C_MUTED};
                      font-weight: 700; letter-spacing: 1.1px; }}
QLabel#Phase       {{ font-size: 13px; font-weight: 600; color: {C_TEXT}; }}
QLabel#RunDetail   {{ font-size: 11px; color: {C_MUTED};
                      font-family: "Cascadia Mono", "Consolas", monospace; }}

QFrame#Header {{
    background: {C_PANEL};
    border: none;
    border-bottom: 1px solid {C_BORDER};
}}
QFrame#ActionBar {{
    background: {C_PANEL};
    border: none;
    border-top: 1px solid {C_BORDER};
}}
QFrame#LogDrawer {{
    background: {C_PANEL};
    border: none;
    border-top: 1px solid {C_BORDER};
}}
QFrame#Card {{
    background: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
}}
QFrame#Tile {{
    background: {C_PANEL_DEEP};
    border: 1px solid {C_BORDER};
    border-radius: 8px;
}}
QFrame#Sep {{ background: {C_BORDER}; border: none; }}

QLineEdit, QComboBox {{
    background: #ffffff;
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 9px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:hover, QComboBox:hover {{ border-color: {C_BORDER_HI}; }}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {C_ACCENT}; }}
QLineEdit[readOnly="true"] {{ background: {C_PANEL_DEEP}; color: {C_MUTED}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: #ffffff;
    border: 1px solid {C_BORDER};
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
    selection-color: #ffffff;
    outline: none;
    padding: 2px;
}}

QPushButton {{
    background: #ffffff;
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    padding: 6px 12px;
    color: {C_TEXT};
}}
QPushButton:hover   {{ border-color: {C_ACCENT}; color: {C_ACCENT}; }}
QPushButton:pressed {{ background: #e9eef8; }}
QPushButton:disabled {{ color: #9aa2b0; border-color: #e0e5ee; background: #f4f6fa; }}

QPushButton#Primary, QPushButton#Secondary {{
    font-size: 12px; font-weight: 600; letter-spacing: 0.6px;
    color: #ffffff; border: none; padding: 0px;
}}
QPushButton#Primary   {{ background: {C_ACCENT}; }}
QPushButton#Primary:hover   {{ background: #2b73ea; }}
QPushButton#Primary:pressed {{ background: #174fae; }}
QPushButton#Secondary {{ background: {C_OK}; }}
QPushButton#Secondary:hover   {{ background: #1a9549; }}
QPushButton#Secondary:pressed {{ background: #10632f; }}
QPushButton#Primary:disabled, QPushButton#Secondary:disabled {{
    background: #dfe4ec; color: #8d95a3;
}}
/* the dataset button doubles as the "which file am I on" readout */
QPushButton#Chip {{
    background: {C_PANEL_DEEP};
    border: 1px solid {C_BORDER_HI};
    padding: 6px 14px;
    font-weight: 600;
    text-align: left;
}}
QPushButton#Chip:hover {{ border-color: {C_ACCENT}; color: {C_ACCENT}; }}

/* Device | Results segmented switch */
QPushButton#SegLeft, QPushButton#SegRight {{
    background: #ffffff;
    border: 1px solid {C_BORDER};
    color: {C_MUTED};
    font-size: 11px; font-weight: 700; letter-spacing: 0.6px;
    padding: 6px 14px;
}}
QPushButton#SegLeft  {{ border-top-right-radius: 0; border-bottom-right-radius: 0; }}
QPushButton#SegRight {{ border-top-left-radius: 0; border-bottom-left-radius: 0;
                        border-left: none; }}
QPushButton#SegLeft:hover, QPushButton#SegRight:hover {{ color: {C_ACCENT}; }}
QPushButton#SegLeft:checked, QPushButton#SegRight:checked {{
    background: {C_ACCENT}; border-color: {C_ACCENT}; color: #ffffff;
}}
QPushButton#SegLeft:disabled, QPushButton#SegRight:disabled {{
    background: #f4f6fa; color: #9aa2b0; border-color: #e0e5ee;
}}

QCheckBox {{ spacing: 9px; padding: 3px 0px; color: {C_TEXT}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {C_BORDER_HI}; border-radius: 4px;
    background: #ffffff;
}}
QCheckBox::indicator:hover {{ border-color: {C_ACCENT}; }}
QCheckBox::indicator:checked {{ background: {C_ACCENT}; border-color: {C_ACCENT}; }}
QCheckBox::indicator:disabled {{ border-color: #dbe0e9; background: #f0f2f6; }}
QCheckBox:disabled {{ color: #9aa2b0; }}

QTextEdit#Log {{
    background: #ffffff;
    border: 1px solid {C_BORDER};
    border-radius: 8px;
    padding: 8px 10px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: {C_TEXT};
    selection-background-color: {C_ACCENT};
    selection-color: #ffffff;
}}

QProgressBar {{
    background: {C_PANEL_DEEP};
    border: 1px solid {C_BORDER};
    border-radius: 6px;
    height: 14px;
    text-align: center;
    font-size: 9px;
    color: {C_MUTED};
}}
QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 5px; }}


QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #b9c2d1; border-radius: 5px; min-height: 26px; }}
QScrollBar::handle:vertical:hover {{ background: #97a3b7; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #b9c2d1; border-radius: 5px; min-width: 26px; }}
QScrollBar::handle:horizontal:hover {{ background: #97a3b7; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; width: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: #ffffff; color: {C_TEXT};
    border: 1px solid {C_BORDER_HI}; border-radius: 6px; padding: 7px;
}}
"""


# ==========================================
# 0b. REUSABLE FRONT-END WIDGETS
# ==========================================
# Presentation only: nothing below reads or changes a training/quantization
# value, each widget is just handed numbers the logic already produced.

def _ui_font(px, weight=QFont.Normal, spacing=0.0, italic=False):
    f = QFont("Segoe UI")
    f.setPixelSize(px)
    f.setWeight(weight)
    f.setItalic(italic)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    return f


class Card(QFrame):
    """A titled panel. Used instead of QGroupBox so the heading is a plain
    label we can retitle, subtitle and restyle without fighting QGroupBox's
    built-in title metrics."""

    def __init__(self, title, subtitle="", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)

        head = QVBoxLayout()
        head.setSpacing(2)
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("CardTitle")
        head.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("CardSub")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        head.addWidget(self.subtitle_label)
        outer.addLayout(head)

        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body, 1)

    def setTitle(self, text):
        self.title_label.setText(text.upper())

    def setSubtitle(self, text):
        self.subtitle_label.setText(text)
        self.subtitle_label.setVisible(bool(text))

    def add(self, item, stretch=0):
        if isinstance(item, QWidget):
            self.body.addWidget(item, stretch)
        else:
            self.body.addLayout(item, stretch)


class StatTile(QFrame):
    """caption / value / note block. compact=True is the run-strip size, the
    default is the results-panel size."""

    def __init__(self, caption, value="--", note="", compact=False, parent=None):
        super().__init__(parent)
        self.setObjectName("Tile")
        self._size = 13 if compact else 19
        lay = QVBoxLayout(self)
        if compact:
            lay.setContentsMargins(9, 6, 9, 6)
        else:
            lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(2)

        self.cap = QLabel(caption.upper())
        self.cap.setObjectName("TileCap")
        self.val = QLabel(value)
        self.val.setObjectName("TileValSm" if compact else "TileVal")
        self.note = QLabel(note)
        self.note.setObjectName("TileNote")
        self.note.setVisible(bool(note))

        lay.addWidget(self.cap)
        lay.addWidget(self.val)
        lay.addWidget(self.note)

    def set_value(self, value, note=None, color=None):
        self.val.setText(str(value))
        self.val.setStyleSheet(
            f"font-size:{self._size}px; font-weight:600; color:{color or C_TEXT};")
        if note is not None:
            self.note.setText(note)
            self.note.setVisible(bool(note))



class DeviceCanvas(QWidget):
    """The hero of the window: the F28379D drawn as two compute domains with
    the inference stages living inside them.

    A stage card is not just a read-out. Each movable stage carries a CPU/CLA
    selector along its bottom edge: you pick the domain you want rather than
    toggling the card, so the current assignment is legible without hovering
    and there is no way to flip a stage by mis-clicking it.

    Every card states what it costs -- the RAM its buffers and weight copy
    occupy and the flash its weights take -- and each block totals its own
    stages, so the two domain gauges below are never a mystery."""

    stage_assigned = pyqtSignal(int, bool)     # (stage id, run on CLA)

    CARD_W, CARD_H = 188, 142
    SEG_H = 26             # the CPU/CLA selector strip along the card bottom
    CARD_GAP = 24          # arrow lives here
    ROW_GAP = 12
    PAD = 18               # inside a domain block
    BLOCK_HEAD = 30
    GAUGE_H = 32
    HANDOFF = 32           # vertical room between the two blocks
    HEAD_H = 32            # device name + flash gauge
    TIME_H = 104           # four headline times, then the frame as a bar
    HINT_H = 66            # the timing verdict (up to three lines) + the hint
    # The device is a fixed-size object, not a panel that should stretch:
    # past this width the blocks are centred instead of pulled apart, so a
    # maximised window does not leave a five-card pipeline swimming in a
    # 2000px-wide box.
    MAX_W = 1120

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set from the selected target by EdgeAiGui; the literal here is only
        # what a bare DeviceCanvas() shows before anything has selected one.
        self.device = "C2000 target"
        self.blocks = []
        self.flash = None      # ("Flash", used_kb, limit_kb)
        self.timing = None     # pipeline_timing() result, or None
        self.advice = None     # (severity, text) from timing_advice()
        self.hint = ""
        self.placeholder = "Select a dataset to map the network onto the device."
        self._hover = None     # (stage id, "cpu"|"cla") under the pointer
        self._hit = []         # [(QRectF, stage id, "cpu"|"cla")]
        # set while a search runs: the plan the thread was handed must not
        # drift away from the picture on screen
        self.locked = False
        self._scale = 1.0      # set by paintEvent when the device is shrunk
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(400)

    # ---------------------------------------------------------------- data
    def set_device(self, blocks, flash=None, hint="", timing=None, advice=None):
        self.blocks, self.flash, self.hint = blocks, flash, hint
        self.timing, self.advice = timing, advice
        self._hover = None
        self.update()

    def clear(self, placeholder=None):
        if placeholder:
            self.placeholder = placeholder
        self.blocks, self.flash, self.hint = [], None, ""
        self.timing, self.advice = None, None
        self._hit = []
        self.update()

    # ----------------------------------------------------------- hit test
    def _target_at(self, pos):
        if self.locked:
            return None
        s = getattr(self, "_scale", 1.0) or 1.0
        pt = QPointF(pos.x() / s, pos.y() / s)
        for rect, sid, side in self._hit:
            if rect.contains(pt):
                return (sid, side)
        return None

    def mouseMoveEvent(self, ev):
        hit = self._target_at(ev.pos())
        if hit != self._hover:
            self._hover = hit
            self.setCursor(Qt.PointingHandCursor if hit else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, _ev):
        self._hover = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def mousePressEvent(self, ev):
        hit = self._target_at(ev.pos())
        if hit:
            self.stage_assigned.emit(hit[0], hit[1] == "cla")

    # -------------------------------------------------------------- layout
    def _per_row(self, inner_w):
        return max(1, int((inner_w + self.CARD_GAP)
                          // (self.CARD_W + self.CARD_GAP)))

    def _block_height(self, n_stages, width):
        inner = max(width - 2 * self.PAD, self.CARD_W)
        rows = max(1, -(-max(n_stages, 1) // self._per_row(inner)))
        return (self.PAD + self.BLOCK_HEAD + rows * self.CARD_H
                + (rows - 1) * self.ROW_GAP + 14 + self.GAUGE_H + self.PAD)

    def _layout(self, view_w):
        """Card width, per-block heights and the height of the whole device,
        for a viewport this wide."""
        cw = min(view_w, float(self.MAX_W))
        heights = [self._block_height(len(b["stages"]), cw) for b in self.blocks]
        time_h = self.TIME_H if self.timing else 0
        total = (self.HEAD_H + 6 + time_h + sum(heights)
                 + self.HANDOFF * (len(self.blocks) - 1))
        return cw, heights, total

    # ------------------------------------------------------------ painting
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        p.fillRect(self.rect(), QColor(C_PANEL))
        self._hit = []

        if not self.blocks:
            # first line is the headline, the rest is the explanation
            head, _, body = self.placeholder.partition("\n\n")
            box = QRectF(self.rect()).adjusted(40, 0, -40, 0)
            p.setFont(_ui_font(16, QFont.DemiBold))
            p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(box.left(), box.center().y() - 40, box.width(), 26),
                       Qt.AlignHCenter | Qt.AlignVCenter, head)
            if body:
                p.setFont(_ui_font(12))
                p.setPen(QColor(C_FAINT))
                p.drawText(QRectF(box.left(), box.center().y() - 8,
                                  box.width(), 60),
                           int(Qt.AlignHCenter | Qt.TextWordWrap), body)
            return

        # The device is drawn at its natural size and, when the window is too
        # short for it, shrunk to fit rather than cut off at the bottom edge.
        # Clicking still works: _target_at divides the pointer by the same
        # factor, so the hit rectangles below stay in drawing coordinates.
        cw, heights, total = self._layout(float(self.width()))
        need = total + self.HINT_H
        self._scale = 1.0
        if self.height() > 0 and need > self.height():
            self._scale = max(0.55, self.height() / need)
            cw, heights, total = self._layout(self.width() / self._scale)
        if self._scale != 1.0:
            p.scale(self._scale, self._scale)
        view_w = self.width() / self._scale
        x0 = (view_w - cw) / 2.0
        room = self.height() / self._scale - self.HINT_H
        # the whole device -- name, flash gauge and both blocks -- is centred
        # as one object rather than pinned to the top edge
        y = max(0.0, (room - total) / 2.0)

        self._draw_head(p, QRectF(x0, y, cw, self.HEAD_H))
        y += self.HEAD_H + 6
        if self.timing:
            self._draw_timeline(p, QRectF(x0, y, cw, self.TIME_H))
            y += self.TIME_H
        for i, (block, h) in enumerate(zip(self.blocks, heights)):
            if i:
                self._draw_handoff(p, QRectF(x0, y, cw, self.HANDOFF))
                y += self.HANDOFF
            self._draw_block(p, QRectF(x0, y, cw, h), block)
            y += h

        # Directly under the device, not pinned to the bottom edge, so it
        # reads as a caption for the thing above it. The timing verdict wins
        # the spot when there is one: 'MFCC is 96% of your frame' is worth
        # more than a reminder of how the stage cards work.
        if self.advice:
            sev, text = self.advice
            col = {"error": C_ERR, "warn": C_WARN,
                   "ok": C_OK}.get(sev, C_MUTED)
            p.setFont(_ui_font(11, QFont.DemiBold if sev in ("warn", "error")
                               else QFont.Normal))
            p.setPen(QColor(col))
            p.drawText(QRectF(x0 + 40, y + 5, cw - 80, 46),
                       int(Qt.AlignHCenter | Qt.TextWordWrap), text)
            y += 44
        if self.hint:
            p.setFont(_ui_font(10))
            p.setPen(QColor(C_FAINT))
            p.drawText(QRectF(x0, y + 6, cw, 16),
                       Qt.AlignHCenter | Qt.AlignVCenter, self.hint)

    def _draw_head(self, p, rect):
        p.setFont(_ui_font(13, QFont.Bold, 1.6))
        p.setPen(QColor(C_MUTED))
        p.drawText(QRectF(rect.left() + 2, rect.top(), rect.width() * 0.5,
                          rect.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, self.device)
        if self.flash:
            self._draw_gauge(p, QRectF(rect.left() + rect.width() * 0.45,
                                       rect.top() + 2,
                                       rect.width() * 0.55 - 2, rect.height()),
                             *self.flash, accent=C_ACCENT)

    def _draw_timeline(self, p, rect):
        """The frame, as four headline numbers over a bar drawn to scale:
        the front end first, then every stage in order, each segment
        coloured by the domain it runs on.

        The numbers lead because they are the answer -- the two gauges say
        whether the model FITS, this says whether it KEEPS UP, and until
        this band existed that question had no place on the screen.

        The bar is the LATENCY -- front end plus all stages, nothing
        overlapping, because inside one window nothing can. What the CLA
        buys shows up as the bracket under the tail: that much of the bar is
        also the next window's feature extraction, so it costs nothing in
        throughput."""
        t = self.timing
        total = max(t["latency_ms"], 1e-6)
        deadline = t.get("deadline_ms")
        over = deadline and t["period_ms"] > deadline

        head = QRectF(rect.left(), rect.top(), rect.width(), 13)
        p.setFont(_ui_font(10, QFont.Bold, 1.0))
        p.setPen(QColor(C_MUTED))
        p.drawText(head, Qt.AlignLeft | Qt.AlignVCenter, "FRAME BUDGET")
        if deadline:
            p.setFont(_ui_font(10))
            p.setPen(QColor(C_ERR if over else C_FAINT))
            p.drawText(head, Qt.AlignRight | Qt.AlignVCenter,
                       f"budget {deadline:.0f} ms"
                       + ("  ·  OVER" if over else
                          f"  ·  {100.0 * t['period_ms'] / deadline:.0f}% used"))

        # four numbers, at the size the rest of the window gives a headline
        infer_col = (C_CLA if all(s["on_cla"] for s in t["stages"])
                     else C_CPU)
        stats = [
            ("features", f"{t['fe_ms']:.1f} ms", C_STAGE,
             "measured" if t["measured"] else "estimated"),
            ("inference", f"{t['infer_ms']:.2f} ms", infer_col,
             f"{t['mac_cost']['cla']:.0f} cyc/MAC"),
            ("latency", f"{t['latency_ms']:.1f} ms", C_TEXT,
             "front end + every stage"),
            ("a result every", f"{t['period_ms']:.1f} ms",
             C_ERR if over else C_OK,
             (f"{t['overlap_ms']:.2f} ms overlapped" if t["overlap_ms"] > 0
              else "nothing overlaps")),
        ]
        cw = rect.width() / len(stats)
        for i, (cap, val, col, note) in enumerate(stats):
            x = rect.left() + i * cw
            p.setFont(_ui_font(9, QFont.Bold, 1.0))
            p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(x, rect.top() + 16, cw - 10, 12),
                       Qt.AlignLeft | Qt.AlignVCenter, cap.upper())
            p.setFont(_ui_font(21, QFont.DemiBold))
            p.setPen(QColor(col))
            p.drawText(QRectF(x, rect.top() + 28, cw - 10, 26),
                       Qt.AlignLeft | Qt.AlignVCenter, val)
            p.setFont(_ui_font(9))
            p.setPen(QColor(C_FAINT))
            p.drawText(QRectF(x, rect.top() + 54, cw - 10, 12),
                       Qt.AlignLeft | Qt.AlignVCenter, note)

        bar = QRectF(rect.left() + 0.5, rect.top() + 72,
                     max(rect.width() - 1, 1), 14)
        tp = QPainterPath()
        tp.addRoundedRect(bar, 4, 4)
        p.fillPath(tp, QColor(C_PANEL_DEEP))

        segs = [(t["fe_ms"], QColor(C_STAGE), "FEATURES")]
        for i, s in enumerate(t["stages"]):
            segs.append((s["ms"], QColor(C_CLA if s["on_cla"] else C_CPU),
                         f"S{i + 1}"))
        p.save()
        p.setClipPath(tp)
        x = bar.left()
        for ms, col, label in segs:
            wdt = bar.width() * (ms / total)
            if wdt <= 0:
                continue
            p.fillRect(QRectF(x, bar.top(), max(wdt, 1.0), bar.height()), col)
            if wdt > 26:
                p.setFont(_ui_font(9, QFont.Bold))
                p.setPen(QColor("#ffffff"))
                p.drawText(QRectF(x, bar.top(), wdt, bar.height()),
                           Qt.AlignCenter, label)
            x += wdt
        p.restore()
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.drawPath(tp)

        # the overlapped tail, marked under the bar where it happens
        p.setFont(_ui_font(9))
        legend = ""
        if t["overlap_ms"] > 0:
            ox = bar.right() - bar.width() * (t["overlap_ms"] / total)
            p.setPen(QPen(QColor(C_CLA), 1.4))
            p.drawLine(QPointF(ox, bar.bottom() + 3),
                       QPointF(bar.right(), bar.bottom() + 3))
            p.drawLine(QPointF(ox, bar.bottom() + 1), QPointF(ox, bar.bottom() + 5))
            legend = (f"the last {t['overlap_ms']:.2f} ms runs on the CLA "
                      f"while the C28x is already extracting the next "
                      f"window's features")
        p.setPen(QColor(C_FAINT))
        p.drawText(QRectF(rect.left(), bar.bottom() + 5, rect.width(), 13),
                   Qt.AlignRight | Qt.AlignVCenter, legend)

    def _draw_handoff(self, p, rect):
        cx = rect.center().x()
        p.setPen(QPen(QColor(C_BORDER_HI), 1.6))
        p.drawLine(QPointF(cx, rect.top() + 4), QPointF(cx, rect.bottom() - 6))
        p.drawLine(QPointF(cx - 5, rect.bottom() - 11), QPointF(cx, rect.bottom() - 5))
        p.drawLine(QPointF(cx + 5, rect.bottom() - 11), QPointF(cx, rect.bottom() - 5))
        p.setFont(_ui_font(9, QFont.Bold, 1.0))
        p.setPen(QColor(C_FAINT))
        p.drawText(QRectF(cx + 12, rect.center().y() - 8, 140, 16),
                   Qt.AlignLeft | Qt.AlignVCenter, "HANDOFF")

    def _draw_block(self, p, rect, block):
        accent = QColor(block.get("accent", C_MUTED))
        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
        p.fillPath(path, QColor(C_LANE))
        p.setPen(QPen(accent, 1.6))
        p.drawPath(path)

        # heading: a filled accent chip, so the two domains are told apart at
        # a glance instead of by reading their titles
        p.setFont(_ui_font(11, QFont.Bold, 1.3))
        tw = p.fontMetrics().horizontalAdvance(block["title"].upper()) + 24
        chip = QRectF(rect.left() + self.PAD, rect.top() + self.PAD - 2, tw, 22)
        cp = QPainterPath()
        cp.addRoundedRect(chip, 11, 11)
        p.fillPath(cp, accent)
        p.setPen(QColor("#ffffff"))
        p.drawText(chip, Qt.AlignCenter, block["title"].upper())

        if block.get("note"):
            p.setFont(_ui_font(11))
            p.setPen(QColor(C_MUTED))
            p.drawText(QRectF(chip.right() + 12, chip.top(),
                              rect.width() - chip.right() - self.PAD, 22),
                       Qt.AlignLeft | Qt.AlignVCenter, block["note"])

        top = rect.top() + self.PAD + self.BLOCK_HEAD
        inner_w = rect.width() - 2 * self.PAD
        stages = block["stages"]
        if not stages:
            p.setFont(_ui_font(12, italic=True))
            p.setPen(QColor(C_FAINT))
            p.drawText(QRectF(rect.left() + self.PAD, top, inner_w, self.CARD_H),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       block.get("empty", "nothing runs here"))
        else:
            per = self._per_row(inner_w)
            for i, st in enumerate(stages):
                r, c = divmod(i, per)
                x = rect.left() + self.PAD + c * (self.CARD_W + self.CARD_GAP)
                yy = top + r * (self.CARD_H + self.ROW_GAP)
                if c:
                    self._draw_arrow(p, x - self.CARD_GAP, x, yy + self.CARD_H / 2)
                self._draw_card(p, QRectF(x, yy, self.CARD_W, self.CARD_H),
                                st, accent, block["is_cla"])

        rows = max(1, -(-max(len(stages), 1) // self._per_row(inner_w)))
        gy = top + rows * self.CARD_H + (rows - 1) * self.ROW_GAP + 14
        gauges = block.get("gauges", [])
        for i, g in enumerate(gauges):
            gw = inner_w / max(len(gauges), 1)
            self._draw_gauge(p, QRectF(rect.left() + self.PAD + i * gw, gy,
                                       gw - 14, self.GAUGE_H), *g, accent=accent)

    def _draw_arrow(self, p, x0, x1, y):
        p.setPen(QPen(QColor(C_BORDER_HI), 1.6))
        p.drawLine(QPointF(x0 + 6, y), QPointF(x1 - 6, y))
        p.drawLine(QPointF(x1 - 11, y - 4), QPointF(x1 - 6, y))
        p.drawLine(QPointF(x1 - 11, y + 4), QPointF(x1 - 6, y))

    def _draw_card(self, p, rect, st, accent, on_cla):
        sid = st.get("id", -1)
        movable = sid >= 0
        path = QPainterPath()
        path.addRoundedRect(rect, 10, 10)
        p.fillPath(path, QColor(C_BOX))

        # a 4px accent spine down the left edge: the card reads as belonging
        # to its domain even when the eye skips the block frame
        p.save()
        p.setClipPath(path)
        p.fillRect(QRectF(rect.left(), rect.top(), 4, rect.height()), accent)
        p.restore()
        p.setPen(QPen(QColor(C_BORDER), 1.2))
        p.drawPath(path)

        left = rect.left() + 14
        wid = rect.width() - 24

        badge = QRectF(left, rect.top() + 11, 19, 19)
        bp = QPainterPath()
        bp.addEllipse(badge)
        p.fillPath(bp, QColor(accent) if movable else QColor(C_FAINT))
        p.setFont(_ui_font(10, QFont.Bold))
        p.setPen(QColor("#ffffff"))
        p.drawText(badge, Qt.AlignCenter, str(st.get("order", "")))

        p.setFont(_ui_font(12, QFont.Bold, 0.6))
        p.setPen(QColor(C_TEXT))
        p.drawText(QRectF(left + 26, rect.top() + 11, wid - 26, 19),
                   Qt.AlignLeft | Qt.AlignVCenter, st.get("caption", ""))

        p.setFont(_ui_font(9))
        p.setPen(QColor(C_FAINT))
        p.drawText(QRectF(left, rect.top() + 33, wid, 13),
                   Qt.AlignLeft | Qt.AlignVCenter, st.get("sub", ""))

        p.setFont(_ui_font(17, QFont.DemiBold))
        p.setPen(QColor(C_TEXT))
        p.drawText(QRectF(left, rect.top() + 48, wid, 24),
                   Qt.AlignLeft | Qt.AlignVCenter, st.get("shape", ""))

        # what this stage costs, on the card itself
        p.setFont(_ui_font(10))
        cost_y = rect.top() + 74
        p.setPen(QColor(C_MUTED))
        p.drawText(QRectF(left, cost_y, wid * 0.52, 14),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   f"RAM {st.get('ram_kb', 0.0):.2f} KB")
        p.setPen(QColor(C_FAINT))
        p.drawText(QRectF(left + wid * 0.52, cost_y, wid * 0.48, 14),
                   Qt.AlignRight | Qt.AlignVCenter,
                   f"Flash {st.get('flash_kb', 0.0):.2f} KB")

        # ...and what it costs in time, which is the half of the picture the
        # gauges cannot show. For a movable stage the other domain's figure
        # is right next to it, so the choice on the strip below is informed.
        if st.get("ms") is not None:
            ms = st["ms"]
            p.setFont(_ui_font(10, QFont.DemiBold))
            p.setPen(QColor(accent))
            p.drawText(QRectF(left, cost_y + 15, wid * 0.52, 14),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       f"{ms:.2f} ms" if ms >= 0.01 else f"{ms * 1000:.0f} µs")
            p.setFont(_ui_font(9))
            p.setPen(QColor(C_FAINT))
            if st.get("alt_ms") is not None:
                other = "CPU" if on_cla else "CLA"
                alt = st["alt_ms"]
                txt = (f"{other} {alt:.2f} ms" if alt >= 0.01
                       else f"{other} {alt * 1000:.0f} µs")
            else:
                txt = st.get("ms_note", "")
            p.drawText(QRectF(left + wid * 0.52, cost_y + 15, wid * 0.48, 14),
                       Qt.AlignRight | Qt.AlignVCenter, txt)

        seg = QRectF(rect.left() + 10, rect.bottom() - self.SEG_H - 8,
                     rect.width() - 20, self.SEG_H)
        if movable:
            self._draw_segment(p, seg, sid, on_cla, accent)
        else:
            p.setFont(_ui_font(10, italic=True))
            p.setPen(QColor(C_FAINT))
            p.drawText(seg, Qt.AlignCenter, st.get("fixed_note", "fixed"))

    def _draw_segment(self, p, rect, sid, on_cla, accent):
        """Two-option domain picker. The selected half is filled, the other
        is a plain click target -- picking the side that is already active
        simply does nothing."""
        half = rect.width() / 2.0
        outline = QPainterPath()
        outline.addRoundedRect(rect, 7, 7)
        p.fillPath(outline, QColor(C_PANEL_DEEP))
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.drawPath(outline)

        for i, (side, label, col) in enumerate(
                (("cpu", "CPU", QColor(C_CPU)), ("cla", "CLA", QColor(C_CLA)))):
            cell = QRectF(rect.left() + i * half, rect.top(), half, rect.height())
            active = (side == "cla") == bool(on_cla)
            hot = self._hover == (sid, side)
            if active:
                p.save()
                p.setClipPath(outline)
                fp = QPainterPath()
                fp.addRoundedRect(cell.adjusted(2, 2, -2, -2), 5, 5)
                p.fillPath(fp, col)
                p.restore()
            elif hot:
                p.save()
                p.setClipPath(outline)
                fp = QPainterPath()
                fp.addRoundedRect(cell.adjusted(2, 2, -2, -2), 5, 5)
                p.fillPath(fp, QColor(C_LANE))
                p.restore()
            p.setFont(_ui_font(10, QFont.Bold, 0.8))
            p.setPen(QColor("#ffffff") if active
                     else (col if hot else QColor(C_MUTED)))
            p.drawText(cell, Qt.AlignCenter, label)
            self._hit.append((cell, sid, side))

    def _draw_gauge(self, p, rect, caption, used, limit, accent=C_ACCENT):
        frac = (used / limit) if limit else 0.0
        col = (QColor(C_ERR) if frac > 1.0
               else QColor(C_WARN) if frac > 0.88 else QColor(accent))

        p.setFont(_ui_font(10, QFont.Bold, 1.0))
        p.setPen(QColor(C_MUTED))
        p.drawText(QRectF(rect.left(), rect.top(), rect.width() * 0.5, 15),
                   Qt.AlignLeft | Qt.AlignVCenter, caption.upper())
        p.setFont(_ui_font(11, QFont.DemiBold))
        p.setPen(QColor(C_TEXT if frac <= 1.0 else C_ERR))
        p.drawText(QRectF(rect.left() + rect.width() * 0.3, rect.top(),
                          rect.width() * 0.7, 15),
                   Qt.AlignRight | Qt.AlignVCenter,
                   f"{used:.1f} / {limit:.0f} KB   {frac * 100:.0f}%")

        track = QRectF(rect.left() + 0.5, rect.top() + 19,
                       max(rect.width() - 1, 1), 8)
        tp = QPainterPath()
        tp.addRoundedRect(track, 4, 4)
        p.fillPath(tp, QColor(C_PANEL))
        p.setPen(QPen(QColor(C_BORDER), 1))
        p.drawPath(tp)
        if frac > 0:
            p.save()
            p.setClipPath(tp)
            fp = QPainterPath()
            fp.addRoundedRect(QRectF(track.left(), track.top(),
                                     max(3.0, min(frac, 1.0) * track.width()),
                                     track.height()), 4, 4)
            p.fillPath(fp, col)
            p.restore()


# ==========================================
# 1. MODEL
# ==========================================
class ScalableDiagramCNN(nn.Module):
    """pool_w=False -> pool along the time axis only (the right choice for MFCC)."""

    def __init__(self, channels, input_h=32, input_w=12, num_classes=3,
                 num_layers=2, pool_w=False, dropout=0.15):
        super().__init__()
        self.channels, self.num_layers, self.pool_w = channels, num_layers, pool_w

        layers, in_ch = [], 1
        for _ in range(num_layers):
            layers += [nn.Conv2d(in_ch, channels, 3, 1, 0),
                       nn.ReLU(),
                       nn.MaxPool2d((2, 2) if pool_w else (2, 1))]
            in_ch = channels
        self.features = nn.Sequential(*layers)

        h, w = self.out_shape(input_h, input_w, num_layers, pool_w)
        if h <= 0 or w <= 0:
            raise ValueError(
                f"Input {input_h}x{input_w} is too small for {num_layers} layers.")

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * h * w, num_classes)

    @staticmethod
    def out_shape(h, w, num_layers, pool_w):
        for _ in range(num_layers):
            h = (h - 2) // 2
            w = (w - 2) // 2 if pool_w else (w - 2)
        return h, w

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.drop(torch.flatten(x, 1)))


# ==========================================
# 2. QUANTIZATION AND C EXPORT
# ==========================================
QMAX = {"INT8": (127.0, np.int8, "int8_t"), "INT16": (32767.0, np.int16, "int16_t")}


def quantize(arr, mode):
    """-> (quantized array, scale). Scale is 1.0 for FLOAT32."""
    if mode == "FLOAT32":
        return arr.astype(np.float32), 1.0
    qmax, dtype, _ = QMAX[mode]
    peak = float(np.max(np.abs(arr)))
    scale = qmax / peak if peak > 0 else 1.0
    return np.round(arr * scale).astype(dtype), scale


def quant_dequant_model(model, mode):
    """Quantize then dequantize weights -> float emulation of the on-device model."""
    import copy
    m = copy.deepcopy(model)
    if mode == "FLOAT32":
        return m
    with torch.no_grad():
        for p in m.parameters():
            a = p.detach().cpu().numpy()
            q, s = quantize(a, mode)
            p.copy_(torch.tensor(q.astype(np.float32) / s).reshape(p.shape))
    return m


def write_array(f, name, arr, mode, is_label=False):
    """Writes the array to f in the model_weights_*.h format, and also
    returns (quantized_flat_array, scale) so a caller can reuse the exact
    same quantized values elsewhere (e.g. baking them into the embedded/
    main.c and weights.h templates) without re-quantizing and risking a
    mismatch. Returns (None, None) for is_label (int32 labels aren't
    quantized)."""
    arr = np.asarray(arr)
    shape = "x".join(map(str, arr.shape))
    arr = arr.flatten()
    if is_label:
        f.write(f"const int32_t {name}[{len(arr)}] = {{ "
                + ", ".join(str(int(v)) for v in arr) + " };\n\n")
        return None, None
    q, scale = quantize(arr.astype(np.float64), mode)
    ctype = "float" if mode == "FLOAT32" else QMAX[mode][2]
    vals = ([f"{v:.8f}f" for v in q] if mode == "FLOAT32"
            else [str(int(v)) for v in q])
    f.write(f"// [{shape}]\n")
    if mode != "FLOAT32":
        f.write(f"const float {name}_scale = {scale:.10f}f;   "
                f"// real = value / {name}_scale\n")
    f.write(f"const {ctype} {name}[{len(q)}] = {{\n")
    for i in range(0, len(vals), 10):
        f.write("    " + ", ".join(vals[i:i + 10])
                + ("," if i + 10 < len(vals) else "") + "\n")
    f.write("};\n\n")
    return q, scale


# ==========================================
# 2b. EMBEDDED FIRMWARE FILE GENERATION
# ==========================================
# CNN_Inference_Edge-AI_with_CLA's own files split into two kinds:
#  - "library" files that never change between exports, because they are
#    parametrized by definitions.h rather than carrying model data:
#    mfcc_extract.c (MFCC stays on the CPU), dense_layer.cla (the fused
#    conv1+pool1 / conv2+pool2 / dense CLA kernels), convolution.c/.h,
#    cla_dense_layer_shared.h, golden_vector*.h.
#  - files that MUST change for every trained model: definitions.h
#    (IMAGE_H/IMAGE_W/NUM_FILTERS/NUM_CLASSES + the six quantization scales),
#    weights.h (conv1/conv2/fc arrays and their *_scale constants) and main.c
#    (weight[], the CLA-resident copy of the dense weights). These are made by
#    reading the real files as templates and patching only the data-bearing
#    declarations -- everything else (the MFCC call, ADC/DMA setup, golden-
#    vector loop, CLA task trigger, comments) stays byte-identical to what is
#    checked in, so hand edits to the logic are never clobbered.
#  - generated from scratch every export: cla_pipeline_config.h (the stage
#    split) and feature_tables.h (MFCC tables from feature_config.json + the
#    training dataset's own mu/sigma; see build_feature_tables_h).
FIRMWARE_DIR = "CNN_Inference_Edge-AI_with_CLA"
# Only these three are read as templates; the rest of the firmware tree is
# never touched, so a build only has to ship these.
FIRMWARE_TEMPLATES = ("definitions.h", "weights.h", "main.c")


def find_firmware_templates():
    """Where to read definitions.h / weights.h / main.c from, in order of
    preference:

      1. EDGE_AI_FIRMWARE_DIR, for a checkout kept somewhere else
      2. CNN_Inference_Edge-AI_with_CLA/ under the current directory
      3. the same, beside the application
      4. the snapshot shipped inside the build

    1-3 mean an installed build still patches YOUR checkout, so hand edits to
    main.c survive, exactly as when running from source. 4 is the fallback
    that keeps the app usable with no checkout at all. Returns None when even
    the snapshot is missing.
    """
    candidates = []
    env = os.environ.get("EDGE_AI_FIRMWARE_DIR")
    if env:
        candidates.append(env)
    candidates += [os.path.join(os.getcwd(), FIRMWARE_DIR),
                   os.path.join(app_dir(), FIRMWARE_DIR),
                   resource_path("firmware_templates")]
    for cand in candidates:
        if all(os.path.isfile(os.path.join(cand, f)) for f in FIRMWARE_TEMPLATES):
            return cand
    return None


def format_int_array_body(values):
    vals = [str(int(v)) for v in values]
    lines = []
    for i in range(0, len(vals), 10):
        chunk = vals[i:i + 10]
        lines.append("    " + ", ".join(chunk)
                      + ("," if i + 10 < len(vals) else ""))
    return "\n".join(lines)


def patch_c_define(text, name, value):
    """Replaces the value token of '#define NAME <value>' (any trailing
    comment on that line is left untouched) -- used for definitions.h's
    architecture constants and quantization scales."""
    pattern = re.compile(rf'(#define\s+{re.escape(name)}\s+)\S+')
    new_text, n = pattern.subn(lambda m: m.group(1) + str(value), text, count=1)
    if n == 0:
        raise ValueError(f"#define {name} not found in template")
    return new_text


def patch_c_array(text, decl, bracket, values):
    """Replaces the initializer (and array size) of a
    '<decl> [<bracket>] = { ... };' or '<decl>[N] = { ... };' declaration --
    used for weights.h's arrays and main.c's weight[]."""
    body = format_int_array_body(values)
    new_block = f"{decl} [{bracket}] = {{\n{body}\n}};"
    pattern = re.compile(
        rf'{re.escape(decl)}\s*\[[^\]]*\]\s*=\s*\{{.*?\}};', re.DOTALL)
    new_text, n = pattern.subn(lambda m: new_block, text, count=1)
    if n == 0:
        raise ValueError(f"array '{decl}' not found in template")
    return new_text


def find_feature_config(dataset_path):
    """Prepare_features.py writes folds/<name>.npz and feature_config.json in
    its working directory -- i.e. next to folds/. Look there, then next to the
    .npz itself. Returns the path or None."""
    d = os.path.dirname(os.path.abspath(dataset_path))
    for cand in (os.path.join(os.path.dirname(d), "feature_config.json"),
                 os.path.join(d, "feature_config.json")):
        if os.path.isfile(cand):
            return cand
    return None


def build_feature_tables_h(cfg, mu, sigma):
    """Returns the text of feature_tables.h (the MFCC tables mfcc_extract.c runs
    on): Hann window, sparse mel filterbank, reduced DCT-II and the z-score
    mu/sigma. A port of Sound_Analyses/generate_c2000_tables.py -- same
    librosa/scipy calls, same order and formatting -- so the output matches
    that script's for the same feature_config.json + dataset. Unlike the
    script (which always reads folds/dataset.npz), mu/sigma come from
    whichever dataset the model was actually trained on."""
    import librosa
    from scipy.fft import dct
    from scipy.signal import get_window

    if cfg["center"]:
        raise ValueError("feature_config.json has center=True; the C2000 cannot "
                         "pad a live stream, features must be built with center=False")
    n_fft, n_mels = cfg["n_fft"], cfg["n_mels"]
    n_out = cfg["n_mfcc"] - int(cfg["drop_c0"])
    n_frames = cfg["n_frames"]
    if len(mu) != n_out or len(sigma) != n_out:
        raise ValueError(f"dataset mu/sigma have {len(mu)} values, feature_config.json "
                         f"expects {n_out} -- not from the same prepare_features run?")

    hann = get_window("hann", n_fft, fftbins=True).astype(np.float64)

    fb = librosa.filters.mel(sr=cfg["sr"], n_fft=n_fft, n_mels=n_mels,
                             fmin=cfg["fmin"], fmax=cfg["fmax"], htk=cfg["htk"],
                             norm="slaney")
    mel_filters = []
    for i in range(fb.shape[0]):
        nz = np.nonzero(fb[i])[0]
        if len(nz) == 0:
            raise ValueError(f"mel filter {i} is empty (n_mels too high for n_fft/fmax?)")
        if nz.max() - nz.min() + 1 != len(nz):
            raise ValueError(f"mel filter {i} is not contiguous; sparse storage invalid")
        mel_filters.append((int(nz.min()), fb[i, nz.min():nz.max() + 1].astype(np.float64)))

    full = dct(np.eye(n_mels), type=2, norm="ortho", axis=0)[:cfg["n_mfcc"]]
    dct_mat = full[1:] if cfg["drop_c0"] else full

    def c_array(name, arr):
        vals = ", ".join(f"{v:.10f}f" for v in arr)
        return f"static const float {name}[{len(arr)}] = {{ {vals} }};\n"

    total_nnz = sum(len(w) for _, w in mel_filters)
    n_fft_bins = n_fft // 2 + 1
    L = []
    L.append("#ifndef FEATURE_TABLES_H")
    L.append("#define FEATURE_TABLES_H")
    L.append("")
    L.append("#include <stdint.h>")
    L.append("")
    L.append("// generate_c2000_tables.py ile feature_config.json'dan uretildi.")
    L.append("// Bu tablolar egitimde kullanilan MFCC boru hattinin BIREBIR ayni")
    L.append("// parametreleriyle hesaplanmistir. Elle degistirmeyin.")
    L.append("")
    L.append(f"#define FEAT_SR          {cfg['sr']}")
    L.append(f"#define FEAT_N_FFT       {n_fft}")
    L.append(f"#define FEAT_HOP         {cfg['hop']}")
    L.append(f"#define FEAT_N_MELS      {n_mels}")
    L.append(f"#define FEAT_N_FFT_BINS  {n_fft_bins}   // n_fft/2 + 1, DC..Nyquist")
    L.append(f"#define FEAT_N_OUT       {n_out}        // MFCC katsayi sayisi (C0 {'HARIC' if cfg['drop_c0'] else 'DAHIL'})")
    L.append(f"#define FEAT_N_FRAMES    {n_frames}     // dilim basina cerceve sayisi")
    L.append(f"#define FEAT_SLICE_LEN   {cfg['slice_len']}   // dilim basina orneksec")
    L.append("#define FEAT_AMIN        1e-10f  // librosa power_to_db varsayilani")
    L.append("#define FEAT_TOP_DB      80.0f   // librosa power_to_db varsayilani")
    L.append(f"#define FEAT_CLIP        {cfg['clip']:.1f}f")
    L.append("")
    L.append("// --- 12-bit ADC on-uc (0..4095) --------------------------------")
    L.append("// DC giderme: tamponun ORTALAMASINI cikar. Filtre yok, yerlesme")
    L.append("// suresi yok, tek gecis. Kartinizin gercek bias'i 2048'den kaysa")
    L.append("// bile otomatik dogru olur.")
    L.append("//")
    L.append("//   float mean = 0; for(i) mean += adc[i]; mean /= FEAT_SLICE_LEN;")
    L.append("//   for(i) x[i] = (adc[i] - mean) / FEAT_ADC_SCALE;")
    L.append("#define FEAT_ADC_SCALE       2048.0f   // (adc-mean)/SCALE -> ~[-1,1]")
    L.append("// Hann penceresi, cerceve orneklerine eleman-eleman carpilir")
    L.append(c_array("feat_hann_window", hann))
    L.append("")
    L.append("// Mel filtre bankasi -- her filtre FFT guc spektrumunun (|X|^2)")
    L.append("// ardisik bir araligiyla nokta carpimidir. start_bin, o araligin")
    L.append("// ilk FFT bin indeksidir (0 = DC). Toplam agirlik sayisi FEAT_N_MELS")
    L.append("// filtreden kucuktur cunku ucgen filtreler seyrektir.")
    L.append(f"#define FEAT_MEL_NNZ {total_nnz}")
    L.append("")
    L.append("static const uint16_t feat_mel_start[FEAT_N_MELS] = {")
    L.append("    " + ", ".join(str(s) for s, _ in mel_filters) + "\n};")
    L.append("")
    L.append("static const uint16_t feat_mel_len[FEAT_N_MELS] = {")
    L.append("    " + ", ".join(str(len(w)) for _, w in mel_filters) + "\n};")
    L.append("")
    all_weights = np.concatenate([w for _, w in mel_filters])
    L.append("// tum filtrelerin agirliklari ucucu ucuca; her filtre icin")
    L.append("// feat_mel_start[i] .. feat_mel_start[i]+feat_mel_len[i] araligindaki")
    L.append("// FFT power bin'leri, feat_mel_weights icindeki karsilik gelen")
    L.append("// feat_mel_len[i] agirlikla carpilip toplanir.")
    offsets = np.cumsum([0] + [len(w) for _, w in mel_filters])[:-1]
    L.append("static const uint16_t feat_mel_offset[FEAT_N_MELS] = {")
    L.append("    " + ", ".join(str(o) for o in offsets) + "\n};")
    L.append("")
    L.append(c_array("feat_mel_weights", all_weights))
    L.append("")
    L.append(f"// Indirgenmis DCT-II (ortho) matrisi, {dct_mat.shape[0]}x{dct_mat.shape[1]}.")
    L.append("// mfcc[i] = sum_j dct_matrix[i][j] * log_mel[j]")
    L.append("static const float feat_dct_matrix[FEAT_N_OUT][FEAT_N_MELS] = {")
    for row in dct_mat:
        L.append("    { " + ", ".join(f"{v:.10f}f" for v in row) + " },")
    L.append("};")
    L.append("")
    L.append("// z-score normalizasyonu: (mfcc - mu) / sigma, sonra +-FEAT_CLIP'e kirp")
    L.append(c_array("feat_mu", np.asarray(mu, dtype=np.float64)))
    L.append(c_array("feat_sigma", np.asarray(sigma, dtype=np.float64)))
    L.append("")
    L.append("#endif // FEATURE_TABLES_H")
    # trailing newline: the script's file ended without one, which made the
    # TI compiler warn (#1 "last line of file ends without a newline")
    return "\n".join(L) + "\n"


def patch_c_float_const(text, name, value):
    """Replaces the value of 'const float <name> = <value>f;' -- weights.h
    carries a companion *_scale constant next to every quantized array."""
    pattern = re.compile(rf'(const\s+float\s+{re.escape(name)}\s*=\s*)[^;]+;')
    new_text, n = pattern.subn(lambda m: m.group(1) + f"{value:.10f}f;",
                                text, count=1)
    if n == 0:
        raise ValueError(f"const float {name} not found in template")
    return new_text


def write_float_array(f, name, arr):
    arr = np.asarray(arr).flatten()
    f.write(f"const float {name}[{len(arr)}] = {{\n    "
            + ", ".join(f"{v:.8f}f" for v in arr) + "\n};\n\n")


# ==========================================
# 3. TRAINING / SEARCH THREAD
# ==========================================
class TrainingThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    optimal_arch_found_signal = pyqtSignal(int)
    # Front-end feedback only. progress_signal carries the run strip's state
    # (phase / channel candidate / seed / epoch / live metrics) and
    # summary_signal the finished run's numbers for the Results tab. Nothing
    # in the search waits on either, so a GUI that ignores them behaves
    # exactly as before.
    progress_signal = pyqtSignal(dict)
    summary_signal = pyqtSignal(dict)

    def __init__(self, max_channels, layers, quant_mode, input_h, input_w,
                 num_classes, target_val, metric, pool_w, dataset_path,
                 cla_stage, timing_note=None):
        super().__init__()
        # Lines the GUI's timing model produced for this plan, written into
        # cla_pipeline_config.h as a comment. Display only: nothing in the
        # search or the export depends on them.
        self.timing_note = timing_note or []
        self.max_channels = max_channels
        self.layers = layers
        self.quant_mode = quant_mode
        self.input_h, self.input_w = input_h, input_w
        self.num_classes = num_classes
        self.target_val = target_val
        self.metric = metric          # "Macro-F1" | "Confident Acc"
        self.pool_w = pool_w
        self.dataset_path = dataset_path
        # Which inference stage should run on the CLA vs the C28x CPU, as set
        # from the GUI checkboxes: {'conv': [layer1, layer2], 'pool': [...],
        # 'dense': bool}. Written out as cla_pipeline_config.h, which
        # CNN_Inference_Edge-AI_with_CLA's main.c reads via CLA_STAGE_*
        # macros to pick the CPU function or the matching Cla1TaskN for each
        # stage. Not build/hardware-tested here -- see that header's comment
        # for the CLA Data RAM budget this plan has to fit in.
        self.cla_stage = cla_stage
        self.max_epochs = 40
        self.patience = 10
        self.class_w = None
        self.pert_rng = np.random.default_rng(SEED)
        self._prog = {"phase": "", "channels": None, "trial": 0, "seed": 0,
                      "seeds": N_SEEDS, "epoch": 0, "epochs": self.max_epochs,
                      "acc": None, "f1": None, "conf": None, "best": None,
                      "done": False}

    # ------------------------------------------------------ gui feedback
    def emit_progress(self, **changed):
        """Merges `changed` into the run state and pushes the whole thing to
        the GUI, so the receiver never has to remember previous partial
        updates. Display only -- see progress_signal."""
        self._prog.update(changed)
        self.progress_signal.emit(dict(self._prog))

    # ---------------------------------------------------------------- data
    def load_data(self):
        d = np.load(self.dataset_path, allow_pickle=True)
        Xtr = torch.tensor(d["X_train"], dtype=torch.float32).unsqueeze(1)
        ytr = torch.tensor(d["y_train"], dtype=torch.long)
        Xva = torch.tensor(d["X_val"], dtype=torch.float32).unsqueeze(1)
        yva = torch.tensor(d["y_val"], dtype=torch.long)
        tr = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
        va = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xva, yva), batch_size=512, shuffle=False)

        # class weights: counters the 8/7/6 file imbalance without discarding data
        cnt = np.bincount(d["y_train"], minlength=self.num_classes).astype(np.float32)
        self.class_w = torch.tensor(cnt.sum() / (self.num_classes * cnt))
        return d, tr, va

    # ----------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        P, Y, C = [], [], []
        for xb, yb in loader:
            prob = torch.softmax(model(xb), dim=1)
            mx, pred = torch.max(prob, 1)
            P.append(pred); Y.append(yb); C.append(mx)
        P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy(); C = torch.cat(C).numpy()
        acc = 100.0 * (P == Y).mean()
        conf = 100.0 * ((P == Y) & (C >= self.target_val / 100.0)).mean()
        f1 = 100.0 * f1_score(Y, P, average="macro", zero_division=0)
        return acc, conf, f1, P, Y

    def score(self, acc, conf, f1):
        return f1 if self.metric == "Macro-F1" else conf

    # ------------------------------------------------------------- training
    def train_one(self, ch, tr, va, seed=SEED, log=True, seed_idx=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = ScalableDiagramCNN(ch, self.input_h, self.input_w,
                                   self.num_classes, self.layers, self.pool_w)
        # No label smoothing: it caps the achievable softmax at (1-s)+s/K,
        # which makes the "Confident Acc" metric unreachable for small K.
        crit = nn.CrossEntropyLoss(weight=self.class_w)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best, best_state, bad = -1.0, None, 0
        for ep in range(1, self.max_epochs + 1):
            model.train()
            for xb, yb in tr:
                opt.zero_grad()
                crit(model(xb), yb).backward()
                opt.step()

            acc, conf, f1, _, _ = self.evaluate(model, va)
            s = self.score(acc, conf, f1)
            if s > best + 1e-6:
                best, bad = s, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            # the run strip follows EVERY seed, not just the logged one, so a
            # long multi-seed candidate never looks stalled
            self.emit_progress(channels=ch, seed=seed_idx + 1, epoch=ep,
                               epochs=self.max_epochs, acc=acc, f1=f1, conf=conf)
            if log:
                self.update_signal.emit(
                    f"  -> ep {ep:2d}/{self.max_epochs} | val acc {acc:.1f}% | "
                    f"F1 {f1:.1f}% | conf {conf:.1f}%{'  *' if bad == 0 else ''}")
            # Train every architecture the same way; the target is checked
            # after training, not used to cut the epoch loop short.
            if bad >= self.patience:
                break

        model.load_state_dict(best_state)
        return model, best

    # ------------------------------------------------------- robustness check
    def perturb_features(self, xb):
        """Feature-space stand-in for a real-world channel shift (speaker/mic
        gain mismatch, frequency-response tilt, extra noise) -- applied only
        at evaluation time, never during training."""
        n, _, h, w = xb.shape
        gain = torch.tensor(self.pert_rng.uniform(*PERTURB_GAIN, size=(n, 1, 1, 1)),
                            dtype=xb.dtype)
        tilt = torch.tensor(self.pert_rng.uniform(-PERTURB_TILT, PERTURB_TILT, size=(n, 1, 1, 1)),
                            dtype=xb.dtype)
        coef_idx = torch.linspace(-1, 1, w, dtype=xb.dtype).view(1, 1, 1, -1)
        noise = torch.tensor(self.pert_rng.standard_normal(xb.shape), dtype=xb.dtype) * PERTURB_NOISE
        return xb * gain * (1.0 + tilt * coef_idx) + noise

    @torch.no_grad()
    def evaluate_perturbed(self, model, loader):
        """evaluate(), but under PERTURB_TRIALS different synthetic channel
        shifts; returns the WORST score seen, not the average."""
        model.eval()
        worst = None
        for _ in range(PERTURB_TRIALS):
            P, Y, C = [], [], []
            for xb, yb in loader:
                prob = torch.softmax(model(self.perturb_features(xb)), dim=1)
                mx, pred = torch.max(prob, 1)
                P.append(pred); Y.append(yb); C.append(mx)
            P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy(); C = torch.cat(C).numpy()
            acc = 100.0 * (P == Y).mean()
            conf = 100.0 * ((P == Y) & (C >= self.target_val / 100.0)).mean()
            f1 = 100.0 * f1_score(Y, P, average="macro", zero_division=0)
            s = self.score(acc, conf, f1)
            worst = s if worst is None else min(worst, s)
        return worst

    def robust_train_and_score(self, ch, tr, va):
        """Trains N_SEEDS models for this channel count; each is scored on
        both clean and perturbed val, and the overall minimum (worst seed x
        worst perturbation) is what phase 1/2 use to accept or shrink further.
        The exported model is always the canonical (first-seed) one, so a
        given channel count's weights stay reproducible."""
        canonical_model, canonical_clean = None, None
        worst = None
        self._prog["trial"] = self._prog.get("trial", 0) + 1
        for i in range(N_SEEDS):
            m, clean = self.train_one(ch, tr, va, seed=SEED + i, log=(i == 0),
                                      seed_idx=i)
            pert = self.evaluate_perturbed(m, va)
            seed_worst = min(clean, pert)
            worst = seed_worst if worst is None else min(worst, seed_worst)
            if i == 0:
                canonical_model, canonical_clean = m, clean
        return canonical_model, canonical_clean, worst

    # ------------------------------------------------- embedded firmware
    def write_cla_pipeline_config(self, path, cs):
        """Writes cla_pipeline_config.h -- the three CLA_STAGE_* switches the
        firmware compiles against (S1 = conv1+pool1 fused, S2 = conv2+pool2
        fused, S3 = dense). MFCC is not a switch: it always runs on the CPU."""
        s1 = int(cs["stages"][0])
        s2 = int(cs["stages"][1]) if self.layers > 1 else 0
        s3 = int(cs["dense"])
        note = ""
        if self.timing_note:
            # the file is written in the default locale encoding, and the GUI
            # strings carry typographic characters the C28x toolchain has no
            # reason to see anyway
            def ascii_only(s):
                return (s.replace("·", "-").replace("→", "->")
                         .replace("≥", ">=").replace("—", "--")
                         .replace("’", "'")
                         .encode("ascii", "replace").decode("ascii"))
            note = ("//\n// --- ZAMAN BUTCESI (GUI cevrim modeli, olculmus "
                    "degil) ------------\n"
                    + "".join(f"// {ascii_only(ln)}\n"
                              for ln in self.timing_note))
        with open(path, "w") as f:
            f.write(
                "#ifndef CLA_PIPELINE_CONFIG_H\n#define CLA_PIPELINE_CONFIG_H\n\n"
                "// ADOT GUI tarafindan uretildi. 1 = asama CLA'da,\n"
                "// 0 = asama C28x CPU'da (convolution.c'deki ikizi calisir --\n"
                "// matematik birebir ayni, sonuc degismez).\n"
                "//\n"
                "//   MFCC : HER ZAMAN CPU (mfcc_extract.c) -- CLA'nin 2KB program\n"
                "//          RAM'ine FFT/log/DCT sigmaz, offload edilebilir degil.\n"
                "//   S1   : conv1 + relu + pool1 (fused)\n"
                "//   S2   : conv2 + relu + pool2 (fused)\n"
                "//   S3   : dense / FC\n"
                "//\n"
                "// BELLEK: CLA program RAM'i (RAMLS5_PROG) 2048 byte ve uc asama\n"
                "// birlikte 2042 byte tutuyor -- bu yuzden ucu de 1 ise hepsi TEK\n"
                "// gorevde toplanir (CLA_SINGLE_TASK). Biri CPU'ya alininca o kod\n"
                "// hic derlenmez, acilan yerle asama basina ayri gorev kullanilir.\n"
                "// CLA veri RAM'i (RAMLS_0_1_2_3_4) 10240 byte; CPU'ya alinan\n"
                "// asamanin agirlik/tampon kopyasi bu havuzdan cikar.\n"
                + note +
                f"#define CLA_STAGE_CONV1_POOL1  {s1}\n"
                f"#define CLA_STAGE_CONV2_POOL2  {s2}\n"
                f"#define CLA_STAGE_DENSE        {s3}\n\n"
                "// --- turetilmis yardimcilar (elle degistirmeyin) ---------------\n"
                "#define CLA_SINGLE_TASK  (CLA_STAGE_CONV1_POOL1 && CLA_STAGE_CONV2_POOL2 && CLA_STAGE_DENSE)\n"
                "#define CLA_ANY_STAGE    (CLA_STAGE_CONV1_POOL1 || CLA_STAGE_CONV2_POOL2 || CLA_STAGE_DENSE)\n"
                "#define CLA_TOUCHES_WORKSPACE  CLA_ANY_STAGE\n\n"
                "#endif // CLA_PIPELINE_CONFIG_H\n")

    def write_feature_tables(self, embedded_dir):
        """Writes embedded_dir/feature_tables.h (MFCC tables + mu/sigma of the
        dataset this model was trained on). Returns a short note for the log,
        or "" if it could not be generated (the reason is logged).

        mfcc_extract.c normalizes every coefficient with feat_mu/feat_sigma.
        They must be the statistics of the training dataset: dataset.npz and
        full.npz differ by up to 0.55 (mu) / 0.92 (sigma), and a stale header
        compiles fine while feeding the network inputs it never saw."""
        try:
            cfg_path = find_feature_config(self.dataset_path)
            if cfg_path is None:
                raise FileNotFoundError(
                    "feature_config.json not found next to the dataset's folds/ "
                    "folder (Prepare_features.py writes it there)")
            with open(cfg_path) as f:
                fcfg = json.load(f)
            n_out = fcfg["n_mfcc"] - int(fcfg["drop_c0"])
            if (fcfg["n_frames"], n_out) != (self.input_h, self.input_w):
                raise ValueError(
                    f"feature_config.json describes {fcfg['n_frames']}x{n_out} "
                    f"features but the model input is {self.input_h}x{self.input_w}")
            ds = np.load(self.dataset_path, allow_pickle=True)
            text = build_feature_tables_h(fcfg, ds["mu"], ds["sigma"])
            with open(f"{embedded_dir}/feature_tables.h", "w", encoding="ascii") as f:
                f.write(text)
            return f"feature_tables.h (mu/sigma from {os.path.basename(self.dataset_path)})"
        except Exception as e:
            self.update_signal.emit(
                f"<span style='color:{C_ERR};'>embedded/feature_tables.h NOT "
                f"generated: {e} -- the firmware's current copy may hold another "
                f"dataset's mu/sigma.</span>")
            return ""

    def write_embedded_files(self, embedded_dir, best_ch, conv_layers,
                              fcw_q, fcw_scale, fcb_q, fcb_scale):
        """Writes ready-to-copy CNN_Inference_Edge-AI_with_CLA/{definitions.h,
        weights.h,main.c} into embedded_dir by patching the real files (read
        as templates) -- see the module-level comment above FIRMWARE_DIR for
        why only these three and not mfcc_extract.c/dense_layer.cla/
        convolution.c/cla_dense_layer_shared.h."""
        os.makedirs(embedded_dir, exist_ok=True)
        # the offload plan travels with the other three files -- main.c,
        # weights.h and dense_layer.cla all compile against it
        self.write_cla_pipeline_config(
            f"{embedded_dir}/cla_pipeline_config.h", self.cla_stage)
        # feature_tables.h depends only on the dataset + MFCC config, not on
        # the model's quantization or shape, so it is written on EVERY export
        # -- before the INT16/2-layer check below, which only gates the files
        # that carry model weights.
        ft_note = self.write_feature_tables(embedded_dir)

        # The firmware's pipeline is fixed: MFCC (CPU, mfcc_extract.c) ->
        # conv1+pool1 fused -> conv2+pool2 fused -> dense, all INT16, 2 conv
        # layers, and 2x1 pooling hard-coded (height only). A model trained
        # outside that shape cannot be dropped into it by patching data.
        why = []
        if self.quant_mode != "INT16":
            why.append(f"quantization is {self.quant_mode}, firmware is INT16-only")
        if self.layers != 2:
            why.append(f"{self.layers} conv layer(s), firmware has exactly 2")
        if self.pool_w:
            why.append("2D pooling is on, firmware pools height only (2x1)")
        if why:
            self.update_signal.emit(
                f"<span style='color:{C_WARN};'>embedded/main.c, weights.h and "
                "definitions.h were NOT regenerated -- " + "; ".join(why)
                + ". cla_pipeline_config.h"
                + (f" and {ft_note}" if ft_note else "")
                + " were still written.</span>")
            return
        if conv_layers.get(1) is None or conv_layers.get(2) is None:
            self.update_signal.emit(
                f"<span style='color:{C_ERR};'>embedded/: missing quantized conv "
                "arrays -- main.c/weights.h/definitions.h skipped.</span>")
            return

        tpl = find_firmware_templates()
        if tpl is None:
            self.update_signal.emit(
                f"<span style='color:{C_ERR};'>embedded/: no firmware "
                f"templates found. Put a {FIRMWARE_DIR}/ checkout beside "
                f"the app, or point EDGE_AI_FIRMWARE_DIR at one. "
                f"main.c / weights.h / definitions.h skipped.</span>")
            return

        w1q, w1s, b1q, b1s = conv_layers[1]
        w2q, w2s, b2q, b2s = conv_layers[2]

        try:
            # ---- definitions.h: architecture sizes + quantization scales --
            # IMAGE_H/IMAGE_W (not INPUT_*) and no POOL_W: the real firmware
            # names them that way and hard-codes 2x1 pooling. Everything else
            # in the file is a formula over these, so the preprocessor
            # recomputes CONV*/POOL*/DENSE_LAYER_* sizes on its own.
            defs = open(os.path.join(tpl, "definitions.h")).read()
            defs = patch_c_define(defs, "IMAGE_H", self.input_h)
            defs = patch_c_define(defs, "IMAGE_W", self.input_w)
            defs = patch_c_define(defs, "NUM_FILTERS", best_ch)
            defs = patch_c_define(defs, "NUM_CLASSES", self.num_classes)
            defs = patch_c_define(defs, "CONV1_WEIGHT_SCALE", f"{w1s:.10f}f")
            defs = patch_c_define(defs, "CONV1_BIAS_SCALE", f"{b1s:.10f}f")
            defs = patch_c_define(defs, "CONV2_WEIGHT_SCALE", f"{w2s:.10f}f")
            defs = patch_c_define(defs, "CONV2_BIAS_SCALE", f"{b2s:.10f}f")
            defs = patch_c_define(defs, "FC_WEIGHT_SCALE", f"{fcw_scale:.10f}f")
            defs = patch_c_define(defs, "FC_BIAS_SCALE", f"{fcb_scale:.10f}f")
            with open(f"{embedded_dir}/definitions.h", "w") as f:
                f.write(defs)

            # ---- weights.h: conv weights/biases + their *_scale constants --
            # input_image is left alone: in this firmware it is the CLA-side
            # buffer that mfcc_extract() fills at runtime (its initializer is
            # only seed data for running without an ADC), and its size follows
            # IMAGE_H*IMAGE_W from definitions.h. fc_weights here is the
            # "for consistency" copy -- the FC weights the CLA actually reads
            # are main.c's weight[], patched below.
            wts = open(os.path.join(tpl, "weights.h")).read()
            for name, q, s in (("conv1_weights", w1q, w1s),
                                ("conv1_bias", b1q, b1s),
                                ("conv2_weights", w2q, w2s),
                                ("conv2_bias", b2q, b2s),
                                ("fc_weights", fcw_q, fcw_scale),
                                ("fc_bias", fcb_q, fcb_scale)):
                wts = patch_c_float_const(wts, f"{name}_scale", s)
                wts = patch_c_array(wts, f"const int16_t {name}", len(q), q)
            with open(f"{embedded_dir}/weights.h", "w") as f:
                f.write(wts)

            # ---- main.c: the CLA-resident copy of the dense weights --------
            # Only weight[] is data; the MFCC call, ADC/DMA setup, golden
            # vector loop and CLA task trigger are left byte-identical.
            mc = open(os.path.join(tpl, "main.c")).read()
            mc = patch_c_array(mc, "const int16_t weight", "DENSE_LAYER_WEIGHTS", fcw_q)
            with open(f"{embedded_dir}/main.c", "w") as f:
                f.write(mc)
        except (FileNotFoundError, ValueError) as e:
            self.update_signal.emit(
                f"<span style='color:{C_ERR};'>embedded/ generation failed: "
                f"{e}</span>")
            return

        with open(f"{embedded_dir}/README.txt", "w") as f:
            f.write(
                "Copy the files in this folder into CNN_Inference_Edge-AI_with_CLA/,\n"
                "overwriting the existing ones, then Project > Clean + Build in CCS:\n"
                "  definitions.h, weights.h, main.c   -- this model's weights/scales\n"
                "  cla_pipeline_config.h              -- CPU/CLA stage split\n"
                + ("  feature_tables.h                   -- MFCC tables, mu/sigma of\n"
                   f"      {os.path.basename(self.dataset_path)}\n" if ft_note else
                   "  (feature_tables.h was NOT generated -- see the GUI log)\n")
                + "\nNOT regenerated (they do not depend on the trained model):\n"
                "  mfcc_extract.c, dense_layer.cla, convolution.c/.h,\n"
                "  cla_dense_layer_shared.h, golden_vector*.h, linker .cmd\n\n"
                f"channels: {best_ch} | layers: {self.layers} | "
                f"input: {self.input_h}x{self.input_w} | "
                f"classes: {self.num_classes}\n"
                "NOT build/hardware-tested by this GUI -- verify in CCS "
                "before flashing.\n")

        self.update_signal.emit(
            f"<span style='color:{C_OK};'>embedded/ ready: definitions.h, "
            f"weights.h, main.c, cla_pipeline_config.h"
            + (f", {ft_note}" if ft_note else "")
            + f" -- copy into {FIRMWARE_DIR}/ and rebuild.</span>")

    # ----------------------------------------------------------------- run
    def run(self):
        try:
            self.emit_progress(phase="Loading dataset")
            self.update_signal.emit(
                f"\n<b>[1/4] Loading dataset: {os.path.basename(self.dataset_path)}</b>")
            data_probe = np.load(self.dataset_path, allow_pickle=True)
            has_test = "X_test" in data_probe.files
            stem = os.path.splitext(os.path.basename(self.dataset_path))[0]
            # Folder name says what the model can and can't be trusted for:
            # a dataset with a real held-out X_test produces an "eval" model
            # (safe to demo/report accuracy on those held-out files); one
            # without X_test (e.g. full.npz) produces a "deployment" model
            # that has already seen every file, so it must not be presented
            # as tested on unseen data.
            tag = "eval_heldout_test" if has_test else "deployment_all_data"
            # GUNCELLEME: her export kendi tarih-saat damgali klasorune yazilir
            # -- ayni dataset'i farkli ayarlarla (kanal/CLA plani/quant) tekrar
            # tekrar denerken eski sonuclar USTUNE YAZILMAZ, hepsi export/
            # altinda ayri ayri durur.
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_dir = os.path.join(output_root(), "export",
                                        f"{stem}__{tag}__{ts}")
            os.makedirs(self.out_dir, exist_ok=True)
            data, tr, va = self.load_data()
            self.update_signal.emit(
                f"train {len(tr.dataset)} | val {len(va.dataset)} samples &mdash; "
                f"selection and early stopping use <b>X_val</b><br>"
                f"class weights: "
                + ", ".join(f"{c}={w:.2f}" for c, w in
                            zip([str(x) for x in data["classes"]], self.class_w.tolist())))

            best_model, best_ch, target_met = None, self.max_channels, False
            best_seen = -1.0

            self.update_signal.emit(
                f"<br><b>[2/4] Architecture search &mdash; target "
                f"{self.metric} >= {self.target_val}%</b>")

            self.emit_progress(phase="Phase 1 · coarse search")
            self.update_signal.emit(
                f"<br><span style='color:{C_ACCENT};'><b>--- PHASE 1: coarse search "
                "(10-channel steps) ---</b></span>")
            ch = min(10, self.max_channels)
            while True:
                self.update_signal.emit(
                    f"<br><i>Trying {ch} channels ({N_SEEDS} seeds x clean+perturbed val, "
                    f"worst-case scored)...</i>")
                m, clean, s = self.robust_train_and_score(ch, tr, va)
                best_seen = max(best_seen, s)
                self.emit_progress(best=best_seen)
                self.update_signal.emit(
                    f"  clean {clean:.1f}% | robust (worst-case) {s:.1f}%")
                if s >= self.target_val:
                    self.update_signal.emit(
                        f"<span style='color:{C_OK};'><b>Target met at {ch} channels "
                        f"&mdash; phase 2</b></span>")
                    best_model, best_ch, target_met = m, ch, True
                    break
                if ch >= self.max_channels:
                    best_model, best_ch = m, ch
                    break
                ch = min(ch + 10, self.max_channels)

            if target_met and best_ch > 1:
                self.emit_progress(phase="Phase 2 · shrinking")
                self.update_signal.emit(
                    f"<br><span style='color:{C_WARN};'><b>--- PHASE 2: step-by-step "
                    "reduction ---</b></span>")
                for fine in range(best_ch - 1, max(0, best_ch - 10), -1):
                    self.update_signal.emit(f"<br><i>{fine} channels...</i>")
                    m, clean, s = self.robust_train_and_score(fine, tr, va)
                    self.update_signal.emit(
                        f"  clean {clean:.1f}% | robust (worst-case) {s:.1f}%")
                    if s < self.target_val:
                        self.update_signal.emit(
                            f"<span style='color:{C_ERR};'>Robust target failed at {fine} "
                            f"channels ({s:.1f}%). Smallest viable: {best_ch} channels.</span>")
                        break
                    if s < best_seen - DEGRADE_TOL:
                        self.update_signal.emit(
                            f"<span style='color:{C_ERR};'>{fine} channels still meet the "
                            f"robust target ({s:.1f}%) but drop more than {DEGRADE_TOL} pts "
                            f"below the best seen ({best_seen:.1f}%). Stopping at "
                            f"{best_ch} channels.</span>")
                        break
                    self.update_signal.emit(
                        f"<span style='color:{C_OK};'>{fine} channels also robust "
                        f"({s:.1f}%), continuing to shrink</span>")
                    best_model, best_ch = m, fine
                    best_seen = max(best_seen, s)
                    self.emit_progress(best=best_seen)

            if not target_met:
                self.update_signal.emit(
                    f"<br><span style='color:{C_ERR};'><b>Target not reached within hardware "
                    "limits; exporting best attempt.</b></span>")

            self.optimal_arch_found_signal.emit(best_ch)

            # ------------- final evaluation + quantization check ------------
            self.emit_progress(phase="Final evaluation", channels=best_ch,
                               seed=0, epoch=0, epochs=0)
            acc, conf, f1, P, Y = self.evaluate(best_model, va)
            cm = confusion_matrix(Y, P, labels=list(range(self.num_classes)))
            classes = [str(c) for c in data["classes"]]
            rows = "<br>".join(
                f"&nbsp;{classes[i][:10]:>10} " + " ".join(f"{v:4d}" for v in cm[i])
                for i in range(len(classes)))
            self.update_signal.emit(
                f"<br><b>[3/4] Selected model: {best_ch} channels</b><br>"
                f"val acc {acc:.1f}% | macro-F1 {f1:.1f}%<br>"
                f"<b>confusion matrix</b> (row = true):<br><pre>{rows}</pre>")

            q_acc = q_f1 = None
            if self.quant_mode != "FLOAT32":
                qm = quant_dequant_model(best_model, self.quant_mode)
                qa, _, qf1, _, _ = self.evaluate(qm, va)
                q_acc, q_f1 = qa, qf1
                delta = qf1 - f1
                col = C_OK if abs(delta) < 1.0 else C_ERR
                self.update_signal.emit(
                    f"<span style='color:{col};'>After {self.quant_mode}: "
                    f"acc {qa:.1f}% | macro-F1 {qf1:.1f}% "
                    f"(delta {delta:+.1f} pts)</span>")

            # -------------------------------------------------- write files -
            torch.save(best_model.state_dict(),
                       f"{self.out_dir}/edge_cnn_{self.quant_mode.lower()}_{best_ch}ch.pth")

            self.emit_progress(phase="Exporting C headers")
            self.update_signal.emit(
                f"<b>[4/4] Generating {self.quant_mode} C headers...</b>")
            sfx, mac = self.quant_mode.lower(), self.quant_mode.upper()
            h, w = ScalableDiagramCNN.out_shape(
                self.input_h, self.input_w, self.layers, self.pool_w)

            with open(f"{self.out_dir}/model_weights_{sfx}.h", "w") as f:
                f.write(f"#ifndef MODEL_WEIGHTS_{mac}_H\n"
                        f"#define MODEL_WEIGHTS_{mac}_H\n\n#include <stdint.h>\n\n")
                f.write(f"#define INPUT_H      {self.input_h}\n"
                        f"#define INPUT_W      {self.input_w}\n"
                        f"#define NUM_CLASSES  {self.num_classes}\n"
                        f"#define NUM_LAYERS   {self.layers}\n"
                        f"#define CHANNELS     {best_ch}\n"
                        f"#define POOL_W       {1 if self.pool_w else 0}\n"
                        f"#define FC_IN        {best_ch * h * w}\n\n")

                f.write("// --- feature normalization: (x - mu) / sigma, "
                        "then clip to [-3,3] ---\n")
                write_float_array(f, "feat_mu", data["mu"])
                write_float_array(f, "feat_sigma", data["sigma"])

                # conv_layers[i] = (weight_q, weight_scale, bias_q, bias_scale)
                # for conv layer i (1-based) -- captured here (INT16 mode)
                # so the embedded/ main.c and weights.h below can reuse the
                # EXACT same quantized values instead of re-quantizing.
                conv_layers = {}
                i = 1
                for layer in best_model.features:
                    if isinstance(layer, nn.Conv2d):
                        wq, wscale = write_array(f, f"conv{i}_weights",
                                    layer.weight.detach().cpu().numpy(), self.quant_mode)
                        bq, bscale = write_array(f, f"conv{i}_bias",
                                    layer.bias.detach().cpu().numpy(), self.quant_mode)
                        conv_layers[i] = (wq, wscale, bq, bscale)
                        i += 1
                fcw_q, fcw_scale = write_array(f, "fc_weights",
                            best_model.fc.weight.detach().cpu().numpy(), self.quant_mode)
                fcb_q, fcb_scale = write_array(f, "fc_bias",
                            best_model.fc.bias.detach().cpu().numpy(), self.quant_mode)
                f.write("#endif\n")

            # -------- CLA offload plan: which stage runs on CLA vs CPU -----
            cs = self.cla_stage
            self.write_cla_pipeline_config(
                f"{self.out_dir}/cla_pipeline_config.h", cs)

            # sample inputs for on-device verification (one per class)
            with open(f"{self.out_dir}/test_samples_{sfx}.h", "w") as f:
                f.write(f"#ifndef TEST_SAMPLES_{mac}_H\n"
                        f"#define TEST_SAMPLES_{mac}_H\n\n#include <stdint.h>\n\n")
                Xva = torch.tensor(data["X_val"], dtype=torch.float32).unsqueeze(1)
                yva = data["y_val"]
                best_model.eval()
                seen = set()
                with torch.no_grad():
                    for idx in range(len(yva)):
                        lab = int(yva[idx])
                        if lab in seen:
                            continue
                        if int(torch.argmax(best_model(Xva[idx:idx + 1]))) == lab:
                            write_array(f, f"test_in_class_{lab}",
                                        Xva[idx].numpy(), self.quant_mode)
                            write_array(f, f"test_lbl_class_{lab}", [lab],
                                        self.quant_mode, is_label=True)
                            seen.add(lab)
                        if len(seen) == self.num_classes:
                            break
                f.write("#endif\n")

            self.write_embedded_files(
                f"{self.out_dir}/embedded", best_ch, conv_layers,
                fcw_q, fcw_scale, fcb_q, fcb_scale)

            # ---- self-documenting info.txt: what this model is/isn't valid for ----
            with open(f"{self.out_dir}/info.txt", "w", encoding="utf-8") as f:
                f.write(f"dataset: {self.dataset_path}\n")
                f.write(f"quant: {self.quant_mode} | channels: {best_ch} | "
                        f"layers: {self.layers} | pool_w: {self.pool_w}\n")
                cla_on = [name for name, on in (
                    ("S1 conv1+pool1", cs['stages'][0]),
                    ("S2 conv2+pool2", cs['stages'][1] and self.layers > 1),
                    ("S3 dense", cs['dense'])) if on]
                f.write(f"CLA offload plan (see cla_pipeline_config.h): "
                        f"{', '.join(cla_on) if cla_on else '(none -- all on CPU)'}"
                        f"   [MFCC always on CPU]\n")
                f.write(f"train slices: {len(tr.dataset)} | val slices: {len(va.dataset)}\n")
                f.write(f"val acc {acc:.1f}% | val macro-F1 {f1:.1f}%\n\n")
                if has_test:
                    f.write(
                        "This model was trained WITHOUT the files below (true held-out\n"
                        "test set, file-level split) -- safe to use for a genuine\n"
                        "unseen-data demo/accuracy report on exactly these files:\n")
                    test_files = sorted(set(str(s) for s in data["src_test"]))
                    for tf in test_files:
                        f.write(f"  {tf}\n")
                else:
                    f.write(
                        "This model was trained on EVERY file in the dataset (no held-out\n"
                        "test set) -- it is the final deployment model, not a fair unseen-\n"
                        "data demo. Do not present it as classifying files it hasn't seen;\n"
                        "use the eval_heldout_test export for that instead.\n")

            self.update_signal.emit(
                f"<br><span style='color:{C_OK};'><b>Done. model_weights, test_samples, "
                f"info.txt and embedded/ are ready in {self.out_dir}.</b></span>")
            self.emit_progress(phase="Done", done=True, seed=0, epoch=0, epochs=0,
                               acc=acc, f1=f1, conf=conf)
            self.summary_signal.emit({
                "channels": best_ch, "layers": self.layers,
                "acc": acc, "f1": f1, "conf": conf,
                "quant": self.quant_mode, "q_acc": q_acc, "q_f1": q_f1,
                "metric": self.metric, "target": self.target_val,
                "target_met": target_met,
                "classes": classes, "cm": cm.tolist(),
                "out_dir": self.out_dir,
            })
            self.finished_signal.emit()

        except Exception as e:
            import traceback
            self.update_signal.emit(
                f"<br><span style='color:{C_ERR};'><b>ERROR: {e}</b></span>"
                f"<pre>{traceback.format_exc()}</pre>")
            self.emit_progress(phase="Failed", done=True, seed=0, epoch=0, epochs=0)
            self.finished_signal.emit()


# ==========================================
# 4. GUI
# ==========================================
class SetupDialog(QDialog):
    """Everything that is set once and then left alone: the target part,
    tensor shape, depth, pooling, the three memory budgets and the real-time
    budget. Keeping it off the main window is what lets the device diagram
    be the window."""

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Target, model, memory and timing setup")
        self.setModal(True)
        self.setStyleSheet(parent.styleSheet() if parent else "")
        self.setMinimumWidth(880)
        self._edits = {}
        self._cfg = dict(cfg)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)
        cols = QHBoxLayout()
        cols.setSpacing(14)
        lay = QVBoxLayout()          # left column
        lay.setSpacing(12)
        right = QVBoxLayout()        # right column
        right.setSpacing(12)
        cols.addLayout(lay, 1)
        cols.addLayout(right, 1)
        outer.addLayout(cols, 1)

        # ---------------------------------------------------- target device
        dev_card = Card("Target device",
                        "Every C2000 part with a CLA. Picking one rewrites "
                        "the budgets below and the clock the timing model runs on.")
        self.dev_combo = QComboBox()
        for d in C2000_DEVICES:
            self.dev_combo.addItem(f"{d['name']}   ({d['family']})", d["key"])
        self.dev_combo.addItem("Custom — keep the values below", CUSTOM_DEVICE)
        idx = self.dev_combo.findData(cfg.get("device", "F28379D"))
        self.dev_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.dev_combo.setFixedHeight(32)
        dev_card.add(self.dev_combo)

        self.dev_spec = QLabel()
        self.dev_spec.setObjectName("Hint")
        self.dev_spec.setWordWrap(True)
        dev_card.add(self.dev_spec)
        self.dev_note = QLabel()
        self.dev_note.setWordWrap(True)
        self.dev_note.setStyleSheet(f"color:{C_MUTED}; font-size:10px;")
        dev_card.add(self.dev_note)
        lay.addWidget(dev_card)

        model = Card("Model", "Shape and class count come from the dataset; "
                              "override only if you know why.")
        model.add(self._field("shape", "Input shape (H,W):",
                              f"{cfg['input_h']},{cfg['input_w']}"))
        model.add(self._field("classes", "Number of classes:", str(cfg["classes"])))
        model.add(self._field("layers", "Conv layers (1-2):", str(cfg["layers"])))

        self.pool_check = QCheckBox("2D pooling")
        self.pool_check.setChecked(cfg["pool_w"])
        self.pool_check.setToolTip(
            "Unchecked: MaxPool(2,1) - pools time only.\n"
            "Checked:   MaxPool(2,2) - pools the coefficient axis as well.\n"
            "With a 32x12 input and 2 layers, 2D pooling collapses the width "
            "from 12 to 1.")
        self.arena_check = QCheckBox("Ultimate Arena")
        self.arena_check.setChecked(cfg["arena"])
        self.arena_check.setToolTip(
            "Reuse one buffer sized to the largest intermediate tensor instead of "
            "summing every layer's output. Input buffer excluded.")
        model.add(self.pool_check)
        model.add(self.arena_check)
        right.addWidget(model)

        mem = Card("Memory budget",
                   "The channel-count ceiling grows until one of these is hit.")
        mem.add(self._field("flash", "Max Flash (KB):", f"{cfg['flash_kb']:g}"))
        mem.add(self._field("cpu", "CPU RAM (KB):", f"{cfg['cpu_kb']:g}"))
        mem.add(self._field("cla", "CLA Data RAM (KB):", f"{cfg['cla_kb']:g}"))
        self._edits["flash"].setToolTip(
            "What is left for weights after the firmware. A device's own "
            "figure is its flash minus a 160 KB reserve: this project's build "
            "measures 131 KB of code, MFCC tables and golden vectors in "
            "Debug/dense_layer_cla.map.")
        self._edits["cpu"].setToolTip(
            "SRAM one C28x can reach. The MFCC front end is charged against "
            "this budget -- its buffers (the raw slice, the log-mel frames and "
            "the FFT scratch) are the largest single item on the CPU side, "
            "~85 KB for a 65-frame window, which is why the C28x RAM and not "
            "the CLA RAM is what a small part runs out of first.")
        self._edits["cla"].setToolTip(
            "CNN_Inference_Edge-AI_with_CLA's linker script (2837xD_FLASH_CLA_lnk_"
            "cpu1.cmd) gives the CLA one combined pool: RAMLS_0_1_2_3_4, length "
            "0x2800. The C2000 is WORD-addressed, so that is 10240 x 16-bit "
            "words = 20 KB of storage (10240 int16 values).\n\n"
            "Measured in Debug/dense_layer_cla.map with everything on the CLA: "
            "6858 of those 10240 words are used (input_image 780, conv weights "
            "390, FC weight 2016, workspace 2532, plus the now-unused "
            "input_vector 672).\n\n"
            "A separate 0x800 = 2048-word block (RAMLS5_PROG) holds CLA program "
            "code and is NOT part of this budget -- that one is nearly full "
            "(2042/2048).")
        lay.addWidget(mem)

        # --------------------------------------------------------- timing
        tim = Card("Timing / real-time budget",
                   "The feature-extraction time lives on the toolbar — it is "
                   "yours to measure. These are the rest.")
        tim.add(self._field("fe_ram", "Feature RAM (KB):",
                            "" if cfg.get("fe_ram_kb") is None
                            else f"{cfg['fe_ram_kb']:g}"))
        self._edits["fe_ram"].setPlaceholderText("auto — this project's MFCC buffers")
        self._edits["fe_ram"].setToolTip(
            "What your feature front end's working buffers occupy in C28x "
            "RAM. Charged against the CPU RAM budget, because on a small "
            "part it -- not the network -- is what runs out of room first.\n\n"
            "Empty: this project's MFCC buffers, derived from the window "
            "(the raw slice, the log-mel frames and the FFT scratch; 80 KB "
            "for a 65-frame window, against 85 KB in the .map). Type your "
            "own if your front end is not that one.")
        tim.add(self._field("infer", "Inference (ms):",
                            "" if cfg.get("infer_ms") is None
                            else f"{cfg['infer_ms']:g}"))
        self._edits["infer"].setPlaceholderText("auto — from the MAC count")
        self._edits["infer"].setToolTip(
            "How long the WHOLE network takes with the stages assigned the "
            "way the diagram has them -- S1 + S2 + S3, not one of them.\n\n"
            "Empty: hand-counted off dense_layer.cla loop by loop, then "
            "corrected by the one factor fitted to the board. That works "
            "out to ~54 effective cycles per MAC on the CLA and ~29 on the "
            "C28x. The CLA figure is measured -- 10 channels, all three "
            "stages in one CLA task, 70 ms at 200 MHz -- "
            "and it is why a MAC is not priced at the two cycles the "
            "multiply and add take: the int16-to-float conversions, the "
            "uint32 index arithmetic and a non-unrolled loop tail are the "
            "cost. The C28x figure is carried over from the measured "
            "feature-extraction time, not measured directly.\n\n"
            "Filled in: your figure re-anchors the per-MAC cost for every "
            "stage, and overrides the MFCC calibration for them.")
        tim.add(self._field("period", "Window period (ms):",
                            "" if cfg.get("period_ms") is None
                            else f"{cfg['period_ms']:g}"))
        self._edits["period"].setPlaceholderText("auto — window length from the dataset")
        self._edits["period"].setToolTip(
            "How often a new feature window arrives, i.e. the deadline the "
            "whole pipeline has to fit inside. Empty means the window's own "
            "length (slice_len / sr), which is the non-overlapping case: 65 "
            "frames at hop 256 and 16 kHz is 1056 ms. Shorten it if you slide "
            "the window faster than that.")
        note = QLabel(
            "The feature time changes nothing but itself. It used to scale "
            "the C28x stage estimates by how far it sat from this project's "
            "MFCC — which was wrong once the front end became yours: a 5 ms "
            "front end means a different algorithm, not a faster core. To "
            "move the stage estimates, measure the inference.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color:{C_FAINT}; font-size:10px;")
        tim.add(note)
        right.addWidget(tim)
        right.addStretch()
        lay.addStretch()

        self.dev_combo.currentIndexChanged.connect(self._device_picked)
        self._show_device()

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color:{C_ERR}; font-size:11px;")
        self.error_label.hide()
        outer.addWidget(self.error_label)

        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setCursor(Qt.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("Primary")
        apply_btn.setFixedHeight(32)
        apply_btn.setFixedWidth(110)
        apply_btn.setCursor(Qt.PointingHandCursor)
        apply_btn.clicked.connect(self._apply)
        row.addWidget(cancel)
        row.addWidget(apply_btn)
        outer.addLayout(row)

    # --------------------------------------------------------- device card
    def _device_picked(self):
        """Selecting a part fills the budget fields with what that part has.
        They stay editable afterwards -- the CLA figures in particular are a
        linker decision, not a datasheet constant."""
        dev = device_by_key(self.dev_combo.currentData())
        self._show_device()
        if dev is None:
            return
        b = device_budgets(dev)
        self._edits["flash"].setText(f"{b['flash_kb']:g}")
        self._edits["cpu"].setText(f"{b['cpu_kb']:g}")
        self._edits["cla"].setText(f"{b['cla_kb']:g}")

    def _show_device(self):
        dev = device_by_key(self.dev_combo.currentData())
        if dev is None:
            self.dev_spec.setText("Custom target — the budgets and the clock "
                                  "below are used exactly as typed.")
            self.dev_note.setText("")
            return
        self.dev_spec.setText(device_spec_line(dev))
        flag = "" if dev["pool_exact"] else (
            "⚠ The CLA RAM split for this family is the usual arrangement, "
            "not a measured one — check it against your linker command file. ")
        self.dev_note.setText(flag + dev["note"])
        self.dev_note.setStyleSheet(
            f"color:{C_MUTED if dev['pool_exact'] else C_WARN}; font-size:10px;")

    def _field(self, key, label, value):
        lay = QHBoxLayout()
        lay.setSpacing(8)
        lbl = QLabel(label)
        lbl.setObjectName("FieldLabel")
        lbl.setMinimumWidth(140)
        edit = QLineEdit(value)
        self._edits[key] = edit
        lay.addWidget(lbl)
        lay.addWidget(edit, 1)
        return lay

    @staticmethod
    def _opt_float(text):
        """An empty timing field means 'let the model fill it in', which is
        not the same as zero."""
        text = text.strip()
        return float(text) if text else None

    def _apply(self):
        try:
            sh = self._edits["shape"].text().split(",")
            h = int(sh[0].strip())
            w = int(sh[1].strip()) if len(sh) > 1 else h
            key = self.dev_combo.currentData()
            dev = device_by_key(key)
            self.result_cfg = {
                "input_h": h, "input_w": w,
                "classes": int(self._edits["classes"].text()),
                "layers": int(self._edits["layers"].text()),
                "pool_w": self.pool_check.isChecked(),
                "arena": self.arena_check.isChecked(),
                "flash_kb": float(self._edits["flash"].text()),
                "cpu_kb": float(self._edits["cpu"].text()),
                "cla_kb": float(self._edits["cla"].text()),
                "device": key,
                "cpu_mhz": float(dev["cpu_mhz"]) if dev
                           else float(self._cfg.get("cpu_mhz", 200.0)),
                "cla_mhz": float(dev["cla_mhz"]) if dev
                           else float(self._cfg.get("cla_mhz", 200.0)),
                "cla_prog_kb": float(dev["cla_prog_kb"]) if dev
                               else float(self._cfg.get("cla_prog_kb", 4.0)),
                # fe_ms is NOT here: the toolbar owns it
                "fe_ram_kb": self._opt_float(self._edits["fe_ram"].text()),
                "infer_ms": self._opt_float(self._edits["infer"].text()),
                "period_ms": self._opt_float(self._edits["period"].text()),
            }
            # a typed-over budget does not stop being that device, but the
            # run configuration has to say the numbers were not its own
            if dev is not None:
                b = device_budgets(dev)
                self.result_cfg["device_edited"] = any(
                    abs(self.result_cfg[k] - b[k]) > 0.01
                    for k in ("flash_kb", "cpu_kb", "cla_kb"))
            else:
                self.result_cfg["device_edited"] = False
        except ValueError as e:
            self.error_label.setText(f"Could not read a value: {e}")
            self.error_label.show()
            return
        for k in ("fe_ram_kb", "infer_ms", "period_ms"):
            v = self.result_cfg[k]
            if v is not None and v <= 0:
                self.error_label.setText(
                    "The timing fields must be positive, or empty to be "
                    "derived from the cycle model.")
                self.error_label.show()
                return
        oh, ow = ScalableDiagramCNN.out_shape(
            h, w, self.result_cfg["layers"], self.result_cfg["pool_w"])
        if oh <= 0 or ow <= 0:
            self.error_label.setText(
                f"Input {h}x{w} is too small for {self.result_cfg['layers']} "
                f"layers with this pooling.")
            self.error_label.show()
            return
        self.accept()


class EdgeAiGui(QMainWindow):
    """One screen: a toolbar, the device, and an action bar.

    The device diagram is the control surface -- clicking a stage moves it
    between the C28x and the CLA and the whole footprint re-derives on the
    spot, so there is no separate 'calculate' step and no checkbox column.
    The log and the results live in panels that stay shut until asked for."""

    def __init__(self):
        super().__init__()
        # No part number in the chrome. The tool searches across every C2000
        # with a CLA now, so branding the window with one of them was a
        # leftover from when there was only one -- and it went stale the
        # moment another device was picked. The selected part is shown where
        # it is actually information: the device diagram's own heading.
        self.setWindowTitle("ADOT  |  BLDC acoustic classification")
        self.setMinimumSize(1080, 720)
        self.resize(1420, 900)
        self.setStyleSheet(APP_QSS)

        self.dataset_path = None
        self.out_dir = None
        # everything the ceiling search needs that is not on the toolbar.
        # The memory numbers and the two clocks are the F28379D's (see
        # C2000_DEVICES); picking another part in the setup dialog replaces
        # all five.
        _d = device_budgets(device_by_key("F28379D"))
        self.cfg = {"input_h": 32, "input_w": 12, "classes": 3, "layers": 2,
                    "pool_w": False, "arena": True,
                    "flash_kb": _d["flash_kb"], "cpu_kb": _d["cpu_kb"],
                    "cla_kb": _d["cla_kb"], "cla_prog_kb": _d["cla_prog_kb"],
                    "device": "F28379D", "device_edited": False,
                    "cpu_mhz": _d["cpu_mhz"], "cla_mhz": _d["cla_mhz"],
                    # timing: None means "derive it from the cycle model"
                    # fe_ms / fe_ram_kb: the feature front end is an INPUT,
                    # not something this tool designs. None falls back to an
                    # estimate of this project's MFCC pipeline.
                    "fe_ms": None, "fe_ram_kb": None,
                    "infer_ms": None, "period_ms": None}
        # parsed feature_config.json of the chosen dataset, when one is found
        # next to it -- the MFCC cost model reads n_fft / hop / n_mels from it
        self.feature_cfg = None
        self.feat_plan = None
        self.feature_ram = 0
        self.timing = None
        # the CLA offload plan now lives here and is edited by clicking the
        # diagram, not by a column of checkboxes
        self.cla_plan = {"stages": [True, True], "dense": True}
        self.max_ch = None
        self._last_footprint = None
        self._t0 = None
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick_elapsed)

        self.init_ui()
        self.refresh()

    # ------------------------------------------------------------------ ui
    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        stage = QWidget()
        stage_lay = QVBoxLayout(stage)
        stage_lay.setContentsMargins(16, 14, 16, 10)
        stage_lay.setSpacing(0)

        # Device and Results are two full-width views of the same run rather
        # than a hero plus a squeezed side panel: the results deserve the
        # whole window, and the device is not useful while you read them.
        self.views = QStackedWidget()

        canvas_card = QFrame()
        canvas_card.setObjectName("Card")
        cc = QVBoxLayout(canvas_card)
        cc.setContentsMargins(16, 14, 16, 14)
        self.canvas = DeviceCanvas()
        self.canvas.stage_assigned.connect(self.assign_stage)
        cc.addWidget(self.canvas)
        self.views.addWidget(canvas_card)
        # The results page keeps its natural height and scrolls when the
        # window is shorter than it needs. Without this the panels fight
        # over the space and QVBoxLayout resolves it by squeezing whichever
        # one can shrink -- which is how the frame-budget table ended up
        # zero pixels tall on a 780 px window.
        results_scroll = QScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setFrameShape(QFrame.NoFrame)
        results_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        results_scroll.setWidget(self._build_results_view())
        self.views.addWidget(results_scroll)

        stage_lay.addWidget(self.views)
        root.addWidget(stage, 1)

        root.addWidget(self._build_action_bar())
        root.addWidget(self._build_log_drawer())

    # ----------------------------------------------------------- toolbar
    def _build_toolbar(self):
        """A proper settings bar: every control sits in a labelled field so
        the top of the window reads as 'this is what the run will use',
        instead of a row of anonymous widgets."""
        bar = QFrame()
        bar.setObjectName("Header")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 10, 18, 12)
        lay.setSpacing(10)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        brand.addSpacing(10)   # line the wordmark up with the field controls
        title = QLabel("ADOT")
        title.setObjectName("Title")
        self.brand_sub = QLabel("C2000  ·  C28x + CLA")
        self.brand_sub.setObjectName("Subtitle")
        brand.addWidget(title)
        brand.addWidget(self.brand_sub)
        lay.addLayout(brand)
        lay.addSpacing(6)
        lay.addWidget(self._vline())
        lay.addSpacing(6)

        self.dataset_btn = QPushButton("Choose dataset…")
        self.dataset_btn.setObjectName("Chip")
        self.dataset_btn.setCursor(Qt.PointingHandCursor)
        self.dataset_btn.setMinimumWidth(210)
        self.dataset_btn.setFixedHeight(32)
        self.dataset_btn.clicked.connect(self.browse_dataset)
        lay.addLayout(self._field("Dataset", self.dataset_btn))

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["FLOAT32", "INT16", "INT8"])
        self.quant_combo.setCurrentText("INT16")
        self.quant_combo.setFixedWidth(112)
        self.quant_combo.setFixedHeight(32)
        self.quant_combo.currentTextChanged.connect(self.refresh)
        lay.addLayout(self._field("Quantization", self.quant_combo))

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Macro-F1", "Confident Acc"])
        self.metric_combo.setFixedWidth(136)
        self.metric_combo.setFixedHeight(32)
        lay.addLayout(self._field("Target metric", self.metric_combo))

        self.target_edit = QLineEdit("95")
        self.target_edit.setFixedWidth(64)
        self.target_edit.setFixedHeight(32)
        self.target_edit.setAlignment(Qt.AlignCenter)
        lay.addLayout(self._field("Target %", self.target_edit))

        # The front end is not this tool's to design -- it is whatever fills
        # the input matrix, MFCC here but a filter bank, an FFT or plain RMS
        # somewhere else -- so its time belongs on the toolbar as an input,
        # next to the dataset, not buried in a dialog as a modelled quantity.
        # The estimator behind it only knows THIS project's mfcc_extract.c.
        self.fe_edit = QLineEdit()
        self.fe_edit.setFixedWidth(88)
        self.fe_edit.setFixedHeight(32)
        self.fe_edit.setAlignment(Qt.AlignCenter)
        self.fe_edit.setPlaceholderText("not set")
        self.fe_edit.setToolTip(
            "How long your feature front end takes to produce one input "
            "matrix, on the C28x, in milliseconds.\n\n"
            "This is a measurement, not something the tool can derive: the "
            "front end is yours. Put a CpuTimer around the call and type the "
            "number -- it is usually the largest term in the frame, so the "
            "CPU/CLA balance is only as real as this field.\n\n"
            "Left empty, no frame budget is shown at all -- not an "
            "estimate, nothing. The front end is code this tool does not "
            "have, and on this pipeline it is the bigger half of the frame, "
            "so guessing it would decide the answer rather than inform it.")
        self.fe_edit.editingFinished.connect(self._fe_time_typed)
        # An option, not a question: off until it is turned on, and while it
        # is off no frame budget is computed or drawn anywhere. The checkbox
        # doubles as the field's caption, so the control says what it is
        # without spending another column of the toolbar.
        self.timing_check = QCheckBox("TIMING ANALYSIS")
        self.timing_check.setToolTip(
            "Work out where the window's time goes.\n\n"
            "Needs one number the tool cannot derive: how long your feature "
            "extraction takes on the board. Measure it once (a CpuTimer or a "
            "GPIO around the call) and type it here.\n\n"
            "Left off, no timing is shown and none is assumed.")
        self.timing_check.setStyleSheet(
            f"QCheckBox {{ font-size:9px; font-weight:700; color:{C_MUTED};"
            f" letter-spacing:1.1px; spacing:5px; padding:0px; }}"
            f"QCheckBox::indicator {{ width:11px; height:11px; }}")
        self.timing_check.toggled.connect(self._timing_toggled)
        lay.addLayout(self._field(self.timing_check, self.fe_edit))
        self._timing_toggled(False)

        self.setup_btn = QPushButton("Model and memory…")
        self.setup_btn.setCursor(Qt.PointingHandCursor)
        self.setup_btn.setFixedHeight(32)
        self.setup_btn.setToolTip("Input shape, conv layers, pooling and the "
                                  "Flash / CPU RAM / CLA RAM budgets")
        self.setup_btn.clicked.connect(self.open_setup)
        lay.addLayout(self._field("Setup", self.setup_btn))

        lay.addStretch()
        self.status_pill = QLabel()
        self.status_pill.setAlignment(Qt.AlignCenter)
        self._set_status("no dataset", C_FAINT)
        lay.addLayout(self._field("Status", self.status_pill))
        return bar

    def _field(self, caption, widget):
        """caption above control -- the unit of the settings bar. `caption`
        is normally text, but a widget may be passed instead when the
        caption is itself the control that governs the field."""
        col = QVBoxLayout()
        col.setSpacing(3)
        if isinstance(caption, QWidget):
            col.addWidget(caption)
        else:
            cap = QLabel(caption.upper())
            cap.setObjectName("ToolCap")
            col.addWidget(cap)
        col.addWidget(widget)
        return col

    def _vline(self):
        line = QFrame()
        line.setObjectName("Sep")
        line.setFixedWidth(1)
        line.setFixedHeight(38)
        return line

    def _set_status(self, text, color):
        self.status_pill.setText(text.upper())
        self.status_pill.setStyleSheet(
            f"color:{color}; border:1px solid {color}; border-radius:8px;"
            f"padding:7px 16px; font-size:10px; font-weight:700;"
            f"letter-spacing:1.2px; background:transparent;")

    # ------------------------------------------------------ results view
    def _build_results_view(self):
        """The finished run, at full width: headline numbers, per-class
        performance derived from the confusion matrix, the matrix itself,
        what the model costs on the device, and what was written out."""
        page = QFrame()
        page.setObjectName("Card")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)

        head = QHBoxLayout()
        cap = QLabel("RUN RESULT")
        cap.setObjectName("CardTitle")
        head.addWidget(cap)
        self.r_context = QLabel("")
        self.r_context.setObjectName("Hint")
        head.addWidget(self.r_context)
        head.addStretch()
        back = QPushButton("‹  Back to device")
        back.setCursor(Qt.PointingHandCursor)
        back.clicked.connect(lambda: self.show_view(0))
        head.addWidget(back)
        outer.addLayout(head)

        self.r_channels = StatTile("channels", "--")
        self.r_f1 = StatTile("macro-F1", "--")
        self.r_acc = StatTile("val accuracy", "--")
        self.r_quant = StatTile("after quantization", "--")
        self.r_target = StatTile("target", "--")
        self.r_deploy = StatTile("deployable", "--")
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        for t in (self.r_channels, self.r_f1, self.r_acc, self.r_quant,
                  self.r_target, self.r_deploy):
            tiles.addWidget(t, 1)
        outer.addLayout(tiles)

        body = QHBoxLayout()
        body.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(self._section("Per-class performance",
                                     "derived from the confusion matrix below"))
        pc_holder = QFrame()
        pc_holder.setObjectName("Tile")
        self.pc_grid = QGridLayout(pc_holder)
        self.pc_grid.setContentsMargins(14, 12, 14, 12)
        self.pc_grid.setHorizontalSpacing(16)
        self.pc_grid.setVerticalSpacing(7)
        left.addWidget(pc_holder)

        left.addSpacing(6)
        left.addWidget(self._section("Confusion matrix", "row = true class"))
        cm_holder = QFrame()
        cm_holder.setObjectName("Tile")
        self.cm_grid = QGridLayout(cm_holder)
        self.cm_grid.setContentsMargins(14, 12, 14, 12)
        self.cm_grid.setHorizontalSpacing(16)
        self.cm_grid.setVerticalSpacing(7)
        left.addWidget(cm_holder)

        left.addSpacing(6)
        left.addWidget(self._section("Run configuration",
                                     "everything needed to reproduce this run"))
        rc_holder = QFrame()
        rc_holder.setObjectName("Tile")
        self.rc_grid = QGridLayout(rc_holder)
        self.rc_grid.setContentsMargins(14, 12, 14, 12)
        self.rc_grid.setHorizontalSpacing(16)
        self.rc_grid.setVerticalSpacing(7)
        left.addWidget(rc_holder)
        left.addStretch()
        body.addLayout(left, 3)

        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(self._section("Device footprint",
                                      "what the exported model occupies"))
        fp_holder = QFrame()
        fp_holder.setObjectName("Tile")
        self.fp_grid = QGridLayout(fp_holder)
        self.fp_grid.setContentsMargins(14, 12, 14, 12)
        self.fp_grid.setHorizontalSpacing(14)
        self.fp_grid.setVerticalSpacing(7)
        right.addWidget(fp_holder)

        right.addSpacing(6)
        right.addWidget(self._section(
            "Frame budget", "where the window's time goes"))
        tm_holder = QFrame()
        tm_holder.setObjectName("Tile")
        tm_outer = QVBoxLayout(tm_holder)
        tm_outer.setContentsMargins(14, 12, 14, 12)
        tm_outer.setSpacing(8)
        self.tm_grid = QGridLayout()
        self.tm_grid.setHorizontalSpacing(14)
        self.tm_grid.setVerticalSpacing(7)
        # the two numbers a write-up quotes, at the size the tiles use --
        # the table underneath is for where they came from
        self.tm_head = QLabel("--")
        self.tm_head.setObjectName("TileVal")
        self.tm_head.setWordWrap(True)
        tm_outer.addWidget(self.tm_head)
        self.tm_sub = QLabel("")
        self.tm_sub.setObjectName("TileNote")
        self.tm_sub.setWordWrap(True)
        tm_outer.addWidget(self.tm_sub)
        tm_outer.addSpacing(4)
        tm_outer.addLayout(self.tm_grid)
        self.tm_foot = QLabel("")
        self.tm_foot.setObjectName("Hint")
        self.tm_foot.setWordWrap(True)
        tm_outer.addWidget(self.tm_foot)
        self.tm_advice = QLabel("--")
        self.tm_advice.setWordWrap(True)
        self.tm_advice.setStyleSheet("font-size:11px;")
        tm_outer.addWidget(self.tm_advice)
        right.addWidget(tm_holder)

        right.addSpacing(6)
        right.addWidget(self._section("Export", "files written by this run"))
        ex_holder = QFrame()
        ex_holder.setObjectName("Tile")
        ex = QVBoxLayout(ex_holder)
        ex.setContentsMargins(14, 12, 14, 12)
        ex.setSpacing(6)
        self.export_label = QLabel("--")
        self.export_label.setObjectName("Hint")
        self.export_label.setWordWrap(True)
        self.export_files = QLabel("--")
        self.export_files.setWordWrap(True)
        self.export_files.setStyleSheet("font-size:11px;")
        ex.addWidget(self.export_label)
        ex.addWidget(self.export_files)
        open_btn = QPushButton("Open export folder")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(self._open_export_dir)
        ex.addWidget(open_btn)
        right.addWidget(ex_holder)
        right.addStretch()
        body.addLayout(right, 2)

        outer.addLayout(body, 1)
        return page

    def _section(self, title, note=""):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        cap = QLabel(title.upper())
        cap.setObjectName("CardTitle")
        lay.addWidget(cap)
        if note:
            n = QLabel(note)
            n.setObjectName("Hint")
            lay.addWidget(n)
        lay.addStretch()
        return w

    def show_view(self, index):
        self.views.setCurrentIndex(index)
        self.device_btn.setChecked(index == 0)
        self.results_btn.setChecked(index == 1)

    def _build_action_bar(self):
        bar = QFrame()
        bar.setObjectName("ActionBar")
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(12)

        # view switch: Device / Results, as a two-button segmented control
        switch = QHBoxLayout()
        switch.setSpacing(0)
        self.device_btn = QPushButton("Device")
        self.device_btn.setObjectName("SegLeft")
        self.results_btn = QPushButton("Results")
        self.results_btn.setObjectName("SegRight")
        for b, idx in ((self.device_btn, 0), (self.results_btn, 1)):
            b.setCheckable(True)
            b.setFixedHeight(32)
            b.setMinimumWidth(92)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _c, i=idx: self.show_view(i))
            switch.addWidget(b)
        self.device_btn.setChecked(True)
        # nothing to show until a run finishes
        self.results_btn.setEnabled(False)
        self.results_btn.setToolTip("Available once a search has finished")
        row.addLayout(switch)

        self.phase_label = QLabel("Ready when you are")
        self.phase_label.setObjectName("Phase")
        row.addWidget(self.phase_label)

        self.ticker = QLabel("")
        self.ticker.setObjectName("Hint")
        row.addWidget(self.ticker, 1)

        self.log_btn = QPushButton("Log ▾")
        self.log_btn.setCursor(Qt.PointingHandCursor)
        self.log_btn.setFixedHeight(32)
        self.log_btn.setToolTip("Show or hide the full training log")
        self.log_btn.clicked.connect(self.toggle_log)
        row.addWidget(self.log_btn)

        self.train_btn = QPushButton("START SEARCH")
        self.train_btn.setObjectName("Primary")
        self.train_btn.setFixedHeight(40)
        self.train_btn.setMinimumWidth(190)
        self.train_btn.setEnabled(False)
        self.train_btn.setCursor(Qt.PointingHandCursor)
        self.train_btn.clicked.connect(self.start_training)
        row.addWidget(self.train_btn)
        lay.addLayout(row)

        self.run_row = QWidget()
        rl = QHBoxLayout(self.run_row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(12)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("")
        self.progress.setFixedHeight(10)
        self.progress.setTextVisible(False)
        rl.addWidget(self.progress, 1)
        self.run_detail = QLabel("")
        self.run_detail.setObjectName("RunDetail")
        rl.addWidget(self.run_detail)
        rl.addSpacing(6)
        self.elapsed_label = QLabel("00:00")
        self.elapsed_label.setObjectName("RunDetail")
        self.elapsed_label.setMinimumWidth(48)
        self.elapsed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.elapsed_label.setToolTip("Time since the search started")
        rl.addWidget(self.elapsed_label)
        self.run_row.hide()
        lay.addWidget(self.run_row)
        return bar

    # -------------------------------------------------------- log drawer
    def _build_log_drawer(self):
        self.log_drawer = QFrame()
        self.log_drawer.setObjectName("LogDrawer")
        lay = QVBoxLayout(self.log_drawer)
        lay.setContentsMargins(16, 0, 16, 12)
        lay.setSpacing(6)
        self.log_output = QTextEdit()
        self.log_output.setObjectName("Log")
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(190)
        lay.addWidget(self.log_output)
        self.log_drawer.hide()
        return self.log_drawer

    def toggle_log(self):
        show = not self.log_drawer.isVisible()
        self.log_drawer.setVisible(show)
        self.log_btn.setText("Log ▴" if show else "Log ▾")

    # -------------------------------------------------------- ui helpers
    def log(self, html):
        """Appends to the drawer and mirrors the line into the one-line
        ticker, so the drawer can stay shut without losing the thread."""
        self.log_output.append(html)
        doc = QTextDocument()
        doc.setHtml(html)
        text = " ".join(doc.toPlainText().split())
        if text:
            self.ticker.setText(text[:150])

    def _open_export_dir(self):
        if self.out_dir and os.path.isdir(self.out_dir):
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(os.path.abspath(self.out_dir)))
        else:
            self.log(f"<span style='color:{C_WARN};'>No export folder yet.</span>")

    # ------------------------------------------------------------ dataset
    def browse_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select training dataset", "", "NumPy archive (*.npz);;All files (*)")
        if not path:
            return
        try:
            d = np.load(path, allow_pickle=True)
            n, h, w = d["X_train"].shape
            classes = [str(c) for c in d["classes"]]
            self.dataset_path = path
            self.cfg["input_h"], self.cfg["input_w"] = h, w
            self.cfg["classes"] = len(classes)
            self.load_feature_cfg(path)
            self.dataset_btn.setText(os.path.basename(path))
            self.dataset_btn.setToolTip(
                f"{path}\n\n{n} train | {len(d['y_val'])} val | {h}x{w} | "
                f"{len(classes)} classes: {', '.join(classes)}")
            self.log(
                f"<b>Dataset:</b> {os.path.basename(path)}<br>"
                f"train {n} | val {len(d['y_val'])} | input {h}x{w} | classes {classes}<br>"
                + ("No X_test &mdash; final-model set (accuracy estimate comes from the folds)"
                   if "X_test" not in d.files else
                   "X_test present but <b>will not be used</b>; selection uses X_val"))
            self.refresh()
        except Exception as e:
            self._set_status("dataset error", C_ERR)
            self.log(f"<span style='color:{C_ERR};'>Could not read file: {e}</span>")

    def _timing_toggled(self, on):
        """Turning the option off does not just grey the field out -- it
        drops the value, so nothing downstream can keep using a number the
        screen is no longer showing."""
        self.fe_edit.setEnabled(on)
        if not on:
            self.fe_edit.clear()
            if self.cfg.get("fe_ms") is not None:
                self.cfg["fe_ms"] = None
                self.refresh()
        else:
            self.fe_edit.setFocus()

    def _fe_time_typed(self):
        """The toolbar's feature-extraction time. Empty means 'fall back to
        the estimator'; anything else is the user's own measurement and wins
        over it everywhere."""
        text = self.fe_edit.text().strip().replace(",", ".")
        if not text:
            new = None
        else:
            try:
                new = float(text)
                if new <= 0:
                    raise ValueError
            except ValueError:
                self.log(f"<span style='color:{C_ERR};'>Feature time must be "
                         f"a positive number of milliseconds, or empty for "
                         f"the estimate.</span>")
                self.fe_edit.setText("" if self.cfg.get("fe_ms") is None
                                     else f"{self.cfg['fe_ms']:g}")
                return
        if new == self.cfg.get("fe_ms"):
            return
        self.cfg["fe_ms"] = new
        self.refresh()

    def load_feature_cfg(self, dataset_path):
        """The MFCC cost model wants n_fft / hop / n_mels, which live in the
        feature_config.json prepare_features.py writes next to folds/. Without
        it the model still runs, on the pipeline that script ships with, and
        says so wherever it shows a number."""
        self.feature_cfg = None
        try:
            p = find_feature_config(dataset_path)
            if p is None:
                self.log(
                    f"<span style='color:{C_FAINT};'>No feature_config.json "
                    f"next to the dataset &mdash; the MFCC timing estimate "
                    f"assumes {FEAT_ASSUMED['n_fft']}-point FFT, hop "
                    f"{FEAT_ASSUMED['hop']}, {FEAT_ASSUMED['n_mels']} mel "
                    f"bands. Type a measured time in Setup to replace it."
                    f"</span>")
                return
            with open(p) as f:
                self.feature_cfg = json.load(f)
            plan = feature_plan(self.cfg["input_h"], self.cfg["input_w"],
                                self.feature_cfg)
            self.log(f"<b>Features:</b> {os.path.basename(p)} &mdash; "
                     f"{plan['n_fft']}-point FFT, hop {plan['hop']}, "
                     f"{plan['n_mels']} mel bands, {plan['n_frames']} frames "
                     f"= {plan['window_ms']:.0f} ms window")
        except Exception as e:
            self.log(f"<span style='color:{C_WARN};'>feature_config.json "
                     f"could not be read ({e}); the MFCC estimate falls back "
                     f"to assumed parameters.</span>")

    def open_setup(self):
        dlg = SetupDialog(self.cfg, self)
        if dlg.exec_() == QDialog.Accepted:
            self.cfg.update(dlg.result_cfg)
            dev = device_by_key(self.cfg.get("device"))
            # Only the diagram's heading follows the selection; the window
            # title and the wordmark stay device-neutral.
            self.canvas.device = dev["name"] if dev else "Custom target"
            self.refresh()

    DENSE_ID = 99      # stage id of the FC stage on the canvas

    def assign_stage(self, sid, to_cla):
        """The diagram is the CLA plan. Stage ids 0..n-1 are the fused
        conv+pool stages and DENSE_ID is the FC stage; `to_cla` is the
        domain the user picked, not a flip, so re-picking the current domain
        is a no-op rather than a surprise."""
        if sid == self.DENSE_ID:
            if self.cla_plan["dense"] == to_cla:
                return
            self.cla_plan["dense"] = to_cla
        elif 0 <= sid < len(self.cla_plan["stages"]):
            if self.cla_plan["stages"][sid] == to_cla:
                return
            self.cla_plan["stages"][sid] = to_cla
        else:
            return
        self.refresh()

    # ------------------------------------------------------------- memory
    def dtype_size(self, mode):
        """Bytes per element ON THE C2000, which is WORD-addressed: the
        smallest addressable unit is 16 bits, so int8_t occupies a full word
        just like int16_t (TI C28x: sizeof(char)==1 but CHAR_BIT==16).
        INT8 therefore buys NO memory on this target -- only INT16 vs FLOAT32
        differ. Verified against Debug/dense_layer_cla.map: workspace
        (MAX_ELEMENTS=2532 int16) occupies 0x9E4 = 2532 address units."""
        return {"FLOAT32": 4, "INT16": 2, "INT8": 2}[mode]

    def get_cla_stage(self):
        """The offload plan in the shape cla_pipeline_config.h wants: one flag
        per FUSED conv+pool stage plus the dense flag. MFCC is not in here: it
        always runs on the CPU."""
        return {"stages": list(self.cla_plan["stages"]),
                "dense": self.cla_plan["dense"]}

    def calculate_metrics(self, ch, layers, arena, mode, ih, iw, ncls, pool_w,
                          cla_stage, feature_ram=0):
        """Models the real firmware's memory, not a generic one:

        * Stages are FUSED (conv+relu+pool in one pass), so a stage's
          unpooled conv output is never stored -- only the pooled outputs
          live in workspace[], exactly like definitions.h::MAX_ELEMENTS.
          That is why the `arena` flag no longer changes the result: the
          fused design already is the arena.
        * Weights always sit in flash; putting a stage on the CLA adds a RAM
          *copy* of that stage's weights (the DATA_SECTION pragmas in
          weights.h/main.c), so moving a stage to the CPU frees CLA RAM.
        * workspace[] and fResult[] need to be CLA-visible as soon as ANY
          stage runs there.
        * `feature_ram` is the MFCC front end's own working set (the raw
          slice, the log-mel frames, the FFT scratch). It is not part of the
          model, but it is the largest thing on the C28x side -- 85 KB of the
          204 KB on this board, measured as 0xa6c0 words in RAMGS_TOTAL_CNN --
          and leaving it out is what made the CPU RAM gauge read zero in
          every configuration. It is charged to the CPU because the C28x is
          what runs mfcc_extract().
        """
        ds = self.dtype_size(mode)
        stages = cla_stage["stages"]
        dense_on_cla = cla_stage["dense"]

        flash, in_ch, h, w = 0, 1, ih, iw
        cpu_extra = int(feature_ram)
        stage_w_bytes, pool_elems = [], []
        for _ in range(layers):
            wb = ((3 * 3 * in_ch * ch) + ch) * ds      # weights + bias
            flash += wb
            h, w = h - 2, w - 2                        # conv (valid, 3x3)
            h, w = h // 2, (w // 2 if pool_w else w)   # pool, fused into conv
            stage_w_bytes.append(wb)
            pool_elems.append(ch * h * w)
            in_ch = ch

        fc_in = ch * h * w
        fc_w_bytes = fc_in * ncls * ds
        flash += fc_w_bytes + ncls * ds                # fc weights + bias

        input_bytes = ih * iw * ds                     # input_image[]
        workspace_bytes = sum(pool_elems) * ds         # pool1 + pool2 outputs
        out_bytes = ncls * 4                           # fResult[] floats

        cla_ram, cpu_ram = 0, cpu_extra
        for i in range(layers):
            if i < len(stages) and stages[i]:
                cla_ram += stage_w_bytes[i]            # RAM copy of the weights
        # input_image is written by the CPU (MFCC) and read by stage 1
        if dense_on_cla:
            # FC weights AND fc_bias: the CLA cannot read flash, so a CLA
            # dense stage needs both copied into CLA RAM (weights.h pragma)
            cla_ram += fc_w_bytes + ncls * ds
        # Buffers (input_image, workspace, fResult) live in CLA data RAM in
        # EVERY configuration, even all-CPU. Measured with the real TI build of
        # all 8 combinations: the C28x RAM region is already ~95% full of MFCC
        # buffers, so moving workspace there fails to link (.ebss #10099),
        # while CLA data RAM never gets tight. Only weights are placement-
        # dependent (a CPU stage leaves them in flash -> no RAM at all).
        cla_ram += input_bytes + workspace_bytes + out_bytes

        return flash, cpu_ram, cla_ram, fc_in

    # -------------------------------------------------------------- timing
    def compute_timing(self, ch):
        """The frame, on the clock of the part that is selected -- or None.

        None when no feature-extraction time has been given, and that is the
        point: the front end is the user's code and the biggest term in the
        frame, so a modelled stand-in for it would decide the answer. The
        estimator for this project's own MFCC still exists and still anchors
        the cycle costs, but nothing shows a frame budget until someone has
        measured their own front end and said so."""
        c = self.cfg
        if c.get("fe_ms") is None:
            return None
        plan = self.feat_plan or feature_plan(self.input_h, self.input_w,
                                              self.feature_cfg)
        period = c.get("period_ms")
        if period is None:
            period = plan["window_ms"]
        return pipeline_timing(
            plan, ch, self.layers_val, self.input_h, self.input_w,
            self.classes_val, self.pool_w, self.cla_stage,
            cpu_mhz=c.get("cpu_mhz", 200.0), cla_mhz=c.get("cla_mhz", 200.0),
            measured_fe_ms=c.get("fe_ms"),
            period_ms=period, measured_infer_ms=c.get("infer_ms"))

    # --------------------------------------------------- live recalculation
    def refresh(self):
        """Re-derives the ceiling and redraws the device. Called whenever
        anything that affects the footprint changes -- picking a dataset,
        switching quantization, applying setup, or clicking a stage -- so the
        old 'calculate hardware ceiling' button is gone: the diagram is never
        out of date with the controls."""
        c = self.cfg
        # the diagram's heading is the one place a part number belongs, and
        # it is derived here rather than pushed from the dialog, so it cannot
        # drift from the configuration the rest of this method uses
        _dev = device_by_key(c.get("device"))
        self.canvas.device = _dev["name"] if _dev else "Custom target"
        self.quant_mode = self.quant_combo.currentText()
        self.input_h, self.input_w = c["input_h"], c["input_w"]
        self.classes_val, self.layers_val = c["classes"], c["layers"]
        self.pool_w, self.use_arena = c["pool_w"], c["arena"]
        self.cla_stage = self.get_cla_stage()
        self.update_firmware_warning()

        # Nothing is drawn until there is a dataset. The defaults would
        # otherwise produce a complete, plausible-looking device -- a channel
        # ceiling, tensor shapes, filled memory gauges -- none of which came
        # from any data. That reads as a result rather than as a placeholder,
        # which is worse than showing nothing.
        if not self.dataset_path:
            self.canvas.clear(
                "No dataset loaded\n\n"
                "Choose one from the toolbar above. The input shape and class "
                "count are read from the file,\nand the channel ceiling is "
                "derived from them and the memory budget.")
            self._set_status("no dataset", C_FAINT)
            self.train_btn.setEnabled(False)
            self.max_ch = None
            self._last_footprint = None
            return

        oh, ow = ScalableDiagramCNN.out_shape(
            self.input_h, self.input_w, self.layers_val, self.pool_w)
        if oh <= 0 or ow <= 0:
            self.canvas.clear(
                f"{self.input_h}x{self.input_w} is too small for "
                f"{self.layers_val} layers. Open Setup to fix it.")
            self._set_status("invalid shape", C_ERR)
            self.train_btn.setEnabled(False)
            return

        # The MFCC front end is a fixed cost on the C28x, independent of the
        # channel count, so it is worked out once and then carried into every
        # candidate's CPU RAM.
        self.feat_plan = feature_plan(self.input_h, self.input_w,
                                      self.feature_cfg)
        self.feature_ram = (int(c["fe_ram_kb"] * 1024)
                            if c.get("fe_ram_kb") is not None
                            else mfcc_cost(self.feat_plan)["ram_bytes"])

        f_lim, c_lim, cla_lim = (c["flash_kb"] * 1024, c["cpu_kb"] * 1024,
                                 c["cla_kb"] * 1024)
        if self.feature_ram > c_lim:
            # Worth saying out loud rather than reporting a 1-channel ceiling:
            # on a part this small the network was never the problem.
            self.canvas.clear(
                f"{self.canvas.device} cannot hold the feature front end\n\n"
                f"The feature front end's buffers alone are "
                f"{self.feature_ram / 1024:.0f} KB (raw slice + log-mel frames "
                f"+ FFT scratch) against a {c['cpu_kb']:.0f} KB CPU RAM "
                f"budget.\nPick a larger part, shorten the window, or raise "
                f"the budget in Setup.")
            self._set_status("no room for MFCC", C_ERR)
            self.train_btn.setEnabled(False)
            self.max_ch = None
            self._last_footprint = None
            return

        self.max_ch = 1
        while self.max_ch < 128:
            f, sc, scla, _ = self.calculate_metrics(
                self.max_ch + 1, self.layers_val, self.use_arena, self.quant_mode,
                self.input_h, self.input_w, self.classes_val, self.pool_w,
                self.cla_stage, self.feature_ram)
            if f > f_lim or sc > c_lim or scla > cla_lim:
                break
            self.max_ch += 1

        # The ceiling above is a MEMORY ceiling, and saying "largest model
        # that fits" while the same screen shows the frame overrunning its
        # budget is the tool contradicting itself. The time ceiling is
        # worked out too and reported next to it -- but it does not shrink
        # the search, because it comes from an estimate and the memory
        # ceiling comes from arithmetic on the linker's own numbers.
        self.max_ch_time = None
        deadline = c.get("period_ms") or self.feat_plan["window_ms"]
        if deadline and c.get("fe_ms") is not None:
            for cand in range(self.max_ch, 0, -1):
                if self.compute_timing(cand)["period_ms"] <= deadline:
                    self.max_ch_time = cand
                    break

        self.draw_device(self.max_ch, "ceiling")
        self.train_btn.setEnabled(True)
        if self.status_pill.text().lower() != "training":
            self._set_status("ready", C_OK)

    # ------------------------------------------------------------- device
    def draw_device(self, ch, mode):
        """Builds the block/stage description the canvas paints. `mode` is
        'ceiling' (largest model that fits) or 'selected' (what training
        picked), which only changes the note next to the CPU block."""
        c = self.cfg
        ds = self.dtype_size(self.quant_mode)
        stages_on_cla = self.cla_stage["stages"]
        dense_cla = self.cla_stage["dense"]
        f, sc, scla, fc_in = self.calculate_metrics(
            ch, self.layers_val, self.use_arena, self.quant_mode,
            self.input_h, self.input_w, self.classes_val, self.pool_w,
            self.cla_stage, self.feature_ram)
        self._last_footprint = (f, sc, scla, fc_in)
        self.timing = self.compute_timing(ch)
        # None when no feature time has been given: every per-stage time
        # below then stays absent rather than being filled with a model
        t_stages = self.timing["stages"] if self.timing else None

        cpu, cla = [], []

        def place(card, on_cla):
            (cla if on_cla else cpu).append(card)

        # The input matrix always enters on the CPU -- the C28x is what fills
        # input_image[] from the MFCC -- so it is not movable. Where those
        # words are STORED is a separate question: the buffer sits in CLA RAM
        # (weights.h's conditional DATA_SECTION), which the tooltip reports.
        place({"order": 1, "caption": "INPUT", "sub": "MFCC feature matrix",
               "shape": f"{self.input_h} × {self.input_w}",
               "flash_kb": 0.0,
               # The front end's working buffers, and ONLY those. The
               # feature matrix itself (input_image[]) is deliberately not
               # added: calculate_metrics charges it to CLA data RAM,
               # because that is where weights.h's DATA_SECTION puts it, and
               # counting it here as well would have this block's total and
               # the CLA gauge both claim the same 1.5 KB.
               "ram_kb": self.feature_ram / 1024,
               "ms": self.timing["fe_ms"] if self.timing else None,
               "ms_note": "measured",
               "id": -1, "fixed_note": "always on the C28x"}, False)

        h, w, in_ch = self.input_h, self.input_w, 1
        for i in range(self.layers_val):
            on_cla = stages_on_cla[i] if i < len(stages_on_cla) else False
            wt = ((3 * 3 * in_ch * ch) + ch) * ds
            h, w = h - 2, w - 2                        # conv
            h, w = h // 2, (w // 2 if self.pool_w else w)   # fused pool
            act = ch * h * w * ds
            st = t_stages[i] if t_stages else None
            place({"order": i + 2, "caption": f"S{i + 1}",
                   "sub": f"conv{i + 1} + ReLU + pool{i + 1}",
                   "shape": f"{ch} × {h} × {w}",
                   "flash_kb": wt / 1024,
                   "ram_kb": (act + (wt if on_cla else 0)) / 1024,
                   "ms": st["ms"] if st else None,
                   "alt_ms": st["alt_ms"] if st else None,
                   "macs": st["macs"] if st else None,
                   "id": i}, on_cla)
            in_ch = ch

        fc_w = fc_in * self.classes_val * ds
        st = t_stages[-1] if t_stages else None
        place({"order": self.layers_val + 2, "caption": f"S{self.layers_val + 1}",
               "sub": "dense / fully connected",
               "shape": f"{self.classes_val} × {fc_in}",
               "flash_kb": (fc_w + self.classes_val * ds) / 1024,
               "ram_kb": ((fc_w + self.classes_val * ds) if dense_cla else 0) / 1024,
               "ms": st["ms"] if st else None,
               "alt_ms": st["alt_ms"] if st else None,
               "macs": st["macs"] if st else None,
               "id": self.DENSE_ID}, dense_cla)
        place({"order": self.layers_val + 3, "caption": "OUTPUT",
               "sub": "class scores", "shape": f"{self.classes_val} × 1",
               "flash_kb": 0.0, "ram_kb": self.classes_val * 4 / 1024,
               "id": -1, "fixed_note": "follows the dense stage"}, dense_cla)

        def totals(cards):
            return (sum(x["ram_kb"] for x in cards),
                    sum(x["flash_kb"] for x in cards))

        cpu_ram_kb, cpu_flash_kb = totals(cpu)
        cla_ram_kb, cla_flash_kb = totals(cla)
        which = ("largest model that fits in memory" if mode == "ceiling"
                 else "trained model")
        limit = getattr(self, "max_ch_time", None)
        if mode == "ceiling" and limit is not None and limit < ch:
            which += f" · the frame budget allows {limit}"
        t = self.timing
        cpu_work = cla_work = ""
        if t:
            cpu_ms = t["fe_ms"] + sum(s["ms"] for s in t["stages"]
                                      if not s["on_cla"])
            cla_ms = sum(s["ms"] for s in t["stages"] if s["on_cla"])
            cpu_work = f" · {cpu_ms:.2f} ms of work"
            cla_work = f" · {cla_ms:.2f} ms of work"
        blocks = [
            {"title": "C28x CPU", "accent": C_CPU, "stages": cpu, "is_cla": False,
             "note": f"{ch} channels · {which} · {len(cpu)} stage(s) here · "
                     f"{cpu_ram_kb:.2f} KB RAM, {cpu_flash_kb:.2f} KB flash"
                     + cpu_work,
             "empty": "every stage is assigned to the CLA",
             "gauges": [("CPU RAM", sc / 1024, c["cpu_kb"])]},
            {"title": "CLA", "accent": C_CLA, "stages": cla, "is_cla": True,
             "note": f"{len(cla)} stage(s) here · {cla_ram_kb:.2f} KB RAM, "
                     f"{cla_flash_kb:.2f} KB flash" + cla_work,
             "empty": "nothing offloaded — pick CLA on a stage card",
             "gauges": [("CLA data RAM", scla / 1024, c["cla_kb"])]},
        ]
        self.canvas.set_device(
            blocks, flash=("Flash", f / 1024, c["flash_kb"]),
            hint=self._current_hint(), timing=t,
            advice=timing_advice(t) if t else None)

    # ----------------------------------------------------- firmware notice
    @staticmethod
    def firmware_incompatibilities(quant_mode, layers, pool_w):
        """Reasons a model with these settings cannot be dropped into
        CNN_Inference_Edge-AI_with_CLA -- the same rule
        TrainingThread.write_embedded_files() enforces at export time.
        `layers` may be an int or the raw text of the layers field."""
        why = []
        if quant_mode != "INT16":
            why.append(f"quantization {quant_mode} (firmware runs INT16 only)")
        try:
            n = int(str(layers).strip())
            if n != 2:
                why.append(f"{n} conv layer(s) (firmware has exactly 2)")
        except ValueError:
            pass
        if pool_w:
            why.append("2D pooling (firmware pools height only, 2x1)")
        return why

    def target_mismatch(self):
        """The exported firmware files are CNN_Inference_Edge-AI_with_CLA's,
        and that project is an F28379D one -- its linker script, its 4 KB of
        CLA program RAM, its F2837xD driverlib. Selecting another part is a
        legitimate thing to explore here (the search, the budgets and the
        timing all follow it), but the headers it writes still have to be
        ported. Returns that sentence, or None when the target is the board
        the firmware is for."""
        key = self.cfg.get("device")
        if key in ("F28379D", None):
            return None
        dev = device_by_key(key)
        name = dev["name"] if dev else "a custom target"
        return (f"the firmware project is an F28379D one, so its headers "
                f"have to be ported to {name} (linker script, CLA program "
                f"RAM, driverlib)")

    def update_firmware_warning(self):
        """No banner any more: the toolbar's quantization control turns amber
        and explains itself, which is where the usual cause sits."""
        why = self.firmware_incompatibilities(
            self.quant_mode, self.layers_val, self.pool_w)
        if why:
            self.quant_combo.setStyleSheet(
                f"QComboBox {{ border: 1px solid {C_WARN}; color: {C_WARN}; }}")
            self.quant_combo.setToolTip(
                "Not deployable to the board: " + "; ".join(why) + ".\n"
                "Training still works, but no firmware weight files "
                "(definitions.h / weights.h / main.c) will be exported.")
        else:
            self.quant_combo.setStyleSheet("")
            self.quant_combo.setToolTip(
                "INT16 is the only mode the firmware kernels run; this "
                "configuration exports firmware files.")

    # ------------------------------------------------------- run feedback
    def _tick_elapsed(self):
        if self._t0 is None:
            return
        s = int(datetime.datetime.now().timestamp() - self._t0)
        self.elapsed_label.setText(f"{s // 60:02d}:{s % 60:02d}")

    def on_progress(self, st):
        """Fed by TrainingThread.progress_signal -- a read-out only, the
        search never waits on it."""
        self.phase_label.setText(st.get("phase") or "Running")
        bits = []
        if st.get("channels"):
            bits.append(f"{st['channels']} ch")
        if st.get("seed"):
            bits.append(f"seed {st['seed']}/{st.get('seeds', N_SEEDS)}")
        for key, lbl in (("acc", "acc"), ("f1", "F1"), ("best", "best")):
            if st.get(key) is not None:
                bits.append(f"{lbl} {st[key]:.1f}%")
        self.run_detail.setText("   ·   ".join(bits))

        ep, eps = st.get("epoch") or 0, st.get("epochs") or 0
        if eps:
            self.progress.setRange(0, eps)
            self.progress.setValue(ep)
        elif st.get("done"):
            self.progress.setRange(0, 1)
            self.progress.setValue(1)

    def on_summary(self, s):
        self.out_dir = s.get("out_dir")
        met = s.get("target_met")
        why = self.firmware_incompatibilities(s["quant"], s["layers"], self.pool_w)

        self.r_context.setText(
            f"{os.path.basename(self.dataset_path or '--')}   ·   "
            f"{s['quant']}   ·   {s['layers']} conv layer(s)   ·   "
            f"{self.input_h}x{self.input_w} input   ·   "
            f"{self.elapsed_label.text()} elapsed")

        self.r_channels.set_value(s["channels"], note="conv width")
        self.r_f1.set_value(f"{s['f1']:.1f}%", note="macro average")
        self.r_acc.set_value(f"{s['acc']:.1f}%", note="validation split")
        if s.get("q_f1") is None:
            self.r_quant.set_value("n/a", note="FLOAT32 export", color=C_MUTED)
        else:
            d = s["q_f1"] - s["f1"]
            self.r_quant.set_value(f"{s['q_f1']:.1f}%",
                                   note=f"macro-F1 {d:+.1f} pts",
                                   color=C_OK if abs(d) < 1.0 else C_WARN)
        self.r_target.set_value(
            "met" if met else "missed",
            note=f"{s['metric']} ≥ {s['target']:.0f}%",
            color=C_OK if met else C_ERR)
        port = self.target_mismatch()
        self.r_deploy.set_value(
            "yes" if not why else "no",
            note=("; ".join(why) if why else
                  "firmware files written" if not port else
                  "firmware files written — " + port),
            color=C_OK if not (why or port) else C_WARN)

        # redraw on the channel count that was actually selected, so the
        # footprint and the frame budget below describe the exported model
        # and not the ceiling the search started from
        if s.get("channels"):
            self.draw_device(int(s["channels"]), "selected")

        self._fill_per_class(s["classes"], s["cm"], s.get("conf"))
        self._fill_confusion(s["classes"], s["cm"])
        self._fill_footprint()
        self._fill_timing()
        self._fill_run_config(s)

        self.export_label.setText(s.get("out_dir", "--"))
        sfx = s["quant"].lower()
        files = [f"model_weights_{sfx}.h", f"test_samples_{sfx}.h",
                 "cla_pipeline_config.h",
                 f"edge_cnn_{sfx}_{s['channels']}ch.pth", "info.txt",
                 "embedded/ (cla_pipeline_config.h, feature_tables.h"
                 + (", definitions.h, weights.h, main.c)" if not why else ")")]
        self.export_files.setText("<br>".join(f"&middot; {f}" for f in files))

        self.results_btn.setEnabled(True)
        self.results_btn.setToolTip("The finished run in full")
        self.show_view(1)

    def _fill_per_class(self, classes, cm, conf=None):
        """Precision / recall / F1 / support straight out of the confusion
        matrix. Presentation only -- the same numbers the matrix already
        carries, just read the way a reviewer wants them."""
        self._clear_grid(self.pc_grid)
        n = len(classes)
        head = ("class", "precision", "recall", "F1", "support")
        for j, h in enumerate(head):
            self.pc_grid.addWidget(
                self._cell(h, "MatrixHead",
                           align=Qt.AlignLeft | Qt.AlignVCenter if j == 0
                           else Qt.AlignRight | Qt.AlignVCenter), 0, j)
        self.pc_grid.setColumnStretch(len(head), 1)

        for i in range(n):
            tp = int(cm[i][i])
            row_sum = sum(int(cm[i][j]) for j in range(n))      # true count
            col_sum = sum(int(cm[j][i]) for j in range(n))      # predicted
            prec = 100.0 * tp / col_sum if col_sum else 0.0
            rec = 100.0 * tp / row_sum if row_sum else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            self.pc_grid.addWidget(
                self._cell(str(classes[i])[:16], "MatrixCell",
                           align=Qt.AlignLeft | Qt.AlignVCenter), i + 1, 0)
            for j, v in enumerate((prec, rec, f1)):
                col = C_TEXT if v >= 90 else (C_WARN if v >= 75 else C_ERR)
                self.pc_grid.addWidget(
                    self._cell(f"{v:.1f}%", "MatrixCell", col,
                               align=Qt.AlignRight | Qt.AlignVCenter), i + 1, j + 1)
            self.pc_grid.addWidget(
                self._cell(str(row_sum), "MatrixCell", C_MUTED,
                           align=Qt.AlignRight | Qt.AlignVCenter), i + 1, 4)

    def _fill_run_config(self, s):
        """What produced these numbers, spelled out -- the search settings a
        write-up has to quote and that are otherwise spread across the
        toolbar, the setup dialog and the diagram."""
        self._clear_grid(self.rc_grid)
        plan = [n for n, on in (("S1", self.cla_plan["stages"][0]),
                                ("S2", self.cla_plan["stages"][1]),
                                ("S3", self.cla_plan["dense"])) if on]
        dev = device_by_key(self.cfg.get("device"))
        target = dev["name"] if dev else "custom target"
        if dev is not None and self.cfg.get("device_edited"):
            target += "  (budgets edited by hand)"
        t = self.timing
        rows = [
            ("Dataset", os.path.basename(self.dataset_path or "--")),
            ("Target device", target),
            ("Budgets",
             f"{self.cfg['flash_kb']:g} KB flash · {self.cfg['cpu_kb']:g} KB "
             f"CPU RAM · {self.cfg['cla_kb']:g} KB CLA data RAM"),
            ("Quantization", s["quant"]),
            ("Conv layers / pooling",
             f"{s['layers']}  ·  "
             + ("MaxPool(2,2), time + coefficient" if self.pool_w
                else "MaxPool(2,1), time only")),
            ("Channel search",
             f"ceiling {self.max_ch}  →  selected {s['channels']}"),
            ("Target", f"{s['metric']} ≥ {s['target']:.0f}%"),
            ("Robustness check",
             f"{N_SEEDS} seeds × (clean + {PERTURB_TRIALS} perturbed val), "
             f"worst case accepted"),
            ("CLA offload plan",
             ", ".join(plan) + " on CLA" if plan else "everything on the C28x"),
            ("Arena", "on" if self.use_arena else "off"),
        ]
        if t:
            p = t["plan"]
            rows.append((
                "Feature front end",
                f"{p['n_fft']}-point FFT, hop {p['hop']}, {p['n_mels']} mel "
                f"bands, {p['n_frames']} frames = {p['window_ms']:.0f} ms window"
                + ("  (assumed — no feature_config.json)" if p["assumed"] else "")))
            rows.append((
                "Frame budget",
                f"features {t['fe_ms']:.1f} ms "
                f"({'measured' if t['measured'] else 'estimated'}) + inference "
                f"{t['infer_ms']:.2f} ms → {t['latency_ms']:.1f} ms latency, "
                f"a window every {t['period_ms']:.1f} ms"))
        self.rc_grid.setColumnStretch(2, 1)
        for i, (k, v) in enumerate(rows):
            self.rc_grid.addWidget(
                self._cell(k, "MatrixHead",
                           align=Qt.AlignLeft | Qt.AlignVCenter), i, 0)
            self.rc_grid.addWidget(
                self._cell(v, "MatrixCell",
                           align=Qt.AlignLeft | Qt.AlignVCenter), i, 1)

    def _fill_footprint(self):
        """Flash / CPU RAM / CLA RAM of the model that was exported, against
        the budgets the search was given."""
        self._clear_grid(self.fp_grid)
        if not self._last_footprint:
            return
        f, sc, scla, fc_in = self._last_footprint
        c = self.cfg
        rows = (("Flash", f / 1024, c["flash_kb"]),
                ("CPU RAM", sc / 1024, c["cpu_kb"]),
                ("CLA data RAM", scla / 1024, c["cla_kb"]))
        for j, h in enumerate(("pool", "used", "budget", "")):
            self.fp_grid.addWidget(
                self._cell(h, "MatrixHead",
                           align=Qt.AlignLeft | Qt.AlignVCenter if j == 0
                           else Qt.AlignRight | Qt.AlignVCenter), 0, j)
        for i, (name, used, limit) in enumerate(rows):
            frac = (used / limit) if limit else 0.0
            col = C_ERR if frac > 1.0 else (C_WARN if frac > 0.88 else C_TEXT)
            self.fp_grid.addWidget(
                self._cell(name, "MatrixCell",
                           align=Qt.AlignLeft | Qt.AlignVCenter), i + 1, 0)
            self.fp_grid.addWidget(
                self._cell(f"{used:.2f} KB", "MatrixCell", col,
                           align=Qt.AlignRight | Qt.AlignVCenter), i + 1, 1)
            self.fp_grid.addWidget(
                self._cell(f"{limit:.0f} KB", "MatrixCell", C_MUTED,
                           align=Qt.AlignRight | Qt.AlignVCenter), i + 1, 2)
            self.fp_grid.addWidget(
                self._cell(f"{frac * 100:.0f}%", "MatrixCell", col,
                           align=Qt.AlignRight | Qt.AlignVCenter), i + 1, 3)
        foot = f"FC input {fc_in}"
        if self.feature_ram:
            foot += (f"   ·   CPU RAM includes the feature front end, "
                     f"{self.feature_ram / 1024:.0f} KB")
        self.fp_grid.addWidget(
            self._cell(foot, "MatrixCell", C_FAINT,
                       align=Qt.AlignLeft | Qt.AlignVCenter),
            len(rows) + 1, 0, 1, 4)

    def _fill_timing(self):
        """Where the frame goes, stage by stage, on the selected part's
        clock. Cycles are shown next to milliseconds because the cycles are
        the derived quantity -- the milliseconds are only those divided by
        the clock, and swapping the part swaps the divisor."""
        self._clear_grid(self.tm_grid)
        t = self.timing
        if not t:
            self.tm_head.setText("not analysed")
            self.tm_sub.setText(
                "No feature-extraction time was given, so no frame budget "
                "was computed. Measure your front end and type it in the "
                "toolbar; nothing here is filled in with an assumed value.")
            self.tm_foot.setText("")
            self.tm_advice.setText("")
            return
        over = t["deadline_ms"] and t["period_ms"] > t["deadline_ms"]
        self.tm_head.setText(
            f"{t['latency_ms']:.1f} ms latency   ·   a result every "
            f"{t['period_ms']:.1f} ms")
        self.tm_head.setStyleSheet(f"color:{C_ERR};" if over else "")
        self.tm_sub.setText(
            f"features {t['fe_ms']:.1f} ms "
            f"({'measured' if t['measured'] else 'estimated'})  +  inference "
            f"{t['infer_ms']:.2f} ms"
            + (f"   ·   {t['overlap_ms']:.2f} ms of it overlapped by the CLA"
               if t["overlap_ms"] > 0 else "   ·   nothing overlaps")
            + (f"   ·   budget {t['deadline_ms']:.0f} ms"
               if t["deadline_ms"] else ""))
        dev = device_by_key(self.cfg.get("device"))
        cpu_mhz = self.cfg.get("cpu_mhz", 200.0)
        cla_mhz = self.cfg.get("cla_mhz", 200.0)

        for j, h in enumerate(("stage", "where", "MACs", "kcycles", "ms")):
            self.tm_grid.addWidget(
                self._cell(h, "MatrixHead",
                           align=Qt.AlignLeft | Qt.AlignVCenter if j < 2
                           else Qt.AlignRight | Qt.AlignVCenter), 0, j)

        rows = [("features", "C28x", "--",
                 t["fe_ms"] * cpu_mhz, t["fe_ms"],
                 C_STAGE)]
        for i, s in enumerate(t["stages"]):
            rows.append((f"S{i + 1}", "CLA" if s["on_cla"] else "C28x",
                         f"{s['macs']:,}", s["cycles"] / 1000.0, s["ms"],
                         C_CLA if s["on_cla"] else C_CPU))
        for i, (name, where, macs, kcyc, ms, col) in enumerate(rows):
            r = i + 1
            self.tm_grid.addWidget(
                self._cell(name, "MatrixCell",
                           align=Qt.AlignLeft | Qt.AlignVCenter), r, 0)
            self.tm_grid.addWidget(
                self._cell(where, "MatrixCell", col,
                           align=Qt.AlignLeft | Qt.AlignVCenter), r, 1)
            self.tm_grid.addWidget(
                self._cell(macs, "MatrixCell", C_MUTED,
                           align=Qt.AlignRight | Qt.AlignVCenter), r, 2)
            self.tm_grid.addWidget(
                self._cell(f"{kcyc:,.0f}", "MatrixCell", C_MUTED,
                           align=Qt.AlignRight | Qt.AlignVCenter), r, 3)
            self.tm_grid.addWidget(
                self._cell(f"{ms:.3f}", "MatrixCell",
                           align=Qt.AlignRight | Qt.AlignVCenter), r, 4)

        # No summary rows here: latency, window period, the budget and
        # the split are the headline above. Repeating them under the
        # table cost it the vertical space it needed and squeezed it
        # to nothing on a short window.

        # provenance in one line rather than four more rows: what clock the
        # milliseconds were divided by, and how much of this is measured
        src = (f"feature time yours, {t['calib']:.2f}x this project's MFCC "
               f"estimate" if t["measured"] else
               "feature time ESTIMATED for this project's MFCC front end"
               + ("" if not t["plan"]["assumed"] else ", assumed FFT parameters"))
        mac = ("inference measured, "
               f"{t['mac_cost']['cla']:.0f} cycles/MAC implied"
               if t["infer_measured"] else
               f"{t['mac_cost']['cla']:.0f} cycles/MAC on the CLA (measured) "
               f"vs {t['mac_cost']['cpu']:.0f} on the C28x (carried over)")
        # The cycle counts were measured on a type-1 CLA. Scaling them to a
        # different CLA generation by clock alone is an extrapolation, and
        # the screen should not pretend otherwise.
        anchor_type = 1
        warn = ("" if dev is None or dev["cla_type"] == anchor_type else
                f"  ·  ⚠ measured on a type-{anchor_type} CLA, scaled to "
                f"type {dev['cla_type']} by clock alone")
        self.tm_foot.setText(
            f"{dev['name'] if dev else 'custom target'}  ·  C28x {cpu_mhz:g} MHz"
            f"  ·  CLA {cla_mhz:g} MHz  ·  {src}  ·  {mac}"
            f"  ·  {t['total_macs']:,} MACs total" + warn
            + "\n" + validation_note())
        self.tm_foot.setStyleSheet(
            f"color:{C_WARN};" if warn else "")

        sev, text = timing_advice(t)
        col = {"error": C_ERR, "warn": C_WARN, "ok": C_OK}.get(sev, C_MUTED)
        self.tm_advice.setText(text)
        self.tm_advice.setStyleSheet(f"font-size:11px; color:{col};")

    def _clear_grid(self, grid):
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    @staticmethod
    def _cell(text, obj, color=None, align=Qt.AlignCenter):
        lab = QLabel(text)
        lab.setObjectName(obj)
        lab.setAlignment(align)
        if color:
            lab.setStyleSheet(f"color:{color};")
        return lab

    def _fill_confusion(self, classes, cm):
        self._clear_grid(self.cm_grid)
        # the grid spans the full panel width, so without this the columns
        # spread out until the matrix stops reading as a matrix
        self.cm_grid.setColumnStretch(len(classes) + 1, 1)
        self.cm_grid.addWidget(
            self._cell("true \\ pred", "MatrixHead",
                       align=Qt.AlignLeft | Qt.AlignVCenter), 0, 0)
        for j, c in enumerate(classes):
            self.cm_grid.addWidget(self._cell(str(c)[:12], "MatrixHead"), 0, j + 1)
            self.cm_grid.setColumnMinimumWidth(j + 1, 78)
        for i, c in enumerate(classes):
            self.cm_grid.addWidget(
                self._cell(str(c)[:12], "MatrixHead",
                           align=Qt.AlignLeft | Qt.AlignVCenter), i + 1, 0)
            for j in range(len(classes)):
                v = int(cm[i][j])
                col = C_OK if i == j else (C_ERR if v else C_FAINT)
                self.cm_grid.addWidget(
                    self._cell(str(v), "MatrixCell", col), i + 1, j + 1)


    # ------------------------------------------------------------ training
    def start_training(self):
        if not self.dataset_path:
            self.log(f"<span style='color:{C_ERR};'>Select a dataset first.</span>")
            return
        try:
            target = float(self.target_edit.text())
        except ValueError:
            self.log(f"<span style='color:{C_ERR};'>Target value must be numeric.</span>")
            return

        port = self.target_mismatch()
        if port:
            self.log(f"<span style='color:{C_WARN};'><b>Target:</b> "
                     f"{port[0].upper() + port[1:]}.</span>")

        if self.timing:
            sev, text = timing_advice(self.timing)
            col = {"error": C_ERR, "warn": C_WARN, "ok": C_OK}.get(sev, C_MUTED)
            self.log(f"<span style='color:{col};'><b>Frame budget:</b> "
                     f"{text}</span>")

        why = self.firmware_incompatibilities(
            self.quant_mode, self.layers_val, self.pool_w)
        if why:
            self.log(
                f"<span style='color:{C_WARN};'><b>Note:</b> this model will NOT be "
                f"deployable to the board -- {'; '.join(why)}. Training continues; "
                f"embedded/ will only get cla_pipeline_config.h and "
                f"feature_tables.h.</span>")

        self.train_btn.setEnabled(False)
        self.set_controls_locked(True)
        self.show_view(0)          # watch the device while the search runs
        self.run_row.show()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.run_detail.setText("")
        self.phase_label.setText("Starting…")
        self._set_status("training", C_WARN)
        self._t0 = datetime.datetime.now().timestamp()
        self.elapsed_label.setText("00:00")
        self._clock.start()

        self.thread = TrainingThread(
            self.max_ch, self.layers_val, self.quant_mode, self.input_h,
            self.input_w, self.classes_val, target,
            self.metric_combo.currentText(), self.pool_w, self.dataset_path,
            self.get_cla_stage(), self._timing_note())
        self.thread.update_signal.connect(self.log)
        self.thread.progress_signal.connect(self.on_progress)
        self.thread.summary_signal.connect(self.on_summary)
        self.thread.optimal_arch_found_signal.connect(self.handle_optimal_architecture)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def _timing_note(self):
        """The frame budget in the form cla_pipeline_config.h carries it:
        Turkish, like the rest of that header's comments, and explicit that
        these are modelled cycles, not a measurement off the board."""
        t = self.timing
        if not t:
            return []
        dev = device_by_key(self.cfg.get("device"))
        lines = [
            f"Hedef: {dev['name'] if dev else 'ozel'}  "
            f"(C28x {self.cfg.get('cpu_mhz', 200):g} MHz, "
            f"CLA {self.cfg.get('cla_mhz', 200):g} MHz)",
            f"Ozellik : {t['fe_ms']:.2f} ms "
            f"({'OLCULEN' if t['measured'] else 'tahmin'})",
            f"MAC basi {t['mac_cost']['cla']:.0f} cevrim (CLA) / "
            f"{t['mac_cost']['cpu']:.0f} (C28x)"
            + f" -- model {CLA_VALIDATION['channels']} kanalda "
              f"%{VALIDATION['error_pct']:+.0f} sapmayla dogrulandi (-O0)"
            + (" -- olculen inference suresinden" if t["infer_measured"]
               else " -- model varsayimi"),
        ]
        for i, s in enumerate(t["stages"]):
            lines.append(
                f"S{i + 1}   : {s['ms']:.3f} ms  "
                f"({'CLA' if s['on_cla'] else 'CPU'}, {s['macs']:,} MAC, "
                f"{s['cycles'] / 1000:.0f} kcevrim) -- diger tarafta "
                f"{s['alt_ms']:.3f} ms")
        lines.append(
            f"Gecikme {t['latency_ms']:.2f} ms, pencere basi "
            f"{t['period_ms']:.2f} ms"
            + (f" (CLA {t['overlap_ms']:.3f} ms'i ortuyor)"
               if t["overlap_ms"] > 0 else ""))
        lines.append(timing_advice(t)[1])
        return lines

    def handle_optimal_architecture(self, ch):
        """Training picked a channel count -- redraw the device as the model
        that will actually be exported, not as the ceiling."""
        self.draw_device(ch, "selected")

    def set_controls_locked(self, locked):
        """While the search runs, everything that feeds it is frozen: the
        thread already holds its own copy of the plan, so letting the toolbar
        or the diagram change would only make the screen lie about the run."""
        self.canvas.locked = locked
        for w in (self.dataset_btn, self.quant_combo, self.metric_combo,
                  self.target_edit, self.timing_check, self.setup_btn):
            w.setEnabled(not locked)
        self.fe_edit.setEnabled(not locked and self.timing_check.isChecked())
        self.canvas.hint = self._current_hint()
        self.canvas.update()

    def _current_hint(self):
        return ("Settings are locked while the search runs."
                if self.canvas.locked else
                "Pick CPU or CLA on a stage card to choose where it runs  "
                "·  the ceiling and both gauges update immediately")

    def training_finished(self):
        self._clock.stop()
        self._tick_elapsed()
        self.set_controls_locked(False)
        self.train_btn.setEnabled(True)
        failed = self.phase_label.text().lower().startswith("fail")
        self._set_status("failed" if failed else "finished",
                         C_ERR if failed else C_OK)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    install_crash_handler()   # needs QApplication for the message box
    win = EdgeAiGui()
    win.show()
    sys.exit(app.exec_())



