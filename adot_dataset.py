"""
adot_dataset.py -- turns a folder of recordings into the dataset ADOT trains on.

    <root>/<class>/<recording>                  class = folder name
    <root>/<class>/<group>/<recording>          optional group (a motor, a load,
                                                a recording day) that can be
                                                held out as the test set
The two levels can be swapped (class_level / group_level) for datasets laid
out the other way round, e.g. MIMII's id_00/normal/*.wav.

What it does, in this order:
  1. finds the recordings and labels them from their folders
  2. splits FILES into train+validation and test (a held-out group, or a
     random share of the files of every class) -- before anything is cut, so
     pieces of one recording never land on both sides
  3. cuts the recordings into windows of slice_len samples: the head of every
     training file gives the training windows (half overlapped, augmented),
     its tail gives the validation windows, test files give windows that do
     not overlap. With val_groups the validation set is whole groups instead
     (an operating point or a unit the training never sees), and training
     files are used from start to end
  4. computes the features of every window:
        mfcc     Hann, FFT, mel filters, log, DCT           (sound)
        logspec  Hann, FFT, linear bands, log -- no mel, no DCT   (vibration)
        raw      the samples themselves, peak-normalised, folded row by row
  5. keeps a few test windows per class as raw ADC codes ("golden" windows:
     golden_adc / golden_y / golden_src / golden_ref in the .npz), so the
     exported firmware can be tested from the ADC code to the class, and
  6. z-scores the features with the TRAINING set's statistics and writes
        <out_dir>/folds/dataset.npz
        <out_dir>/feature_config.json
     which is what ADOT's "dataset" button and its firmware-table generator
     read. The file's "feature" key says which front end it describes.

No GUI and no torch in here, so it also runs from a terminal:
    python adot_dataset.py --help

Recordings: .wav/.flac (any rate and channel count: resampled, mixed to mono)
and .npy/.csv/.txt holding one signal as a column of numbers. The latter carry
no sample rate, so input_sr has to be given. They carry no scale either:
full_scale is the signal value that would drive the board's ADC to its rail.
Left unset it is taken from the recordings themselves -- 1.2 times their
largest peak, rounded up to a round number -- which is what a recorder does
when it writes a .wav: the file's full scale is whatever its ADC clipped at.
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

AUDIO_EXT = (".wav", ".flac")
TABLE_EXT = (".npy", ".csv", ".txt")
FEATURES = ("mfcc", "logspec", "raw")


class PrepError(ValueError):
    """Something about the folder or the settings the user has to fix."""


class Cancelled(Exception):
    """The progress callback asked to stop."""


@dataclass
class PrepConfig:
    root: str = ""
    out_dir: str = ""
    name: str = "dataset"                # folds/<name>.npz
    feature: str = "mfcc"

    # --- where the labels are -------------------------------------------
    class_level: int = 0                 # folder depth below root that names the class
    group_level: Optional[int] = 1       # ... that names the group; None = no groups
    classes: Optional[List[str]] = None  # label order; None = alphabetical

    # --- reading ----------------------------------------------------------
    sr: int = 16000                      # rate the features are computed at
    input_sr: Optional[int] = None       # rate of .npy/.csv files (None = sr)
    full_scale: Optional[float] = None   # .npy/.csv: value mapped to +-1;
                                         # None = from the recordings' peak
    column: int = 0                      # .npy/.csv with several columns

    # --- windows ------------------------------------------------------------
    n_fft: int = 512
    hop: int = 256
    slice_len: int = 16896               # samples per window

    # --- mfcc -----------------------------------------------------------------
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 0.0
    fmax: Optional[float] = None         # None = sr / 2
    htk: bool = False
    drop_c0: bool = True
    # --- logspec ---------------------------------------------------------------
    n_bands: Optional[int] = None        # None = default_bands(n_fft)
    # --- raw ---------------------------------------------------------------------
    raw_cols: int = 20                   # slice_len must be a multiple of this

    clip: float = 3.0

    # --- split -------------------------------------------------------------------
    test_groups: Optional[List[str]] = None   # these groups are the test set ...
    test_fraction: float = 0.20          # ... otherwise this share of each class's files
    val_tail: float = 0.32               # last part of every training file -> validation
    val_groups: Optional[List[str]] = None    # ... or these whole groups instead
    val_from_test: bool = False          # ... or the test set itself: nothing is
                                         # taken from the training files, but then
                                         # no score on the test set is independent
    max_files: Optional[int] = None      # per (group, class); None = all

    # --- augmentation of the training windows --------------------------------------
    n_aug: int = 1
    gain_db: float = 6.0
    snr_db: Tuple[float, float] = (25.0, 45.0)
    tilt: float = 0.30
    speed: float = 0.03
    seed: int = 1337

    extra: dict = field(default_factory=dict)   # copied into feature_config.json

    # ----------------------------------------------------------------------------
    def resolved(self):
        """Fills the settings that depend on others and checks the rest."""
        c = PrepConfig(**asdict(self))
        if c.feature not in FEATURES:
            raise PrepError(f"feature '{c.feature}' is not one of {', '.join(FEATURES)}")
        if c.fmax is None:
            c.fmax = c.sr / 2.0
        if c.input_sr is None:
            c.input_sr = c.sr
        if c.feature == "raw":
            if c.raw_cols < 3 or c.slice_len % c.raw_cols or c.slice_len // c.raw_cols < 3:
                raise PrepError(f"raw: slice_len {c.slice_len} is not rows x {c.raw_cols} "
                                f"columns with at least 3 of each")
        else:
            if c.n_fft < 32 or c.n_fft & (c.n_fft - 1):
                raise PrepError(f"n_fft {c.n_fft} must be a power of two, 32 or larger "
                                f"(the firmware's FFT)")
            if c.hop < 1 or c.hop > c.n_fft:
                raise PrepError(
                    f"hop {c.hop} is larger than the FFT length {c.n_fft}: the samples "
                    f"between two frames would never be looked at. Half the FFT "
                    f"length ({c.n_fft // 2}) is the usual choice.")
            if c.n_bands is None:
                c.n_bands = default_bands(c.n_fft)
            if c.slice_len < c.n_fft or (c.slice_len - c.n_fft) % c.hop:
                raise PrepError(
                    f"slice_len {c.slice_len} is not n_fft + a whole number of hops; "
                    f"nearest: {fit_slice_len(1000.0 * c.slice_len / c.sr, c.sr, c.n_fft, c.hop)}")
        if c.feature == "logspec" and not 1 <= c.n_bands <= c.n_fft // 2:
            raise PrepError(f"{c.n_bands} bands do not fit {c.n_fft // 2} FFT bins")
        if not 0.0 < c.val_tail < 1.0:
            raise PrepError("val_tail must be between 0 and 1")
        if c.val_from_test and c.val_groups:
            raise PrepError("validation is either the test set or a group, not both")
        if c.val_from_test and not (c.test_groups or c.test_fraction > 0):
            raise PrepError("validation = the test set needs a test set")
        if c.group_level is not None and c.group_level == c.class_level:
            raise PrepError("class and group cannot be the same folder level")
        return c

    @property
    def n_frames(self):
        if self.feature == "raw":
            return self.slice_len // self.raw_cols
        return (self.slice_len - self.n_fft) // self.hop + 1


def default_bands(n_fft):
    """Linear bands for a log-spectrum front end when none is asked for: at
    least two FFT bins per band, and an input no wider than 32 columns.
    (24 at n_fft 256 is what the CWRU bearing data needed; 12 was too few.)"""
    bins = n_fft // 2
    if bins >= 256:
        return 32
    if bins >= 128:
        return 24
    return max(4, min(16, bins // 2))


def nice_full_scale(peak, headroom=1.2):
    """A round full-scale value with some room above the largest peak."""
    target = float(peak) * headroom
    if not target > 0:
        return 1.0
    exp = np.floor(np.log10(target))
    m = target / 10.0 ** exp
    for step in (1, 1.25, 1.6, 2, 2.5, 3.2, 4, 5, 6.4, 8, 10):
        if m <= step * (1 + 1e-9):
            return float(step * 10.0 ** exp)
    return float(10.0 ** (exp + 1))


def fit_slice_len(ms, sr, n_fft, hop):
    """The window length closest to `ms` that is n_fft + a whole number of
    hops -- with center=False that is exactly what the frames span, and the
    only kind of window the firmware can cut."""
    n_frames = max(1, int(round((ms * sr / 1000.0 - n_fft) / hop)) + 1)
    return hop * (n_frames - 1) + n_fft


# ============================================================================
# 1. find and label
# ============================================================================
def discover(cfg):
    """-> (rows, classes, skipped). A row is (path, label, group, source);
    source is the path below root and ends up in the dataset's src_* arrays.
    (Not '<group>/<file name>': MIMII numbers the files of id_00/normal and
    id_00/abnormal alike, so that would name two recordings the same.)"""
    root = Path(cfg.root)
    if not root.is_dir():
        raise PrepError(f"{root} is not a folder")
    found, skipped = [], 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXT + TABLE_EXT:
            continue
        dirs = path.relative_to(root).parts[:-1]
        if len(dirs) <= cfg.class_level:
            skipped += 1                 # not inside a class folder
            continue
        cls = dirs[cfg.class_level]
        grp = (dirs[cfg.group_level]
               if cfg.group_level is not None and len(dirs) > cfg.group_level else "")
        found.append((path, cls, grp))
    if not found:
        raise PrepError(f"no {'/'.join(AUDIO_EXT + TABLE_EXT)} recordings in class "
                        f"folders under {root}")
    names = sorted({c for _, c, _ in found})
    if cfg.classes:
        missing = [c for c in cfg.classes if c not in names]
        if missing:
            raise PrepError(f"class folder(s) not found: {missing} (found: {names})")
        skipped += sum(1 for _, c, _ in found if c not in cfg.classes)
        found = [f for f in found if f[1] in cfg.classes]
        names = list(cfg.classes)
    if len(names) < 2:
        raise PrepError(f"only one class folder ({names}); check class_level")
    rows = [(p, names.index(c), g, p.relative_to(root).as_posix()) for p, c, g in found]
    rows.sort(key=lambda r: (r[1], r[2], str(r[0])))
    return rows, names, skipped


def describe(rows, classes):
    """{class: {group: file count}} for a summary table."""
    out = {c: {} for c in classes}
    for _, label, grp, _ in rows:
        d = out[classes[label]]
        d[grp] = d.get(grp, 0) + 1
    return out


# ============================================================================
# 2. read
# ============================================================================
def read_table(path, cfg):
    """A .npy/.csv/.txt recording in its own unit, at cfg.sr (float64)."""
    from scipy.signal import resample_poly
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        x = np.load(str(path))
    else:
        x = np.genfromtxt(str(path), delimiter="," if ext == ".csv" else None)
        x = x[~np.isnan(x).any(axis=tuple(range(1, x.ndim)))] if x.ndim > 1 else x[~np.isnan(x)]
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        if x.shape[0] < x.shape[1]:
            x = x.T                      # one row per channel -> one column per channel
        if cfg.column >= x.shape[1]:
            raise PrepError(f"{path.name}: column {cfg.column} of {x.shape[1]}")
        x = x[:, cfg.column]
    elif x.ndim != 1:
        raise PrepError(f"{path.name}: {x.ndim}-dimensional, expected one signal")
    if cfg.input_sr != cfg.sr:
        g = int(np.gcd(int(cfg.input_sr), int(cfg.sr)))
        x = resample_poly(x, cfg.sr // g, cfg.input_sr // g)   # low-pass + rate change
    return x


def load_signal(path, cfg):
    """One recording as float32 in [-1, 1] at cfg.sr."""
    path = Path(path)
    if path.suffix.lower() in AUDIO_EXT:
        import librosa
        y, _ = librosa.load(str(path), sr=cfg.sr, mono=True)
        return y.astype(np.float32)
    if cfg.full_scale is None:
        raise PrepError("full_scale is not set (prepare() measures it when left out)")
    x = read_table(path, cfg)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > cfg.full_scale:
        raise PrepError(f"{path.name}: peak {peak:.4g} exceeds full_scale {cfg.full_scale:g}")
    return (x / cfg.full_scale).astype(np.float32)


# ============================================================================
# 3. cut, augment
# ============================================================================
def take_portion(y, portion, slice_len):
    """('head', f): all but the last share f. ('tail', f): the last share f,
    less one window, so the last training window and the first validation
    window never share samples."""
    if portion is None:
        return y
    kind, frac = portion
    cut = int(len(y) * (1.0 - frac))
    return y[:cut] if kind == "head" else y[cut + slice_len:]


def slice_wave(y, slice_len, hop):
    if len(y) < slice_len:
        return []
    return [y[i:i + slice_len] for i in range(0, len(y) - slice_len + 1, hop)]


def augment(seg, cfg, rng):
    """Speed, spectral tilt, gain, added noise and polarity -- what differs
    between two installations of the same machine and sensor."""
    from scipy.signal import lfilter, resample_poly
    n = cfg.slice_len
    out = seg
    if cfg.speed > 0:
        up = int(round(1000 * rng.uniform(1 - cfg.speed, 1 + cfg.speed)))
        out = resample_poly(out, up, 1000).astype(np.float32)
        if len(out) >= n:
            o = rng.integers(0, len(out) - n + 1)
            out = out[o:o + n]
        else:
            out = np.pad(out, (0, n - len(out)))
    if cfg.tilt > 0:
        out = lfilter([1.0, -rng.uniform(-cfg.tilt, cfg.tilt)], [1.0], out).astype(np.float32)
    out = out * 10.0 ** (rng.uniform(-cfg.gain_db, cfg.gain_db) / 20.0)
    p = np.mean(out ** 2) + 1e-12
    out = out + rng.normal(
        0.0, np.sqrt(p / 10.0 ** (rng.uniform(*cfg.snr_db) / 10.0)), out.shape
    ).astype(np.float32)
    if rng.random() < 0.5:
        out = -out
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ============================================================================
# 4. features
# ============================================================================
def band_edges(cfg):
    """logspec: band b averages the power bins edges[b] .. edges[b+1]-1.
    Bin 0 (DC) is in no band; the last band ends at Nyquist."""
    e = np.round(np.linspace(1, cfg.n_fft // 2 + 1, cfg.n_bands + 1)).astype(int)
    if np.any(np.diff(e) < 1):
        raise PrepError(f"{cfg.n_bands} bands are too many for {cfg.n_fft // 2} bins")
    return e


def make_featurizer(cfg):
    """-> f(window, quantized=False) -> [n_frames, n_columns] float32, not yet
    normalised.

    Training features are computed from the recordings as they are: nothing
    is rounded to ADC codes here. The board test is where the ADC comes in
    (golden_windows below): there the window is first turned into 12-bit
    codes, and quantized=True says it already is what the firmware computes
    from them, (code - mean) / half-scale."""
    import librosa
    if cfg.feature == "mfcc":
        def f(seg, quantized=False):
            m = librosa.feature.mfcc(
                y=seg, sr=cfg.sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop,
                win_length=cfg.n_fft, window="hann", center=False, n_mels=cfg.n_mels,
                fmin=cfg.fmin, fmax=cfg.fmax, htk=cfg.htk, power=2.0, dct_type=2,
                norm="ortho")
            return m.T.astype(np.float32)
        return f
    if cfg.feature == "logspec":
        edges = band_edges(cfg)
        fb = np.zeros((cfg.n_bands, cfg.n_fft // 2 + 1))
        for b in range(cfg.n_bands):
            fb[b, edges[b]:edges[b + 1]] = 1.0 / (edges[b + 1] - edges[b])

        def f(seg, quantized=False):
            # The firmware removes the slice mean; the Hann window leaks DC
            # into bin 1, so the lowest band differs if training does not.
            if not quantized:
                seg = seg - seg.mean()
            S = np.abs(librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop,
                                    win_length=cfg.n_fft, window="hann",
                                    center=False)) ** 2
            return librosa.power_to_db(fb @ S).T.astype(np.float32)
        return f

    def f(seg, quantized=False):         # raw
        v = seg.astype(np.float64) - seg.mean()
        v /= np.abs(v).max() + 1e-12
        return v.reshape(-1, cfg.raw_cols).astype(np.float32)
    return f


def build(rows, cfg, rng, featurize, do_aug, portion=None, dense=False, tick=None):
    """Features of every window of `rows`, NOT normalised.
    -> (X [n, frames, columns], y, src, files that were too short)."""
    X, y, src, short = [], [], [], 0
    hop = cfg.slice_len // 2 if (do_aug or dense) else cfg.slice_len
    for path, label, _grp, source in rows:
        if tick:
            tick(source)
        segs = slice_wave(take_portion(load_signal(path, cfg), portion, cfg.slice_len),
                          cfg.slice_len, hop)
        if not segs:
            short += 1
            continue
        for s in segs:
            for v in ([s] + [augment(s, cfg, rng) for _ in range(cfg.n_aug)] if do_aug else [s]):
                X.append(featurize(v)); y.append(label); src.append(source)
    if not X:
        return np.zeros((0, 0, 0), np.float32), np.zeros(0, np.int64), np.zeros(0, str), short
    return np.stack(X), np.asarray(y, np.int64), np.asarray(src), short


GOLDEN_PER_CLASS = 4
GOLDEN_BITS = 12                 # the firmware's ADC: codes 0..4095, centre 2048


def golden_windows(rows, cfg, classes, featurize, portion=None):
    """A few windows per class as the ADC codes a board would sample, with
    the features the firmware has to compute from exactly those codes.

    Taken from `rows` (the test files; the validation tails when there is no
    test set), spread over as many recordings as the class has: the first
    window of each file, then second windows. Codes are 12-bit and centred
    on half scale -- code = round(2048 + x * 2047) -- which is the inverse of
    the firmware's (code - mean) / 2048 up to the mean. The features kept
    with them are computed from the CODES, so they include the rounding the
    training features do not have: they are what the board must reproduce.
    -> (codes uint16 [n, slice_len], labels, sources, features [n, h, w])"""
    mid = float(1 << (GOLDEN_BITS - 1))
    codes, labels, srcs, feats = [], [], [], []
    for li in range(len(classes)):
        mine = [r for r in rows if r[1] == li]
        segs = []                        # per file: its windows, in order
        for path, _l, _g, source in mine[:GOLDEN_PER_CLASS]:
            y = take_portion(load_signal(path, cfg), portion, cfg.slice_len)
            segs.append((source, slice_wave(y, cfg.slice_len, cfg.slice_len)))
        took, k = 0, 0
        while took < GOLDEN_PER_CLASS and any(k < len(w) for _, w in segs):
            for source, wins in segs:
                if k < len(wins) and took < GOLDEN_PER_CLASS:
                    adc = np.clip(np.round(mid + wins[k].astype(np.float64) * (mid - 1.0)),
                                  0, 2 * mid - 1)
                    seen = ((adc - adc.mean()) / mid).astype(np.float32)
                    codes.append(adc.astype(np.uint16)); labels.append(li)
                    srcs.append(f"{source} @ sample {k * cfg.slice_len}")
                    feats.append(featurize(seen, quantized=True))
                    took += 1
            k += 1
    if not codes:
        return None
    return (np.stack(codes), np.asarray(labels, np.int64), np.asarray(srcs),
            np.stack(feats))


# ============================================================================
# 5. split, normalise, write
# ============================================================================
def split_files(rows, classes, cfg, rng):
    """-> (training rows, validation rows, test rows, note). Whole files,
    never windows. The validation rows are empty unless cfg.val_groups is
    set: by default validation is the tail of every training file."""
    if cfg.max_files:
        by_key = {}
        for r in rows:
            by_key.setdefault((r[2], r[1]), []).append(r)
        rows = []
        for lst in by_key.values():
            rng.shuffle(lst)
            rows += lst[:cfg.max_files]
    groups = sorted({r[2] for r in rows})
    va = []
    if cfg.val_groups:
        missing = [g for g in cfg.val_groups if g not in groups]
        if missing:
            raise PrepError(f"validation group(s) not found: {missing} (groups: {groups})")
        both = [g for g in cfg.val_groups if g in (cfg.test_groups or [])]
        if both:
            raise PrepError(f"group(s) {both} cannot be both the validation and the test set")
        va = [r for r in rows if r[2] in cfg.val_groups]
        rows = [r for r in rows if r[2] not in cfg.val_groups]
    if cfg.test_groups:
        missing = [g for g in cfg.test_groups if g not in groups]
        if missing:
            raise PrepError(f"test group(s) not found: {missing} (groups: {groups})")
        te = [r for r in rows if r[2] in cfg.test_groups]
        tr = [r for r in rows if r[2] not in cfg.test_groups]
        note = f"test = group(s) {', '.join(cfg.test_groups)}"
    elif cfg.test_fraction > 0:
        te, tr = [], []
        for li in range(len(classes)):
            f = [r for r in rows if r[1] == li]
            rng.shuffle(f)
            n_te = max(1, int(round(len(f) * cfg.test_fraction)))
            te += f[:n_te]; tr += f[n_te:]
        note = f"test = {100 * cfg.test_fraction:.0f}% of each class's files, at random"
    else:
        te, tr, note = [], list(rows), "no test set"
    if cfg.val_from_test:
        note = ("held out" + note[len("test"):] + "; never trained on, and the "
                "search is scored on them")
    elif cfg.val_groups:
        note += f"; validation = group(s) {', '.join(cfg.val_groups)}"
    else:
        note += f"; validation = last {100 * cfg.val_tail:.0f}% of each training file"
    for li, c in enumerate(classes):
        if not any(r[1] == li for r in tr):
            raise PrepError(f"class '{c}' has no training file left ({note})")
        if cfg.val_groups and not any(r[1] == li for r in va):
            raise PrepError(f"class '{c}' has no recording in the validation group(s)")
    return tr, va, te, note


def mel_nnz(cfg):
    """Sum of the mel filters' lengths (FEAT_MEL_NNZ in the firmware header),
    after checking what the firmware's sparse storage needs of every filter."""
    import librosa
    fb = librosa.filters.mel(sr=cfg.sr, n_fft=cfg.n_fft, n_mels=cfg.n_mels,
                             fmin=cfg.fmin, fmax=cfg.fmax, htk=cfg.htk)
    nnz = 0
    for i, row in enumerate(fb):
        idx = np.nonzero(row)[0]
        if len(idx) == 0:
            raise PrepError(f"mel filter {i} is empty: {cfg.n_mels} mel bands are too "
                            f"many for a {cfg.n_fft}-point FFT")
        if idx[-1] - idx[0] + 1 != len(idx):
            raise PrepError(f"mel filter {i} is not contiguous")
        nnz += len(idx)
    return int(nnz)


def feature_config(cfg):
    """(file name, dict) describing the front end, in the keys ADOT reads."""
    c = cfg
    if c.feature == "raw":
        return "feature_config.json", {
            "feature": "raw", "sr": c.sr, "slice_len": c.slice_len,
            "rows": c.n_frames, "cols": c.raw_cols, "order": "row-major",
            "normalize": "(x - mean) / max|x - mean|"}
    if c.feature == "logspec":
        e = band_edges(c)
        d = {"feature": "logspec", "sr": c.sr, "n_fft": c.n_fft, "hop": c.hop,
             "n_bands": c.n_bands, "band_edges": [int(v) for v in e],
             # the keys ADOT reads for an MFCC front end, with the same meaning:
             # as many "filters" as bands, no coefficient dropped
             "n_mels": c.n_bands, "n_mfcc": c.n_bands, "drop_c0": False,
             "mel_nnz": int(e[-1] - e[0]),
             "center": False, "window": "hann", "power": 2.0,
             "slice_len": c.slice_len, "n_frames": c.n_frames, "clip": c.clip}
    else:
        d = {"feature": "mfcc", "sr": c.sr, "n_fft": c.n_fft, "hop": c.hop,
             "n_mels": c.n_mels, "n_mfcc": c.n_mfcc, "fmin": c.fmin, "fmax": c.fmax,
             "htk": c.htk, "center": False, "window": "hann", "power": 2.0,
             "dct_type": 2, "dct_norm": "ortho", "slice_len": c.slice_len,
             "n_frames": c.n_frames, "drop_c0": c.drop_c0, "clip": c.clip,
             "mel_nnz": mel_nnz(c)}
    d.update(c.extra)
    return "feature_config.json", d


def normalise(cfg, Xtr, others):
    """z-score with the training set's per-column statistics, then clip.
    -> (mu, sigma, [Xtr, *others] normalised)."""
    sets = [Xtr] + list(others)
    if cfg.feature == "mfcc" and cfg.drop_c0:
        sets = [A[:, :, 1:] if A.size else A for A in sets]
    w = sets[0].shape[-1]
    if cfg.feature == "raw":             # every window is already in [-1, 1]
        return np.zeros(w, np.float32), np.ones(w, np.float32), sets
    flat = sets[0].reshape(-1, w)
    mu, sigma = flat.mean(0), flat.std(0) + 1e-8
    sets = [np.clip((A - mu) / sigma, -cfg.clip, cfg.clip).astype(np.float32)
            if A.size else A for A in sets]
    return mu.astype(np.float32), sigma.astype(np.float32), sets


def prepare(cfg, progress=None, log=print):
    """Runs the whole preparation. `progress(done, total, text)` is called once
    per file read and may return False to cancel. -> summary dict."""
    cfg = cfg.resolved()
    if not cfg.out_dir:
        raise PrepError("no output folder")
    rng = np.random.default_rng(cfg.seed)
    cfg_name, cfg_dict = feature_config(cfg)          # also validates the filters
    rows, classes, skipped = discover(cfg)
    tr_rows, va_rows, te_rows, note = split_files(rows, classes, cfg, rng)
    log(f"{len(rows)} recordings, {len(classes)} classes ({', '.join(classes)}); {note}")

    tables = [r for r in rows if r[0].suffix.lower() in TABLE_EXT]
    measure = bool(tables) and cfg.full_scale is None
    whole = bool(va_rows) or cfg.val_from_test     # training files used start to end
    total = ((len(tr_rows) + len(va_rows) if whole else 2 * len(tr_rows))
             + len(te_rows) + (len(tables) if measure else 0))
    state = {"done": 0}

    def tick(source):
        state["done"] += 1
        if progress and progress(state["done"], total, source) is False:
            raise Cancelled()

    peak = None
    if measure:                          # the recordings' own full scale
        peak = 0.0
        for path, _l, _g, source in tables:
            tick("peak of " + source)
            x = read_table(path, cfg)
            peak = max(peak, float(np.abs(x).max()) if len(x) else 0.0)
        cfg.full_scale = nice_full_scale(peak)
        log(f"full scale {cfg.full_scale:g} (largest peak in the recordings {peak:.4g})")
    if tables:
        cfg_dict["full_scale"] = cfg.full_scale

    featurize = make_featurizer(cfg)
    if cfg.val_from_test:                # validation = the test windows (filled in below)
        Xtr, ytr, s_tr, short_tr = build(tr_rows, cfg, rng, featurize, True, tick=tick)
        Xva = None
    elif va_rows:                        # validation = whole held-out groups
        Xtr, ytr, s_tr, short_tr = build(tr_rows, cfg, rng, featurize, True, tick=tick)
        Xva, yva, s_va, _ = build(va_rows, cfg, rng, featurize, False, dense=True, tick=tick)
    else:                                # validation = the tail of every training file
        Xtr, ytr, s_tr, short_tr = build(tr_rows, cfg, rng, featurize, True,
                                         ("head", cfg.val_tail), tick=tick)
        Xva, yva, s_va, _ = build(tr_rows, cfg, rng, featurize, False,
                                  ("tail", cfg.val_tail), dense=True, tick=tick)
    Xte, yte, s_te, short_te = build(te_rows, cfg, rng, featurize, False, tick=tick)
    if Xva is None:
        Xva, yva, s_va = Xte, yte, s_te
    if not len(Xtr):
        raise PrepError(f"no training window: every recording is shorter than "
                        f"{cfg.slice_len / cfg.sr / (1 - cfg.val_tail):.2f} s")
    if not len(Xva):
        raise PrepError(
            f"no validation window: the last {100 * cfg.val_tail:.0f}% of a recording "
            f"must hold two windows ({2 * cfg.slice_len / cfg.sr / cfg.val_tail:.1f} s "
            f"recordings at this window length); use longer recordings or a shorter window")
    for li, c in enumerate(classes):
        if not (ytr == li).any() or not (yva == li).any():
            raise PrepError(f"class '{c}' has no training or no validation window "
                            f"(recordings too short?)")

    gold = (golden_windows(te_rows, cfg, classes, featurize) if te_rows else
            golden_windows(va_rows, cfg, classes, featurize) if va_rows else
            golden_windows(tr_rows, cfg, classes, featurize, ("tail", cfg.val_tail)))
    Xg = gold[3] if gold else np.zeros((0, 0, 0), np.float32)
    mu, sigma, (Xtr, Xva, Xte, Xg) = normalise(cfg, Xtr, [Xva, Xte, Xg])
    out_dir = Path(cfg.out_dir)
    (out_dir / "folds").mkdir(parents=True, exist_ok=True)
    out = dict(X_train=Xtr, y_train=ytr, src_train=s_tr,
               X_val=Xva, y_val=yva, src_val=s_va,
               mu=mu, sigma=sigma, classes=np.array(classes))
    if len(Xte):
        out.update(X_test=Xte, y_test=yte, src_test=s_te)
    if cfg.val_from_test:                # readers must not call X_test "unseen"
        out["val_is_test"] = np.bool_(True)
    if gold:
        out.update(golden_adc=gold[0], golden_y=gold[1], golden_src=gold[2],
                   golden_ref=Xg, golden_bits=np.int64(GOLDEN_BITS))
    npz = out_dir / "folds" / f"{cfg.name}.npz"
    np.savez_compressed(npz, **out)
    with open(out_dir / cfg_name, "w") as f:
        json.dump(cfg_dict, f, indent=2)

    counts = {c: {"train": int((ytr == li).sum()), "val": int((yva == li).sum()),
                  "test": int((yte == li).sum()) if len(Xte) else 0}
              for li, c in enumerate(classes)}
    # A sanity check, NOT a result: nearest class mean. With time-frequency
    # features even this should be far above chance if the labels are right.
    # (With raw input it sits near chance, and that is normal.)
    cent = np.stack([Xtr[ytr == li].reshape(-1, Xtr[0].size).mean(0)
                     for li in range(len(classes))])
    sanity = {}
    for nm, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        if len(X):
            d = ((X.reshape(len(X), 1, -1) - cent[None]) ** 2).sum(-1)
            sanity[nm] = float(100.0 * (d.argmin(1) == y).mean())
    summary = {"dataset": str(npz), "config": str(out_dir / cfg_name),
               "input_h": int(Xtr.shape[1]), "input_w": int(Xtr.shape[2]),
               "classes": classes, "counts": counts, "split": note,
               "files": {"train_val": len(tr_rows) + len(va_rows), "test": len(te_rows),
                         "outside_class_folders": skipped,
                         "too_short": short_tr + short_te},
               "window_ms": 1000.0 * cfg.slice_len / cfg.sr,
               "held_out_scoring": bool(cfg.val_from_test),
               "golden_windows": int(len(gold[0])) if gold else 0,
               "full_scale": cfg.full_scale if tables else None, "peak": peak,
               "nearest_mean_pct": sanity}
    log(f"{npz.name}: input {summary['input_h']}x{summary['input_w']} | "
        f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}"
        + (f" | {short_tr + short_te} recording(s) too short" if short_tr + short_te else ""))
    return summary


# ============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Recordings in class folders -> ADOT dataset.npz + feature_config.json")
    ap.add_argument("root", nargs="?", help="folder holding the class folders")
    ap.add_argument("out_dir", nargs="?", help="where folds/ and feature_config.json go")
    ap.add_argument("--config", help="JSON file with any PrepConfig fields")
    ap.add_argument("--feature", choices=FEATURES)
    ap.add_argument("--sr", type=int)
    ap.add_argument("--input-sr", type=int, dest="input_sr")
    ap.add_argument("--full-scale", type=float, dest="full_scale")
    ap.add_argument("--n-fft", type=int, dest="n_fft")
    ap.add_argument("--hop", type=int)
    ap.add_argument("--slice-len", type=int, dest="slice_len")
    ap.add_argument("--window-ms", type=float,
                    help="window length; rounded to n_fft + whole hops")
    ap.add_argument("--bands", type=int, dest="n_bands")
    ap.add_argument("--raw-cols", type=int, dest="raw_cols")
    ap.add_argument("--class-level", type=int, dest="class_level")
    ap.add_argument("--group-level", type=int, dest="group_level")
    ap.add_argument("--classes", help="comma-separated label order")
    ap.add_argument("--test-groups", dest="test_groups", help="comma-separated")
    ap.add_argument("--val-from-test", dest="val_from_test", action="store_const", const=True,
                    help="validate on the test set itself (no independent test left)")
    ap.add_argument("--val-groups", dest="val_groups",
                    help="comma-separated; default: the tail of every training file")
    ap.add_argument("--test-fraction", type=float, dest="test_fraction")
    ap.add_argument("--max-files", type=int, dest="max_files")
    a = vars(ap.parse_args(argv))
    d = {}
    config = a.pop("config")
    if config:
        with open(config) as f:
            d.update(json.load(f))
    window_ms = a.pop("window_ms")
    for k in ("classes", "test_groups", "val_groups"):
        if a.get(k):
            a[k] = [s.strip() for s in a[k].split(",")]
    d.update({k: v for k, v in a.items() if v is not None})
    cfg = PrepConfig(**d)
    if window_ms and cfg.feature != "raw":
        cfg.slice_len = fit_slice_len(window_ms, cfg.sr, cfg.n_fft, cfg.hop)
    if not cfg.root or not cfg.out_dir:
        ap.error("root and out_dir are required (as arguments or in --config)")
    try:
        s = prepare(cfg, progress=lambda i, n, t: print(f"\r{i}/{n}", end="", flush=True),
                    log=lambda m: print("\n" + m))
    except PrepError as e:
        raise SystemExit(f"ERROR: {e}")
    for c, n in s["counts"].items():
        print(f"  {c:12s} train {n['train']:6d} | val {n['val']:5d} | test {n['test']:5d}")
    print("  nearest-class-mean check:",
          ", ".join(f"{k} {v:.1f}%" for k, v in s["nearest_mean_pct"].items()))
    print(f"  {s['dataset']}\n  {s['config']}")


if __name__ == "__main__":
    main()
