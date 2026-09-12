"""
Edge AI NAS tool - BLDC motor acoustic classification (C2000 / F28379D target)

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

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog,
                             QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QPushButton, QTextEdit, QVBoxLayout,
                             QWidget)
from sklearn.metrics import confusion_matrix, f1_score

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
    arr = np.asarray(arr)
    shape = "x".join(map(str, arr.shape))
    arr = arr.flatten()
    if is_label:
        f.write(f"const int32_t {name}[{len(arr)}] = {{ "
                + ", ".join(str(int(v)) for v in arr) + " };\n\n")
        return
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

    def __init__(self, max_channels, layers, quant_mode, input_h, input_w,
                 num_classes, target_val, metric, pool_w, dataset_path):
        super().__init__()
        self.max_channels = max_channels
        self.layers = layers
        self.quant_mode = quant_mode
        self.input_h, self.input_w = input_h, input_w
        self.num_classes = num_classes
        self.target_val = target_val
        self.metric = metric          # "Macro-F1" | "Confident Acc"
        self.pool_w = pool_w
        self.dataset_path = dataset_path
        self.max_epochs = 40
        self.patience = 10
        self.class_w = None
        self.pert_rng = np.random.default_rng(SEED)

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
    def train_one(self, ch, tr, va, seed=SEED, log=True):
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
        for i in range(N_SEEDS):
            m, clean = self.train_one(ch, tr, va, seed=SEED + i, log=(i == 0))
            pert = self.evaluate_perturbed(m, va)
            seed_worst = min(clean, pert)
            worst = seed_worst if worst is None else min(worst, seed_worst)
            if i == 0:
                canonical_model, canonical_clean = m, clean
        return canonical_model, canonical_clean, worst

    # ----------------------------------------------------------------- run
    def run(self):
        try:
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
            self.out_dir = f"./export/{stem}__{tag}"
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

            self.update_signal.emit(
                "<br><span style='color:#3498db;'><b>--- PHASE 1: coarse search "
                "(10-channel steps) ---</b></span>")
            ch = min(10, self.max_channels)
            while True:
                self.update_signal.emit(
                    f"<br><i>Trying {ch} channels ({N_SEEDS} seeds x clean+perturbed val, "
                    f"worst-case scored)...</i>")
                m, clean, s = self.robust_train_and_score(ch, tr, va)
                best_seen = max(best_seen, s)
                self.update_signal.emit(
                    f"  clean {clean:.1f}% | robust (worst-case) {s:.1f}%")
                if s >= self.target_val:
                    self.update_signal.emit(
                        f"<span style='color:#f1c40f;'><b>Target met at {ch} channels "
                        f"&mdash; phase 2</b></span>")
                    best_model, best_ch, target_met = m, ch, True
                    break
                if ch >= self.max_channels:
                    best_model, best_ch = m, ch
                    break
                ch = min(ch + 10, self.max_channels)

            if target_met and best_ch > 1:
                self.update_signal.emit(
                    "<br><span style='color:#e67e22;'><b>--- PHASE 2: step-by-step "
                    "reduction ---</b></span>")
                for fine in range(best_ch - 1, max(0, best_ch - 10), -1):
                    self.update_signal.emit(f"<br><i>{fine} channels...</i>")
                    m, clean, s = self.robust_train_and_score(fine, tr, va)
                    self.update_signal.emit(
                        f"  clean {clean:.1f}% | robust (worst-case) {s:.1f}%")
                    if s < self.target_val:
                        self.update_signal.emit(
                            f"<span style='color:#e74c3c;'>Robust target failed at {fine} "
                            f"channels ({s:.1f}%). Smallest viable: {best_ch} channels.</span>")
                        break
                    if s < best_seen - DEGRADE_TOL:
                        self.update_signal.emit(
                            f"<span style='color:#e74c3c;'>{fine} channels still meet the "
                            f"robust target ({s:.1f}%) but drop more than {DEGRADE_TOL} pts "
                            f"below the best seen ({best_seen:.1f}%). Stopping at "
                            f"{best_ch} channels.</span>")
                        break
                    self.update_signal.emit(
                        f"<span style='color:#2ecc71;'>{fine} channels also robust "
                        f"({s:.1f}%), continuing to shrink</span>")
                    best_model, best_ch = m, fine
                    best_seen = max(best_seen, s)

            if not target_met:
                self.update_signal.emit(
                    "<br><span style='color:#e74c3c;'><b>Target not reached within hardware "
                    "limits; exporting best attempt.</b></span>")

            self.optimal_arch_found_signal.emit(best_ch)

            # ------------- final evaluation + quantization check ------------
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

            if self.quant_mode != "FLOAT32":
                qm = quant_dequant_model(best_model, self.quant_mode)
                qa, _, qf1, _, _ = self.evaluate(qm, va)
                delta = qf1 - f1
                col = "#2ecc71" if abs(delta) < 1.0 else "#e74c3c"
                self.update_signal.emit(
                    f"<span style='color:{col};'>After {self.quant_mode}: "
                    f"acc {qa:.1f}% | macro-F1 {qf1:.1f}% "
                    f"(delta {delta:+.1f} pts)</span>")

            # -------------------------------------------------- write files -
            torch.save(best_model.state_dict(),
                       f"{self.out_dir}/edge_cnn_{self.quant_mode.lower()}_{best_ch}ch.pth")

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

                i = 1
                for layer in best_model.features:
                    if isinstance(layer, nn.Conv2d):
                        write_array(f, f"conv{i}_weights",
                                    layer.weight.detach().cpu().numpy(), self.quant_mode)
                        write_array(f, f"conv{i}_bias",
                                    layer.bias.detach().cpu().numpy(), self.quant_mode)
                        i += 1
                write_array(f, "fc_weights",
                            best_model.fc.weight.detach().cpu().numpy(), self.quant_mode)
                write_array(f, "fc_bias",
                            best_model.fc.bias.detach().cpu().numpy(), self.quant_mode)
                f.write("#endif\n")

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

            # ---- self-documenting info.txt: what this model is/isn't valid for ----
            with open(f"{self.out_dir}/info.txt", "w", encoding="utf-8") as f:
                f.write(f"dataset: {self.dataset_path}\n")
                f.write(f"quant: {self.quant_mode} | channels: {best_ch} | "
                        f"layers: {self.layers} | pool_w: {self.pool_w}\n")
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
                f"<br><span style='color:#2ecc71;'><b>Done. model_weights, test_samples and "
                f"info.txt are ready in {self.out_dir}.</b></span>")
            self.finished_signal.emit()

        except Exception as e:
            import traceback
            self.update_signal.emit(
                f"<br><span style='color:#e74c3c;'><b>ERROR: {e}</b></span>"
                f"<pre>{traceback.format_exc()}</pre>")
            self.finished_signal.emit()


# ==========================================
# 4. GUI
# ==========================================
class EdgeAiGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge AI NAS - BLDC acoustic classification")
        self.setMinimumSize(1200, 850)
        self.dataset_path = None
        self.init_ui()

    # ------------------------------------------------------------------ ui
    def init_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        main = QHBoxLayout(cw)

        left = QVBoxLayout()
        box = QGroupBox("Dataset, hardware limits and network parameters")
        grid = QVBoxLayout()

        ds = QHBoxLayout()
        self.dataset_edit = QLineEdit(); self.dataset_edit.setReadOnly(True)
        self.dataset_edit.setPlaceholderText("Select folds/fold0.npz ...")
        bbtn = QPushButton("Browse"); bbtn.setFixedWidth(70)
        bbtn.clicked.connect(self.browse_dataset)
        ds.addWidget(QLabel("Dataset:")); ds.addWidget(self.dataset_edit); ds.addWidget(bbtn)
        grid.addLayout(ds)

        q = QHBoxLayout(); q.addWidget(QLabel("Quantization:"))
        self.quant_combo = QComboBox(); self.quant_combo.addItems(["FLOAT32", "INT16", "INT8"])
        q.addWidget(self.quant_combo); grid.addLayout(q)

        mt = QHBoxLayout(); mt.addWidget(QLabel("Target metric:"))
        self.metric_combo = QComboBox(); self.metric_combo.addItems(["Macro-F1", "Confident Acc"])
        mt.addWidget(self.metric_combo); grid.addLayout(mt)

        self.target_input = self.create_input("Target value (%):", "95")
        self.input_shape_input = self.create_input("Input shape (H,W):", "32,12")
        self.num_classes_input = self.create_input("Number of classes:", "3")
        self.layers_input = self.create_input("Number of layers (1-2):", "2")
        self.flash_input = self.create_input("Max Flash (KB):", "128")
        self.conv_sram_input = self.create_input("Conv SRAM (KB):", "32")
        self.fc_sram_input = self.create_input("Dense SRAM (KB):", "8")
        for i in [self.target_input, self.input_shape_input, self.num_classes_input,
                  self.layers_input, self.flash_input,
                  self.conv_sram_input, self.fc_sram_input]:
            grid.addLayout(i[0])

        self.arena_check = QCheckBox("Ultimate Arena")
        self.arena_check.setChecked(True)
        self.arena_check.setToolTip(
            "Reuse one buffer sized to the largest intermediate tensor instead of "
            "summing every layer's output. Input buffer excluded.")

        self.poolw_check = QCheckBox("2D pooling")
        self.poolw_check.setToolTip(
            "Unchecked: MaxPool(2,1) - pools time only.\n"
            "Checked:   MaxPool(2,2) - pools the coefficient axis as well.\n"
            "With a 32x12 MFCC input and 2 layers, 2D pooling collapses the width "
            "from 12 to 1.")

        for c in (self.arena_check, self.poolw_check):
            grid.addWidget(c)

        box.setLayout(grid)

        self.calc_btn = QPushButton("1. CALCULATE HARDWARE CEILING")
        self.calc_btn.setFixedHeight(50); self.calc_btn.setEnabled(False)
        self.calc_btn.setStyleSheet("background-color:#27ae60;color:white;font-weight:bold;")
        self.calc_btn.clicked.connect(self.run_logic)

        self.train_btn = QPushButton("2. START SEARCH / TRAINING")
        self.train_btn.setFixedHeight(50); self.train_btn.setEnabled(False)
        self.train_btn.setStyleSheet("background-color:#2980b9;color:white;font-weight:bold;")
        self.train_btn.clicked.connect(self.start_training)

        left.addWidget(box); left.addWidget(self.calc_btn); left.addWidget(self.train_btn)
        left.addStretch()

        right = QVBoxLayout()
        self.log_output = QTextEdit(); self.log_output.setFixedHeight(260)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color:#121212;color:#00FF00;font-family:Consolas;")
        self.diagram_container = QGroupBox("Architecture and step-wise memory usage")
        self.diagram_layout = QVBoxLayout(); self.diagram_container.setLayout(self.diagram_layout)
        self.arena_status_label = QLabel("Status: waiting for dataset")
        self.arena_status_label.setAlignment(Qt.AlignCenter)

        right.addWidget(QLabel("<b>Analysis and training log:</b>"))
        right.addWidget(self.log_output)
        right.addWidget(self.diagram_container)
        right.addWidget(self.arena_status_label)

        main.addLayout(left, 1); main.addLayout(right, 3)

    def create_input(self, label, default):
        lay = QHBoxLayout(); edit = QLineEdit(default)
        lay.addWidget(QLabel(label)); lay.addWidget(edit)
        return lay, edit

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
            self.dataset_edit.setText(path)
            self.input_shape_input[1].setText(f"{h},{w}")
            self.num_classes_input[1].setText(str(len(classes)))
            self.log_output.append(
                f"<b>Dataset:</b> {os.path.basename(path)}<br>"
                f"train {n} | val {len(d['y_val'])} | input {h}x{w} | classes {classes}<br>"
                + ("No X_test &mdash; final-model set (accuracy estimate comes from the folds)"
                   if "X_test" not in d.files else
                   "X_test present but <b>will not be used</b>; selection uses X_val"))
            self.calc_btn.setEnabled(True); self.train_btn.setEnabled(False)
        except Exception as e:
            self.log_output.append(
                f"<span style='color:#e74c3c;'>Could not read file: {e}</span>")

    # ------------------------------------------------------------- memory
    def dtype_size(self, mode):
        return {"FLOAT32": 4, "INT16": 2, "INT8": 1}[mode]

    def calculate_metrics(self, ch, layers, arena, mode, ih, iw, ncls, pool_w):
        ds = self.dtype_size(mode)
        flash, in_ch, h, w = 0, 1, ih, iw
        input_ram = ih * iw * ds
        peak, cumul = 0, 0
        for _ in range(layers):
            flash += ((3 * 3 * in_ch * ch) + ch) * ds
            ch_h, ch_w = h - 2, w - 2
            conv_out = ch * ch_h * ch_w * ds
            ph, pw = ch_h // 2, (ch_w // 2 if pool_w else ch_w)
            pool_out = ch * ph * pw * ds
            if arena:
                peak = max(peak, conv_out, pool_out)
            else:
                cumul += conv_out + pool_out
            h, w, in_ch = ph, pw, ch
        conv_sram = input_ram + (peak if arena else cumul)
        fc_in = ch * h * w
        fc_sram = (fc_in + fc_in * ncls + ncls + ncls) * ds
        flash += (fc_in * ncls + ncls) * ds
        return flash, conv_sram, fc_sram, fc_in

    def clear_layout(self, lay):
        while lay.count():
            it = lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
            elif it.layout(): self.clear_layout(it.layout())

    def create_box(self, title, sub, f_kb, s_kb, color="#f1c40f", tag="SRAM"):
        b = QFrame()
        b.setStyleSheet(f"background-color:#222;border:2px solid {color};border-radius:8px;")
        b.setFixedWidth(190); b.setFixedHeight(120)
        l = QVBoxLayout(b)
        for txt, style in ((title, "font-weight:bold;color:white;border:none;"),
                           (sub, "font-size:10px;color:#bdc3c7;border:none;"),
                           (f"Flash: {f_kb:.2f}K\n{tag}: {s_kb:.2f}K",
                            "color:#2ecc71;border:none;font-size:9px;")):
            q = QLabel(txt); q.setAlignment(Qt.AlignCenter); q.setStyleSheet(style)
            l.addWidget(q)
        return b

    def update_diagram(self, ch, layers, arena, mode, ih, iw, ncls, pool_w):
        self.clear_layout(self.diagram_layout)
        ds = self.dtype_size(mode)

        cg = QGroupBox("Convolutional memory domain (Conv SRAM)")
        crow = QHBoxLayout(); crow.setAlignment(Qt.AlignLeft); cg.setLayout(crow)
        dg = QGroupBox("Dense memory domain (Dense SRAM)")
        drow = QHBoxLayout(); drow.setAlignment(Qt.AlignLeft); dg.setLayout(drow)

        crow.addWidget(self.create_box(f"{ih}x{iw}", "INPUT", 0,
                                       ih * iw * ds / 1024, "#95a5a6", "Conv SRAM"))
        h, w, in_ch = ih, iw, 1
        for i in range(layers):
            crow.addWidget(QLabel("->"))
            h, w = h - 2, w - 2
            crow.addWidget(self.create_box(
                f"{ch}x{h}x{w}", f"CONV {i+1}",
                ((3 * 3 * in_ch * ch) + ch) * ds / 1024,
                ch * h * w * ds / 1024, "#f1c40f", "Conv SRAM"))
            crow.addWidget(QLabel("->"))
            h, w = h // 2, (w // 2 if pool_w else w)
            crow.addWidget(self.create_box(
                f"{ch}x{h}x{w}", f"POOL {i+1}", 0,
                ch * h * w * ds / 1024, "#e67e22", "Conv SRAM"))
            in_ch = ch

        fc_in = ch * h * w
        drow.addWidget(self.create_box(f"{fc_in}x1", "FLATTEN", 0,
                                       fc_in * ds / 1024, "#3498db", "Dense SRAM"))
        drow.addWidget(QLabel(" X "))
        drow.addWidget(self.create_box(f"{ncls}x{fc_in}", "DENSE W", 0,
                                       (fc_in * ncls + ncls) * ds / 1024,
                                       "#9b59b6", "Dense SRAM"))
        drow.addWidget(QLabel(" = "))
        drow.addWidget(self.create_box(f"{ncls}x1", "OUTPUT", 0,
                                       ncls * ds / 1024, "#2ecc71", "Dense SRAM"))
        self.diagram_layout.addWidget(cg); self.diagram_layout.addWidget(dg)

    # -------------------------------------------------------------- actions
    def run_logic(self):
        try:
            sh = self.input_shape_input[1].text().split(",")
            self.input_h = int(sh[0].strip())
            self.input_w = int(sh[1].strip()) if len(sh) > 1 else self.input_h
            self.classes_val = int(self.num_classes_input[1].text())
            self.layers_val = int(self.layers_input[1].text())
            self.use_arena = self.arena_check.isChecked()
            self.pool_w = self.poolw_check.isChecked()
            self.quant_mode = self.quant_combo.currentText()
            f_lim = float(self.flash_input[1].text()) * 1024
            c_lim = float(self.conv_sram_input[1].text()) * 1024
            d_lim = float(self.fc_sram_input[1].text()) * 1024

            h, w = ScalableDiagramCNN.out_shape(
                self.input_h, self.input_w, self.layers_val, self.pool_w)
            if h <= 0 or w <= 0:
                self.log_output.append(
                    f"<span style='color:#e74c3c;'>Input {self.input_h}x{self.input_w} is too "
                    f"small for {self.layers_val} layers.</span>")
                return

            self.max_ch = 1
            while self.max_ch < 128:
                f, sc, sd, _ = self.calculate_metrics(
                    self.max_ch + 1, self.layers_val, self.use_arena, self.quant_mode,
                    self.input_h, self.input_w, self.classes_val, self.pool_w)
                if f > f_lim or sc > c_lim or sd > d_lim:
                    break
                self.max_ch += 1

            f, sc, sd, fi = self.calculate_metrics(
                self.max_ch, self.layers_val, self.use_arena, self.quant_mode,
                self.input_h, self.input_w, self.classes_val, self.pool_w)
            self.update_diagram(self.max_ch, self.layers_val, self.use_arena,
                                self.quant_mode, self.input_h, self.input_w,
                                self.classes_val, self.pool_w)
            self.diagram_container.setTitle(
                f"Hardware ceiling ({self.quant_mode}, KB)")
            self.log_output.append(
                f"<br><b>Ceiling: {self.max_ch} channels</b> &mdash; "
                f"Flash {f/1024:.2f}K | Conv SRAM {sc/1024:.2f}K | "
                f"Dense SRAM {sd/1024:.2f}K | FC input {fi}")
            self.arena_status_label.setText(
                f"{'Arena active' if self.use_arena else 'Arena disabled'} "
                f"({self.quant_mode})")
            self.train_btn.setEnabled(True)
        except Exception as e:
            self.log_output.append(
                f"<span style='color:#e74c3c;'>Input error: {e}</span>")

    def handle_optimal_architecture(self, ch):
        f, sc, sd, _ = self.calculate_metrics(
            ch, self.layers_val, self.use_arena, self.quant_mode,
            self.input_h, self.input_w, self.classes_val, self.pool_w)
        self.update_diagram(ch, self.layers_val, self.use_arena, self.quant_mode,
                            self.input_h, self.input_w, self.classes_val, self.pool_w)
        self.diagram_container.setTitle(f"Selected architecture ({self.quant_mode}, KB)")
        self.log_output.append(
            f"<br><b>Final footprint ({ch} channels):</b> Flash {f/1024:.2f}K | "
            f"Conv SRAM {sc/1024:.2f}K | Dense SRAM {sd/1024:.2f}K")

    def start_training(self):
        if not self.dataset_path:
            self.log_output.append(
                "<span style='color:#e74c3c;'>Select a dataset first.</span>")
            return
        self.train_btn.setEnabled(False); self.calc_btn.setEnabled(False)
        self.quant_mode = self.quant_combo.currentText()
        try:
            target = float(self.target_input[1].text())
        except ValueError:
            self.log_output.append(
                "<span style='color:#e74c3c;'>Target value must be numeric.</span>")
            self.train_btn.setEnabled(True); self.calc_btn.setEnabled(True)
            return

        self.thread = TrainingThread(
            self.max_ch, self.layers_val, self.quant_mode, self.input_h,
            self.input_w, self.classes_val, target,
            self.metric_combo.currentText(), self.pool_w, self.dataset_path)
        self.thread.update_signal.connect(self.log_output.append)
        self.thread.optimal_arch_found_signal.connect(self.handle_optimal_architecture)
        self.thread.finished_signal.connect(self.training_finished)
        self.thread.start()

    def training_finished(self):
        self.train_btn.setEnabled(True); self.calc_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = EdgeAiGui(); win.show()
    sys.exit(app.exec_())
