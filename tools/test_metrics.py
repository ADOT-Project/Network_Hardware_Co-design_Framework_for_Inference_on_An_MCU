"""Checks the in-tree confusion_matrix / f1_score against scikit-learn.

scikit-learn is NOT a runtime dependency any more -- dropping it (and scipy
with it) took 152 MB out of the installer. This test is the record that the
replacements behave identically, so run it in a dev environment that still
has scikit-learn installed:

    python tools/test_metrics.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from sklearn.metrics import confusion_matrix as sk_cm
    from sklearn.metrics import f1_score as sk_f1
except ImportError:
    sys.exit("scikit-learn is not installed here; nothing to compare against.")

import edge_ai_train_model_gui_quantization_according_succes_rate as M


def main():
    rng = np.random.default_rng(0)
    cases = []

    # the shapes the search actually produces: 3 classes, a few hundred
    # validation slices, plus degenerate runs a collapsing architecture hits
    for n_cls in (2, 3, 5):
        for n in (1, 7, 300, 2000):
            cases.append((rng.integers(0, n_cls, n), rng.integers(0, n_cls, n)))

    # a model that predicts one class only -- the degenerate case that makes
    # precision undefined for every other class (zero_division territory)
    y = rng.integers(0, 3, 400)
    cases.append((y, np.zeros_like(y)))
    # a perfect model
    cases.append((y, y.copy()))
    # a class that never appears in y_true but does in y_pred
    cases.append((np.zeros(50, dtype=int), rng.integers(0, 3, 50)))

    bad = 0
    for i, (yt, yp) in enumerate(cases):
        n_cls = int(max(yt.max(initial=0), yp.max(initial=0))) + 1
        labels = list(range(n_cls))

        mine_cm = M.confusion_matrix(yt, yp, labels=labels)
        theirs_cm = sk_cm(yt, yp, labels=labels)
        if not np.array_equal(mine_cm, theirs_cm):
            print(f"  case {i}: confusion_matrix differs\n{mine_cm}\n{theirs_cm}")
            bad += 1

        mine_f1 = M.f1_score(yt, yp, average="macro", zero_division=0)
        theirs_f1 = sk_f1(yt, yp, average="macro", zero_division=0)
        if abs(mine_f1 - theirs_f1) > 1e-12:
            print(f"  case {i}: f1 {mine_f1!r} vs sklearn {theirs_f1!r}")
            bad += 1

    print(f"{len(cases)} cases checked, {bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
