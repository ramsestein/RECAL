"""RECAL 0.4 selection: picks recalibration when the frozen model is miscalibrated."""

import pickle

import numpy as np

from recal_cli import adapt_select as A


class _ToyModel:
    def __init__(self, shift):
        self.shift = shift

    def predict_proba(self, X):
        z = X[:, 0] * 1.5 + self.shift
        return 1 / (1 + np.exp(-z))


class _Pair:
    def __init__(self, X, y):
        self.X_t, self.y_t = X, y


def test_calibrator_picklable_and_exact():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.01, 0.99, 500)
    y = rng.binomial(1, s)
    for k in A.CALIBRATORS:
        c = A.fit_calibrator(k, y, s)
        c2 = pickle.loads(pickle.dumps(c))
        np.testing.assert_allclose(c(s), c2(s))


def test_selection_prefers_recalibration_for_shifted_model(monkeypatch):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 1))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, 0] * 1.5 - 1.5))))
    model = _ToyModel(shift=+1.0)  # predicts far too high risk
    monkeypatch.setattr(A, "TRANSFORMS", ["identity"])
    monkeypatch.setattr(A, "_make_pair_with_target_subset", lambda p, idx: _Pair(p.X_t[idx], p.y_t[idx]))
    sel = A.select_adaptation(_Pair(X, y), model, ["x"], drift_fn=None, n_splits=5, seed=0)
    assert sel["chosen"][1] in ("intercept", "logistic")
    assert sel["chosen"][1] != "none"
