"""Tests for recal_cli.verdict (decision rules V1/V2, protocol §5)."""

import numpy as np
import pytest

from recal_cli.verdict import (
    AMBER,
    GREEN,
    INDETERMINATE,
    RED,
    bootstrap_recovery_ratio,
    calibration_slope_intercept,
    decide,
)


def _d(**kw):
    base = dict(n_events=50, auroc_raw=0.70, auroc_in=0.81, auroc_oof=0.80,
                auroc_oracle=0.82, rr_ci=(0.75, 0.90), slope_oof=1.0)
    base.update(kw)
    return decide(**base)


def test_green_when_all_conditions_hold():
    r = _d()
    assert r["v2"]["verdict"] == GREEN
    assert r["v1"]["verdict"] == GREEN
    assert r["recovery_ratio"] == pytest.approx(0.10 / 0.12)


def test_insufficient_events_is_indeterminate():
    r = _d(n_events=9)
    assert r["v2"]["verdict"] == INDETERMINATE
    assert r["v1"]["verdict"] == INDETERMINATE


def test_large_optimism_is_red():
    r = _d(auroc_in=0.90)  # optimism 0.10
    assert r["v2"]["verdict"] == RED
    assert r["v1"]["verdict"] == RED


def test_ci_crossing_threshold_is_indeterminate_in_v2_only():
    r = _d(rr_ci=(0.60, 0.95))
    assert r["v1"]["verdict"] == GREEN
    assert r["v2"]["verdict"] == INDETERMINATE


def test_ci_inside_mixed_band_is_amber():
    r = _d(auroc_oof=0.76, auroc_in=0.765, rr_ci=(0.35, 0.65))
    assert r["v2"]["verdict"] == AMBER


def test_upper_ci_below_low_threshold_is_red():
    r = _d(auroc_oof=0.72, auroc_in=0.725, rr_ci=(-0.1, 0.25))
    assert r["v2"]["verdict"] == RED


def test_slope_out_of_range_blocks_green_in_v2():
    r = _d(slope_oof=1.6)
    assert r["v1"]["verdict"] == GREEN
    assert r["v2"]["verdict"] != GREEN


def test_no_discrimination_gap():
    r = _d(auroc_oracle=0.69)
    assert r["recovery_ratio"] is None
    assert r["v2"]["verdict"] == GREEN
    r = _d(auroc_oracle=0.69, slope_oof=2.0)
    assert r["v2"]["verdict"] == AMBER


def test_calibration_slope_recovers_known_values():
    rng = np.random.default_rng(0)
    lp = rng.normal(0, 1.5, 20000)
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.5 + 0.6 * lp))))
    p = 1 / (1 + np.exp(-lp))
    slope, _ = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(0.6, abs=0.05)
    y2 = rng.binomial(1, 1 / (1 + np.exp(-(lp - 0.8))))
    _, citl = calibration_slope_intercept(y2, p)
    assert citl == pytest.approx(-0.8, abs=0.06)


def test_bootstrap_ratio_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.binomial(1, 0.3, n)
    signal = y + rng.normal(0, 1, n)
    raw = signal + rng.normal(0, 1.5, n)
    oof = signal + rng.normal(0, 0.8, n)
    oracle = signal + rng.normal(0, 0.5, n)
    from sklearn.metrics import roc_auc_score
    a_r, a_o, a_c = (roc_auc_score(y, s) for s in (raw, oof, oracle))
    rr = (a_o - a_r) / (a_c - a_r)
    b = bootstrap_recovery_ratio(y, raw, oof, oracle, n_boot=300, seed=0)
    lo, hi = b["ci"]
    assert lo < rr < hi


def test_fold_mean_auroc_ignores_cross_fold_calibration():
    from recal_cli.verdict import fold_mean_auroc
    rng = np.random.default_rng(3)
    n = 400
    y = rng.binomial(1, 0.4, n)
    s = y + rng.normal(0, 1, n)
    f = np.repeat(np.arange(4), n // 4)
    shifted = s + f * 5.0  # fold-specific monotone shift: within-fold ranking unchanged
    from sklearn.metrics import roc_auc_score
    assert fold_mean_auroc(y, shifted, f) == pytest.approx(fold_mean_auroc(y, s, f))
    assert roc_auc_score(y, shifted) != pytest.approx(roc_auc_score(y, s), abs=1e-3)
