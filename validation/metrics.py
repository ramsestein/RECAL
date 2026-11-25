"""Hold-out metrics and ground-truth rule (protocol v1.0, §5.6 and §7)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from recal_cli.oracle import _delong_auroc_ci
from recal_cli.verdict import calibration_slope_intercept

NB_THRESHOLDS = np.round(np.arange(0.05, 0.3001, 0.01), 2)


def ece_equal_frequency(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    n = len(p)
    return float(sum(len(b) / n * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def ici(y: np.ndarray, p: np.ndarray) -> float:
    """Integrated calibration index (Austin & Steyerberg): mean |loess(y~p) − p|."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    rng = float(p.max() - p.min()) or 1.0
    fit = lowess(y, p, frac=2 / 3, it=0, delta=0.005 * rng, return_sorted=False)
    return float(np.mean(np.abs(fit - p)))


def threshold_metrics(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"threshold": float(t), "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)}


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds=NB_THRESHOLDS) -> dict:
    n = len(y)
    out = {}
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)) / n
        fp = np.sum(pred & (y == 0)) / n
        out[f"{t:.2f}"] = float(tp - fp * t / (1 - t))
    return out


def holdout_metrics(y: np.ndarray, p: np.ndarray, class_threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    auc, lo, hi = _delong_auroc_ci(y, p)
    slope, citl = calibration_slope_intercept(y, p)
    return {
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "auroc": auc, "auroc_ci": [lo, hi],
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "citl": citl, "slope": slope,
        "ece10": ece_equal_frequency(y, p),
        "ici": ici(y, p),
        **threshold_metrics(y, p, class_threshold),
        "net_benefit": net_benefit(y, p),
    }


def deployable(m: dict, auroc_reference: float, delta: float = 0.02,
               slope_range=(0.7, 1.3), citl_max: float = 0.2) -> bool:
    """Ground truth (§5.6): discrimination within δ of the Reference model and acceptable
    calibration on the Target-Holdout."""
    return bool(m["auroc"] >= auroc_reference - delta
                and slope_range[0] <= m["slope"] <= slope_range[1]
                and abs(m["citl"]) <= citl_max)
