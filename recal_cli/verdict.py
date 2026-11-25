"""
recal_cli.verdict
=================
Deploy-or-retrain verdict (decision rules V1 and V2 of the pre-registered validation
protocol, validation/PREREGISTRATION.md §5).

All quantities are computed from out-of-fold (OOF) predictions on the target cohort, as
means of per-fold AUROCs over one common stratified partition (raw model, wrapper and
oracle evaluated on the same folds):

    optimism_gap = AUROC_in − AUROC_oof
    total_gap    = AUROC_oracle − AUROC_raw          (oracle: target-trained, OOF)
    rr           = (AUROC_oof − AUROC_raw) / total_gap

rr_CI is a percentile bootstrap (patients resampled with replacement within folds,
stratified by outcome, together with their raw, OOF and oracle-OOF scores). It does NOT include the
variability of refitting the wrapper.

The thresholds below are PROVISIONAL HEURISTICS. They are not derived from data; their
validation is the object of the pre-registered study. Do not tune them on the cohort
being assessed.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Provisional thresholds (protocol §5.3–5.4)
OPTIMISM_OK = 0.02
OPTIMISM_MAX = 0.05
RR_LOW = 0.3
RR_HIGH = 0.7
SLOPE_RANGE = (0.7, 1.3)
MIN_EVENTS = 10

GREEN, AMBER, RED, INDETERMINATE = "GREEN", "AMBER", "RED", "INDETERMINATE"

INTENDED_USE = (
    "RECAL assesses whether a frozen binary classifier for numeric tabular data can be "
    "reused at a new site after input alignment and recalibration, using a small labelled "
    "target cohort and the source cohort. The verdict is decision support for the team "
    "responsible for the model; it is not a clinical validation. Thresholds are provisional "
    "heuristics. The verdict does not detect concept shift arising after the target cohort "
    "was collected, does not replace prospective monitoring, and is uncertain when the "
    "target cohort has few events."
)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration slope (logistic regression of y on logit p) and calibration-in-the-large
    intercept (slope fixed at 1, logit p as offset)."""
    y = np.asarray(y, dtype=int)
    lp = _logit(p)
    if y.min() == y.max():
        return np.nan, np.nan
    lr = LogisticRegression(C=1e12, max_iter=1000)  # effectively unpenalised, version-robust
    lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])
    # intercept with offset: Newton iterations on a single parameter
    a = 0.0
    for _ in range(200):
        mu = 1 / (1 + np.exp(-(a + lp)))
        g = np.sum(y - mu)
        h = np.sum(mu * (1 - mu))
        if h <= 1e-12:
            break
        step = float(np.clip(g / h, -5.0, 5.0))  # damped Newton (v0.4): no divergence on extreme p
        a += step
        if abs(step) < 1e-10:
            break
    return slope, float(a)


def fold_mean_auroc(y: np.ndarray, s: np.ndarray, fold_ids: np.ndarray | None) -> float:
    """Mean of per-fold AUROCs (folds with a single class are skipped). Pooling OOF
    predictions across folds mixes models with fold-specific calibration and alignment,
    which distorts the ranking; the fold mean avoids that (v0.3.0, Amendment A1)."""
    y = np.asarray(y, dtype=int)
    if fold_ids is None:
        return float(roc_auc_score(y, s))
    vals = []
    for k in np.unique(fold_ids[fold_ids >= 0]):
        m = (fold_ids == k) & ~np.isnan(s)
        if len(np.unique(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_recovery_ratio(
    y: np.ndarray,
    scores_raw: np.ndarray,
    scores_oof: np.ndarray,
    scores_oracle_oof: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    fold_ids: np.ndarray | None = None,
) -> dict:
    """Percentile bootstrap CI of the recovery ratio, keeping the correlation between the
    three AUROCs. With ``fold_ids`` the three AUROCs are fold means and patients are
    resampled within folds (stratified by outcome). Resamples with total_gap ≤ 0 have no
    defined ratio; they are excluded from the percentiles and their share is reported."""
    y = np.asarray(y, dtype=int)
    ok = ~(np.isnan(scores_raw) | np.isnan(scores_oof) | np.isnan(scores_oracle_oof))
    y, r, o, c = y[ok], scores_raw[ok], scores_oof[ok], scores_oracle_oof[ok]
    f = np.zeros(len(y), dtype=int) if fold_ids is None else np.asarray(fold_ids)[ok]
    strata = [np.where((f == k) & (y == v))[0] for k in np.unique(f) for v in (0, 1)]
    strata = [st for st in strata if len(st)]
    rng = np.random.default_rng(seed)
    ratios, undefined = [], 0
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(st, len(st)) for st in strata])
        fb = None if fold_ids is None else f[idx]
        a_r = fold_mean_auroc(y[idx], r[idx], fb)
        a_o = fold_mean_auroc(y[idx], o[idx], fb)
        a_c = fold_mean_auroc(y[idx], c[idx], fb)
        tot = a_c - a_r
        if not np.isfinite(tot) or tot <= 0:
            undefined += 1
            continue
        ratios.append((a_o - a_r) / tot)
    if not ratios:
        return {"ci": (np.nan, np.nan), "n_boot": n_boot, "share_undefined": 1.0}
    lo, hi = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci": (float(lo), float(hi)), "n_boot": n_boot,
            "share_undefined": undefined / n_boot}


def decide(
    n_events: int,
    auroc_raw: float,
    auroc_in: float,
    auroc_oof: float,
    auroc_oracle: float | None,
    rr_ci: tuple[float, float] | None,
    slope_oof: float,
) -> dict:
    """Apply rules V2 (primary) and V1 (secondary). Returns both verdicts with reasons."""
    optimism = auroc_in - auroc_oof
    res = {"optimism_gap": optimism, "slope_oof": slope_oof, "n_events": n_events,
           "thresholds": {"optimism_ok": OPTIMISM_OK, "optimism_max": OPTIMISM_MAX,
                          "rr_low": RR_LOW, "rr_high": RR_HIGH, "slope_range": SLOPE_RANGE,
                          "min_events": MIN_EVENTS, "status": "provisional heuristics"}}

    if n_events < MIN_EVENTS or auroc_oracle is None or np.isnan(auroc_oof):
        reason = (f"insufficient events (E={n_events} < {MIN_EVENTS})" if n_events < MIN_EVENTS
                  else "oracle or OOF estimate unavailable")
        res.update(total_gap=None, recovery_ratio=None,
                   v2={"verdict": INDETERMINATE, "reason": reason},
                   v1={"verdict": INDETERMINATE, "reason": reason})
        return res

    total = auroc_oracle - auroc_raw
    slope_ok = SLOPE_RANGE[0] <= slope_oof <= SLOPE_RANGE[1]
    res["total_gap"] = total

    if total <= 0:
        v = GREEN if (optimism <= OPTIMISM_OK and slope_ok) else AMBER
        why = "no discrimination gap (oracle ≤ raw); " + (
            "optimism and calibration acceptable" if v == GREEN else "optimism or calibration out of range")
        res.update(recovery_ratio=None, v2={"verdict": v, "reason": why}, v1={"verdict": v, "reason": why})
        return res

    rr = (auroc_oof - auroc_raw) / total
    res["recovery_ratio"] = rr
    res["recovery_ratio_ci"] = rr_ci

    # V1 — point estimates
    if optimism > OPTIMISM_MAX or rr < RR_LOW:
        v1 = (RED, "optimism > 0.05" if optimism > OPTIMISM_MAX else "rr < 0.3")
    elif optimism <= OPTIMISM_OK and rr >= RR_HIGH:
        v1 = (GREEN, "optimism ≤ 0.02 and rr ≥ 0.7")
    else:
        v1 = (AMBER, "between thresholds")
    res["v1"] = {"verdict": v1[0], "reason": v1[1]}

    # V2 — interval-aware
    lo, hi = rr_ci if rr_ci is not None else (np.nan, np.nan)
    if np.isnan(lo) or np.isnan(hi):
        v2 = (INDETERMINATE, "recovery-ratio CI unavailable")
    elif optimism > OPTIMISM_MAX or hi < RR_LOW:
        v2 = (RED, "optimism > 0.05" if optimism > OPTIMISM_MAX else "rr CI entirely < 0.3")
    elif optimism <= OPTIMISM_OK and lo >= RR_HIGH and slope_ok:
        v2 = (GREEN, "optimism ≤ 0.02, rr CI ≥ 0.7, OOF slope in range")
    elif lo >= RR_LOW and hi <= RR_HIGH:
        v2 = (AMBER, "rr CI within [0.3, 0.7] (mixed drift)")
    else:
        v2 = (INDETERMINATE, "interval crosses a threshold or another condition fails")
    res["v2"] = {"verdict": v2[0], "reason": v2[1]}
    return res
