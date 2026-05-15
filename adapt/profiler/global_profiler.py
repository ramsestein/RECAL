"""
adapt.profiler.global_profiler
================================
Calcula los campos globales del DriftProfile: MMD², test de Fisher para
prevalencia, AUROC baseline con CI, calibración slope, ECE, CITL.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.stats import fisher_exact
from scipy.special import logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from adapt.profiler.constants import (
    BOOTSTRAP_N_AUROC,
    MMD2_N_PERMUTATIONS,
)

logger = logging.getLogger(__name__)


def profile_global(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    model,
) -> dict:
    """
    Calcula los campos globales del DriftProfile.

    Parameters
    ----------
    X_source : np.ndarray (n_s, p)
        Features source (puede contener NaN).
    y_source : np.ndarray (n_s,)
        Labels binarios source.
    X_target : np.ndarray (n_t, p)
        Features target (puede contener NaN).
    y_target : np.ndarray (n_t,)
        Labels binarios target.
    model : ModelWrapper
        Modelo source con predict_proba().

    Returns
    -------
    dict
        Campos globales del DriftProfile.
    """
    n_s = len(y_source)
    n_t = len(y_target)
    n_s_events = int(y_source.sum())
    n_t_events = int(y_target.sum())

    prev_s = n_s_events / n_s if n_s > 0 else float("nan")
    prev_t = n_t_events / n_t if n_t > 0 else float("nan")

    # Test de Fisher para prevalencia
    fisher_pvalue = _fisher_prevalence_test(n_s_events, n_s, n_t_events, n_t)

    # Ratio p/n en target
    n_t_neg = n_t - n_t_events
    p_n_ratio = n_t_events / n_t_neg if n_t_neg > 0 else float("inf")

    # MMD² con permutación
    mmd2, mmd2_pval = _mmd2_with_permutation(
        X_source, X_target, n_permutations=MMD2_N_PERMUTATIONS
    )

    # PCA varianza explicada (source estandarizado)
    pca_var = _pca_variance_explained(X_source, n_components=5)

    # Métricas baseline (modelo source → target raw)
    baseline_auroc, auroc_ci_low, auroc_ci_high = _bootstrap_auroc(
        y_target, model, X_target, n_boot=BOOTSTRAP_N_AUROC
    )

    cal_slope, ece, citl = _calibration_metrics(y_target, model, X_target)

    return {
        "n_source_obs": n_s,
        "n_target_obs": n_t,
        "n_source_events": n_s_events,
        "n_target_events": n_t_events,
        "prevalence_source": prev_s,
        "prevalence_target": prev_t,
        "prevalence_shift_pvalue": fisher_pvalue,
        "p_n_ratio_target": p_n_ratio,
        "mmd2_source_target": mmd2,
        "mmd2_pvalue": mmd2_pval,
        "pca_variance_explained": pca_var,
        "baseline_auroc": baseline_auroc,
        "baseline_auroc_ci_low": auroc_ci_low,
        "baseline_auroc_ci_high": auroc_ci_high,
        "baseline_calibration_slope": cal_slope,
        "baseline_ece": ece,
        "baseline_citl": citl,
    }


# ── Implementaciones internas ─────────────────────────────────────────────────

def _fisher_prevalence_test(
    n_pos_s: int,
    n_s: int,
    n_pos_t: int,
    n_t: int,
) -> float:
    """Test exacto de Fisher H0: prevalencia source == prevalencia target."""
    table = [
        [n_pos_s, n_s - n_pos_s],
        [n_pos_t, n_t - n_pos_t],
    ]
    _, pvalue = fisher_exact(table, alternative="two-sided")
    return float(pvalue)


def _mmd2_with_permutation(
    X_s: np.ndarray,
    X_t: np.ndarray,
    n_permutations: int = MMD2_N_PERMUTATIONS,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Calcula MMD² con kernel RBF y p-valor por permutación.

    Usa una submuestra de max_n=500 puntos por cohorte para velocidad
    (el MMD² es un U-statistic consistente bajo submuestreo).

    Returns
    -------
    (mmd2, pvalue)
    """
    rng = np.random.default_rng(seed)
    max_n = 500

    # Imputar NaN con media de source
    mu_s = np.nanmean(X_s, axis=0)
    X_s_imp = np.where(np.isnan(X_s), mu_s[np.newaxis, :], X_s)
    X_s_imp = np.nan_to_num(X_s_imp, nan=0.0)
    X_t_imp = np.where(np.isnan(X_t), mu_s[np.newaxis, :], X_t)
    X_t_imp = np.nan_to_num(X_t_imp, nan=0.0)

    # Estandarizar por source
    scaler = StandardScaler()
    X_s_std = scaler.fit_transform(X_s_imp)
    X_t_std = scaler.transform(X_t_imp)

    # Submuestrear
    n_s = min(len(X_s_std), max_n)
    n_t = min(len(X_t_std), max_n)
    idx_s = rng.choice(len(X_s_std), size=n_s, replace=False)
    idx_t = rng.choice(len(X_t_std), size=n_t, replace=False)
    Xs = X_s_std[idx_s]
    Xt = X_t_std[idx_t]

    # Bandwidth = mediana de las distancias pairwise
    bandwidth = _median_bandwidth(Xs, Xt)

    observed_mmd2 = _compute_mmd2(Xs, Xt, bandwidth)

    # Permutación
    combined = np.vstack([Xs, Xt])
    n_total = len(combined)
    perm_mmd2 = np.zeros(n_permutations)
    for i in range(n_permutations):
        idx = rng.permutation(n_total)
        perm_s = combined[idx[:n_s]]
        perm_t = combined[idx[n_s : n_s + n_t]]
        perm_mmd2[i] = _compute_mmd2(perm_s, perm_t, bandwidth)

    pvalue = float((perm_mmd2 >= observed_mmd2).mean())
    return float(observed_mmd2), pvalue


def _median_bandwidth(Xs: np.ndarray, Xt: np.ndarray) -> float:
    """Heurística de la mediana para el bandwidth del kernel RBF."""
    combined = np.vstack([Xs, Xt])
    # Submuestrear si es grande
    if len(combined) > 200:
        idx = np.random.choice(len(combined), 200, replace=False)
        combined = combined[idx]
    diff = combined[:, np.newaxis, :] - combined[np.newaxis, :, :]
    sq_dists = (diff ** 2).sum(axis=-1)
    upper_tri = sq_dists[np.triu_indices(len(combined), k=1)]
    median_dist = float(np.median(upper_tri))
    return max(median_dist, 1e-8)


def _compute_mmd2(Xs: np.ndarray, Xt: np.ndarray, bandwidth: float) -> float:
    """MMD² con kernel RBF (implementación O(n²) simplificada)."""
    def rbf_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
        sq = (diff ** 2).sum(axis=-1)
        return np.exp(-sq / (2 * bandwidth))

    Kss = rbf_kernel(Xs, Xs)
    Ktt = rbf_kernel(Xt, Xt)
    Kst = rbf_kernel(Xs, Xt)

    ns, nt = len(Xs), len(Xt)
    # Unbiased MMD²
    np.fill_diagonal(Kss, 0.0)
    np.fill_diagonal(Ktt, 0.0)
    mmd2 = (
        Kss.sum() / (ns * (ns - 1))
        + Ktt.sum() / (nt * (nt - 1))
        - 2 * Kst.mean()
    )
    return float(max(mmd2, 0.0))


def _pca_variance_explained(X_source: np.ndarray, n_components: int = 5) -> list:
    """
    Varianza explicada acumulada de los primeros n_components PCs sobre
    X_source estandarizado.
    """
    mu = np.nanmean(X_source, axis=0)
    X_imp = np.where(np.isnan(X_source), mu[np.newaxis, :], X_source)
    X_imp = np.nan_to_num(X_imp, nan=0.0)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_imp)

    k = min(n_components, X_std.shape[1], X_std.shape[0] - 1)
    pca = PCA(n_components=k, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pca.fit(X_std)
    cumvar = list(np.cumsum(pca.explained_variance_ratio_).tolist())
    # Pad si k < n_components
    while len(cumvar) < n_components:
        cumvar.append(cumvar[-1] if cumvar else 0.0)
    return cumvar[:n_components]


def _bootstrap_auroc(
    y_target: np.ndarray,
    model,
    X_target: np.ndarray,
    n_boot: int = BOOTSTRAP_N_AUROC,
    seed: int = 42,
) -> tuple[float, float, float]:
    """AUROC baseline + CI bootstrap estratificado."""
    proba = model.predict_proba(X_target)

    if len(np.unique(y_target)) < 2:
        logger.warning("_bootstrap_auroc: solo una clase en y_target.")
        return float("nan"), float("nan"), float("nan")

    auroc = float(roc_auc_score(y_target, proba))

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_target == 1)[0]
    neg_idx = np.where(y_target == 0)[0]
    aucs = []
    for _ in range(n_boot):
        ip = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        in_ = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ip, in_])
        try:
            aucs.append(float(roc_auc_score(y_target[idx], proba[idx])))
        except ValueError:
            pass
    if not aucs:
        return auroc, float("nan"), float("nan")
    aucs_arr = np.array(aucs)
    return auroc, float(np.percentile(aucs_arr, 2.5)), float(np.percentile(aucs_arr, 97.5))


def _calibration_metrics(
    y_target: np.ndarray,
    model,
    X_target: np.ndarray,
) -> tuple[float, float, float]:
    """
    Retorna (calibration_slope, ECE, CITL).

    calibration_slope: slope de regresión logística en logit scale.
    ECE: Expected Calibration Error (10 bins iguales).
    CITL: Calibration-in-the-large = media(proba) - prevalencia observada.
    """
    proba = model.predict_proba(X_target)

    # Calibration slope
    p_clip = np.clip(proba, 1e-7, 1 - 1e-7)
    slope = float("nan")
    try:
        lr = LogisticRegression(C=1e9, fit_intercept=True, max_iter=1000, solver="lbfgs")
        lr.fit(logit(p_clip).reshape(-1, 1), y_target)
        slope = float(lr.coef_[0][0])
    except Exception:
        pass

    # ECE
    ece = _compute_ece(y_target, proba, n_bins=10)

    # CITL
    prev_obs = float(y_target.mean())
    prev_pred = float(proba.mean())
    citl = prev_pred - prev_obs

    return slope, ece, citl


def _compute_ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        mask = (proba >= bins[i]) & (proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(float(y_true[mask].mean()) - float(proba[mask].mean()))
    return float(ece)


def _bootstrap_auroc_from_scores(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_boot: int = BOOTSTRAP_N_AUROC,
    seed: int = 42,
) -> tuple[float, float, float]:
    """AUROC + CI bootstrap a partir de scores pre-calculados."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan")
    auroc = float(roc_auc_score(y_true, proba))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_boot):
        ip = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        in_ = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([ip, in_])
        try:
            aucs.append(float(roc_auc_score(y_true[idx], proba[idx])))
        except ValueError:
            pass
    if not aucs:
        return auroc, float("nan"), float("nan")
    aucs_arr = np.array(aucs)
    return auroc, float(np.percentile(aucs_arr, 2.5)), float(np.percentile(aucs_arr, 97.5))


def _calibration_slope(
    y_true: np.ndarray, proba: np.ndarray
) -> tuple[float, float]:
    """Slope e intercept de calibración (regresión logística en logit scale).

    Returns
    -------
    (slope, intercept)
    """
    p_clip = np.clip(proba, 1e-7, 1 - 1e-7)
    slope, intercept = float("nan"), float("nan")
    try:
        lr = LogisticRegression(C=1e9, fit_intercept=True, max_iter=1000, solver="lbfgs")
        lr.fit(logit(p_clip).reshape(-1, 1), y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        pass
    return slope, intercept


def _ece_score(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """Wrapper de _compute_ece con firma simplificada."""
    return _compute_ece(y_true, proba, n_bins=n_bins)
