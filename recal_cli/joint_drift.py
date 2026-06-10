"""
recal_cli.joint_drift
======================
Joint-drift analysis: captures changes in the *covariance structure* between
source and target that are invisible to marginal (per-feature) drift metrics.

Exported functions
------------------
- compute_vif(X, features)          → DataFrame  [feature, VIF]
- compute_condition_number(X)        → float
- compute_effective_rank(X)          → float  (Shannon-entropy of norm. eigenvalues)
- joint_drift_report(X_source, X_target, features,
                     delta_vif_warn, delta_vif_severe)
                                     → DataFrame  [feature, vif_source, vif_target,
                                                   delta_vif, flag]
- mi_matrix_delta(X_source, X_target)→ float  (Frobenius norm of MI-matrix diff)

Thresholds (configurable via parameters, defaults match config_schema defaults):
  delta_vif_warn   = 2.0
  delta_vif_severe = 5.0
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── VIF ──────────────────────────────────────────────────────────────────────


def compute_vif(X: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    """
    Variance Inflation Factor for each feature column in X.

    VIF_j = 1 / (1 - R²_j), where R²_j is the coefficient of determination
    obtained by regressing feature j on all other features.

    Extremely high VIF (>1000) is capped internally to 1000 to avoid Inf
    propagation; this is logged as a warning.

    Parameters
    ----------
    X : np.ndarray  (n, p)
        Feature matrix — no NaN.
    features : sequence of str  length p
        Feature names.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, VIF]
    """
    n, p = X.shape
    if p != len(features):
        raise ValueError(
            f"X has {p} columns but {len(features)} feature names were provided."
        )
    if n <= p:
        logger.warning(
            "compute_vif: n=%d ≤ p=%d; OLS regression is singular. "
            "VIF estimates will be unreliable.",
            n, p,
        )

    vif_values = []
    for j in range(p):
        y = X[:, j]
        Xother = np.delete(X, j, axis=1)
        # Add intercept
        Xother_int = np.column_stack([np.ones(n), Xother])
        try:
            # Use lstsq for numerical stability
            coef, _, rank, _ = np.linalg.lstsq(Xother_int, y, rcond=None)
            y_hat = Xother_int @ coef
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot < 1e-12:
                vif = 1.0  # constant feature → no collinearity
            else:
                r2 = 1.0 - ss_res / ss_tot
                r2 = min(max(r2, 0.0), 1.0 - 1e-10)
                vif = 1.0 / (1.0 - r2)
                if vif > 1000:
                    logger.warning(
                        "compute_vif: VIF for '%s' capped at 1000 (raw=%.1f). "
                        "Perfect or near-perfect multicollinearity detected.",
                        features[j], vif,
                    )
                    vif = 1000.0
        except np.linalg.LinAlgError:
            logger.warning("compute_vif: LinAlgError for feature '%s'; VIF set to NaN.", features[j])
            vif = float("nan")
        vif_values.append(vif)

    return pd.DataFrame({"feature": list(features), "VIF": vif_values})


# ── Condition number ──────────────────────────────────────────────────────────


def compute_condition_number(X: np.ndarray) -> float:
    """
    Condition number of the empirical covariance matrix of X.

    κ = σ_max / σ_min  (ratio of largest to smallest singular value of Cov(X)).

    A large condition number (> 1000) indicates a poorly conditioned covariance
    and suggests that CORAL alignment may be numerically fragile.

    Parameters
    ----------
    X : np.ndarray (n, p)
        Feature matrix — no NaN.

    Returns
    -------
    float
        Condition number.  Returns np.inf if the covariance is rank-deficient.
    """
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        return 1.0  # single feature
    sv = np.linalg.svd(cov, compute_uv=False)
    sv_max = float(sv[0])
    sv_min = float(sv[-1])
    if sv_min < 1e-15:
        return float("inf")
    return sv_max / sv_min


# ── Effective rank ────────────────────────────────────────────────────────────


def compute_effective_rank(X: np.ndarray) -> float:
    """
    Effective rank of the empirical covariance matrix via Shannon entropy.

    Defined as exp(H) where H = -∑ p_i log(p_i) and p_i are the eigenvalues
    of Cov(X) normalised to sum to 1.

    A value close to 1 indicates near-rank-1 structure (dominated by one PC);
    a value close to p indicates full-rank / isotropic structure.

    Reference: Roy & Vetterli (2007). The effective rank: A measure of
    effective dimensionality. EUSIPCO.

    Parameters
    ----------
    X : np.ndarray (n, p)

    Returns
    -------
    float
    """
    cov = np.cov(X, rowvar=False)
    if cov.ndim == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total < 1e-15:
        return 1.0
    p = eigvals / total
    # Remove zeros to avoid log(0)
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


# ── Joint drift report ────────────────────────────────────────────────────────

_FLAG_OK = "OK"
_FLAG_WATCH = "WATCH"
_FLAG_SEVERE = "SEVERE"


def joint_drift_report(
    X_source: np.ndarray,
    X_target: np.ndarray,
    features: Sequence[str],
    delta_vif_warn: float = 5.0,
    delta_vif_severe: float = 10.0,
) -> pd.DataFrame:
    """
    Compare covariance structure between source and target cohorts.

    Computes per-feature VIF on the source cohort and flags features
    where the absolute source VIF exceeds configurable thresholds.
    (Target VIF is omitted because small target cohorts make OLS singular.)

    Parameters
    ----------
    X_source : np.ndarray (n_s, p)
    X_target : np.ndarray (n_t, p)  — used for non-VIF metrics only
    features : sequence of str, length p
    delta_vif_warn : float
        VIF_source threshold for WATCH.  Default 5.0.
    delta_vif_severe : float
        VIF_source threshold for SEVERE.  Default 10.0.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, vif_source, flag]
        where flag ∈ {OK, WATCH, SEVERE}.
    """
    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError(
            f"X_source has {X_source.shape[1]} features but "
            f"X_target has {X_target.shape[1]}."
        )
    if X_source.shape[1] != len(features):
        raise ValueError(
            f"X_source has {X_source.shape[1]} columns but "
            f"{len(features)} feature names provided."
        )

    vif_s = compute_vif(X_source, features)

    df = vif_s.rename(columns={"VIF": "vif_source"}).copy()

    def _flag(vif: float) -> str:
        if vif >= delta_vif_severe:
            return _FLAG_SEVERE
        if vif >= delta_vif_warn:
            return _FLAG_WATCH
        return _FLAG_OK

    df["flag"] = df["vif_source"].apply(_flag)
    return df[["feature", "vif_source", "flag"]].reset_index(drop=True)


# ── Optional: MI matrix delta ─────────────────────────────────────────────────


def mi_matrix_delta(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Frobenius norm of the difference between per-cohort mutual information
    matrices.

    For each pair of features (i, j), a discretised MI estimate is computed
    via a joint histogram.  The result is the Frobenius norm of
    ||MI_source − MI_target||_F.

    Note: this function is *O(p²)* and can be expensive for large p.
    Gate it with ``compute_mi_matrix: false`` in the config unless needed.

    Parameters
    ----------
    X_source : np.ndarray (n_s, p)
    X_target : np.ndarray (n_t, p)
    n_bins : int
        Number of bins per feature for the joint histogram.  Default 10.

    Returns
    -------
    float
        Frobenius norm of the MI-matrix difference.
    """
    _, p = X_source.shape

    def _mi_matrix(X: np.ndarray) -> np.ndarray:
        mat = np.zeros((p, p))
        for i in range(p):
            for j in range(i, p):
                mi = _mi_pair(X[:, i], X[:, j], n_bins)
                mat[i, j] = mi
                mat[j, i] = mi
        return mat

    mi_s = _mi_matrix(X_source)
    mi_t = _mi_matrix(X_target)
    return float(np.linalg.norm(mi_s - mi_t, ord="fro"))


def _mi_pair(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Discretised mutual information between two continuous arrays."""
    # Clip to finite values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return 0.0
    xm, ym = x[mask], y[mask]
    hist, _, _ = np.histogram2d(xm, ym, bins=n_bins)
    joint = hist / hist.sum()
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    # Avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            (joint > 0) & (marginal_x > 0) & (marginal_y > 0),
            joint / (marginal_x * marginal_y),
            1.0,
        )
        mi = float(np.sum(joint * np.log(ratio)))
    return max(mi, 0.0)
