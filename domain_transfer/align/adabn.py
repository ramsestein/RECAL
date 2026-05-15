"""
domain_transfer.align.adabn
=============================
AdaBN diagonal — per-feature mean and standard-deviation alignment.

AdaBN (Adaptive Batch Normalisation) was originally proposed to re-normalise
the internal batch-norm statistics of a deep neural network using target
domain statistics.  The diagonal variant applied here is a feature-level
analogue: it re-standardises each feature independently by the target mean
and std, then re-scales to match the source distribution.

Algorithm (per feature j)
--------------------------
    X_aligned[:, j] = (X_t[:, j] − μ_t[j]) / σ_t[j] · σ_s[j] + μ_s[j]

This is equivalent to standardising both cohorts to zero-mean / unit-variance
using their own statistics, which removes marginal covariate shift (mean and
scale mismatch) independently for each feature.

Limitations
-----------
* Corrects only marginal shift (mean/std per feature) — no cross-feature
  covariance is touched.
* More conservative than CORAL but also much more robust when n is small.
* Does not correct concept shift or nonlinear drift.
"""

from __future__ import annotations

import logging

import numpy as np

from domain_transfer.align.base import Aligner, _restore_nan

logger = logging.getLogger(__name__)


class AdaBNAligner(Aligner):
    """
    Diagonal AdaBN: per-feature mean/std re-alignment.

    Parameters
    ----------
    eps : float
        Minimum std value to avoid division by zero.  Default 1e-8.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._mu_s: np.ndarray | None = None
        self._std_s: np.ndarray | None = None
        self._mu_t: np.ndarray | None = None
        self._std_t: np.ndarray | None = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "AdaBNAligner":
        """
        Compute per-feature mean and std for source and target.

        Parameters
        ----------
        X_source : np.ndarray
            Shape (n_s, q).
        X_target : np.ndarray
            Shape (n_t, q).
        """
        self._mu_s = X_source.mean(axis=0)
        self._std_s = X_source.std(axis=0).clip(self.eps)
        self._mu_t = X_target.mean(axis=0)
        self._std_t = X_target.std(axis=0).clip(self.eps)
        return self

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Re-normalise each feature to match source distribution.

        X_aligned[:, j] = (X_t[:, j] − μ_t) / σ_t · σ_s + μ_s
        """
        if self._mu_s is None:
            raise RuntimeError("AdaBNAligner must be fitted before transform.")

        X_aligned = (X_target - self._mu_t) / self._std_t * self._std_s + self._mu_s
        return _restore_nan(X_aligned, nan_mask)
