"""
domain_transfer.align.coral
=============================
CORAL — CORrelation ALignment (Sun & Saenko, 2016).

Aligns the second-order statistics (covariance) of the target distribution
to match the source distribution via a linear transformation:

    X_t_aligned = (X_t - μ_t) @ A^T + μ_s

where the alignment matrix is:

    A = Σ_s^{1/2} · Σ_t^{-1/2}

This removes linear covariate shift without requiring any labels.

Limitations
-----------
* Only corrects linear structure — non-linear relationships are not affected.
* Requires p < n to estimate a full-rank covariance.  When p ≈ n (as with
  Clínic, n=105, p≈107), the covariance estimate is poorly conditioned and
  CORAL can *degrade* performance.  Use :class:`PCACoralAligner` instead.
* ``CohortPair.align()`` logs a warning when p/n > 0.5.

References
----------
Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep
domain adaptation. ECCV Workshops. https://arxiv.org/abs/1612.01939
"""

from __future__ import annotations

import logging

import numpy as np

from domain_transfer.align.base import Aligner, _restore_nan, safe_invsqrtm, safe_sqrtm

logger = logging.getLogger(__name__)


class CoralAligner(Aligner):
    """
    Full CORAL alignment on all provided features.

    Parameters
    ----------
    reg : float
        Regularisation added to the diagonal of both covariance matrices
        before inversion.  Prevents numerical singularity.
        Default: 1e-4 (matches legacy ``w_alignment_eval.py``).
    """

    def __init__(self, reg: float = 1e-4) -> None:
        self.reg = reg
        self._A: np.ndarray | None = None
        self._mu_s: np.ndarray | None = None
        self._mu_t: np.ndarray | None = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "CoralAligner":
        """
        Estimate covariance matrices and compute alignment matrix A.

        Parameters
        ----------
        X_source : np.ndarray
            Shape (n_s, q).  No NaN.
        X_target : np.ndarray
            Shape (n_t, q).  No NaN.
        """
        q = X_source.shape[1]
        p_over_n = q / X_target.shape[0]
        if p_over_n > 0.5:
            logger.warning(
                "CoralAligner.fit: p/n = %.2f (p=%d, n=%d). "
                "Covariance estimate is poorly conditioned. "
                "CORAL global may degrade. Consider PCACoralAligner(k=5).",
                p_over_n, q, X_target.shape[0],
            )

        self._mu_s = X_source.mean(axis=0)
        self._mu_t = X_target.mean(axis=0)

        Sig_s = np.cov(X_source, rowvar=False) + self.reg * np.eye(q)
        Sig_t = np.cov(X_target, rowvar=False) + self.reg * np.eye(q)
        self._A = safe_sqrtm(Sig_s) @ safe_invsqrtm(Sig_t)

        return self

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply CORAL transformation: X_aligned = (X_t - μ_t) @ A^T + μ_s.
        """
        if self._A is None:
            raise RuntimeError("CoralAligner must be fitted before transform.")

        X_aligned = (X_target - self._mu_t) @ self._A.T + self._mu_s
        return _restore_nan(X_aligned, nan_mask)
