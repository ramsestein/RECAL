"""
recal.align.coral
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

from recal.align.base import Aligner, _restore_nan, safe_invsqrtm, safe_sqrtm

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
    shrinkage : str, float, or None
        Shrinkage estimator for the target covariance matrix.

        - ``"auto"`` : use :class:`sklearn.covariance.LedoitWolf` to
          estimate the optimal shrinkage coefficient automatically.
          Recommended when ``n_target`` is small.
        - ``float ∈ [0, 1]`` : apply manual shrinkage
          ``Σ̂ = (1 − α) · S + α · μ_trace · I``, where ``S`` is the sample
          covariance, ``μ_trace = trace(S) / p``, and ``α`` is the given
          coefficient.
        - ``None`` or ``0`` : no shrinkage (legacy behaviour).

        The estimated Ledoit-Wolf coefficient is stored in
        ``lw_coef_source_`` and ``lw_coef_target_`` after fitting.
        Default: ``"auto"``.
    """

    def __init__(
        self,
        reg: float = 1e-4,
        shrinkage: str | float | None = "auto",
    ) -> None:
        self.reg = reg
        self.shrinkage = shrinkage
        self._A: np.ndarray | None = None
        self._mu_s: np.ndarray | None = None
        self._mu_t: np.ndarray | None = None
        # Reported after fit — LW shrinkage coefficients (None if not used)
        self.lw_coef_source_: float | None = None
        self.lw_coef_target_: float | None = None

    # ── Covariance estimation ─────────────────────────────────────────────────

    def _estimate_cov(
        self, X: np.ndarray, label: str
    ) -> tuple[np.ndarray, float | None]:
        """
        Estimate regularised covariance of X.

        Returns
        -------
        cov : np.ndarray (q, q)
        lw_coef : float or None
            Ledoit-Wolf shrinkage coefficient when shrinkage='auto';
            the manual alpha when shrinkage is a float; None otherwise.
        """
        q = X.shape[1]
        shrinkage = self.shrinkage

        if shrinkage == "auto":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(X)
            cov = lw.covariance_ + self.reg * np.eye(q)
            lw_coef = float(lw.shrinkage_)
            logger.info(
                "CoralAligner [%s]: LedoitWolf shrinkage=%.4f (n=%d, p=%d).",
                label, lw_coef, X.shape[0], q,
            )
            return cov, lw_coef

        S = np.cov(X, rowvar=False)
        if isinstance(shrinkage, (int, float)) and shrinkage not in (None, 0, 0.0):
            alpha = float(shrinkage)
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(
                    f"shrinkage must be in [0, 1] or 'auto' or None; got {alpha}."
                )
            mu = float(np.trace(S)) / q
            cov = (1.0 - alpha) * S + alpha * mu * np.eye(q)
            cov += self.reg * np.eye(q)
            logger.info(
                "CoralAligner [%s]: manual shrinkage alpha=%.4f applied.",
                label, alpha,
            )
            return cov, alpha

        # None / 0 → legacy behaviour
        cov = S + self.reg * np.eye(q)
        return cov, None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> CoralAligner:
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

        Sig_s, self.lw_coef_source_ = self._estimate_cov(X_source, "source")
        Sig_t, self.lw_coef_target_ = self._estimate_cov(X_target, "target")
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
