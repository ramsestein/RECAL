"""
domain_transfer.align.pca_coral
=================================
PCA-CORAL — PCA compression followed by CORAL in the latent space.

Motivation
----------
Standard CORAL requires a well-conditioned covariance estimate, which in turn
requires n >> p.  For the Clínic cohort (n=105, p≈107), the ratio p/n ≈ 1
makes the full covariance matrix rank-deficient and CORAL degrades.

PCA-CORAL solves this by compressing the feature space to k << n orthogonal
principal components (fit on SNUH), performing CORAL in that k-dimensional
latent space, and projecting back to the original space.

Algorithm
---------
1. Standardise both matrices by the SOURCE mean and std:
   ``Xs_std = (Xs - μ_s) / σ_s``,  ``Xt_std = (Xt - μ_s) / σ_s``.

2. Fit PCA on ``Xs_std``, keep k components.

3. Project: ``Zs = PCA.transform(Xs_std)``,  ``Zt = PCA.transform(Xt_std)``.

4. Apply CORAL in the k-dim latent space:
   ``Zt_aligned = (Zt - μ_zt) @ A_pca^T + μ_zs``
   where ``A_pca = Σ_zs^{1/2} · Σ_zt^{-1/2}``.

5. Inverse-transform: ``Xt_pca_std = PCA.inverse_transform(Zt_aligned)``.

6. De-standardise: ``X_aligned = Xt_pca_std * σ_s + μ_s``.

With k=5 and n_s=7554, the latent covariance is well-conditioned
(5×5 matrix from 7554 samples).  With n_t=105 and k=5, the target latent
covariance is also reasonable (5 << 105).

This implementation bit-for-bit reproduces ``_sweep_n.py::safe_pca_coral_align``
and ``w_alignment_eval.py``'s PCA-CORAL block with the same seed and k.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from domain_transfer.align.base import Aligner, _restore_nan, safe_invsqrtm, safe_sqrtm

logger = logging.getLogger(__name__)


class PCACoralAligner(Aligner):
    """
    PCA-CORAL alignment.

    Parameters
    ----------
    k : int
        Number of PCA components (latent dimension).
        Default k=5 matches the legacy optimal found by sweep in
        ``legacy/scripts/_sweep_n.py``.
    reg_pca : float
        Regularisation for the k×k covariance matrices in latent space.
        Default 1e-6 (matches legacy scripts).
    random_state : int
        Seed for PCA's randomised SVD.
    """

    def __init__(
        self,
        k: int = 5,
        reg_pca: float = 1e-6,
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.reg_pca = reg_pca
        self.random_state = random_state

        self._pca: PCA | None = None
        self._mu_xs: np.ndarray | None = None
        self._std_xs: np.ndarray | None = None
        self._A_pca: np.ndarray | None = None
        self._mu_zs: np.ndarray | None = None
        self._mu_zt: np.ndarray | None = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "PCACoralAligner":
        """
        Standardise by source stats, fit PCA on source, CORAL in latent space.

        Parameters
        ----------
        X_source : np.ndarray
            Shape (n_s, q).  No NaN.
        X_target : np.ndarray
            Shape (n_t, q).  No NaN.
        """
        # Step 1 — standardise by source mean/std
        self._mu_xs = X_source.mean(axis=0)
        self._std_xs = X_source.std(axis=0).clip(1e-8)

        Xs_std = (X_source - self._mu_xs) / self._std_xs
        Xt_std = (X_target - self._mu_xs) / self._std_xs

        # Step 2 — PCA fit on source
        k = min(self.k, Xs_std.shape[1], Xs_std.shape[0] - 1)
        if k < self.k:
            logger.warning(
                "PCACoralAligner: requested k=%d but only %d components available. "
                "Using k=%d.", self.k, k, k,
            )
        self._pca = PCA(n_components=k, random_state=self.random_state)
        self._pca.fit(Xs_std)

        # Step 3 — project
        Zs = self._pca.transform(Xs_std)
        Zt = self._pca.transform(Xt_std)

        # Step 4 — CORAL in latent space
        self._mu_zs = Zs.mean(axis=0)
        self._mu_zt = Zt.mean(axis=0)

        Sg_zs = np.cov(Zs, rowvar=False) + self.reg_pca * np.eye(k)
        Sg_zt = np.cov(Zt, rowvar=False) + self.reg_pca * np.eye(k)
        self._A_pca = safe_sqrtm(Sg_zs) @ safe_invsqrtm(Sg_zt)

        return self

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply PCA-CORAL transformation.

        Steps: standardise → project → latent CORAL → inverse project →
        de-standardise → restore NaN.
        """
        if self._pca is None:
            raise RuntimeError("PCACoralAligner must be fitted before transform.")

        Xt_std = (X_target - self._mu_xs) / self._std_xs
        Zt = self._pca.transform(Xt_std)

        Zt_aligned = (Zt - self._mu_zt) @ self._A_pca.T + self._mu_zs
        Xt_pca_std = self._pca.inverse_transform(Zt_aligned)

        X_aligned = Xt_pca_std * self._std_xs + self._mu_xs
        return _restore_nan(X_aligned, nan_mask)
