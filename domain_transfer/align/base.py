"""
domain_transfer.align.base
===========================
Abstract Aligner interface and shared mathematical utilities.

All aligners implement the same fit/transform/fit_transform interface so they
can be swapped transparently in ``CohortPair.align()``.

Mathematical conventions
------------------------
All aligners:
* Receive **imputed** matrices (no NaN) for ``fit`` and ``transform``.
* Accept an optional ``nan_mask`` in ``transform`` to restore original NaN
  positions after alignment (so the XGBoost model can apply its native
  missing-value handling rather than acting on imputed values).
* Operate on the sub-matrix of correctable features provided by
  ``CohortPair.align()`` — they do NOT need to know about idx_corr.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Aligner(ABC):
    """
    Abstract domain alignment algorithm.

    Fit is performed on the SOURCE and TARGET sub-matrices (correctable
    features only, imputed with source mean).  Transform takes the imputed
    target sub-matrix and optionally restores NaN positions.
    """

    @abstractmethod
    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "Aligner":
        """
        Estimate alignment parameters from source and target distributions.

        Parameters
        ----------
        X_source : np.ndarray
            Source feature matrix, shape (n_s, q).  No NaN, no Inf.
        X_target : np.ndarray
            Target feature matrix, shape (n_t, q).  No NaN, no Inf.

        Returns
        -------
        Aligner
            self (for chaining).
        """

    @abstractmethod
    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply alignment to the target matrix.

        Parameters
        ----------
        X_target : np.ndarray
            Imputed target matrix, shape (n_t, q).  No NaN, no Inf.
        nan_mask : np.ndarray, optional
            Boolean mask of shape (n_t, q).  Where True, the original value
            was NaN and will be restored as NaN in the output.

        Returns
        -------
        np.ndarray
            Aligned target matrix, shape (n_t, q).  NaN restored where
            ``nan_mask`` is True.
        """

    def fit_transform(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        return self.fit(X_source, X_target).transform(X_target, nan_mask=nan_mask)


# ── Shared matrix utilities ───────────────────────────────────────────────────

def safe_sqrtm(M: np.ndarray) -> np.ndarray:
    """
    Symmetric positive-definite matrix square root via eigendecomposition.

    Numerically stable: eigenvalues are clipped to ``1e-10`` before taking
    the square root.

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix of shape (p, p).

    Returns
    -------
    np.ndarray
        Matrix ``S`` such that ``S @ S ≈ M``.
    """
    S = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def safe_invsqrtm(M: np.ndarray) -> np.ndarray:
    """
    Symmetric positive-definite matrix inverse square root via eigendecomposition.

    Numerically stable: eigenvalues are clipped to ``1e-10`` before inversion.

    Parameters
    ----------
    M : np.ndarray
        Square symmetric matrix of shape (p, p).

    Returns
    -------
    np.ndarray
        Matrix ``S`` such that ``S @ M @ S ≈ I``.
    """
    S = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _restore_nan(X_aligned: np.ndarray, nan_mask: np.ndarray | None) -> np.ndarray:
    """Restore NaN at positions where the original target had NaN."""
    if nan_mask is None:
        return X_aligned
    return np.where(nan_mask, np.nan, X_aligned)
