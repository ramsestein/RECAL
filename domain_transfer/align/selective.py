"""
domain_transfer.align.selective
=================================
SelectiveAligner — decorator that restricts alignment to a feature sub-set.

Motivation
----------
Global alignment (CORAL, PCA-CORAL, etc.) applies to all features
simultaneously.  In the presence of features with stable distributions
(aligned features), touching them may introduce noise.  Selective alignment
only corrects the features identified as drifted (via
:class:`~domain_transfer.select.CombinedScoreSelector` or similar) while
leaving stable features untouched.

Design
------
SelectiveAligner wraps any other :class:`Aligner` and:
1. Extracts the sub-matrix of ``feature_indices`` columns.
2. Calls the inner aligner's ``fit`` / ``transform`` on that sub-matrix.
3. Writes the aligned columns back to the result, keeping all other columns
   from the *original* (un-aligned) target matrix.
4. NaN positions within the selected columns are restored via ``nan_mask``.

The ``feature_indices`` are *local* indices into the q-column sub-matrix
passed to SelectiveAligner (i.e., already relative to the correctable feature
set selected by ``CohortPair``).

Usage
-----
>>> base = PCACoralAligner(k=5)
>>> sel = SelectiveAligner(base_aligner=base, feature_indices=bottom_10_idx)
>>> X_aligned = pair.align(sel)
"""

from __future__ import annotations

import numpy as np

from domain_transfer.align.base import Aligner, _restore_nan


class SelectiveAligner(Aligner):
    """
    Applies a base aligner only to a selected subset of features.

    Parameters
    ----------
    base_aligner : Aligner
        Any fitted-or-unfitted Aligner instance (e.g. PCACoralAligner(k=5)).
        A new ``fit`` call will be forwarded to a COPY of this aligner (each
        call to ``fit`` re-uses the same object in place — do not share a
        single instance across multiple ``fit`` calls).
    feature_indices : array-like of int
        Local column indices (relative to the q-feature sub-matrix) to align.
    """

    def __init__(
        self,
        base_aligner: Aligner,
        feature_indices: np.ndarray | list[int],
    ) -> None:
        self.base_aligner = base_aligner
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self._fitted = False

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "SelectiveAligner":
        """
        Fit the inner aligner on the selected feature columns only.
        """
        idx = self.feature_indices
        self.base_aligner.fit(X_source[:, idx], X_target[:, idx])
        self._fitted = True
        return self

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Align only the selected columns; copy all others unchanged.

        Parameters
        ----------
        X_target : np.ndarray
            Full target sub-matrix, shape (n_t, q).  Imputed (no NaN).
        nan_mask : np.ndarray, optional
            Boolean mask shape (n_t, q).  NaN positions are restored in the
            aligned columns after transformation.
        """
        if not self._fitted:
            raise RuntimeError("SelectiveAligner must be fitted before transform.")

        idx = self.feature_indices
        X_out = X_target.copy()

        # NaN mask for selected columns only
        sub_nan_mask = nan_mask[:, idx] if nan_mask is not None else None

        X_out[:, idx] = self.base_aligner.transform(
            X_target[:, idx], nan_mask=sub_nan_mask
        )

        # Restore NaN for unselected columns
        if nan_mask is not None:
            other_idx = np.setdiff1d(np.arange(X_target.shape[1]), idx)
            if other_idx.size > 0:
                X_out[:, other_idx] = np.where(
                    nan_mask[:, other_idx], np.nan, X_out[:, other_idx]
                )

        return X_out
