"""
recal.align.identity
================================
IdentityAligner — the Raw baseline (no alignment).

Returns the target matrix unchanged.  Used as the baseline in all comparison
tables so that every configuration goes through the same pipeline code.
"""

from __future__ import annotations

import numpy as np

from recal.align.base import Aligner, _restore_nan


class IdentityAligner(Aligner):
    """
    No-op aligner (Raw baseline).

    Does not modify the target distribution.  All subsequent scores measure
    the degradation due to dataset shift alone.
    """

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> IdentityAligner:
        return self

    def transform(
        self,
        X_target: np.ndarray,
        nan_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        return _restore_nan(X_target.copy(), nan_mask)
