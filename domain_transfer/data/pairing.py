"""
domain_transfer.data.pairing
=============================
CohortPair: pairs a source and target CohortLoader, provides aligned numpy
arrays, and exposes the main API for applying domain alignment.

Design rationale
----------------
The ``CohortPair`` centralises all array manipulation so that alignment
algorithms (see :mod:`domain_transfer.align`) can be implemented without
knowing about NaN handling, feature subsetting, or source-mean imputation.

Key concepts
~~~~~~~~~~~~
* **idx_corr** — indices of features that are NOT 100% NaN in the target.
  Only these features are passed to aligners.  Features structurally absent
  in target (100% NaN) are left untouched.

* **Imputation** — NaN values are filled with the SOURCE mean before passing
  to aligners (CORAL requires full matrices).  Original NaN positions are
  restored after alignment.  Imputation with source mean avoids introducing
  target-distribution information into the alignment fit.

* **filter_target** — filters target rows whose NaN rate exceeds a threshold.
  Must be called BEFORE ``mask_features`` to match the legacy evaluation order
  (legacy scripts compute the 50%-missing filter on unmasked data).

* **mask_features** — sets specified features to NaN in BOTH source and
  target, forcing aligners and the model to ignore them.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from domain_transfer.align.base import Aligner
    from domain_transfer.data.base import CohortLoader

logger = logging.getLogger(__name__)


class CohortPair:
    """
    Container for a (source, target) pair of cohorts.

    Parameters
    ----------
    source : CohortLoader
        The training/reference cohort (SNUH).
    target : CohortLoader
        The external validation cohort (Clínic).

    Attributes
    ----------
    schema : list[str]
        Feature names (shared between both cohorts).
    """

    def __init__(self, source: "CohortLoader", target: "CohortLoader") -> None:
        # NOTE(v0.2.0): single-target only.  Multi-target extension point:
        # accept list[CohortLoader] for target; construct one CohortPair per
        # destination inside a MultiCohortPair wrapper.  See OPEN_QUESTIONS OQ-5.
        if source.schema != target.schema:
            raise ValueError("Source and target must share the same feature schema.")
        self._source = source
        self._target = target
        self.schema: list[str] = source.schema

        # Load DataFrames once
        df_s = source._get_df()
        df_t = target._get_df()

        self._X_s: np.ndarray = df_s[self.schema].values.astype(float)
        self._X_t: np.ndarray = df_t[self.schema].values.astype(float)
        self._y_s: np.ndarray = df_s[source.label_col].values.astype(int)
        self._y_t: np.ndarray = df_t[target.label_col].values.astype(int)

        self._p = len(self.schema)

    # ── Raw arrays (with NaN) ─────────────────────────────────────────────────

    @property
    def X_s(self) -> np.ndarray:
        """Source feature matrix (n_s, p), may contain NaN."""
        return self._X_s

    @property
    def X_t(self) -> np.ndarray:
        """Target feature matrix (n_t, p), may contain NaN."""
        return self._X_t

    @property
    def y_s(self) -> np.ndarray:
        """Source labels (n_s,), binary int."""
        return self._y_s

    @property
    def y_t(self) -> np.ndarray:
        """Target labels (n_t,), binary int."""
        return self._y_t

    # ── Derived arrays (recomputed on demand) ─────────────────────────────────

    @property
    def mu_s(self) -> np.ndarray:
        """Per-feature mean over the source cohort (ignoring NaN). Shape (p,)."""
        return np.nanmean(self._X_s, axis=0)

    @property
    def X_s_imp(self) -> np.ndarray:
        """
        Source matrix imputed with source mean (NaN → mu_s).
        If mu_s[j] itself is NaN (100% missing feature), imputed to 0.
        Shape (n_s, p).
        """
        mu = self.mu_s
        X = np.where(np.isnan(self._X_s), mu[np.newaxis, :], self._X_s)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def X_t_imp(self) -> np.ndarray:
        """
        Target matrix imputed with SOURCE mean (NaN → mu_s).
        Using source mean for imputation avoids leaking target distribution
        into the alignment covariance estimates.
        Shape (n_t, p).
        """
        mu = self.mu_s
        X = np.where(np.isnan(self._X_t), mu[np.newaxis, :], self._X_t)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def nan_mask_t(self) -> np.ndarray:
        """Boolean mask of original NaN positions in target. Shape (n_t, p)."""
        return np.isnan(self._X_t)

    @property
    def idx_corr(self) -> list[int]:
        """
        Indices of features that are correctable (not 100% NaN in target).
        These are the features passed to aligners.
        """
        nan_rate = np.isnan(self._X_t).mean(axis=0)
        return [j for j in range(self._p) if nan_rate[j] < 1.0]

    @property
    def idx_missing(self) -> list[int]:
        """Indices of features that are 100% NaN in target (structurally absent)."""
        nan_rate = np.isnan(self._X_t).mean(axis=0)
        return [j for j in range(self._p) if nan_rate[j] == 1.0]

    # ── Filtering and masking ─────────────────────────────────────────────────

    def filter_target(self, max_missing_rate: float = 0.5) -> "CohortPair":
        """
        Return a new CohortPair with only target rows whose per-patient NaN
        rate is below *max_missing_rate*.

        **Must be called BEFORE** :meth:`mask_features` to reproduce the legacy
        evaluation order: the 50%-missing filter is computed on unmasked data.

        Parameters
        ----------
        max_missing_rate : float
            Rows in the target with NaN rate ≥ this value are excluded.
            Legacy default: 0.5 (keeps 105/655 Clínic patients).

        Returns
        -------
        CohortPair
            New pair with filtered target (source unchanged).
        """
        nan_per_patient = np.isnan(self._X_t).mean(axis=1)
        mask = nan_per_patient < max_missing_rate
        n_kept = int(mask.sum())
        n_dropped = int((~mask).sum())
        logger.info(
            "filter_target(%.2f): kept %d, dropped %d target patients",
            max_missing_rate, n_kept, n_dropped,
        )
        new = object.__new__(CohortPair)
        new._source = self._source
        new._target = self._target
        new.schema = self.schema
        new._X_s = self._X_s
        new._X_t = self._X_t[mask]
        new._y_s = self._y_s
        new._y_t = self._y_t[mask]
        new._p = self._p
        return new

    def mask_features(self, feature_names: list[str]) -> "CohortPair":
        """
        Return a new CohortPair where the specified features are set to NaN
        in **both** source and target matrices.

        Masking silences features that harm transfer (e.g. high drift or low
        L_base score).  The model receives NaN for these features and applies
        its learned default direction (equivalent to removing the feature's
        contribution from the prediction).

        Parameters
        ----------
        feature_names : list[str]
            Names of features to mask (must be a subset of ``self.schema``).

        Returns
        -------
        CohortPair
            New pair with masked features.
        """
        feat2idx = {f: i for i, f in enumerate(self.schema)}
        idxs = [feat2idx[f] for f in feature_names if f in feat2idx]
        unknown = [f for f in feature_names if f not in feat2idx]
        if unknown:
            logger.warning("mask_features: unknown features (ignored): %s", unknown)

        new = object.__new__(CohortPair)
        new._source = self._source
        new._target = self._target
        new.schema = self.schema
        new._X_s = self._X_s.copy()
        new._X_t = self._X_t.copy()
        new._y_s = self._y_s
        new._y_t = self._y_t
        new._p = self._p

        for j in idxs:
            new._X_s[:, j] = np.nan
            new._X_t[:, j] = np.nan

        logger.info("mask_features: masked %d features", len(idxs))
        return new

    # ── Alignment API ─────────────────────────────────────────────────────────

    def align(
        self,
        aligner: "Aligner",
        feature_names: list[str] | None = None,
    ) -> np.ndarray:
        """
        Fit *aligner* on the source/target pair and return an aligned target matrix.

        Workflow
        --------
        1. Extract the ``idx_corr`` column subset (or a subset thereof if
           *feature_names* is given).
        2. Pass imputed sub-matrices to ``aligner.fit()``.
        3. Call ``aligner.transform()`` with the original NaN mask to restore
           missing positions after alignment.
        4. Re-insert aligned columns into a copy of the full target matrix.

        Parameters
        ----------
        aligner : Aligner
            Fitted/unfitted aligner instance.
        feature_names : list[str], optional
            If given, only align these features (must be correctable).  Useful
            for ``SelectiveAligner``-style experiments without the decorator.

        Returns
        -------
        np.ndarray
            Aligned target matrix of shape (n_t, p), with original NaN
            positions preserved.
        """
        if feature_names is not None:
            feat_set = set(feature_names)
            active_idx = [j for j in self.idx_corr if self.schema[j] in feat_set]
        else:
            active_idx = self.idx_corr

        if not active_idx:
            logger.warning("align(): no correctable features selected — returning raw target")
            return self._X_t.copy()

        # OQ-7: guard against calling align() before filter_target().  If the
        # caller forgot to remove high-missingness rows, covariance estimates
        # will be contaminated.  Threshold = 10% of target rows with ≥50% NaN.
        nan_per_row = np.isnan(self._X_t).mean(axis=1)
        high_missing_frac = float((nan_per_row >= 0.5).mean())
        if high_missing_frac > 0.10:
            raise ValueError(
                f"align() detected {high_missing_frac:.1%} of target rows with "
                "\u226550% missing values (threshold: 10%). "
                "Call filter_target(max_missing_rate=0.5) before align() to "
                "remove them and avoid contaminating covariance estimates."
            )

        Xs_sub = self.X_s_imp[:, active_idx]   # (n_s, q) — no NaN
        Xt_sub = self.X_t_imp[:, active_idx]   # (n_t, q) — no NaN
        nan_sub = self.nan_mask_t[:, active_idx]  # (n_t, q) — original NaN positions

        p_ratio = Xt_sub.shape[1] / Xt_sub.shape[0]
        if p_ratio > 0.5:
            logger.warning(
                "align(): p/n = %.2f (p=%d, n=%d). CORAL global may be ill-conditioned. "
                "Consider PCACoralAligner.",
                p_ratio, Xt_sub.shape[1], Xt_sub.shape[0],
            )

        aligner.fit(Xs_sub, Xt_sub)
        Xt_aligned_sub = aligner.transform(Xt_sub, nan_mask=nan_sub)

        # Sanitise Inf introduced by the alignment transform
        Xt_aligned_sub = np.where(
            np.isfinite(Xt_aligned_sub) | np.isnan(Xt_aligned_sub),
            Xt_aligned_sub,
            np.nan,
        )

        X_t_out = self._X_t.copy()
        for li, gi in enumerate(active_idx):
            X_t_out[:, gi] = Xt_aligned_sub[:, li]

        return X_t_out

    # ── Metadata ──────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a dict summarising both cohorts and their missingness."""
        nan_rate_s = np.isnan(self._X_s).mean(axis=0)
        nan_rate_t = np.isnan(self._X_t).mean(axis=0)
        return {
            "source": {
                "n": len(self._X_s),
                "n_pos": int(self._y_s.sum()),
                "prevalence": round(float(self._y_s.mean()), 4),
            },
            "target": {
                "n": len(self._X_t),
                "n_pos": int(self._y_t.sum()),
                "prevalence": round(float(self._y_t.mean()), 4),
            },
            "n_features": self._p,
            "n_correctable": len(self.idx_corr),
            "n_structurally_absent_target": len(self.idx_missing),
        }
