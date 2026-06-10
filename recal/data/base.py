"""
recal.data.base
==========================
Abstract base class for cohort loaders.

A CohortLoader encapsulates the I/O and preprocessing for a single cohort
(one hospital / one dataset). It must return a DataFrame whose feature columns
match the provided schema exactly, plus a ``label`` column with binary labels.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CohortLoader(ABC):
    """
    Abstract cohort loader.

    Subclasses must implement :meth:`load` which performs all I/O and
    preprocessing (unit conversions, column alignment, etc.) and returns a
    clean DataFrame.  All methods below are derived from that result and cached
    after the first call.
    """

    def __init__(self, schema: list[str], label_col: str = "label") -> None:
        self._schema = schema
        self._label_col = label_col
        self._df: pd.DataFrame | None = None

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load and preprocess the cohort.

        Returns
        -------
        pd.DataFrame
            Columns = ``self.schema`` + [``label_col``].
            Features may contain NaN for missing values.
            Label is binary integer (0/1).
        """

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def schema(self) -> list[str]:
        return self._schema

    @property
    def label_col(self) -> str:
        return self._label_col

    def _get_df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self.load()
        return self._df

    def features(self) -> pd.DataFrame:
        """Return the feature sub-DataFrame (schema columns only)."""
        return self._get_df()[self._schema]

    def labels(self) -> pd.Series:
        """Return the binary label Series."""
        return self._get_df()[self._label_col]

    def metadata(self) -> dict:
        """
        Return a summary dictionary for logging and reporting.

        Keys: ``n``, ``n_pos``, ``prevalence``, ``n_features``,
        ``missing_rate_mean``, ``features_100pct_nan``.
        """
        df = self._get_df()
        X = df[self._schema].values.astype(float)
        nan_rates = np.isnan(X).mean(axis=0)
        n_total_missing = int((nan_rates == 1.0).sum())
        label_counts = df[self._label_col].value_counts().sort_index().to_dict()
        n = len(df)
        n_pos = int(df[self._label_col].sum())
        return {
            "n": n,
            "n_pos": n_pos,
            "prevalence": round(n_pos / n, 4) if n > 0 else float("nan"),
            "n_features": len(self._schema),
            "missing_rate_mean": round(float(nan_rates.mean()), 4),
            "features_100pct_nan": n_total_missing,
            "label_distribution": label_counts,
        }


# ── Shared preprocessing utilities (used by both SNUH and Clínic loaders) ────

def strip_units(df: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    """
    Normalise column names to match the schema by removing unit suffixes.

    Normalisation priorities (in order):
    1. Exact match after strip whitespace.
    2. Strip `` (unit)`` suffix — ``"intraop_Na_q25 (mEq/L)"`` → ``"intraop_Na_q25"``.
    3. Remove space before ``(`` — ``"CPB_time (min)"`` → ``"CPB_time(min)"``.
    4. Fallback: priority-2 form.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with raw column names from the CSV.
    schema : list[str]
        Target feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    schema_set = set(schema)
    df = df.copy()
    new_cols: list[str] = []
    for c in df.columns:
        c_stripped = c.strip()
        if c_stripped in schema_set:
            new_cols.append(c_stripped)
            continue
        c2 = c_stripped.split(" (")[0].strip()
        if c2 in schema_set:
            new_cols.append(c2)
            continue
        c3 = c_stripped.replace(" (", "(")
        if c3 in schema_set:
            new_cols.append(c3)
            continue
        new_cols.append(c2 if c2 else c_stripped)
    df.columns = new_cols
    return df


def align_to_schema(
    df: pd.DataFrame,
    schema: list[str],
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Align *df* to *schema*: add missing columns as NaN, drop extra columns,
    reorder to ``schema + [label_col]``.

    Parameters
    ----------
    df : pd.DataFrame
    schema : list[str]
    label_col : str

    Returns
    -------
    pd.DataFrame
        Columns = ``schema + [label_col]``.
    """
    for f in schema:
        if f not in df.columns:
            df = df.copy()
            df[f] = np.nan

    extra = [c for c in df.columns if c not in schema and c != label_col]
    if extra:
        df = df.drop(columns=extra)

    return df[schema + [label_col]]
