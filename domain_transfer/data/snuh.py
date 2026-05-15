"""
domain_transfer.data.snuh
==========================
CohortLoader adapter for the SNUH (Seoul National University Hospital) cohort.

Reads the raw CSV ``datasets/SNUH_AKI(SNUH_AKI).csv``, normalises column names,
aligns to the feature schema, and returns a clean DataFrame.  No unit
conversions are required — SNUH is the training cohort and defines the reference
unit system used by the XGBoost model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from domain_transfer.data.base import CohortLoader, align_to_schema, strip_units

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _REPO_ROOT / "datasets" / "SNUH_AKI(SNUH_AKI).csv"


class SNUHLoader(CohortLoader):
    """
    Loader for the SNUH surgical AKI cohort (n ≈ 7 554, AKI+ ≈ 25.7%).

    SNUH is the training/reference cohort for the XGBoost AKI model.  It
    defines the reference unit system.  No unit conversions are applied.

    Parameters
    ----------
    csv_path : Path, optional
        Path to the raw SNUH CSV.  Defaults to
        ``datasets/SNUH_AKI(SNUH_AKI).csv`` relative to the repo root.
    schema : list[str]
        Ordered feature schema (114 features for the AKI model).
    label_col : str
        Name of the binary outcome column in the CSV.
    """

    def __init__(
        self,
        schema: list[str],
        csv_path: Path | None = None,
        label_col: str = "label",
    ) -> None:
        super().__init__(schema=schema, label_col=label_col)
        self._csv_path = Path(csv_path) if csv_path else _DEFAULT_CSV

    def load(self) -> pd.DataFrame:
        """
        Load SNUH CSV → normalise columns → align to schema.

        Returns
        -------
        pd.DataFrame
            Columns: schema features + ``label``.  Features may have partial NaN.
        """
        if not self._csv_path.exists():
            raise FileNotFoundError(f"SNUH CSV not found at {self._csv_path}")

        logger.info("Loading SNUH from %s", self._csv_path)
        raw = pd.read_csv(self._csv_path)
        logger.debug("  Raw shape: %s", raw.shape)

        df = strip_units(raw, self._schema)

        # Drop rows without a valid label
        before = len(df)
        df = df.dropna(subset=[self._label_col])
        if len(df) < before:
            logger.warning("  Dropped %d rows with missing label", before - len(df))

        df = align_to_schema(df, self._schema, self._label_col)
        df[self._label_col] = df[self._label_col].astype(int)

        # Replace ±Inf with NaN (rare in SNUH but safe to handle)
        feat_vals = df[self._schema].values.astype(float)
        feat_vals[~np.isfinite(feat_vals)] = np.nan
        df[self._schema] = feat_vals

        logger.info(
            "  SNUH loaded: n=%d, AKI+=%d (%.1f%%)",
            len(df),
            int(df[self._label_col].sum()),
            100 * df[self._label_col].mean(),
        )
        return df
