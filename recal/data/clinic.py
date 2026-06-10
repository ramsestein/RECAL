"""
recal.data.clinic
============================
CohortLoader adapter for the Hospital Clínic de Barcelona cohort.

Reads the raw CSV ``datasets/Clínic_AKI.csv``, normalises column names,
applies **unit corrections** to map Clínic's measurement units to the SNUH
reference system expected by the XGBoost model, and returns a clean DataFrame.

Unit corrections (verified via units_audit.csv and cross-cohort EDA):

* **Haemoglobin** (``postop_Hb_avg``, ``postop_Hb_intercept``):
  Clínic records in g/L; SNUH (and the model) expect g/dL → divide by 10.

* **Albumin** (``postop_Albumin_avg``, ``postop_Albumin_intercept``):
  Clínic records in g/L; SNUH expects g/dL → divide by 10.

* **Base excess BEecf** (``postop_BEecf_avg``, ``postop_BEecf_intercept``):
  Clínic records signed values; SNUH convention is absolute value → ``abs()``.

* **Calcium** (``preop_Ca``):
  Clínic records in mEq/L; SNUH expects mg/dL.  Conversion: 1 mEq/L = 2 mg/dL
  (verified: median ratio exactly 2.0 in the cohort).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recal.data.base import CohortLoader, align_to_schema, strip_units

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _REPO_ROOT / "datasets" / "Clínic_AKI.csv"

# ── Unit correction constants ─────────────────────────────────────────────────
_DIVIDE_BY_10 = ("postop_Hb_avg", "postop_Hb_intercept",
                 "postop_Albumin_avg", "postop_Albumin_intercept")
_TAKE_ABS     = ("postop_BEecf_avg", "postop_BEecf_intercept")
_MULTIPLY_BY_2 = ("preop_Ca",)


class ClinicLoader(CohortLoader):
    """
    Loader for the Hospital Clínic de Barcelona AKI cohort (n ≈ 655).

    Raw prevalence is ~6.3% (41/655 AKI+).  After filtering to patients with
    <50% missing features the filtered cohort has n=105, AKI+=29 (27.6%) —
    this filtering is NOT applied here; it is the responsibility of
    :class:`~recal.data.pairing.CohortPair`.

    Parameters
    ----------
    csv_path : Path, optional
        Path to the raw Clínic CSV.  Defaults to
        ``datasets/Clínic_AKI.csv`` relative to the repo root.
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
        Load Clínic CSV → normalise columns → unit corrections → align to schema.

        Returns
        -------
        pd.DataFrame
            Columns: schema features + ``label``.  Values are in the same unit
            system as SNUH.  Features may have substantial NaN (up to 100% for
            structurally absent measurements).
        """
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Clínic CSV not found at {self._csv_path}")

        logger.info("Loading Clínic from %s", self._csv_path)
        raw = pd.read_csv(self._csv_path)
        logger.debug("  Raw shape: %s", raw.shape)

        df = strip_units(raw, self._schema)

        before = len(df)
        df = df.dropna(subset=[self._label_col])
        if len(df) < before:
            logger.warning("  Dropped %d rows with missing label", before - len(df))

        df = align_to_schema(df, self._schema, self._label_col)
        df[self._label_col] = df[self._label_col].astype(int)

        # ── Apply unit corrections ────────────────────────────────────────────
        df = self._apply_unit_corrections(df)

        # Replace ±Inf with NaN
        feat_vals = df[self._schema].values.astype(float)
        feat_vals[~np.isfinite(feat_vals)] = np.nan
        df[self._schema] = feat_vals

        logger.info(
            "  Clínic loaded: n=%d, AKI+=%d (%.1f%%)",
            len(df),
            int(df[self._label_col].sum()),
            100 * df[self._label_col].mean(),
        )
        return df

    def _apply_unit_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all unit conversions in-place (returns modified copy)."""
        df = df.copy()
        for feat in _DIVIDE_BY_10:
            if feat in df.columns:
                df[feat] = df[feat] / 10.0
                logger.debug("  Unit correction: %s ÷ 10", feat)

        for feat in _TAKE_ABS:
            if feat in df.columns:
                df[feat] = df[feat].abs()
                logger.debug("  Unit correction: abs(%s)", feat)

        for feat in _MULTIPLY_BY_2:
            if feat in df.columns:
                df[feat] = df[feat] * 2.0
                logger.debug("  Unit correction: %s × 2", feat)

        return df
