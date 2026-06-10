"""
recal.data.schema
============================
Feature schema loading and validation utilities.

The schema is a fixed ordered list of 114 feature names that defines the column
layout used both during training (SNUH) and inference. All loaders must return
DataFrames whose feature columns match this list exactly, in this order.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default location relative to repository root
_DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "inputs" / "feature_schema.json"


def load_schema(path: Path | None = None) -> list[str]:
    """
    Load the ordered feature schema from a JSON file.

    Parameters
    ----------
    path : Path, optional
        Path to the feature_schema.json file. Defaults to
        ``inputs/feature_schema.json`` relative to the repository root.

    Returns
    -------
    list[str]
        Ordered list of feature names (114 entries for the AKI model).
    """
    p = Path(path) if path else _DEFAULT_SCHEMA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Feature schema not found at {p}."

        )
    schema: list[str] = json.loads(p.read_text())
    logger.debug("Loaded schema with %d features from %s", len(schema), p)
    return schema


def validate_dataframe(df: pd.DataFrame, schema: list[str], label_col: str = "label") -> None:
    """
    Assert that *df* contains all schema columns plus the label column.

    Parameters
    ----------
    df : pd.DataFrame
    schema : list[str]
    label_col : str

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [f for f in schema if f not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing {len(missing)} schema columns: {missing[:5]} ...")
    if label_col not in df.columns:
        raise ValueError(f"DataFrame missing label column '{label_col}'")


# Features structurally absent in Clínic Barcelona (100% NaN by design —
# these labs/measurements are not collected in the Clínic protocol).
# NOTE: This set is dynamic — update based on your actual feature schema.
STRUCTURALLY_ABSENT_CLINIC: frozenset[str] = frozenset()
