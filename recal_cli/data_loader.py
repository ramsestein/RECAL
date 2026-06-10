"""
recal_cli.data_loader
======================
GenericCohortLoader: carga cualquier CSV/Parquet alineado a un schema,
encapsulado como CohortLoader compatible con CohortPair del paquete recal_core.

Espera un CSV/Parquet con columnas:
    - Todas las features del schema (en cualquier orden, casing exacto)
    - Una columna outcome (binaria 0/1)

Las columnas faltantes se rellenan con NaN. Las columnas extras se ignoran.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from recal.data.base import CohortLoader

try:
    from recal.data.base import strip_units
except ImportError:
    strip_units = None  # type: ignore

logger = logging.getLogger(__name__)


class GenericCohortLoader(CohortLoader):
    """CohortLoader que carga desde CSV o Parquet con un outcome configurable.

    Si se proporciona un ``pipeline_preprocessor``, se aplica al DataFrame
    **antes** de alinear al schema. Esto permite que el CSV contenga features
    crudas y el pipeline (PCA + feature selection) las transforme a las
    features que el modelo espera.
    """

    def __init__(
        self,
        path: str | Path,
        schema: list[str],
        outcome_col: str,
        label_positive_value=1,
        unit_corrections: dict | None = None,
        pipeline_preprocessor=None,
    ) -> None:
        super().__init__(schema=schema, label_col="label")
        self.path = Path(path)
        self.outcome_col = outcome_col
        self.label_positive_value = label_positive_value
        self.unit_corrections = unit_corrections or {}
        self._pipeline = pipeline_preprocessor

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {self.path}")

        ext = self.path.suffix.lower()
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(self.path, sep=sep)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(self.path)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(self.path)
        else:
            raise ValueError(f"Formato no soportado: {ext}")

        if self.outcome_col not in df.columns:
            raise KeyError(
                f"Columna outcome '{self.outcome_col}' no encontrada en {self.path}. "
                f"Disponibles: {list(df.columns)[:20]}..."
            )

        # ── Normalizar nombres de columnas (quitar sufijos de unidades) ──
        # Debe ejecutarse antes de unit_corrections y pipeline para que los
        # nombres de features en la config coincidan con las columnas del CSV.
        if strip_units is not None:
            try:
                df = strip_units(df, self._schema)
            except Exception:
                pass

        # ── Unit corrections (apply to raw columns BEFORE pipeline) ───────
        if self.unit_corrections:
            for feat, op in self.unit_corrections.items():
                if feat not in df.columns:
                    logger.warning("[unit_corrections] feature '%s' not in raw data; ignored", feat)
                    continue
                if isinstance(op, str):
                    op_lc = op.lower()
                    if op_lc in ("div10", "÷10", "/10"):
                        df[feat] = pd.to_numeric(df[feat], errors="coerce") / 10.0
                    elif op_lc in ("mul10", "×10", "*10"):
                        df[feat] = pd.to_numeric(df[feat], errors="coerce") * 10.0
                    elif op_lc in ("mul2", "×2", "*2"):
                        df[feat] = pd.to_numeric(df[feat], errors="coerce") * 2.0
                    elif op_lc in ("div2", "÷2", "/2"):
                        df[feat] = pd.to_numeric(df[feat], errors="coerce") / 2.0
                    elif op_lc == "abs":
                        df[feat] = pd.to_numeric(df[feat], errors="coerce").abs()
                    elif op_lc == "neg":
                        df[feat] = -pd.to_numeric(df[feat], errors="coerce")
                    else:
                        logger.warning("[unit_corrections] op '%s' unknown for %s", op, feat)
                        continue
                elif isinstance(op, (int, float)):
                    df[feat] = pd.to_numeric(df[feat], errors="coerce") * float(op)
                else:
                    logger.warning("[unit_corrections] invalid value for %s: %r", feat, op)
                    continue
                logger.debug("[unit_corrections] %s ← %s", feat, op)
            logger.info("[%s] %d unit_corrections applied", self.path.name, len(self.unit_corrections))

        # ── Pipeline preprocessing (PCA + feature selection) ──────────────
        if self._pipeline is not None:
            logger.info(
                "[%s] applying pipeline preprocessor (%d → %d features)",
                self.path.name,
                self._pipeline.n_features_in,
                self._pipeline.n_features_out,
            )
            df = self._pipeline.transform(df)

        # Construir la matriz de features alineada al schema (faltantes → NaN)
        out = pd.DataFrame(index=df.index, columns=self._schema, dtype=float)
        present, missing = [], []
        for feat in self._schema:
            if feat in df.columns:
                out[feat] = pd.to_numeric(df[feat], errors="coerce")
                present.append(feat)
            else:
                missing.append(feat)

        if missing:
            logger.warning(
                "[%s] %d/%d features missing (filled with NaN): %s%s",
                self.path.name, len(missing), len(self._schema),
                missing[:5], "..." if len(missing) > 5 else "",
            )

        # Sanitise ±Inf → NaN (common in CSVs with miscalculated ratios)
        n_inf = int(np.isinf(out[self._schema].values).sum())
        if n_inf:
            logger.warning("[%s] %d Inf values replaced with NaN", self.path.name, n_inf)
            out[self._schema] = out[self._schema].replace([np.inf, -np.inf], np.nan)

        # Label binaria
        raw = df[self.outcome_col]
        if self.label_positive_value == 1:
            label = pd.to_numeric(raw, errors="coerce")
        else:
            label = (raw == self.label_positive_value).astype("Int64")

        # Drop rows with NaN label (do not convert to 0; that would introduce bias)
        valid_mask = label.notna()
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            logger.warning("[%s] %d rows with NaN label dropped", self.path.name, n_dropped)
            out = out.loc[valid_mask].reset_index(drop=True)
            label = label.loc[valid_mask].reset_index(drop=True)
        out["label"] = label.astype(int).clip(0, 1)

        n = len(out)
        n_pos = int(out["label"].sum())
        logger.info(
            "[%s] loaded: n=%d, events=%d (%.1f%%), features present=%d/%d",
            self.path.name, n, n_pos, 100 * n_pos / max(n, 1),
            len(present), len(self._schema),
        )
        return out


def load_schema_from_file(path: str | Path) -> list[str]:
    """
    Carga la lista de features desde JSON o TXT.

    JSON: lista plana o dict con clave 'features' / 'schema'.
    TXT:  una feature por línea.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema no encontrado: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            for key in ("features", "schema", "columns", "feature_schema"):
                if key in data:
                    return [str(x) for x in data[key]]
        raise ValueError(f"No se pudo extraer lista de features de {path}")

    # TXT / CSV de una columna
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
