"""
adapt_cli.data_loader
======================
GenericCohortLoader: carga cualquier CSV/Parquet alineado a un schema,
encapsulado como CohortLoader compatible con CohortPair del paquete adapt.

Espera un CSV/Parquet con columnas:
    - Todas las features del schema (en cualquier orden, casing exacto)
    - Una columna outcome (binaria 0/1)

Las columnas faltantes se rellenan con NaN. Las columnas extras se ignoran.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from domain_transfer.data.base import CohortLoader

try:
    from domain_transfer.data.base import strip_units
except ImportError:
    strip_units = None  # type: ignore

logger = logging.getLogger(__name__)


class GenericCohortLoader(CohortLoader):
    """CohortLoader que carga desde CSV o Parquet con un outcome configurable."""

    def __init__(
        self,
        path: str | Path,
        schema: list[str],
        outcome_col: str,
        label_positive_value=1,
        unit_corrections: Optional[dict] = None,
    ) -> None:
        super().__init__(schema=schema, label_col="label")
        self.path = Path(path)
        self.outcome_col = outcome_col
        self.label_positive_value = label_positive_value
        self.unit_corrections = unit_corrections or {}

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

        # Normalizar nombres de columnas (quitar sufijos de unidades) si hay solape
        if strip_units is not None:
            try:
                df = strip_units(df, self._schema)
            except Exception:
                pass

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
                "[%s] %d/%d features ausentes (rellenadas con NaN): %s%s",
                self.path.name, len(missing), len(self._schema),
                missing[:5], "..." if len(missing) > 5 else "",
            )

        # Aplicar correcciones de unidades configurables
        if self.unit_corrections:
            for feat, op in self.unit_corrections.items():
                if feat not in out.columns:
                    logger.warning("[unit_corrections] feature '%s' no en schema; ignorada", feat)
                    continue
                if isinstance(op, str):
                    op_lc = op.lower()
                    if op_lc in ("div10", "÷10", "/10"):
                        out[feat] = out[feat] / 10.0
                    elif op_lc in ("mul10", "×10", "*10"):
                        out[feat] = out[feat] * 10.0
                    elif op_lc in ("mul2", "×2", "*2"):
                        out[feat] = out[feat] * 2.0
                    elif op_lc in ("div2", "÷2", "/2"):
                        out[feat] = out[feat] / 2.0
                    elif op_lc == "abs":
                        out[feat] = out[feat].abs()
                    elif op_lc == "neg":
                        out[feat] = -out[feat]
                    else:
                        logger.warning("[unit_corrections] op '%s' desconocida para %s", op, feat)
                        continue
                elif isinstance(op, (int, float)):
                    # Multiplicador numérico
                    out[feat] = out[feat] * float(op)
                else:
                    logger.warning("[unit_corrections] valor inválido para %s: %r", feat, op)
                    continue
                logger.debug("[unit_corrections] %s ← %s", feat, op)
            logger.info("[%s] %d unit_corrections aplicadas", self.path.name, len(self.unit_corrections))

        # Sanitizar ±Inf → NaN (común en CSVs con ratios mal calculados)
        n_inf = int(np.isinf(out[self._schema].values).sum())
        if n_inf:
            logger.warning("[%s] %d valores Inf reemplazados por NaN", self.path.name, n_inf)
            out[self._schema] = out[self._schema].replace([np.inf, -np.inf], np.nan)

        # Label binaria
        raw = df[self.outcome_col]
        if self.label_positive_value == 1:
            label = pd.to_numeric(raw, errors="coerce")
        else:
            label = (raw == self.label_positive_value).astype("Int64")

        # Drop filas con label NaN (no convertir a 0; sería sesgo)
        valid_mask = label.notna()
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            logger.warning("[%s] %d filas con label NaN descartadas", self.path.name, n_dropped)
            out = out.loc[valid_mask].reset_index(drop=True)
            label = label.loc[valid_mask].reset_index(drop=True)
        out["label"] = label.astype(int).clip(0, 1)

        n = len(out)
        n_pos = int(out["label"].sum())
        logger.info(
            "[%s] cargado: n=%d, eventos=%d (%.1f%%), features presentes=%d/%d",
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
