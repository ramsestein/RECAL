"""
adapt_cli.config_schema
========================
Esquema y validación de los YAML de configuración.

Ejemplo mínimo:

    model:
      path: inputs/models/aki.json
    source:
      path: inputs/source/snuh.csv
      schema: inputs/source/feature_schema.json
      outcome_col: aki
    target:
      path: inputs/target/clinic.csv
      outcome_col: aki
    output:
      report: outputs/reports/snuh_to_clinic.html
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    path: str
    type: Optional[str] = None              # auto-detect por extensión
    custom_loader: Optional[str] = None     # ruta a .py BYOM


@dataclass
class DatasetConfig:
    path: str
    outcome_col: str
    schema: Optional[str] = None            # path a JSON/TXT con features
    label_positive_value: object = 1
    unit_corrections: dict = field(default_factory=dict)  # {feature: "div10"|"mul2"|"abs"|float}


@dataclass
class AdaptConfig:
    pca_k: int = 5
    max_n_sweep: int = 30
    apply_qt: bool = True                   # True | False
    apply_woe: str = "auto"                 # "auto" | True | False
    max_missing_rate: float = 0.5
    drift_csv: Optional[str] = None         # CSV precomputado con drift_type, shap_importance_main_model, L_base


@dataclass
class OverfittingConfig:
    enabled: bool = True
    method: str = "kfold"                   # "kfold" | "none"
    n_splits: int = 5
    random_state: int = 42


@dataclass
class OutputConfig:
    report: str = "outputs/reports/adapt_report.html"
    adapted_model: Optional[str] = "outputs/adapted_models/adapted.joblib"
    metrics_json: Optional[str] = None
    source_name: str = "Source"
    target_name: str = "Target"


@dataclass
class FullConfig:
    model: ModelConfig
    source: DatasetConfig
    target: DatasetConfig
    adapt: AdaptConfig = field(default_factory=AdaptConfig)
    overfitting_check: OverfittingConfig = field(default_factory=OverfittingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _build(dc_class, data: dict, name: str):
    """Construye una dataclass tolerando claves desconocidas (con warning)."""
    if data is None:
        data = {}
    valid = {f.name for f in dc_class.__dataclass_fields__.values()}
    extra = set(data) - valid
    if extra:
        import logging
        logging.getLogger(__name__).warning(
            "[%s] claves desconocidas ignoradas: %s", name, sorted(extra)
        )
    filtered = {k: v for k, v in data.items() if k in valid}
    return dc_class(**filtered)


def load_config(path: str | Path) -> FullConfig:
    """Lee y valida un YAML de configuración."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrado: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML inválido en {path} (debe ser un dict)")

    for required in ("model", "source", "target"):
        if required not in raw:
            raise KeyError(f"Falta sección obligatoria '{required}' en {path}")

    return FullConfig(
        model=_build(ModelConfig, raw["model"], "model"),
        source=_build(DatasetConfig, raw["source"], "source"),
        target=_build(DatasetConfig, raw["target"], "target"),
        adapt=_build(AdaptConfig, raw.get("adapt"), "adapt"),
        overfitting_check=_build(OverfittingConfig, raw.get("overfitting_check"), "overfitting_check"),
        output=_build(OutputConfig, raw.get("output"), "output"),
    )
