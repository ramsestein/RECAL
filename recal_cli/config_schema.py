"""
recal_cli.config_schema
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

import yaml


@dataclass
class ModelConfig:
    path: str
    type: str | None = None              # auto-detect por extensión
    custom_loader: str | None = None     # ruta a .py BYOM
    pipeline: str | None = None          # ruta a _pipeline.json (PCA+feature selection)
                                         # Si no se especifica, se busca junto al modelo


@dataclass
class DatasetConfig:
    path: str
    outcome_col: str
    schema: str | None = None            # path a JSON/TXT con features
    label_positive_value: object = 1
    unit_corrections: dict = field(default_factory=dict)  # {feature: "div10"|"mul2"|"abs"|float}


@dataclass
class AdaptConfig:
    pca_k: int = 5
    max_n_sweep: int = 30
    apply_qt: bool = True                   # True | False
    apply_woe: str = "auto"                 # "auto" | True | False
    max_missing_rate: float = 0.5
    drift_csv: str | None = None         # CSV precomputado con drift_type, shap_importance_main_model, L_base


@dataclass
class OverfittingConfig:
    enabled: bool = True
    method: str = "kfold"                   # "kfold" | "none"
    n_splits: int = 5
    random_state: int = 42


@dataclass
class OutputConfig:
    report: str = "outputs/reports/recal_report.html"
    recal_model: str | None = "outputs/recal_models/recal.joblib"
    metrics_json: str | None = None
    source_name: str = "Source"
    target_name: str = "Target"
    timestamp: bool = True   # True → añade _YYYYMMDD_HHMMSS al stem de cada fichero


@dataclass
class JointDriftConfig:
    """Configuration for joint (covariance-structure) drift analysis."""
    delta_vif_warn: float = 2.0
    delta_vif_severe: float = 5.0
    compute_mi_matrix: bool = False
    severe_share_threshold: float = 0.20


@dataclass
class RegularizationConfig:
    """Regularisation settings for CORAL, Platt calibration, and WOE."""
    # "auto" | float ∈ [0,1] | None  — forwarded to CoralAligner / PCACoralAligner
    shrinkage: str | float | None = "auto"
    # Inverse of L2 strength for Platt calibration; forwarded to
    # StratifiedPlattRecalibrator(C=...)
    calibration_C: float = 1.0
    # Laplace smoothing for WOE encoder; forwarded to WOEEncoder(smoothing=...)
    woe_smoothing: float = 0.5


@dataclass
class EvalConfig:
    """Configuración para las evaluaciones opcionales (oracle, attribution, counterfactuals)."""
    oracle_eval: bool = True                  # Entrenar oracle con k-fold en target
    oracle_cv: int = 5                        # Número de folds para el oracle
    feature_attribution: bool = True          # Atribución por feature del recoverable_gap
    feature_attribution_top_n: int = 10       # Top-N features en atribución
    counterfactual_alternatives: int = 3      # Nº de alternativas por decisión clave
    brier_decompose: bool = True              # Descomponer Brier score en Murphy components


@dataclass
class FullConfig:
    model: ModelConfig
    source: DatasetConfig
    target: DatasetConfig
    recal_core: AdaptConfig = field(default_factory=AdaptConfig)
    overfitting_check: OverfittingConfig = field(default_factory=OverfittingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    joint_drift: JointDriftConfig = field(default_factory=JointDriftConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


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
        raise FileNotFoundError(f"Config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML in {path} (expected a dict)")

    for required in ("model", "source", "target"):
        if required not in raw:
            raise KeyError(f"Missing required section '{required}' in {path}")

    return FullConfig(
        model=_build(ModelConfig, raw["model"], "model"),
        source=_build(DatasetConfig, raw["source"], "source"),
        target=_build(DatasetConfig, raw["target"], "target"),
        recal_core=_build(AdaptConfig, raw.get("recal_core"), "recal_core"),
        overfitting_check=_build(OverfittingConfig, raw.get("overfitting_check"), "overfitting_check"),
        output=_build(OutputConfig, raw.get("output"), "output"),
        joint_drift=_build(JointDriftConfig, raw.get("joint_drift"), "joint_drift"),
        regularization=_build(RegularizationConfig, raw.get("regularization"), "regularization"),
        evaluation=_build(EvalConfig, raw.get("evaluation"), "evaluation"),
    )
