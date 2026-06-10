"""
recal_cli.audit_serializer
============================
Serialización completa del pipeline RECAL en formato YAML auditado.

Contenido del audit YAML
------------------------
- run_id : UUID4 del run
- timestamp : ISO 8601
- input_hash : SHA-256 de (schema + n_rows_source + n_rows_target + sample_hash)
              SIN PII — solo identifica la versión de datos, no los datos mismos
- versions : RECAL + dependencias críticas (numpy, sklearn, xgboost, scipy)
- config_used : copia del FullConfig (sin paths sensibles)
- designer_audit_trail : lista de DesignerDecisions serializadas
- feature_log : log por feature (stats de distribución pre/post + KS)
- drift_decomposition : recoverable_gap, irreducible_gap, recovery_ratio + CIs
- oracle_result : AUROC oracle + CI + metadata
- counterfactuals : resultado del análisis de sensibilidad
- calibration_decomposition_raw : descomposición Brier raw
- calibration_decomposition_adapted : descomposición Brier adaptado
- calibration_delta : deltas

El audit YAML se guarda en outputs/audit/<run_id>.yaml.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Dependencias a reportar
_TRACKED_PACKAGES = [
    "numpy", "scikit-learn", "xgboost", "scipy", "pandas",
    "joblib", "pyyaml",
]


def _pkg_version(name: str) -> str:
    """Obtiene la versión de un paquete instalado."""
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _adapt_version() -> str:
    """Versión del propio paquete RECAL."""
    try:
        return importlib.metadata.version("recal_core")
    except Exception:
        try:
            from recal_core import __version__  # type: ignore
            return str(__version__)
        except Exception:
            return "unknown"


def _compute_input_hash(
    schema: list[str],
    n_source: int,
    n_target: int,
    X_source: np.ndarray | None = None,
    X_target: np.ndarray | None = None,
) -> str:
    """
    Calcula SHA-256 de (schema + n_rows + sample_hash) sin PII.

    El sample_hash es el hash del vector de medias por columna (no permite
    reconstruir datos individuales).
    """
    h = hashlib.sha256()
    h.update(json.dumps(sorted(schema)).encode("utf-8"))
    h.update(f"|n_source={n_source}".encode())
    h.update(f"|n_target={n_target}".encode())

    if X_source is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            col_means_s = np.nanmean(X_source, axis=0)
        h.update(np.round(col_means_s, 6).tobytes())

    if X_target is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            col_means_t = np.nanmean(X_target, axis=0)
        h.update(np.round(col_means_t, 6).tobytes())

    return h.hexdigest()


def _safe_serialize(obj: Any) -> Any:
    """Convierte objetos numpy/etc. a tipos Python nativos para YAML/JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v  # NaN → None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def serialize_run_audit(
    schema: list[str],
    pair,
    auto_adapter,
    oracle_result: dict | None,
    drift_decomp: dict | None,
    counterfactuals: dict | None,
    calibration_raw: dict | None,
    calibration_adapted: dict | None,
    calibration_delta: dict | None,
    feature_attribution: list | None,
    config_dict: dict | None,
    output_dir: str = "outputs/audit",
    run_id: str | None = None,
) -> dict:
    """
    Serializa el audit completo del pipeline a un YAML y devuelve el dict.

    Parameters
    ----------
    schema : list[str]
    pair : CohortPair
    auto_adapter : AutoAdapter (ya fitted)
    oracle_result : dict o None
    drift_decomp : dict o None
    counterfactuals : dict o None
    calibration_raw, calibration_adapted, calibration_delta : dict o None
    feature_attribution : list[dict] o None
    config_dict : dict o None — FullConfig serializado (sin secrets)
    output_dir : str
    run_id : str o None — si None se genera UUID4

    Returns
    -------
    dict — el documento de audit completo (también guardado en disco)
    """
    import yaml  # tipo yaml

    if run_id is None:
        run_id = str(uuid.uuid4())

    timestamp = datetime.now(timezone.utc).isoformat()

    # Hash de inputs
    input_hash = _compute_input_hash(
        schema=schema,
        n_source=int(len(pair.y_s)),
        n_target=int(len(pair.y_t)),
        X_source=pair.X_s,
        X_target=pair.X_t,
    )

    # Versiones
    versions = {
        "recal_core": _adapt_version(),
        **{pkg: _pkg_version(pkg) for pkg in _TRACKED_PACKAGES},
    }

    # Audit trail del Designer
    audit_trail = []
    if auto_adapter.audit is not None:
        try:
            audit_trail = auto_adapter.audit.to_dict()
        except Exception as e:
            logger.warning("No se pudo serializar audit trail: %s", e)

    # Feature log
    feature_log = {}
    try:
        feature_log = auto_adapter.feature_log or {}
    except Exception as e:
        logger.warning("No se pudo obtener feature_log: %s", e)

    # Config (sin paths sensibles, solo sección recal_core + overfitting)
    safe_config = {}
    if config_dict:
        safe_config = {
            k: v for k, v in config_dict.items()
            if k in ("recal_core", "overfitting_check", "regularization", "joint_drift")
        }

    audit_doc = {
        "run_id": run_id,
        "timestamp": timestamp,
        "input_hash": input_hash,
        "versions": versions,
        "config_used": _safe_serialize(safe_config),
        "designer_audit_trail": _safe_serialize(audit_trail),
        "feature_log": _safe_serialize(feature_log),
        "drift_decomposition": _safe_serialize(drift_decomp),
        "oracle_result": _safe_serialize(
            {k: v for k, v in (oracle_result or {}).items() if k != "oof_scores"}
        ),
        "feature_attribution": _safe_serialize(feature_attribution),
        "counterfactuals": _safe_serialize(counterfactuals),
        "calibration_decomposition_raw": _safe_serialize(calibration_raw),
        "calibration_decomposition_adapted": _safe_serialize(calibration_adapted),
        "calibration_delta": _safe_serialize(calibration_delta),
    }

    # Guardar a disco
    out_path = Path(output_dir) / f"{run_id}.yaml"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(audit_doc, f, allow_unicode=True, default_flow_style=False,
                      sort_keys=False)
        logger.info("Audit YAML saved to %s", out_path)
    except Exception as e:
        logger.warning("Could not save audit YAML: %s", e)

    audit_doc["_audit_path"] = str(out_path)
    return audit_doc
