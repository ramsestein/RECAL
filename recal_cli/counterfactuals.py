"""
recal_cli.counterfactuals
==========================
Análisis de sensibilidad contrafactual de las decisiones del Designer.

Para cada decisión principal (mask_n, pca_coral_k, calibration_method),
se corren N alternativas vecinas a la elegida y se reportan sus métricas.

Esto permite evaluar:
- ¿Cómo de sensible es el resultado al valor exacto de mask_n?
- ¿Importa mucho el número de componentes PCA-CORAL?
- ¿Qué calibración habría dado mejores resultados?

Config: `counterfactual_alternatives` (default 3) por decisión.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _auroc_safe(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """Calcula AUROC con manejo de errores."""
    try:
        return float(roc_auc_score(y_true, scores))
    except (ValueError, Exception):
        return None


def _run_pipeline_variant(
    pair,
    model,
    schema: list[str],
    mask_n: int,
    mask_features: list[str],
    pca_k: int,
    calibration_method: str | None,
    apply_pca_coral: bool,
    apply_calibration: bool,
    base_adapter,
) -> float | None:
    """
    Ejecuta una variante de la pipeline con parámetros específicos.
    Reutiliza el perfil del adapter base (no re-perfila).
    """
    from recal_core.designer.base import AdapterConfig
    from recal_core.pipeline.auto_adapter import AutoAdapter

    config_variant = AdapterConfig()
    config_variant.apply_mask = mask_n > 0
    config_variant.mask_n = mask_n
    config_variant.mask_features = mask_features[:mask_n] if mask_n > 0 else []
    config_variant.mask_selection_method = "counterfactual"

    # Heredar QT y WOE del adapter base
    base_cfg = base_adapter._config
    config_variant.apply_quantile = base_cfg.apply_quantile
    config_variant.quantile_features = base_cfg.quantile_features
    config_variant.quantile_output_distribution = base_cfg.quantile_output_distribution
    config_variant.apply_woe = base_cfg.apply_woe
    config_variant.woe_features = base_cfg.woe_features
    config_variant.woe_n_bins = base_cfg.woe_n_bins

    config_variant.apply_pca_coral = apply_pca_coral
    config_variant.pca_coral_k = pca_k
    config_variant.pca_coral_k_selection_method = "counterfactual"

    config_variant.apply_calibration = apply_calibration
    config_variant.calibration_method = calibration_method or "platt_loo"
    config_variant.rationale = {"source": "counterfactual"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aa_v = AutoAdapter(
            model=model,
            schema=schema,
            drift_type_dict=base_adapter._drift_type_dict,
            shap_dict=base_adapter._shap_dict,
            lbase_dict=base_adapter._lbase_dict,
        )
        aa_v._profile = base_adapter._profile  # reusa perfil
        aa_v._config = config_variant

        try:
            aa_v.fit(pair)
            scores = aa_v.predict(pair)
            return _auroc_safe(pair.y_t, scores)
        except Exception as e:
            logger.debug("Variante fallida (mask_n=%d, k=%d): %s", mask_n, pca_k, e)
            return None


def run_counterfactual_sensitivity(
    auto_adapter,
    pair,
    model,
    schema: list[str],
    n_alternatives: int = 3,
) -> dict:
    """
    Corre análisis de sensibilidad contrafactual sobre las decisiones principales.

    Para cada decisión: genera n_alternatives vecinos y computa AUROC.

    Parameters
    ----------
    auto_adapter : AutoAdapter
        AutoAdapter ya fitted con perfil y config disponibles.
    pair : CohortPair
        Par (source, target) para evaluar.
    model : object
        Modelo source.
    schema : list[str]
        Schema de features.
    n_alternatives : int
        Número de alternativas por decisión (default 3).

    Returns
    -------
    dict con secciones:
        mask_n     : list[dict] — [{choice, auroc, selected}, ...]
        pca_coral_k: list[dict]
        calibration: list[dict]
    """
    config = auto_adapter._config
    profile = auto_adapter._profile

    if config is None or profile is None:
        logger.warning("AutoAdapter no fitted; no se puede correr counterfactuals.")
        return {}

    # Features base para mask (ordenadas por combined_score)
    sorted_features = sorted(profile.features, key=lambda f: f.combined_score)
    sorted_feat_names = [f.name for f in sorted_features]

    # AUROC del elegido (línea base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores_base = auto_adapter.predict(pair)
            auroc_base = _auroc_safe(pair.y_t, scores_base)
        except Exception:
            auroc_base = None

    results = {}

    # ── 1. Mask N ─────────────────────────────────────────────────────────────
    chosen_mask_n = config.mask_n
    p = len(sorted_feat_names)
    max_mask = max(1, min(30, p // 4))

    mask_alternatives = _generate_neighbors(
        chosen_mask_n, n_alternatives, lo=0, hi=max_mask, step=1
    )

    mask_results = []
    for n_val in sorted(set([chosen_mask_n] + mask_alternatives)):
        if n_val == chosen_mask_n:
            mask_results.append({
                "choice": n_val, "auroc": auroc_base, "selected": True,
            })
            continue
        auroc_v = _run_pipeline_variant(
            pair=pair, model=model, schema=schema,
            mask_n=n_val, mask_features=sorted_feat_names,
            pca_k=config.pca_coral_k if config.apply_pca_coral else 5,
            calibration_method=config.calibration_method if config.apply_calibration else None,
            apply_pca_coral=config.apply_pca_coral,
            apply_calibration=config.apply_calibration,
            base_adapter=auto_adapter,
        )
        mask_results.append({
            "choice": n_val, "auroc": auroc_v, "selected": False,
        })
    results["mask_n"] = sorted(mask_results, key=lambda x: x["choice"])

    # ── 2. PCA-CORAL k ────────────────────────────────────────────────────────
    if config.apply_pca_coral:
        chosen_k = config.pca_coral_k
        max_k = min(20, pair.X_t_imp.shape[1], pair.X_s_imp.shape[0] - 1)
        k_alternatives = _generate_neighbors(
            chosen_k, n_alternatives, lo=1, hi=max_k, step=1
        )

        k_results = []
        for k_val in sorted(set([chosen_k] + k_alternatives)):
            if k_val == chosen_k:
                k_results.append({
                    "choice": k_val, "auroc": auroc_base, "selected": True,
                })
                continue
            auroc_v = _run_pipeline_variant(
                pair=pair, model=model, schema=schema,
                mask_n=chosen_mask_n, mask_features=sorted_feat_names,
                pca_k=k_val,
                calibration_method=config.calibration_method if config.apply_calibration else None,
                apply_pca_coral=True,
                apply_calibration=config.apply_calibration,
                base_adapter=auto_adapter,
            )
            k_results.append({
                "choice": k_val, "auroc": auroc_v, "selected": False,
            })
        results["pca_coral_k"] = sorted(k_results, key=lambda x: x["choice"])

    # ── 3. Calibración ────────────────────────────────────────────────────────
    if config.apply_calibration:
        chosen_cal = config.calibration_method
        all_cal_methods = ["platt_loo", "platt_stratified", "isotonic_loo"]
        cal_alternatives = [m for m in all_cal_methods if m != chosen_cal][:n_alternatives]

        cal_results = []
        for method in ([chosen_cal] + cal_alternatives):
            if method == chosen_cal:
                cal_results.append({
                    "choice": method, "auroc": auroc_base, "selected": True,
                })
                continue
            auroc_v = _run_pipeline_variant(
                pair=pair, model=model, schema=schema,
                mask_n=chosen_mask_n, mask_features=sorted_feat_names,
                pca_k=config.pca_coral_k if config.apply_pca_coral else 5,
                calibration_method=method,
                apply_pca_coral=config.apply_pca_coral,
                apply_calibration=True,
                base_adapter=auto_adapter,
            )
            cal_results.append({
                "choice": method, "auroc": auroc_v, "selected": False,
            })
        results["calibration"] = cal_results

    return results


def _generate_neighbors(
    value: int,
    n: int,
    lo: int,
    hi: int,
    step: int = 1,
) -> list[int]:
    """
    Genera n vecinos enteros alrededor de `value` dentro de [lo, hi].

    Distribuye equitativamente hacia abajo y hacia arriba.
    """
    neighbors = []
    radius = 1
    while len(neighbors) < n and (value - radius * step >= lo or value + radius * step <= hi):
        for sign in (-1, 1):
            candidate = value + sign * radius * step
            if lo <= candidate <= hi and candidate != value:
                neighbors.append(candidate)
                if len(neighbors) >= n:
                    break
        radius += 1
    return neighbors[:n]
