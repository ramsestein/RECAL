"""
recal_core.profiler.quadrant
========================
Asignación de cuadrantes SHAP × L_base para cada feature.

Cuadrantes
----------
La normalización y los umbrales se calculan sobre la distribución completa
de source. El umbral es la mediana de cada eje (percentil 50).

    A_core            : SHAP alto,  L_base alto  → núcleo transferible
    B_noisy_important : SHAP alto,  L_base bajo  → relevante pero ruidoso
    C_redundant       : SHAP bajo,  L_base alto  → redundante, prescindible
    D_ponzonous       : SHAP bajo,  L_base bajo  → candidata a máscara
"""

from __future__ import annotations

import numpy as np

from recal_core.profiler.constants import QUADRANT_THRESHOLD_PERCENTILE


def assign_quadrants(
    shap_importance: np.ndarray,
    lbase_scores: np.ndarray,
    percentile: float = QUADRANT_THRESHOLD_PERCENTILE,
) -> list[str]:
    """
    Asigna un cuadrante a cada feature basándose en sus scores SHAP y L_base.

    Parameters
    ----------
    shap_importance : np.ndarray, shape (p,)
        Mean |SHAP values| del modelo en source.
    lbase_scores : np.ndarray, shape (p,)
        R² o AUROC del predictor LASSO en source.
    percentile : float
        Percentil usado como umbral (default: 50 = mediana).

    Returns
    -------
    list[str]
        Cuadrante por feature: 'A_core' | 'B_noisy_important' |
        'C_redundant' | 'D_ponzonous'
    """
    shap = np.asarray(shap_importance, dtype=float)
    lbase = np.asarray(lbase_scores, dtype=float)

    # Normalizar a [0, 1] por percentiles
    shap_norm = _percentile_normalize(shap)
    lbase_norm = _percentile_normalize(lbase)

    # Umbral = percentil indicado (default mediana)
    shap_threshold = np.percentile(shap_norm, percentile)
    lbase_threshold = np.percentile(lbase_norm, percentile)

    quadrants = []
    for s, l in zip(shap_norm, lbase_norm):
        if s >= shap_threshold and l >= lbase_threshold:
            quadrants.append("A_core")
        elif s >= shap_threshold and l < lbase_threshold:
            quadrants.append("B_noisy_important")
        elif s < shap_threshold and l >= lbase_threshold:
            quadrants.append("C_redundant")
        else:
            quadrants.append("D_ponzonous")

    return quadrants


def _percentile_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normaliza array a [0, 1] usando los percentiles 1 y 99 como límites
    (robusto a outliers).
    """
    lo = float(np.nanpercentile(arr, 1))
    hi = float(np.nanpercentile(arr, 99))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    normalized = (arr - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0)
