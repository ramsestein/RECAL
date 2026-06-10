"""
recal_cli.calibration_decomposition
=====================================
Descomposición del Brier score en reliability − resolution + uncertainty.

Marco teórico (Murphy 1973)
---------------------------
    BS = reliability − resolution + uncertainty

    uncertainty  = p̄ * (1 − p̄)           — varianza de y (irreducible)
    resolution   = Σ_k n_k/n * (ȳ_k − p̄)² — variabilidad del forecast por bin
    reliability  = Σ_k n_k/n * (ȳ_k − p̄_k)² — calibration error por bin

Donde:
    p̄ = prevalencia del outcome en la muestra
    ȳ_k = fracción de outcomes en el bin k
    p̄_k = predicción media en el bin k
    n_k = número de observaciones en el bin k

Esta descomposición permite entender si una mejora en Brier viene de:
    - mejor reliability (calibración) → RECAL puede influir directamente
    - mejor resolution (discriminación) → capturada también por AUROC
    - uncertainty (irreducible) → fijo dado el outcome

API pública
-----------
    brier_decompose(y_true, scores, n_bins=10) → dict
    brier_delta(decomp_raw, decomp_adapted) → dict
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def brier_decompose(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Descompone el Brier score en sus componentes Murphy.

    Parameters
    ----------
    y_true : np.ndarray (n,)
        Labels binarios.
    scores : np.ndarray (n,)
        Probabilidades predichas ∈ [0, 1].
    n_bins : int
        Número de bins de calibración (default 10).

    Returns
    -------
    dict con:
        brier_score : float  — BS total = reliability - resolution + uncertainty
        reliability : float  — componente de miscalibración (↓ mejor)
        resolution  : float  — componente de discriminación (↑ mejor)
        uncertainty : float  — componente irreducible = p̄*(1−p̄)
        n_bins_used : int    — bins no vacíos efectivamente usados
        bin_stats   : list[dict] — [{bin_lo, bin_hi, n, mean_pred, mean_obs}, ...]
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)

    n = len(y)
    if n == 0:
        return {
            "brier_score": np.nan, "reliability": np.nan,
            "resolution": np.nan, "uncertainty": np.nan,
            "n_bins_used": 0, "bin_stats": [],
        }

    prevalence = float(np.mean(y))
    brier_total = float(np.mean((y - p) ** 2))

    # Bins uniformes en [0, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(p, bins, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    reliability = 0.0
    resolution = 0.0
    bin_stats = []

    for k in range(1, n_bins + 1):
        mask = bin_indices == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        mean_pred_k = float(np.mean(p[mask]))
        mean_obs_k = float(np.mean(y[mask]))

        reliability += (n_k / n) * (mean_obs_k - mean_pred_k) ** 2
        resolution += (n_k / n) * (mean_obs_k - prevalence) ** 2

        bin_stats.append({
            "bin_lo": float(bins[k - 1]),
            "bin_hi": float(bins[k]),
            "n": n_k,
            "mean_pred": mean_pred_k,
            "mean_obs": mean_obs_k,
        })

    uncertainty = prevalence * (1.0 - prevalence)

    return {
        "brier_score": brier_total,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "n_bins_used": len(bin_stats),
        "bin_stats": bin_stats,
        # Verificación interna
        "_check_bs": float(reliability - resolution + uncertainty),
    }


def brier_delta(
    decomp_raw: dict,
    decomp_adapted: dict,
) -> dict:
    """
    Calcula los deltas entre la descomposición raw y adaptada.

    Un delta negativo en reliability indica mejor calibración post-adaptación.
    Un delta positivo en resolution indica mejor discriminación post-adaptación.
    Un delta negativo en brier_score es siempre deseable.

    Parameters
    ----------
    decomp_raw : dict
        Salida de brier_decompose para el modelo raw (sin adaptar).
    decomp_adapted : dict
        Salida de brier_decompose para el modelo adaptado.

    Returns
    -------
    dict con:
        delta_brier_score : float   — adaptado - raw (negativo = mejora)
        delta_reliability : float   — adaptado - raw (negativo = mejor calibración)
        delta_resolution  : float   — adaptado - raw (positivo = mejor discriminación)
        delta_uncertainty : float   — siempre ≈ 0 (invariante a la adaptación)
        interpretation    : str
    """
    def _d(key: str) -> float:
        a = decomp_adapted.get(key)
        r = decomp_raw.get(key)
        if a is None or r is None or np.isnan(a) or np.isnan(r):
            return np.nan
        return float(a) - float(r)

    d_bs = _d("brier_score")
    d_rel = _d("reliability")
    d_res = _d("resolution")
    d_unc = _d("uncertainty")

    # Automatic interpretation
    parts = []
    if not np.isnan(d_bs):
        if d_bs < -1e-4:
            parts.append(f"Brier improved ({d_bs:+.4f})")
        elif d_bs > 1e-4:
            parts.append(f"Brier worsened ({d_bs:+.4f})")
        else:
            parts.append("Brier: no significant change")
    if not np.isnan(d_rel):
        if d_rel < -1e-4:
            parts.append(f"calibration improved (\u0394rel={d_rel:+.4f})")
        elif d_rel > 1e-4:
            parts.append(f"calibration worsened (\u0394rel={d_rel:+.4f})")
    if not np.isnan(d_res):
        if d_res > 1e-4:
            parts.append(f"discrimination improved (\u0394res={d_res:+.4f})")
        elif d_res < -1e-4:
            parts.append(f"discrimination worsened (\u0394res={d_res:+.4f})")

    interpretation = "; ".join(parts) if parts else "No significant change."

    return {
        "delta_brier_score": d_bs,
        "delta_reliability": d_rel,
        "delta_resolution": d_res,
        "delta_uncertainty": d_unc,
        "interpretation": interpretation,
    }
