"""
adapt.reporter.figures
=======================
Genera 4 figuras matplotlib para el reporte ADAPT.

Figura 1: Mapa de cuadrantes (SHAP vs L_base, color = drift_type)
Figura 2: Calibration curve before/after (reliability diagram)
Figura 3: Combined score distribution (horizontal bar, top-20)
Figura 4: Feature missing rates (source vs target, top-20 por miss_target)

Cada función devuelve una figura matplotlib.Figure. El caller es responsable
de cerrar la figura con plt.close(fig) después de exportarla.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # backend sin GUI (compatible con cualquier entorno)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from adapt.profiler.base import DriftProfile

# Paleta de colores por tipo de drift
_DRIFT_COLORS = {
    "STABLE": "#4CAF50",
    "LINEAR_RECOVERABLE": "#8BC34A",
    "NONLINEAR_DRIFT": "#FF9800",
    "PARTIAL_RECOVERY": "#FF5722",
    "CONCEPT_RELATIONAL": "#9C27B0",
    "DEGENERATE": "#607D8B",
    "INSUFFICIENT_DATA": "#9E9E9E",
    "INSUFFICIENT_CLINIC_DATA": "#9E9E9E",
    "LOW_VARIANCE_TARGET": "#00BCD4",
    "unknown": "#BDBDBD",
}

# Paleta de cuadrantes
_QUADRANT_COLORS = {
    "A_core": "#2196F3",
    "B_noisy_important": "#FF9800",
    "C_redundant": "#9E9E9E",
    "D_ponzonous": "#F44336",
}


def figure_quadrant_map(profile: DriftProfile, figsize=(9, 7)) -> plt.Figure:
    """
    Figura 1: Mapa de cuadrantes SHAP vs L_base.

    Puntos coloreados por drift_type. Cuadrantes marcados con líneas punteadas
    en los umbrales medianos.
    """
    features = profile.features
    shap = np.array([f.shap_importance for f in features])
    lbase = np.array([f.lbase_score for f in features])
    types = [f.drift_type_v for f in features]
    names = [f.name for f in features]

    fig, ax = plt.subplots(figsize=figsize)

    # Colorear por drift_type
    unique_types = sorted(set(types))
    for dt in unique_types:
        idx = [i for i, t in enumerate(types) if t == dt]
        color = _DRIFT_COLORS.get(dt, "#BDBDBD")
        ax.scatter(
            lbase[idx], shap[idx],
            c=color, label=dt, alpha=0.75, edgecolors="white",
            linewidths=0.5, s=60,
        )

    # Líneas de umbral (medianas)
    lbase_med = float(np.nanmedian(lbase))
    shap_med = float(np.nanmedian(shap))
    ax.axvline(lbase_med, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(shap_med, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # Anotaciones de cuadrante
    xmax = float(np.nanmax(lbase)) * 1.02
    ymax = float(np.nanmax(shap)) * 1.02
    xmin = float(np.nanmin(lbase)) * 0.98
    ymin = float(np.nanmin(shap)) * 0.98

    for label, x, y, ha, va in [
        ("A_core", xmax, ymax, "right", "top"),
        ("B_noisy_important", xmin, ymax, "left", "top"),
        ("C_redundant", xmin, ymin, "left", "bottom"),
        ("D_ponzonous", xmax, ymin, "right", "bottom"),
    ]:
        ax.text(x, y, label, ha=ha, va=va, fontsize=8,
                color=_QUADRANT_COLORS.get(label, "gray"), alpha=0.7)

    ax.set_xlabel("L_base (LASSO logístico en source)", fontsize=10)
    ax.set_ylabel("SHAP importance (modelo source)", fontsize=10)
    ax.set_title("ADAPT: Mapa de cuadrantes de features", fontsize=12)
    ax.legend(title="Drift type", fontsize=7, title_fontsize=8,
              bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    return fig


def figure_calibration_curve(
    y_true: np.ndarray,
    scores_before: np.ndarray,
    scores_after: Optional[np.ndarray] = None,
    n_bins: int = 10,
    figsize=(7, 6),
) -> plt.Figure:
    """
    Figura 2: Reliability diagram (calibration curve) antes/después.

    Bins de frecuencia iguales.
    """
    fig, ax = plt.subplots(figsize=figsize)

    def _plot_calibration(ax, y, scores, label, color, marker):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(scores, bins) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)

        frac_pos = []
        mean_pred = []
        counts = []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() >= 3:
                frac_pos.append(y[mask].mean())
                mean_pred.append(scores[mask].mean())
                counts.append(mask.sum())

        frac_pos = np.array(frac_pos)
        mean_pred = np.array(mean_pred)
        counts = np.array(counts)

        ax.plot(mean_pred, frac_pos, marker=marker, color=color, label=label,
                linewidth=1.8, markersize=8)
        # Tamaño de punto proporcional al n
        for xp, yp, c in zip(mean_pred, frac_pos, counts):
            ax.scatter(xp, yp, s=c * 0.5, color=color, alpha=0.25, zorder=2)

    _plot_calibration(ax, y_true, scores_before, "Antes (raw)", "#F44336", "o")
    if scores_after is not None:
        _plot_calibration(ax, y_true, scores_after, "Después (ADAPT)", "#2196F3", "^")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6, label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha (media del bin)", fontsize=10)
    ax.set_ylabel("Fracción de positivos observados", fontsize=10)
    ax.set_title("Reliability Diagram (calibración)", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig


def figure_combined_score_bar(
    profile: DriftProfile,
    n_top: int = 20,
    figsize=(9, 7),
) -> plt.Figure:
    """
    Figura 3: Barras horizontales de combined_score para las top-20 features
    con menor score (las candidatas a enmascarar).
    """
    features = sorted(profile.features, key=lambda f: f.combined_score)[:n_top]
    names = [f.name for f in features]
    scores = [f.combined_score for f in features]
    quadrants = [f.quadrant for f in features]
    colors = [_QUADRANT_COLORS.get(q, "#BDBDBD") for q in quadrants]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Combined Score (L_base + SHAP norm.)", fontsize=10)
    ax.set_title(f"Top-{n_top} features por combined score (candidatas a máscara)", fontsize=11)

    # Leyenda de cuadrantes
    patches = [
        mpatches.Patch(color=c, label=q)
        for q, c in _QUADRANT_COLORS.items()
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def figure_missing_rates(
    profile: DriftProfile,
    n_top: int = 20,
    figsize=(9, 7),
) -> plt.Figure:
    """
    Figura 4: Missing rates source vs target para las top-20 features
    con mayor missing rate en target.
    """
    features = sorted(
        profile.features, key=lambda f: f.missing_rate_target, reverse=True
    )[:n_top]
    names = [f.name for f in features]
    miss_s = [f.missing_rate_source for f in features]
    miss_t = [f.missing_rate_target for f in features]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, miss_s, width, label="Source (SNUH)", color="#4CAF50", alpha=0.8)
    ax.bar(x + width / 2, miss_t, width, label="Target (Clínic)", color="#F44336", alpha=0.8)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Umbral filtro (50%)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Missing rate (fracción)", fontsize=10)
    ax.set_title(f"Top-{n_top} features por missing rate en target", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig
