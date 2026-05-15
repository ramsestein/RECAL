"""
domain_transfer.select.sweep
============================
sweep_mask_n: barre el número N de features ponzoñosas a enmascarar y devuelve
la curva AUROC-alineado / GAP-dominio para encontrar el óptimo.

Filosofía
---------
Las features con bajo score combinado (L_base bajo + SHAP bajo) son
*ponzoñosas para la transferencia de dominio*: no aportan señal predictiva en
el source y tienen alta inestabilidad cross-domain.  Cuando se incluyen en la
PCA de CORAL, los componentes principales capturan covarianzas espurias de
drift → el alineamiento se contamina → el AUROC alineado cae.

Enmascarar las N peores libera a la PCA para capturar las dimensiones de
señal real → el alineamiento mejora.  El N óptimo se encuentra donde el AUROC
alineado se maximiza (o el GAP se minimiza).

Nota: esta función requiere y_t (labels del target) y es por tanto una
utilidad de *investigación/ajuste*, no del pipeline operacional.  El pipeline
operacional usa MetaDriftPredictor (sin y_t) para obtener los scores.

Uso típico
----------
    from domain_transfer.select.sweep import sweep_mask_n
    from domain_transfer.select.combined_score import CombinedScoreSelector

    selector = CombinedScoreSelector(n_to_mask=1).fit(lbase_arr, shap_arr)
    df, n_opt = sweep_mask_n(pair_base, model, selector.scores_, schema)
    pair = pair_base.mask_features(selector.get_mask_names(schema, n=n_opt))
"""

from __future__ import annotations

from typing import Sequence, Type

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sweep_mask_n(
    pair_base,
    model,
    scores: np.ndarray,
    schema: list[str],
    n_range: Sequence[int] = range(0, 21),
    aligner_cls: Type | None = None,
    aligner_kwargs: dict | None = None,
    criterion: str = "max_aligned",
) -> tuple[pd.DataFrame, int]:
    """
    Barre N en ``n_range``, enmascarando las N features con menor
    ``scores``, y evalúa el AUROC del modelo alineado en el target.

    Parameters
    ----------
    pair_base : CohortPair
        Par de cohorts SIN máscara (ya con ``filter_target`` aplicado).
        Debe tener ``y_t`` disponible para calcular el AUROC (uso analítico).
    model : XGBoostWrapper
        Modelo con ``predict_proba()``.
    scores : np.ndarray, shape (p,)
        Score combinado por feature (menor = más ponzoñosa para la transfer).
        Típicamente ``CombinedScoreSelector.scores_``.
    schema : list[str]
        Nombres de las p features en el mismo orden que ``scores``.
    n_range : Sequence[int]
        Valores de N a evaluar.  Default: ``range(0, 21)``.
    aligner_cls : class, optional
        Clase del aligner a instanciar.  Default: ``PCACoralAligner``.
    aligner_kwargs : dict, optional
        Kwargs pasados al constructor del aligner.  Default: ``{"k": 5}``.
    criterion : {"max_aligned", "min_gap"}
        Criterio para elegir el N óptimo:

        - ``"max_aligned"`` — maximiza AUROC alineado en target.
        - ``"min_gap"``     — minimiza ``AUROC_s − AUROC_aligned`` (domain gap).

    Returns
    -------
    df : pd.DataFrame
        Una fila por N con columnas:
        ``n, feature_removed, auroc_s, auroc_raw, auroc_aligned,
        gap_raw, gap_aligned, delta_auroc``.
    optimal_n : int
        El N que optimiza el criterio indicado.

    Raises
    ------
    ValueError
        Si ``criterion`` no es reconocido.
    """
    if aligner_cls is None:
        from domain_transfer.align.pca_coral import PCACoralAligner
        aligner_cls = PCACoralAligner
    aligner_kwargs = aligner_kwargs or {"k": 5}

    if criterion not in ("max_aligned", "min_gap"):
        raise ValueError(
            f"criterion debe ser 'max_aligned' o 'min_gap', recibido: {criterion!r}"
        )

    sorted_indices = np.argsort(scores)  # ascendente: 0 = más ponzoñosa
    rows: list[dict] = []

    for n in n_range:
        if n == 0:
            pair_n = pair_base
            last_feat = "—"
        else:
            mask_names = [schema[int(sorted_indices[i])] for i in range(n)]
            pair_n = pair_base.mask_features(mask_names)
            last_feat = schema[int(sorted_indices[n - 1])]

        aligner = aligner_cls(**aligner_kwargs)
        X_aligned = pair_n.align(aligner)

        auroc_s = float(roc_auc_score(pair_n.y_s, model.predict_proba(pair_n.X_s)))
        auroc_raw = float(roc_auc_score(pair_n.y_t, model.predict_proba(pair_n.X_t)))
        auroc_aligned = float(roc_auc_score(pair_n.y_t, model.predict_proba(X_aligned)))

        rows.append(
            {
                "n": n,
                "feature_removed": last_feat,
                "auroc_s": auroc_s,
                "auroc_raw": auroc_raw,
                "auroc_aligned": auroc_aligned,
                "gap_raw": auroc_s - auroc_raw,
                "gap_aligned": auroc_s - auroc_aligned,
                "delta_auroc": auroc_aligned - auroc_raw,
            }
        )

    df = pd.DataFrame(rows)

    if criterion == "max_aligned":
        optimal_n = int(df.loc[df["auroc_aligned"].idxmax(), "n"])
    else:  # min_gap
        optimal_n = int(df.loc[df["gap_aligned"].idxmin(), "n"])

    return df, optimal_n
