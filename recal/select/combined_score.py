"""
recal.select.combined_score
======================================
CombinedScoreSelector: enmascara las N features con menor score combinado.

Score combinado por feature j:
    score_j = normalize(lbase_score_j) + normalize(shap_importance_j) [+ normalize(d_j)]

Donde normalize() escala cada array a [0, 1].  Un score más bajo indica que
la feature es menos informativa o más perjudicial para la transferencia →
candidata a ser enmascarada.
"""

from __future__ import annotations

import numpy as np


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Escala arr a [0, 1].  Devuelve arr de ceros si rango == 0."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


class CombinedScoreSelector:
    """
    Selector de features basado en la suma normalizada de scores de importancia.

    Parameters
    ----------
    n_to_mask : int
        Número de features a enmascarar (las N con menor score combinado).

    Attributes
    ----------
    scores_ : np.ndarray, shape (p,)
        Score combinado por feature (disponible tras ``fit``).
    bottom_indices_ : np.ndarray, shape (n_to_mask,)
        Índices (enteros) de las features a enmascarar, ordenados de menor
        a mayor score.
    """

    def __init__(self, n_to_mask: int) -> None:
        if n_to_mask < 1:
            raise ValueError(f"n_to_mask debe ser ≥ 1, recibido: {n_to_mask}")
        self.n_to_mask = n_to_mask
        self.scores_: np.ndarray | None = None
        self._sorted_indices_: np.ndarray | None = None  # orden completo ascendente
        self.bottom_indices_: np.ndarray | None = None

    def fit(
        self,
        lbase_scores: np.ndarray,
        shap_importance: np.ndarray,
        meta_drift_scores: np.ndarray | None = None,
    ) -> CombinedScoreSelector:
        """
        Calcula el score combinado y selecciona las N peores features.

        Parameters
        ----------
        lbase_scores : np.ndarray, shape (p,)
            Valores absolutos de los coeficientes LASSO logístico
            (y_s ~ X_s) — mayor = más discriminativo en source.
        shap_importance : np.ndarray, shape (p,)
            mean|SHAP values| del modelo en source — mayor = más importante.
        meta_drift_scores : np.ndarray, shape (p,), optional
            Scores d_j de MetaDriftPredictor — mayor = más beneficioso para
            la transferencia.  Si None, no se incluye en la suma.

        Returns
        -------
        self
        """
        p = len(lbase_scores)
        if len(shap_importance) != p:
            raise ValueError("lbase_scores y shap_importance deben tener la misma longitud.")

        score = _minmax(np.asarray(lbase_scores, dtype=float)) \
              + _minmax(np.asarray(shap_importance, dtype=float))

        if meta_drift_scores is not None:
            if len(meta_drift_scores) != p:
                raise ValueError("meta_drift_scores debe tener longitud p.")
            score = score + _minmax(np.asarray(meta_drift_scores, dtype=float))

        self.scores_ = score
        # Orden ascendente completo (menor score = más ponzoñosa)
        self._sorted_indices_ = np.argsort(score)
        self.bottom_indices_ = self._sorted_indices_[: self.n_to_mask]
        return self

    def get_mask_indices(self, n: int | None = None) -> np.ndarray:
        """
        Devuelve los índices de features a enmascarar.

        Parameters
        ----------
        n : int, optional
            Número de features a enmascarar.  Si None, usa ``self.n_to_mask``.

        Returns
        -------
        np.ndarray, shape (n,)

        Raises
        ------
        RuntimeError
            Si ``fit`` no ha sido llamado.
        ValueError
            Si ``n`` excede el número de features.
        """
        if self._sorted_indices_ is None:
            raise RuntimeError(
                "CombinedScoreSelector is not fitted.  Call fit() first."
            )
        k = self.n_to_mask if n is None else int(n)
        if k > len(self.scores_):
            raise ValueError(
                f"n={k} exceeds the number of features ({len(self.scores_)})."
            )
        return self._sorted_indices_[:k]

    def get_mask_names(self, schema: list[str], n: int | None = None) -> list[str]:
        """
        Devuelve los nombres de features a enmascarar.

        Parameters
        ----------
        schema : list[str]
            Lista de nombres de features en el mismo orden que los arrays
            pasados a ``fit``.
        n : int, optional
            Número de features.  Si None, usa ``self.n_to_mask``.

        Returns
        -------
        list[str]
        """
        return [schema[i] for i in self.get_mask_indices(n=n)]
