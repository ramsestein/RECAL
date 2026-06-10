"""
recal.drift.analyzer
================================
MetaDriftAnalyzer: componente diagnóstico de drift.

Diseño (Opción C, Componente 2 — ver OPEN_QUESTIONS.md §B3)
-------------------------------------------------------------
Calcula señales de drift per-feature que **sí requieren labels de target**
(y_t).  Vive en ``drift/``, nunca en ``select/``.  Su output no alimenta
ningún selector; va al reporte como tabla explicativa.

Propósito
---------
1. Diagnóstico retrospectivo: "¿qué features driftaron realmente y cómo?"
2. Puente de validación (Componente 3, B8): correlacionar las predicciones
   de ``MetaDriftPredictor`` (sintéticas, sin y_t) con el drift real
   observado en Clínic.

Señales calculadas
-------------------
- Todas las de ``compute_drift_features`` (P(X) de source y target).
- ``spearman_flip`` : 1 si el signo de Spearman(feature, label) se invierte
  entre source y target.  **Requiere y_t.**
- ``delta_auroc_real`` : ΔAUROC real al enmascarar la feature j en el target.
  ΔAUROC_j = AUROC(completo) − AUROC(feature j → source mean).
  **Requiere y_t.**

Por qué ``spearman_flip`` solo es analítico
--------------------------------------------
Si spearman_flip estuviese en el predictor operacional, el argumento "el
selector funciona en un hospital nuevo sin labels" se cae: para computarlo
necesitas y_t.  El componente operacional (MetaDriftPredictor) usa solo
señales de P(X) — ver ``recal.select.meta_drift``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from recal.select.meta_drift import compute_drift_features

logger = logging.getLogger(__name__)


# ── MetaDriftAnalyzer ─────────────────────────────────────────────────────────

class MetaDriftAnalyzer:
    """
    Analizador retrospectivo de drift por feature (requiere y_t).

    Uso típico
    ----------
    >>> analyzer = MetaDriftAnalyzer()
    >>> report = analyzer.analyze(X_s, X_t, y_s, y_t, model, schema=schema)
    >>> # report es un DataFrame con una fila por feature

    Parameters
    ----------
    Ninguno en el constructor.  Toda la configuración se pasa a ``analyze``.
    """

    @staticmethod
    def _spearman_flip(
        X_s: np.ndarray,
        X_t: np.ndarray,
        y_s: np.ndarray,
        y_t: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula spearman_flip_j = 1 si el signo de Spearman(feature_j, label)
        se invierte entre source y target.

        Devuelve array de floats (0.0 o 1.0) de shape (p,).
        Se asigna 0 si alguno de los dos Spearman no es computable.
        """
        p = X_s.shape[1]
        flip = np.zeros(p)
        for j in range(p):
            xs_j = X_s[:, j]
            xt_j = X_t[:, j]

            xs_valid = ~np.isnan(xs_j)
            xt_valid = ~np.isnan(xt_j)

            if xs_valid.sum() < 3 or xt_valid.sum() < 3:
                continue
            # Necesita variación en ambas variables
            if np.std(xs_j[xs_valid]) < 1e-12 or len(np.unique(y_s[xs_valid])) < 2:
                continue
            if np.std(xt_j[xt_valid]) < 1e-12 or len(np.unique(y_t[xt_valid])) < 2:
                continue

            rho_s, _ = spearmanr(xs_j[xs_valid], y_s[xs_valid])
            rho_t, _ = spearmanr(xt_j[xt_valid], y_t[xt_valid])

            if np.isnan(rho_s) or np.isnan(rho_t):
                continue

            # Inversión si los signos difieren (tolerancia: ambos deben ser
            # suficientemente distintos de cero para no ser ruido)
            if np.sign(rho_s) != np.sign(rho_t) and (abs(rho_s) > 0.05 or abs(rho_t) > 0.05):
                flip[j] = 1.0

        return flip

    @staticmethod
    def _delta_auroc_real(
        model,
        X_t: np.ndarray,
        y_t: np.ndarray,
        mu_s: np.ndarray,
    ) -> np.ndarray:
        """
        ΔAUROC_j real: AUROC(completo) − AUROC(feature j → source mean) sobre
        los datos reales de target.

        Parameters
        ----------
        model : duck-typed, necesita ``predict_proba(X) → array``
        X_t : np.ndarray, shape (n_t, p) — con NaN donde corresponda
        y_t : np.ndarray, shape (n_t,)
        mu_s : np.ndarray, shape (p,) — medias de source para imputación

        Returns
        -------
        np.ndarray, shape (p,)
        """
        p = X_t.shape[1]

        # Imputa NaN con source mean
        X_imp = np.where(np.isnan(X_t), mu_s[np.newaxis, :], X_t)
        X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

        if len(np.unique(y_t)) < 2:
            logger.warning("_delta_auroc_real: solo una clase en y_t — ΔAUROC=0.")
            return np.zeros(p)

        proba_base = model.predict_proba(X_imp)
        if proba_base.ndim == 2:
            proba_base = proba_base[:, 1]
        auroc_base = roc_auc_score(y_t, proba_base)

        delta = np.zeros(p)
        for j in range(p):
            X_j = X_imp.copy()
            X_j[:, j] = mu_s[j]
            proba_j = model.predict_proba(X_j)
            if proba_j.ndim == 2:
                proba_j = proba_j[:, 1]
            delta[j] = auroc_base - roc_auc_score(y_t, proba_j)

        return delta

    def analyze(
        self,
        X_s: np.ndarray,
        X_t: np.ndarray,
        y_s: np.ndarray,
        y_t: np.ndarray,
        model,
        schema: list[str] | None = None,
        shap_importance_s: np.ndarray | None = None,
        lbase_score_s: np.ndarray | None = None,
        temporal_domain: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Analiza el drift por feature y devuelve un DataFrame diagnóstico.

        Parameters
        ----------
        X_s : np.ndarray, shape (n_s, p)
        X_t : np.ndarray, shape (n_t, p)
        y_s : np.ndarray, shape (n_s,) — labels de source
        y_t : np.ndarray, shape (n_t,) — labels de target  **[requiere y_t]**
        model : duck-typed, necesita ``predict_proba(X) → array``
        schema : list[str], optional
            Nombres de features.  Si None, usa "f0", "f1", ...
        shap_importance_s, lbase_score_s, temporal_domain : opcionales
            Pasados a ``compute_drift_features``.

        Returns
        -------
        pd.DataFrame
            Una fila por feature.  Columnas:
            - Todas las de DRIFT_FEATURE_COLS (señales P(X)).
            - ``spearman_flip``: 1 si el signo Spearman se invierte.
            - ``delta_auroc_real``: ΔAUROC al enmascarar en target real.
            - ``feature_name``: nombre de la feature (index del DataFrame).
        """
        p = X_s.shape[1]
        if schema is None:
            schema = [f"f{j}" for j in range(p)]
        assert len(schema) == p

        mu_s = np.nanmean(X_s, axis=0)

        # Señales P(X) — sin labels de target
        df_px = compute_drift_features(
            X_s, X_t,
            shap_importance_s=shap_importance_s,
            lbase_score_s=lbase_score_s,
            temporal_domain=temporal_domain,
        )

        # Señales que requieren y_t
        logger.info("MetaDriftAnalyzer.analyze: calculando spearman_flip (requiere y_t).")
        flip = self._spearman_flip(X_s, X_t, y_s, y_t)

        logger.info("MetaDriftAnalyzer.analyze: calculando delta_auroc_real (requiere y_t).")
        delta_real = self._delta_auroc_real(model, X_t, y_t, mu_s)

        df_px["spearman_flip"] = flip
        df_px["delta_auroc_real"] = delta_real
        df_px.index = schema
        df_px.index.name = "feature_name"

        return df_px
