"""
domain_transfer.select.meta_drift
==================================
MetaDriftPredictor: operacional feature-drift scorer.

Diseño (Opción C, Componente 1 — ver OPEN_QUESTIONS.md §B3)
-------------------------------------------------------------
Entrena un meta-regresor que mapea señales de drift por feature (calculadas
exclusivamente desde P(X) de source y target, sin labels de target) al
ΔAUROC esperado cuando esa feature se enmascara.

Calibrado íntegramente en pseudo-targets sintéticos derivados de los datos
de source.  No requiere y_t en ningún momento — válido en un hospital nuevo
sin labels etiquetados.

Lo que este módulo NO usa
--------------------------
* y_t  — labels del hospital destino.
* spearman_flip — inversión del signo de Spearman(feature, label) entre
  source y target.  Requiere y_t.  Ver ``MetaDriftAnalyzer`` en
  ``domain_transfer.drift.analyzer`` para el componente diagnóstico.

Inputs por feature j (todos desde P(X))
-----------------------------------------
ks_stat, wasserstein_dist, mean_shift (normalizado por σ_s), std_ratio
(σ_t/σ_s), nan_rate_t, nan_rate_s, shap_importance_s, lbase_score_s
(coef. LASSO logístico y_s ~ X_s), variance_s, skewness_s, kurtosis_s,
temporal_domain (0=preop / 1=intraop / 2=postop, opcional).

Workflow de entrenamiento sintético
-------------------------------------
1. Para cada simulación k de K=50:
   a. Genera pseudo-target X_t_k desde X_s aplicando:
      - mean-shift_j ~ U(-2σ_j, 2σ_j)
      - scale_j      ~ U(0.5, 2.0)
      - inyección de NaN según nan_rate_t (si se proporciona)
   b. Calcula el vector de drift features por cada feature j.
   c. Calcula ΔAUROC_j = AUROC(X_t_k completo) − AUROC(X_t_k con feature j
      igualada a la media de source — sin señal target-específica).
2. Pool: K × p muestras (features de drift → ΔAUROC).
3. Ajusta un regresor XGBoost (o GradientBoostingRegressor como fallback).

Workflow de inferencia (sin y_t)
----------------------------------
1. Calcula drift features para cada feature j sobre (X_s, X_t_real).
2. Predice d_j = ΔAUROC esperado.
   d_j alto  → feature beneficiosa para la transferencia (conservar).
   d_j bajo  → candidata a enmascarar.
3. Pasa d_j a ``CombinedScoreSelector``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.stats import skew as scipy_skew
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# ── Feature names (columnas del DataFrame de entrenamiento del meta-modelo) ──

DRIFT_FEATURE_COLS = [
    "ks_stat",
    "wasserstein_dist",
    "mean_shift",
    "std_ratio",
    "nan_rate_t",
    "nan_rate_s",
    "shap_importance_s",
    "lbase_score_s",
    "variance_s",
    "skewness_s",
    "kurtosis_s",
    "temporal_domain",  # 0/1/2 — preop/intraop/postop; 0 si no se proporciona
]


# ── Compute drift features ────────────────────────────────────────────────────

def compute_drift_features(
    X_s: np.ndarray,
    X_t: np.ndarray,
    shap_importance_s: Optional[np.ndarray] = None,
    lbase_score_s: Optional[np.ndarray] = None,
    temporal_domain: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Calcula el vector de drift features para cada feature j.

    Solo usa P(X) de source y target — NO usa labels de target.

    Parameters
    ----------
    X_s : np.ndarray, shape (n_s, p)
        Matriz de source (puede contener NaN).
    X_t : np.ndarray, shape (n_t, p)
        Matriz de target (puede contener NaN).
    shap_importance_s : np.ndarray, shape (p,), optional
        mean|SHAP values| del modelo evaluado en source.  Si no se
        proporciona, se rellena con 0.
    lbase_score_s : np.ndarray, shape (p,), optional
        Coeficientes del LASSO logístico (y_s ~ X_s) en valores absolutos.
        Si no se proporciona, se rellena con 0.
    temporal_domain : np.ndarray of int, shape (p,), optional
        Dominio temporal por feature: 0=preop, 1=intraop, 2=postop.
        Si no se proporciona, se rellena con 0.

    Returns
    -------
    pd.DataFrame, shape (p, len(DRIFT_FEATURE_COLS))
    """
    n_s, p = X_s.shape
    n_t = X_t.shape[0]
    assert X_t.shape[1] == p, "X_s y X_t deben tener el mismo número de features."

    if shap_importance_s is None:
        shap_importance_s = np.zeros(p)
    if lbase_score_s is None:
        lbase_score_s = np.zeros(p)
    if temporal_domain is None:
        temporal_domain = np.zeros(p, dtype=int)

    rows = []
    for j in range(p):
        xs_j = X_s[:, j]
        xt_j = X_t[:, j]

        # Valores no-NaN para cálculos distribucionales
        xs_valid = xs_j[~np.isnan(xs_j)]
        xt_valid = xt_j[~np.isnan(xt_j)]

        nan_rate_s = float(np.isnan(xs_j).mean())
        nan_rate_t = float(np.isnan(xt_j).mean())

        if len(xs_valid) < 2 or len(xt_valid) < 2:
            # Feature structurally absent — drift features sin sentido
            rows.append({
                "ks_stat": 1.0,
                "wasserstein_dist": 0.0,
                "mean_shift": 0.0,
                "std_ratio": 1.0,
                "nan_rate_t": nan_rate_t,
                "nan_rate_s": nan_rate_s,
                "shap_importance_s": float(shap_importance_s[j]),
                "lbase_score_s": float(lbase_score_s[j]),
                "variance_s": 0.0,
                "skewness_s": 0.0,
                "kurtosis_s": 0.0,
                "temporal_domain": int(temporal_domain[j]),
            })
            continue

        mu_s = float(np.mean(xs_valid))
        sigma_s = float(np.std(xs_valid))
        mu_t = float(np.mean(xt_valid))
        sigma_t = float(np.std(xt_valid))

        ks_stat, _ = ks_2samp(xs_valid, xt_valid)
        w1 = wasserstein_distance(xs_valid, xt_valid)

        mean_shift = (mu_t - mu_s) / (sigma_s + 1e-8)
        std_ratio = (sigma_t + 1e-8) / (sigma_s + 1e-8)

        rows.append({
            "ks_stat": float(ks_stat),
            "wasserstein_dist": float(w1),
            "mean_shift": float(mean_shift),
            "std_ratio": float(std_ratio),
            "nan_rate_t": nan_rate_t,
            "nan_rate_s": nan_rate_s,
            "shap_importance_s": float(shap_importance_s[j]),
            "lbase_score_s": float(lbase_score_s[j]),
            "variance_s": float(np.var(xs_valid)),
            "skewness_s": float(scipy_skew(xs_valid)),
            "kurtosis_s": float(scipy_kurtosis(xs_valid)),
            "temporal_domain": int(temporal_domain[j]),
        })

    return pd.DataFrame(rows, columns=DRIFT_FEATURE_COLS)


# ── Synthetic simulation ───────────────────────────────────────────────────────

def _simulate_pseudo_target(
    X_s: np.ndarray,
    rng: np.random.Generator,
    nan_rate_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Genera un pseudo-target a partir de X_s aplicando perturbaciones realistas.

    Perturbaciones por feature j:
        X_t[:,j] = X_s[:,j] * scale_j + shift_j
        scale_j  ~ U(0.5, 2.0)
        shift_j  ~ U(-2 * sigma_j, 2 * sigma_j)

    Si se proporciona nan_rate_t, inyecta NaN según esa tasa empírica.

    Parameters
    ----------
    X_s : np.ndarray, shape (n_s, p)
    rng : np.random.Generator
    nan_rate_t : np.ndarray, shape (p,), optional
        Tasa de NaN observada en el target real.  Se usa para simular
        missingness realista.  Si None, no se inyecta NaN.

    Returns
    -------
    np.ndarray, shape (n_s, p)  — pseudo-target (con valores válidos)
    """
    n_s, p = X_s.shape
    sigma_s = np.nanstd(X_s, axis=0)  # (p,)

    scale = rng.uniform(0.5, 2.0, size=p)                  # (p,)
    shift = rng.uniform(-2 * sigma_s, 2 * sigma_s)          # (p,)

    X_t = X_s * scale[np.newaxis, :] + shift[np.newaxis, :]  # (n_s, p)

    if nan_rate_t is not None:
        for j in range(p):
            rate = float(nan_rate_t[j])
            if rate > 0:
                nan_idx = rng.choice(n_s, size=int(n_s * rate), replace=False)
                X_t[nan_idx, j] = np.nan

    return X_t


# ── MetaDriftPredictor ────────────────────────────────────────────────────────

class MetaDriftPredictor:
    """
    Predictor operacional de drift por feature.

    Entrena un meta-regresor que predice ΔAUROC_j (beneficio de conservar
    la feature j para la transferencia) a partir de señales de drift que
    solo requieren P(X) de source y target.

    Parameters
    ----------
    n_sims : int
        Número de simulaciones sintéticas.  Por defecto 50.
    random_state : int
        Semilla para reproducibilidad.
    regressor : object, optional
        Regresor sklearn-compatible con ``fit`` y ``predict``.  Si None,
        usa XGBRegressor si está disponible, o GradientBoostingRegressor
        como fallback.

    Attributes
    ----------
    regressor_ : fitted regressor
        Meta-regresor ajustado (disponible tras ``fit``).
    feature_names_in_ : list[str]
        Columnas de drift features usadas (= DRIFT_FEATURE_COLS).
    """

    def __init__(
        self,
        n_sims: int = 50,
        random_state: int = 42,
        regressor=None,
    ) -> None:
        self.n_sims = n_sims
        self.random_state = random_state
        self._regressor_proto = regressor
        self.regressor_ = None
        self.feature_names_in_: list[str] = DRIFT_FEATURE_COLS

    # ── Private helpers ───────────────────────────────────────────────────────

    def _make_regressor(self):
        """Instancia el meta-regresor (XGBoost si disponible, GBT si no)."""
        if self._regressor_proto is not None:
            return self._regressor_proto
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
            )
        except ImportError:
            logger.warning(
                "XGBoost no disponible — usando GradientBoostingRegressor como fallback."
            )
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.random_state,
            )

    @staticmethod
    def _impute(X: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
        """Imputa NaN con la media de source (mu_s).  Shape preservada."""
        X_imp = np.where(np.isnan(X), mu_s[np.newaxis, :], X)
        return np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _compute_delta_auroc(
        model,
        X_t_imp: np.ndarray,
        y: np.ndarray,
        mu_s: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula ΔAUROC_j = AUROC(completo) − AUROC(feature j → source mean).

        Mascarar una feature j al valor de source mean elimina su señal
        target-específica.  Un ΔAUROC_j positivo significa que la feature j
        aporta información útil para la predicción en el pseudo-target.

        Parameters
        ----------
        model : duck-typed, necesita ``predict_proba(X) → np.ndarray``
        X_t_imp : np.ndarray, shape (n_t, p) — ya imputado, sin NaN
        y : np.ndarray, shape (n_t,) — labels (y_s reutilizados)
        mu_s : np.ndarray, shape (p,) — medias de source

        Returns
        -------
        np.ndarray, shape (p,)
        """
        p = X_t_imp.shape[1]

        proba_base = model.predict_proba(X_t_imp)
        # predict_proba puede devolver (n,) o (n, 2) — tomar la clase positiva
        if proba_base.ndim == 2:
            proba_base = proba_base[:, 1]

        # Guard: si AUROC no es computable (solo una clase presente), devolvemos ceros
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning("_compute_delta_auroc: solo una clase en y — ΔAUROC=0 para todas las features.")
            return np.zeros(p)

        auroc_base = roc_auc_score(y, proba_base)

        delta = np.zeros(p)
        for j in range(p):
            X_masked = X_t_imp.copy()
            X_masked[:, j] = mu_s[j]  # feature j → source mean (sin señal target)
            proba_j = model.predict_proba(X_masked)
            if proba_j.ndim == 2:
                proba_j = proba_j[:, 1]
            auroc_j = roc_auc_score(y, proba_j)
            delta[j] = auroc_base - auroc_j

        return delta

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X_s: np.ndarray,
        y_s: np.ndarray,
        model,
        nan_rate_t: Optional[np.ndarray] = None,
        shap_importance_s: Optional[np.ndarray] = None,
        lbase_score_s: Optional[np.ndarray] = None,
        temporal_domain: Optional[np.ndarray] = None,
    ) -> "MetaDriftPredictor":
        """
        Entrena el meta-regresor con K simulaciones sintéticas sobre source.

        NO accede a X_t real ni a y_t.  El pseudo-target se genera desde X_s.
        Las labels del pseudo-target son y_s (mismos pacientes, features
        perturbadas — simula covariate shift manteniendo P(Y|X)).

        Parameters
        ----------
        X_s : np.ndarray, shape (n_s, p)
        y_s : np.ndarray, shape (n_s,)
        model : duck-typed, necesita ``predict_proba(X) → array``
            Modelo ya entrenado en source (XGBoostWrapper u otro).
        nan_rate_t : np.ndarray, shape (p,), optional
            Tasa de NaN observada en el target real.  Si se proporciona,
            se inyecta missingness realista en las simulaciones.
        shap_importance_s : np.ndarray, shape (p,), optional
        lbase_score_s : np.ndarray, shape (p,), optional
        temporal_domain : np.ndarray, shape (p,), optional

        Returns
        -------
        self
        """
        n_s, p = X_s.shape
        mu_s = np.nanmean(X_s, axis=0)
        rng = np.random.default_rng(self.random_state)

        all_features: list[pd.DataFrame] = []
        all_targets: list[np.ndarray] = []

        logger.info("MetaDriftPredictor.fit: iniciando %d simulaciones (p=%d).", self.n_sims, p)

        for k in range(self.n_sims):
            X_t_k = _simulate_pseudo_target(X_s, rng, nan_rate_t=nan_rate_t)
            X_t_k_imp = self._impute(X_t_k, mu_s)

            df_features = compute_drift_features(
                X_s, X_t_k,
                shap_importance_s=shap_importance_s,
                lbase_score_s=lbase_score_s,
                temporal_domain=temporal_domain,
            )

            delta = self._compute_delta_auroc(model, X_t_k_imp, y_s, mu_s)

            all_features.append(df_features)
            all_targets.append(delta)

        X_meta = pd.concat(all_features, ignore_index=True)
        y_meta = np.concatenate(all_targets)

        logger.info(
            "MetaDriftPredictor.fit: pool de entrenamiento (%d muestras). Ajustando meta-regresor.",
            len(y_meta),
        )
        self.regressor_ = self._make_regressor()
        self.regressor_.fit(X_meta.values, y_meta)
        logger.info("MetaDriftPredictor.fit: completado.")
        return self

    def predict(
        self,
        X_s: np.ndarray,
        X_t: np.ndarray,
        shap_importance_s: Optional[np.ndarray] = None,
        lbase_score_s: Optional[np.ndarray] = None,
        temporal_domain: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predice d_j = ΔAUROC esperado al conservar (no enmascarar) feature j.

        Requiere que ``fit`` haya sido llamado primero.

        Parameters
        ----------
        X_s : np.ndarray, shape (n_s, p)
        X_t : np.ndarray, shape (n_t, p)
        shap_importance_s, lbase_score_s, temporal_domain : opcionales

        Returns
        -------
        np.ndarray, shape (p,)
            Score d_j por feature.  Valores más altos → feature más
            beneficiosa para la transferencia.
        """
        if self.regressor_ is None:
            raise RuntimeError(
                "MetaDriftPredictor no está ajustado.  Llama a fit() primero."
            )
        df_features = compute_drift_features(
            X_s, X_t,
            shap_importance_s=shap_importance_s,
            lbase_score_s=lbase_score_s,
            temporal_domain=temporal_domain,
        )
        return self.regressor_.predict(df_features.values)
