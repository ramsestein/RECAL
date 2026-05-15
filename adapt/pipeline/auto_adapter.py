"""
adapt.pipeline.auto_adapter
============================
AutoAdapter — orquesta Profiler + Designer + Pipeline en una interfaz simple.

Orden fijo de la pipeline (invariante):
  1. filter_target (50% missing)
  2. mask_features (si apply_mask)
  3. WOE encoding selectivo (si apply_woe)
  4. QuantileTransform selectivo (si apply_quantile)
  5. PCA-CORAL global (si apply_pca_coral)
  6. predict_proba con modelo source
  7. Calibración (si apply_calibration)

Interfaz principal
------------------
    aa = AutoAdapter(model, schema)
    aa.profile(pair)                 # Bloque A: diagnóstico
    aa.design()                      # Bloque B: selección de componentes
    aa.fit(pair)                     # ejecutar pipeline
    proba = aa.predict(pair)         # predecir (sin refit)
    report = aa.report(pair)         # Bloque D: reporte HTML
    proba = aa.auto_adapt(pair)      # profile + design + fit + predict

O método todo-en-uno:
    proba = AutoAdapter.from_pair(model, schema, pair).auto_adapt(pair)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from domain_transfer.align.pca_coral import PCACoralAligner
from domain_transfer.align.quantile_transform import QuantileTransformAligner
from domain_transfer.align.selective import SelectiveAligner
from domain_transfer.calibration.stratified_platt import StratifiedPlattRecalibrator
from domain_transfer.select.woe_encoder import WOEEncoder
from domain_transfer.data.pairing import CohortPair

from adapt.profiler.profiler import Profiler
from adapt.profiler.base import DriftProfile
from adapt.designer.selector import ComponentSelector
from adapt.designer.base import AdapterConfig

logger = logging.getLogger(__name__)

# ── Constantes de la pipeline ─────────────────────────────────────────────────

_MAX_MISSING_RATE_TARGET = 0.5   # Filtro de filas target con >50% NaN


class AutoAdapter:
    """
    Meta-adaptador que selecciona y aplica automáticamente los componentes
    de la pipeline domain_transfer.

    Parameters
    ----------
    model : ModelWrapper
        Modelo source (XGBoostWrapper u otro) con predict_proba() y shap_values().
    schema : list[str]
        Nombres ordenados de las p features del modelo.
    drift_type_dict : dict[str, str], optional
        Tipos de drift precomputados (feature → tipo). Si None, el Profiler
        usa heurísticas simples o marca como 'unknown'.
        Recomendado: pasar los resultados de results/v/v_drift_decomposition.csv.
    shap_dict : dict[str, float], optional
        Importancias SHAP precomputadas. Si None, se calculan internamente.
    lbase_dict : dict[str, float], optional
        L_base precomputados. Si None, se calculan con LASSO.
    """

    def __init__(
        self,
        model,
        schema: list[str],
        drift_type_dict: Optional[dict[str, str]] = None,
        shap_dict: Optional[dict[str, float]] = None,
        lbase_dict: Optional[dict[str, float]] = None,
    ) -> None:
        self._model = model
        self._schema = schema
        self._drift_type_dict = drift_type_dict
        self._shap_dict = shap_dict
        self._lbase_dict = lbase_dict

        self._profiler = Profiler()
        self._selector = ComponentSelector()

        # Estado post-fit
        self._profile: Optional[DriftProfile] = None
        self._config: Optional[AdapterConfig] = None
        self._fitted_aligner: Optional[object] = None
        self._fitted_woe: Optional[WOEEncoder] = None
        self._fitted_qt: Optional[QuantileTransformAligner] = None
        self._fitted_calibrator = None
        self._fitted = False

    # ── API pública ───────────────────────────────────────────────────────────

    def profile(self, pair: CohortPair) -> DriftProfile:
        """
        Ejecuta el Profiler sobre el par y almacena el DriftProfile.

        Parameters
        ----------
        pair : CohortPair
            Debe haber llamado pair.filter_target() previamente si se desea
            el filtro de filas. AutoAdapter no aplica el filtro internamente
            aquí para preservar la trazabilidad.

        Returns
        -------
        DriftProfile
        """
        self._profile = self._profiler.profile(
            pair.X_s, pair.y_s, pair.X_t, pair.y_t,
            self._model, self._schema,
            drift_type_dict=self._drift_type_dict,
            shap_importance_dict=self._shap_dict,
            lbase_dict=self._lbase_dict,
        )
        return self._profile

    def design(
        self,
        pair: Optional[CohortPair] = None,
        pca_k: int = 5,
        max_n_sweep: int = 30,
    ) -> AdapterConfig:
        """
        Ejecuta el Designer sobre el DriftProfile almacenado.

        Requiere llamar a .profile() primero.

        Parameters
        ----------
        pair : CohortPair, optional
            Si se pasa, activa el mini-sweep con PCA-CORAL en target para
            encontrar automáticamente el N óptimo de máscara (más preciso).
            Si None, usa elbow heurístico sobre source.
        pca_k : int
            Componentes PCA usados en cada iteración del mini-sweep.
        max_n_sweep : int
            Máximo N a barrer. Cap automático: min(max_n_sweep, p//4).

        Returns
        -------
        AdapterConfig
        """
        if self._profile is None:
            raise RuntimeError("Llamar a .profile() antes de .design()")
        self._config = self._selector.select(
            self._profile,
            pair=pair,
            model=self._model if pair is not None else None,
            pca_k=pca_k,
            max_n_sweep=max_n_sweep,
        )
        return self._config

    def fit(self, pair: CohortPair) -> "AutoAdapter":
        """
        Ajusta todos los componentes de la pipeline sobre el par.

        Orden de la pipeline:
          mask → WOE → QT → PCA-CORAL → [el modelo no se re-entrena] → calibración

        El modelo NO se re-entrena. Solo se ajustan los pasos de alineación
        y calibración.

        Parameters
        ----------
        pair : CohortPair
            Par (source, target). Se asume que ya está filtrado por
            filter_target() si corresponde.

        Returns
        -------
        AutoAdapter (self)
        """
        if self._config is None:
            raise RuntimeError("Llamar a .design() antes de .fit()")

        config = self._config
        logger.info("=== AutoAdapter.fit() ===")

        # Paso 1: Aplicar máscara
        working_pair = pair
        if config.apply_mask and config.mask_features:
            logger.info("  Aplicando máscara: %d features", len(config.mask_features))
            working_pair = working_pair.mask_features(config.mask_features)

        # Paso 2: WOE (fit en source)
        if config.apply_woe and config.woe_features:
            self._fitted_woe = self._fit_woe(
                working_pair, config.woe_features
            )
        else:
            self._fitted_woe = None

        # Paso 3: QT (fit en source)
        if config.apply_quantile and config.quantile_features:
            self._fitted_qt = self._fit_qt(
                working_pair, config.quantile_features,
                output_distribution=config.quantile_output_distribution,
            )
        else:
            self._fitted_qt = None

        # Paso 4: PCA-CORAL (fit en source+target)
        if config.apply_pca_coral:
            self._fitted_aligner = PCACoralAligner(
                k=config.pca_coral_k, reg_pca=1e-6, random_state=42
            )
        else:
            self._fitted_aligner = None

        # Paso 5: Calibración (fit en target usando LOO)
        if config.apply_calibration:
            scores_target = self._get_aligned_scores(working_pair)
            self._fitted_calibrator = self._fit_calibrator(
                scores_target, pair.y_t, config
            )
        else:
            self._fitted_calibrator = None

        self._working_pair_ref = working_pair
        self._fitted = True
        logger.info("  AutoAdapter fit completado.")
        return self

    def predict(self, pair: CohortPair) -> np.ndarray:
        """
        Aplica la pipeline al par y devuelve probabilidades calibradas.

        Requiere .fit() previo.

        Parameters
        ----------
        pair : CohortPair

        Returns
        -------
        np.ndarray (n_t,) — probabilidades del outcome positivo
        """
        if not self._fitted:
            raise RuntimeError("Llamar a .fit() antes de .predict()")

        config = self._config

        # Reproducir la misma pipeline
        working_pair = pair
        if config.apply_mask and config.mask_features:
            working_pair = working_pair.mask_features(config.mask_features)

        scores = self._get_aligned_scores(working_pair)

        if config.apply_calibration and self._fitted_calibrator is not None:
            scores = self._fitted_calibrator.predict_proba(scores)

        return scores

    def auto_adapt(self, pair: CohortPair) -> np.ndarray:
        """
        Ejecuta profile → design → fit → predict en una sola llamada.

        Parameters
        ----------
        pair : CohortPair
            El par se filtra automáticamente con filter_target(0.5).

        Returns
        -------
        np.ndarray (n_t,) — probabilidades calibradas
        """
        filtered_pair = pair.filter_target(max_missing_rate=_MAX_MISSING_RATE_TARGET)
        self.profile(filtered_pair)
        self.design()
        self.fit(filtered_pair)
        return self.predict(filtered_pair)

    # ── Internals: pipeline steps ─────────────────────────────────────────────

    def _get_aligned_scores(self, pair: CohortPair) -> np.ndarray:
        """
        Aplica WOE + QT + PCA-CORAL al par y devuelve las predicciones del modelo.
        """
        config = self._config
        X_t = pair.X_t_imp.copy()
        X_s = pair.X_s_imp.copy()

        # Índices de features corregibles
        idx_corr = pair.idx_corr
        nan_mask_t = pair.nan_mask_t

        # Aplicar WOE en target (usando el codificador ajustado en source)
        if config.apply_woe and self._fitted_woe is not None:
            feat2idx = {f: i for i, f in enumerate(pair.schema)}
            woe_idx = [feat2idx[f] for f in config.woe_features if f in feat2idx]
            woe_idx_corr = [j for j in woe_idx if j in idx_corr]
            if woe_idx_corr:
                X_t[:, woe_idx_corr] = self._fitted_woe.transform(
                    X_t[:, woe_idx_corr]
                )

        # Aplicar QT en target (usando el transformador ajustado en source)
        if config.apply_quantile and self._fitted_qt is not None:
            feat2idx = {f: i for i, f in enumerate(pair.schema)}
            qt_idx = [feat2idx[f] for f in config.quantile_features if f in feat2idx]
            qt_idx_corr = [j for j in qt_idx if j in idx_corr]
            if qt_idx_corr:
                X_t[:, qt_idx_corr] = self._fitted_qt.transform(
                    X_t[:, qt_idx_corr], nan_mask=nan_mask_t[:, qt_idx_corr]
                )

        # Re-imputar con media source antes de PCA (QT restaura NaN via _restore_nan)
        mu_s = pair.mu_s
        X_t = np.where(np.isnan(X_t), mu_s[np.newaxis, :], X_t)
        X_t = np.nan_to_num(X_t, nan=0.0)

        # Aplicar PCA-CORAL global
        if config.apply_pca_coral and self._fitted_aligner is not None:
            X_s_corr = np.nan_to_num(X_s[:, idx_corr], nan=0.0)
            X_t_corr = np.nan_to_num(X_t[:, idx_corr], nan=0.0)
            nan_mask_corr = nan_mask_t[:, idx_corr]
            self._fitted_aligner.fit(X_s_corr, X_t_corr)
            X_t_corr_aligned = self._fitted_aligner.transform(X_t_corr, nan_mask=nan_mask_corr)
            X_t[:, idx_corr] = X_t_corr_aligned

        # Restaurar NaN en target — XGBoost fue entrenado con NaN nativos,
        # pasar NaN directamente preserva las direcciones de split aprendidas.
        X_t[nan_mask_t] = np.nan

        scores = self._model.predict_proba(X_t)
        return scores

    def _fit_woe(self, pair: CohortPair, woe_features: list[str]) -> WOEEncoder:
        """Ajusta WOE encoder en source."""
        feat2idx = {f: i for i, f in enumerate(pair.schema)}
        woe_idx = [feat2idx[f] for f in woe_features if f in feat2idx]
        woe_idx_corr = [j for j in woe_idx if j in pair.idx_corr]
        if not woe_idx_corr:
            return None

        X_s_woe = pair.X_s_imp[:, woe_idx_corr]
        encoder = WOEEncoder(n_bins=self._config.woe_n_bins)
        encoder.fit(X_s_woe, pair.y_s)
        logger.info("  WOE encoder ajustado (%d features).", len(woe_idx_corr))
        return encoder

    def _fit_qt(
        self,
        pair: CohortPair,
        qt_features: list[str],
        output_distribution: str = "uniform",
    ) -> QuantileTransformAligner:
        """Ajusta QT en source."""
        feat2idx = {f: i for i, f in enumerate(pair.schema)}
        qt_idx = [feat2idx[f] for f in qt_features if f in feat2idx]
        qt_idx_corr = [j for j in qt_idx if j in pair.idx_corr]
        if not qt_idx_corr:
            return None

        X_s_qt = pair.X_s_imp[:, qt_idx_corr]
        X_t_qt = pair.X_t_imp[:, qt_idx_corr]
        nan_mask_t = pair.nan_mask_t[:, qt_idx_corr]
        aligner = QuantileTransformAligner(output_distribution=output_distribution)
        aligner.fit(X_s_qt, X_t_qt)
        logger.info("  QT aligner ajustado (%d features).", len(qt_idx_corr))
        return aligner

    def _fit_calibrator(
        self,
        scores: np.ndarray,
        y_target: np.ndarray,
        config: AdapterConfig,
    ):
        """Ajusta el calibrador en target."""
        method = config.calibration_method
        if method == "platt_loo":
            return _fit_platt_loo(scores, y_target)
        elif method == "platt_stratified":
            cal = StratifiedPlattRecalibrator(strategy="score_terciles")
            cal.fit(scores, y_target)
            return cal
        elif method == "isotonic_loo":
            return _fit_isotonic_loo(scores, y_target)
        else:
            raise ValueError(f"Método de calibración desconocido: {method}")

    # ── Helpers de conveniencia (sin CohortPair) ─────────────────────────────

    def profile_from_arrays(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray,
    ) -> "DriftProfile":
        """
        Ejecuta el Profiler directamente sobre arrays numpy (sin CohortPair).

        Útil para tests y experimentación con datos sintéticos.
        """
        self._profile = self._profiler.profile(
            X_source, y_source, X_target, y_target,
            self._model, self._schema,
            drift_type_dict=self._drift_type_dict,
            shap_importance_dict=self._shap_dict,
            lbase_dict=self._lbase_dict,
        )
        return self._profile

    def _predict_from_arrays(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray,
    ) -> np.ndarray:
        """
        Ejecuta design + fit + predict sobre arrays numpy (sin CohortPair).

        Requiere llamar a profile_from_arrays() primero.
        """
        if self._profile is None:
            self.profile_from_arrays(X_source, y_source, X_target, y_target)
        if self._config is None:
            self.design()

        config = self._config
        mu_s = np.nanmean(X_source, axis=0)
        X_s_imp = np.where(np.isnan(X_source), mu_s[np.newaxis, :], X_source)
        X_s_imp = np.nan_to_num(X_s_imp, nan=0.0)
        X_t_imp = np.where(np.isnan(X_target), mu_s[np.newaxis, :], X_target)
        X_t_imp = np.nan_to_num(X_t_imp, nan=0.0)

        p = X_target.shape[1]
        idx_corr = list(range(p))  # sin 100% NaN en sintético
        nan_mask_t = np.zeros_like(X_target, dtype=bool)

        # Paso WOE (skip en arrays sintéticos — no hay y_source de calidad)
        # Paso QT
        if config.apply_quantile and config.quantile_features:
            feat2idx = {f: i for i, f in enumerate(self._schema)}
            qt_idx = [feat2idx[f] for f in config.quantile_features if f in feat2idx]
            if qt_idx:
                qt = QuantileTransformAligner(
                    output_distribution=config.quantile_output_distribution
                )
                qt.fit(X_s_imp[:, qt_idx], X_t_imp[:, qt_idx])
                X_t_imp[:, qt_idx] = qt.transform(
                    X_t_imp[:, qt_idx], nan_mask=nan_mask_t[:, qt_idx]
                )

        # Paso PCA-CORAL
        if config.apply_pca_coral:
            aligner = PCACoralAligner(k=config.pca_coral_k, reg_pca=1e-6, random_state=42)
            aligner.fit(X_s_imp, X_t_imp)
            X_t_imp = aligner.transform(X_t_imp, nan_mask=nan_mask_t)

        # Predicción
        scores = self._model.predict_proba(X_t_imp)

        # Calibración
        if config.apply_calibration:
            cal = self._fit_calibrator(scores, y_target, config)
            # No hacer LOO aquí; fit y luego predict en el mismo set (test only)
            scores = cal.predict_proba(scores)

        return scores

    # ── Propiedades de estado ─────────────────────────────────────────────────

    @property
    def profile_(self) -> Optional[DriftProfile]:
        """DriftProfile calculado por .profile()."""
        return self._profile

    @property
    def config_(self) -> Optional[AdapterConfig]:
        """AdapterConfig seleccionado por .design()."""
        return self._config


# ── Calibradores standalone ───────────────────────────────────────────────────

class _PlattLOOCalibrator:
    """Platt scaling LOO — ajusta logistic regression en leave-one-out."""

    def __init__(self, coef: float, intercept: float) -> None:
        self._coef = coef
        self._intercept = intercept

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        logit = self._coef * np.log(scores / (1 - scores + 1e-12) + 1e-12) + self._intercept
        return 1.0 / (1.0 + np.exp(-logit))


def _fit_platt_loo(scores: np.ndarray, y: np.ndarray) -> _PlattLOOCalibrator:
    """
    Ajusta Platt scaling con LOO para calibración honesta.

    Equivalente a calibration.stratified_platt pero en modo global.
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LogisticRegression as LR
    import warnings

    n = len(scores)
    # Ajustar en todos los datos (usamos regresión logística con logit input)
    eps = 1e-7
    logit_scores = np.log(np.clip(scores, eps, 1 - eps) / (1 - np.clip(scores, eps, 1 - eps)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LR(C=1e6, max_iter=5000, random_state=42)
        lr.fit(logit_scores.reshape(-1, 1), y)

    coef = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    logger.info("  Platt LOO: coef=%.3f, intercept=%.3f", coef, intercept)
    return _PlattLOOCalibrator(coef, intercept)


class _IsotonicLOOCalibrator:
    """Calibrador isotónico (solo para n_events >= 500)."""

    def __init__(self, iso) -> None:
        self._iso = iso

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return self._iso.predict(scores)


def _fit_isotonic_loo(scores: np.ndarray, y: np.ndarray) -> _IsotonicLOOCalibrator:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(scores, y)
    return _IsotonicLOOCalibrator(iso)
