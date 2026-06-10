"""
recal_core.profiler.profiler
========================
Clase principal Profiler: combina GlobalProfiler y FeatureProfiler en un
DriftProfile completo.

Interfaz principal
------------------
    profiler = Profiler()
    profile = profiler.profile(X_s, y_s, X_t, y_t, model, schema)

Sin hiperparámetros: todos los thresholds están en recal_core/profiler/constants.py.
Los datos precomputados (drift_type_dict, shap_dict, lbase_dict) pueden pasarse
opcionalmente para acelerar el diagnóstico si ya están calculados (e.g., desde
results/v/v_drift_decomposition.csv).
"""

from __future__ import annotations

import logging

import numpy as np

from recal_core.profiler.base import DriftProfile
from recal_core.profiler.feature_profiler import profile_features
from recal_core.profiler.global_profiler import profile_global

logger = logging.getLogger(__name__)


class Profiler:
    """
    Diagnostica el par (source, target) y produce un DriftProfile.

    No tiene hiperparámetros en el constructor. Todos los thresholds
    están auditables en recal_core/profiler/constants.py.

    Parameters
    ----------
    Ninguno.
    """

    def profile(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target: np.ndarray,
        y_target: np.ndarray,
        model,
        schema: list[str],
        drift_type_dict: dict[str, str] | None = None,
        shap_importance_dict: dict[str, float] | None = None,
        lbase_dict: dict[str, float] | None = None,
    ) -> DriftProfile:
        """
        Diagnostica el par y devuelve un DriftProfile.

        Parameters
        ----------
        X_source : np.ndarray (n_s, p)
            Features source (puede contener NaN).
        y_source : np.ndarray (n_s,)
            Labels binarios source.
        X_target : np.ndarray (n_t, p)
            Features target (puede contener NaN).
        y_target : np.ndarray (n_t,)
            Labels binarios target.
        model : ModelWrapper
            Modelo source con predict_proba() y opcionalmente shap_values().
        schema : list[str]
            Nombres ordenados de las p features.
        drift_type_dict : dict[str, str], optional
            Tipos de drift precomputados por feature.
            Si None, se usan heurísticas o se marca como 'unknown'.
            Recomendado: pasar los resultados de la descomposición V.
        shap_importance_dict : dict[str, float], optional
            Importancias SHAP precomputadas.
            Si None, se calculan internamente usando model.shap_values().
        lbase_dict : dict[str, float], optional
            L_base precomputados.
            Si None, se calculan con LASSO logístico en source.

        Returns
        -------
        DriftProfile
        """
        logger.info("=== RECAL Profiler ===")
        logger.info(
            "Source: n=%d, events=%d | Target: n=%d, events=%d | Features: %d",
            len(y_source), int(y_source.sum()),
            len(y_target), int(y_target.sum()),
            len(schema),
        )

        # ── Globales ──────────────────────────────────────────────────────────
        logger.info("Computing global metrics...")
        globals_dict = profile_global(
            X_source, y_source, X_target, y_target, model
        )

        # ── Features ──────────────────────────────────────────────────────────
        features = profile_features(
            X_source, y_source, X_target, y_target,
            model, schema,
            drift_type_dict=drift_type_dict,
            shap_importance_dict=shap_importance_dict,
            lbase_dict=lbase_dict,
        )

        profile = DriftProfile(
            features=features,
            **globals_dict,
        )

        logger.info(
            "Profiling complete. AUROC baseline=%.4f [%.4f, %.4f]. "
            "Slope=%.2f. MMD²=%.4f (p=%.3f).",
            profile.baseline_auroc,
            profile.baseline_auroc_ci_low,
            profile.baseline_auroc_ci_high,
            profile.baseline_calibration_slope,
            profile.mmd2_source_target,
            profile.mmd2_pvalue,
        )
        logger.info(
            "Concept relational SHAP%%=%.1f%%. Flip-of-sign=%d features.",
            100 * profile.concept_relational_shap_pct(),
            profile.flip_of_sign_count(),
        )

        return profile
