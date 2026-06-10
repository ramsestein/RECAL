"""
recal_core.profiler.base
====================
Dataclasses DriftProfile y FeatureProfile — estructuras de datos del diagnóstico
del par (source, target).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureProfile:
    """
    Perfil diagnóstico de una feature individual.

    Attributes
    ----------
    name : str
        Nombre de la feature.
    domain : str
        Dominio temporal: 'preop' | 'intraop' | 'postop' | 'base' | 'demographic'.
    drift_type_v : str
        Tipo de drift según descomposición V:
        'STABLE' | 'NONLINEAR_DRIFT' | 'LINEAR_RECOVERABLE' |
        'PARTIAL_RECOVERY' | 'CONCEPT_RELATIONAL' | 'INSUFFICIENT_DATA' |
        'DEGENERATE' | 'LOW_VARIANCE_TARGET' | 'unknown'
    lbase_score : float
        R² o AUROC del predictor LASSO de la feature en source.
        Mayor = más discriminativa/predecible en el contexto source.
    shap_importance : float
        Mean |SHAP value| de la feature en el modelo source.
    combined_score : float
        norm(lbase) + norm(shap). Score combinado de importancia.
    quadrant : str
        Cuadrante SHAP × L_base:
        'A_core' | 'B_noisy_important' | 'C_redundant' | 'D_ponzonous'
    univariate_concept_shift_beta3 : float
        Coeficiente β₃ de la interacción feature × cohort en regresión
        logística pooled. β₃ ≠ 0 indica concept shift univariado.
    univariate_concept_shift_qbh : float
        Q-valor BH del test β₃ = 0.
    flip_of_sign : bool
        True si el signo de Spearman(feature, label) se invierte entre
        source y target.
    cv_target : float
        Coeficiente de variación en target (std / |mean|). NaN si mean≈0.
    cv_source : float
        Coeficiente de variación en source.
    var_ratio : float
        var_target / var_source. NaN si var_source≈0.
    near_constant_target : bool
        True si cv_target < constants.CV_TARGET_NEAR_CONSTANT_THRESHOLD.
    missing_rate_source : float
        Fracción de valores NaN en source.
    missing_rate_target : float
        Fracción de valores NaN en target.
    """

    name: str
    domain: str
    drift_type_v: str
    lbase_score: float
    shap_importance: float
    combined_score: float
    quadrant: str
    univariate_concept_shift_beta3: float
    univariate_concept_shift_qbh: float
    flip_of_sign: bool
    cv_target: float
    cv_source: float
    var_ratio: float
    near_constant_target: bool
    missing_rate_source: float
    missing_rate_target: float


@dataclass
class DriftProfile:
    """
    Perfil completo del par (source, target).

    Attributes
    ----------
    Globales:
        n_source_obs, n_target_obs : int
        n_source_events, n_target_events : int
        prevalence_source, prevalence_target : float
        prevalence_shift_pvalue : float
            P-valor del test exacto de Fisher para H0: prevalencia igual.
        p_n_ratio_target : float
            Ratio positivos/negativos en target. Indicador de balance.
        mmd2_source_target : float
            MMD² entre source y target (features estandarizadas).
        mmd2_pvalue : float
            P-valor de permutación para MMD² (1000 réplicas).
        pca_variance_explained : list[float]
            Varianza explicada acumulada de los 5 primeros PCs sobre source
            estandarizado.
        baseline_auroc : float
            AUROC del modelo source aplicado a target raw (sin adaptación).
        baseline_auroc_ci_low, baseline_auroc_ci_high : float
            IC 95% bootstrap del AUROC baseline.
        baseline_calibration_slope : float
            Slope de calibración Platt del modelo source en target raw.
        baseline_ece : float
            ECE del modelo source en target raw.
        baseline_citl : float
            Calibration-in-the-large (diferencia entre prevalencia predicha
            y observada).
    Por feature:
        features : list[FeatureProfile]
    """

    # Globales
    n_source_obs: int
    n_target_obs: int
    n_source_events: int
    n_target_events: int
    prevalence_source: float
    prevalence_target: float
    prevalence_shift_pvalue: float
    p_n_ratio_target: float
    mmd2_source_target: float
    mmd2_pvalue: float
    pca_variance_explained: list
    baseline_auroc: float
    baseline_auroc_ci_low: float
    baseline_auroc_ci_high: float
    baseline_calibration_slope: float
    baseline_ece: float
    baseline_citl: float

    # Por feature
    features: list = field(default_factory=list)

    # ── Propiedades derivadas ─────────────────────────────────────────────────

    def feature_dict(self) -> dict[str, FeatureProfile]:
        """Devuelve dict name → FeatureProfile para acceso rápido."""
        return {f.name: f for f in self.features}

    def features_by_quadrant(self, quadrant: str) -> list[FeatureProfile]:
        """Devuelve features del cuadrante indicado."""
        return [f for f in self.features if f.quadrant == quadrant]

    def features_by_drift_type(self, drift_type: str) -> list[FeatureProfile]:
        """Devuelve features del tipo de drift indicado."""
        return [f for f in self.features if f.drift_type_v == drift_type]

    def shap_total(self) -> float:
        """Suma total de SHAP importances."""
        return sum(f.shap_importance for f in self.features)

    def concept_relational_shap_pct(self) -> float:
        """
        Fracción del SHAP total en features CONCEPT_RELATIONAL.
        Indicador de techo teórico de UDA no supervisada.
        """
        total = self.shap_total()
        if total <= 0:
            return 0.0
        cr_shap = sum(
            f.shap_importance
            for f in self.features
            if f.drift_type_v == "CONCEPT_RELATIONAL"
        )
        return cr_shap / total

    def flip_of_sign_count(self) -> int:
        """Número de features con inversión de signo (flip_of_sign=True)."""
        return sum(1 for f in self.features if f.flip_of_sign)

    def ponzonous_features(self) -> list[FeatureProfile]:
        """Features del cuadrante D_ponzonous (candidatas a máscara)."""
        return self.features_by_quadrant("D_ponzonous")
