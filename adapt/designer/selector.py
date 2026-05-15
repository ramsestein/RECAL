"""
adapt.designer.selector
========================
ComponentSelector: aplica todas las reglas del Designer y construye
el AdapterConfig con justificaciones por componente.
"""

from __future__ import annotations

import logging

from adapt.profiler.base import DriftProfile
from adapt.designer.base import AdapterConfig
from adapt.designer import rules

logger = logging.getLogger(__name__)


class ComponentSelector:
    """
    Selecciona los componentes de la pipeline ADAPT a partir del DriftProfile.

    Aplica las reglas determinísticas de adapt.designer.rules y registra
    la justificación de cada decisión en AdapterConfig.rationale.

    No tiene hiperparámetros. Las reglas están en adapt/designer/rules.py
    y los thresholds en adapt/profiler/constants.py.
    """

    def select(
        self,
        profile: DriftProfile,
        pair=None,
        model=None,
        pca_k: int = 5,
        max_n_sweep: int = 30,
    ) -> AdapterConfig:
        """
        Construye el AdapterConfig para el perfil dado.

        Parameters
        ----------
        profile : DriftProfile
            Perfil del par (source, target) producido por el Profiler.
        pair : CohortPair, optional
            Si se proporciona (junto con model), activa el mini-sweep de N
            con PCA-CORAL en target para encontrar el N óptimo de máscara.
        model : ModelWrapper, optional
            Modelo source con predict_proba(). Requerido si pair se pasa.
        pca_k : int
            Número de componentes PCA para el mini-sweep del Designer.
        max_n_sweep : int
            Máximo N a barrer en el mini-sweep del Designer.

        Returns
        -------
        AdapterConfig
        """
        config = AdapterConfig()
        rationale: dict[str, str] = {}

        # ── 1. Máscara ────────────────────────────────────────────────────────
        apply_mask, mask_reason = rules.should_mask_features(profile)
        config.apply_mask = apply_mask
        rationale["mask_activate"] = mask_reason

        if apply_mask:
            n, n_reason = rules.select_mask_n(
                profile, pair=pair, model=model,
                pca_k=pca_k, max_n_sweep=max_n_sweep,
            )
            config.mask_n = n
            config.mask_selection_method = "sweep_pca_coral" if pair is not None else "elbow_source"
            rationale["mask_n"] = n_reason

            # Seleccionar las N features con menor combined_score
            sorted_features = sorted(profile.features, key=lambda f: f.combined_score)
            config.mask_features = [f.name for f in sorted_features[:n]]
            rationale["mask_features"] = (
                f"Bottom-{n} por combined_score: "
                + ", ".join(config.mask_features[:5])
                + ("..." if n > 5 else "")
            )
        else:
            config.mask_n = 0
            config.mask_features = []

        # ── 2. QuantileTransform ──────────────────────────────────────────────
        qt_decisions, qt_reason = rules.should_apply_quantile_transform_per_feature(profile)
        qt_features = [name for name, apply in qt_decisions.items() if apply]
        config.apply_quantile = len(qt_features) > 0
        config.quantile_features = qt_features
        config.quantile_output_distribution = "uniform"
        rationale["quantile"] = qt_reason

        # ── 3. WOE ────────────────────────────────────────────────────────────
        woe_decisions, woe_reason = rules.should_apply_woe_per_feature(profile)
        woe_features = [name for name, apply in woe_decisions.items() if apply]
        config.apply_woe = len(woe_features) > 0
        config.woe_features = woe_features
        config.woe_n_bins = 10
        rationale["woe"] = woe_reason

        # ── 4. PCA-CORAL ──────────────────────────────────────────────────────
        apply_pcacoral, pcacoral_reason = rules.should_apply_pca_coral(profile)
        config.apply_pca_coral = apply_pcacoral
        rationale["pca_coral_activate"] = pcacoral_reason

        if apply_pcacoral:
            k, k_reason = rules.select_pca_coral_k(profile)
            config.pca_coral_k = k
            config.pca_coral_k_selection_method = "sqrt_n_target"
            rationale["pca_coral_k"] = k_reason

        # ── 5. Calibración ────────────────────────────────────────────────────
        apply_cal, cal_reason = rules.should_recalibrate(profile)
        config.apply_calibration = apply_cal
        rationale["calibration_activate"] = cal_reason

        if apply_cal:
            method, method_reason = rules.select_calibration_method(profile)
            config.calibration_method = method
            config.calibration_strata_fn = (
                "score_terciles" if method == "platt_stratified" else None
            )
            rationale["calibration_method"] = method_reason

        config.rationale = rationale

        logger.info("AdapterConfig seleccionada:\n%s", config.summary())
        return config
