"""
recal_core.designer.selector
========================
ComponentSelector: aplica todas las reglas del Designer y construye
el AdapterConfig con justificaciones por componente.
"""

from __future__ import annotations

import logging

from recal_core.designer import rules
from recal_core.designer.base import AdapterConfig
from recal_core.designer_audit import (
    AlternativeChoice,
    DesignerAuditTrail,
    DesignerDecision,
)
from recal_core.profiler.base import DriftProfile

logger = logging.getLogger(__name__)


class ComponentSelector:
    """
    Selecciona los componentes de la pipeline RECAL a partir del DriftProfile.

    Aplica las reglas determinísticas de recal_core.designer.rules y registra
    la justificación de cada decisión en AdapterConfig.rationale.

    No tiene hiperparámetros. Las reglas están en recal_core/designer/rules.py
    y los thresholds en recal_core/profiler/constants.py.
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
        audit = DesignerAuditTrail()

        # ── 1. Máscara ────────────────────────────────────────────────────────
        apply_mask, mask_reason = rules.should_mask_features(profile)
        config.apply_mask = apply_mask
        rationale["mask_activate"] = mask_reason
        audit.record(DesignerDecision(
            step="mask_activate",
            criterion="n_target_events >= N_EVENTS_MINIMUM_MASK",
            alternatives=[
                AlternativeChoice(choice=True, metric_name="activated", metric_value=None,
                                  selected=apply_mask),
                AlternativeChoice(choice=False, metric_name="activated", metric_value=None,
                                  selected=not apply_mask),
            ],
            final_choice=apply_mask,
            justification=mask_reason,
        ))

        mask_sweep_history: list[dict] = []

        if apply_mask:
            n, n_reason, mask_sweep_history = rules.select_mask_n(
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

            # Alternativas del sweep
            sweep_alternatives = [
                AlternativeChoice(
                    choice=entry["n"],
                    metric_name="auroc_target",
                    metric_value=entry["auroc"],
                    selected=(entry["n"] == n),
                )
                for entry in mask_sweep_history
            ]
            audit.record(DesignerDecision(
                step="mask_n",
                criterion="max AUROC target con PCA-CORAL (sweep)" if pair is not None else "elbow del combined_score",
                alternatives=sweep_alternatives,
                final_choice=n,
                justification=n_reason,
            ))
            audit.record(DesignerDecision(
                step="mask_features",
                criterion="bottom-N por combined_score",
                alternatives=[
                    AlternativeChoice(choice=f.name, metric_name="combined_score",
                                      metric_value=f.combined_score,
                                      selected=(f.name in config.mask_features))
                    for f in sorted(profile.features, key=lambda x: x.combined_score)[:max(n + 3, 5)]
                ],
                final_choice=config.mask_features,
                justification=rationale["mask_features"],
            ))
        else:
            config.mask_n = 0
            config.mask_features = []

        # Guardar sweep_history en config para acceso externo
        config.mask_sweep_history = mask_sweep_history

        # ── 2. QuantileTransform ──────────────────────────────────────────────
        qt_decisions, qt_reason = rules.should_apply_quantile_transform_per_feature(profile)
        qt_features = [name for name, apply in qt_decisions.items() if apply]
        config.apply_quantile = len(qt_features) > 0
        config.quantile_features = qt_features
        config.quantile_output_distribution = "uniform"
        rationale["quantile"] = qt_reason
        audit.record(DesignerDecision(
            step="quantile_transform",
            criterion="drift_type_v in {NONLINEAR_DRIFT, PARTIAL_RECOVERY} AND cv_target >= threshold AND var_ratio out of range",
            alternatives=[
                AlternativeChoice(choice=name, metric_name="apply_qt",
                                  metric_value=float(apply), selected=apply)
                for name, apply in qt_decisions.items()
            ],
            final_choice=qt_features,
            justification=qt_reason,
        ))

        # ── 3. WOE ────────────────────────────────────────────────────────────
        woe_decisions, woe_reason = rules.should_apply_woe_per_feature(profile)
        woe_features = [name for name, apply in woe_decisions.items() if apply]
        config.apply_woe = len(woe_features) > 0
        config.woe_features = woe_features
        config.woe_n_bins = 10
        rationale["woe"] = woe_reason
        audit.record(DesignerDecision(
            step="woe_encoding",
            criterion="drift_type_v in {STABLE, LINEAR_RECOVERABLE} AND shap_importance >= SHAP_WOE_MINIMUM",
            alternatives=[
                AlternativeChoice(choice=name, metric_name="apply_woe",
                                  metric_value=float(apply), selected=apply)
                for name, apply in woe_decisions.items()
            ],
            final_choice=woe_features,
            justification=woe_reason,
        ))

        # ── 4. PCA-CORAL ──────────────────────────────────────────────────────
        apply_pcacoral, pcacoral_reason = rules.should_apply_pca_coral(profile)
        config.apply_pca_coral = apply_pcacoral
        rationale["pca_coral_activate"] = pcacoral_reason
        audit.record(DesignerDecision(
            step="pca_coral_activate",
            criterion="default enabled (robust aligner for any p/n regime)",
            alternatives=[
                AlternativeChoice(choice=True, metric_name="activated", metric_value=None,
                                  selected=apply_pcacoral),
                AlternativeChoice(choice=False, metric_name="activated", metric_value=None,
                                  selected=not apply_pcacoral),
            ],
            final_choice=apply_pcacoral,
            justification=pcacoral_reason,
        ))

        if apply_pcacoral:
            k, k_reason = rules.select_pca_coral_k(profile)
            config.pca_coral_k = k
            config.pca_coral_k_selection_method = "sqrt_n_target"
            rationale["pca_coral_k"] = k_reason
            audit.record(DesignerDecision(
                step="pca_coral_k",
                criterion="floor(sqrt(n_target)) capped by p and n constraints",
                alternatives=[
                    AlternativeChoice(choice=k, metric_name="n_components",
                                      metric_value=None, selected=True),
                ],
                final_choice=k,
                justification=k_reason,
            ))

        # ── 5. Calibración ────────────────────────────────────────────────────
        apply_cal, cal_reason = rules.should_recalibrate(profile)
        config.apply_calibration = apply_cal
        rationale["calibration_activate"] = cal_reason
        audit.record(DesignerDecision(
            step="calibration_activate",
            criterion="n_target_events >= threshold AND (slope deviation OR heterogeneity)",
            alternatives=[
                AlternativeChoice(choice=True, metric_name="activated", metric_value=None,
                                  selected=apply_cal),
                AlternativeChoice(choice=False, metric_name="activated", metric_value=None,
                                  selected=not apply_cal),
            ],
            final_choice=apply_cal,
            justification=cal_reason,
        ))

        if apply_cal:
            method, method_reason = rules.select_calibration_method(profile)
            config.calibration_method = method
            config.calibration_strata_fn = (
                "score_terciles" if method == "platt_stratified" else None
            )
            rationale["calibration_method"] = method_reason
            all_methods = ["platt_loo", "platt_stratified", "isotonic_loo"]
            audit.record(DesignerDecision(
                step="calibration_method",
                criterion="n_target_events threshold for isotonic; heterogeneity p-value for stratified",
                alternatives=[
                    AlternativeChoice(choice=m, metric_name="method",
                                      metric_value=None, selected=(m == method))
                    for m in all_methods
                ],
                final_choice=method,
                justification=method_reason,
            ))

        config.rationale = rationale
        config.audit = audit

        logger.info("AdapterConfig selected:\n%s", config.summary())
        return config
