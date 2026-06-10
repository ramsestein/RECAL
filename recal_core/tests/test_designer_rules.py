"""
recal_core/tests/test_designer_rules.py
====================================
Tests unitarios de las reglas del Designer.

Prueba cada regla de forma aislada con perfiles sintéticos.
"""

from __future__ import annotations

import numpy as np

from recal_core.profiler.base import DriftProfile, FeatureProfile
from recal_core.profiler.constants import (
    CALIBRATION_SLOPE_RECAL_THRESHOLD,
    N_EVENTS_MINIMUM_CALIBRATION,
    N_EVENTS_MINIMUM_MASK,
    N_EVENTS_MINIMUM_WOE,
)

# ── Helper ────────────────────────────────────────────────────────────────────

def _make_profile(
    n_target_events=30,
    n_source_events=500,
    n_target_obs=100,
    n_source_obs=1000,
    slope=9.0,
    features=None,
) -> DriftProfile:
    if features is None:
        # Features por defecto: 10 features con scores variados
        features = [
            FeatureProfile(
                name=f"f{i:02d}", domain="preop",
                drift_type_v=(
                    "STABLE" if i < 4
                    else "CONCEPT_RELATIONAL" if i < 7
                    else "NONLINEAR_DRIFT"
                ),
                lbase_score=float(i) / 10,
                shap_importance=float(10 - i) / 10 * 0.1,
                combined_score=float(i) / 10,
                quadrant="D_ponzonous" if i < 2 else ("C_redundant" if i < 5 else "A_core"),
                univariate_concept_shift_beta3=0.1,
                univariate_concept_shift_qbh=0.5,
                flip_of_sign=False,
                cv_target=0.3 if i > 1 else 0.01,  # f00, f01 near-constant
                cv_source=0.5,
                var_ratio=3.0 if i < 4 else 1.0,  # primeras 4 con var_ratio alto
                near_constant_target=(i < 2),
                missing_rate_source=0.0,
                missing_rate_target=0.0,
            )
            for i in range(10)
        ]
    return DriftProfile(
        n_source_obs=n_source_obs,
        n_target_obs=n_target_obs,
        n_source_events=n_source_events,
        n_target_events=n_target_events,
        prevalence_source=n_source_events / n_source_obs,
        prevalence_target=n_target_events / n_target_obs,
        prevalence_shift_pvalue=0.5,
        p_n_ratio_target=10 / n_target_obs,
        mmd2_source_target=0.5,
        mmd2_pvalue=0.01,
        pca_variance_explained=[0.35, 0.55, 0.70, 0.82, 0.92],
        baseline_auroc=0.63,
        baseline_auroc_ci_low=0.52,
        baseline_auroc_ci_high=0.74,
        baseline_calibration_slope=slope,
        baseline_ece=0.15,
        baseline_citl=0.02,
        features=features,
    )


# ── Regla 1: Máscara ──────────────────────────────────────────────────────────

class TestShouldMaskFeatures:

    def test_activates_above_threshold(self):
        from recal_core.designer.rules import should_mask_features
        profile = _make_profile(n_target_events=N_EVENTS_MINIMUM_MASK)
        apply, reason = should_mask_features(profile)
        assert apply is True
        assert "mask" in reason.lower() or str(N_EVENTS_MINIMUM_MASK) in reason

    def test_deactivates_below_threshold(self):
        from recal_core.designer.rules import should_mask_features
        profile = _make_profile(n_target_events=N_EVENTS_MINIMUM_MASK - 1)
        apply, reason = should_mask_features(profile)
        assert apply is False

    def test_boundary_equals_threshold(self):
        from recal_core.designer.rules import should_mask_features
        profile = _make_profile(n_target_events=N_EVENTS_MINIMUM_MASK)
        apply, _ = should_mask_features(profile)
        assert apply is True


class TestSelectMaskN:

    def test_returns_positive_int(self):
        from recal_core.designer.rules import select_mask_n
        profile = _make_profile()
        n, reason, _sweep = select_mask_n(profile)
        assert isinstance(n, int)
        assert n >= 1

    def test_does_not_exceed_20pct(self):
        from recal_core.designer.rules import select_mask_n
        profile = _make_profile()
        n, _, _sweep = select_mask_n(profile)
        max_allowed = max(1, int(0.20 * len(profile.features)))
        assert n <= max_allowed

    def test_at_least_ponzonous_count(self):
        from recal_core.designer.rules import select_mask_n
        profile = _make_profile()
        n_ponzonous = len(profile.ponzonous_features())
        n, _, _sweep = select_mask_n(profile)
        if n_ponzonous > 0:
            assert n >= min(1, n_ponzonous)


# ── Regla 2: QuantileTransform ────────────────────────────────────────────────

class TestQuantileTransformRule:

    def test_near_constant_excluded(self):
        from recal_core.designer.rules import should_apply_quantile_transform_per_feature
        profile = _make_profile()
        decisions, _ = should_apply_quantile_transform_per_feature(profile)
        # f00 y f01 tienen near_constant_target=True → deben estar False
        for f in profile.features:
            if f.near_constant_target:
                assert decisions[f.name] is False, (
                    f"{f.name} es near-constant pero QT se activó"
                )

    def test_stable_features_excluded(self):
        from recal_core.designer.rules import should_apply_quantile_transform_per_feature
        profile = _make_profile()
        decisions, _ = should_apply_quantile_transform_per_feature(profile)
        # Features STABLE no deben activar QT
        for f in profile.features:
            if f.drift_type_v == "STABLE":
                assert decisions[f.name] is False

    def test_nonlinear_drift_with_high_cv_included(self):
        from recal_core.designer.rules import should_apply_quantile_transform_per_feature
        from recal_core.profiler.constants import CV_TARGET_QT_MINIMUM, VAR_RATIO_QT_UPPER
        # Feature NONLINEAR_DRIFT con cv_target alto y var_ratio alto → QT activa
        features = [
            FeatureProfile(
                name="nl_feature", domain="intraop",
                drift_type_v="NONLINEAR_DRIFT",
                lbase_score=0.5, shap_importance=0.05, combined_score=0.5,
                quadrant="B_noisy_important",
                univariate_concept_shift_beta3=0.0, univariate_concept_shift_qbh=1.0,
                flip_of_sign=False,
                cv_target=CV_TARGET_QT_MINIMUM + 0.1,  # por encima del mínimo
                cv_source=0.5, var_ratio=VAR_RATIO_QT_UPPER + 0.5,  # por encima del máximo
                near_constant_target=False,
                missing_rate_source=0.0, missing_rate_target=0.0,
            )
        ]
        profile = _make_profile(features=features)
        decisions, _ = should_apply_quantile_transform_per_feature(profile)
        assert decisions["nl_feature"] is True


# ── Regla 3: WOE ─────────────────────────────────────────────────────────────

class TestWOERule:

    def test_deactivates_below_n_events_target(self):
        from recal_core.designer.rules import should_apply_woe_per_feature
        profile = _make_profile(n_target_events=N_EVENTS_MINIMUM_WOE - 1)
        decisions, reason = should_apply_woe_per_feature(profile)
        assert all(not v for v in decisions.values())
        assert str(N_EVENTS_MINIMUM_WOE) in reason

    def test_activates_stable_feature_above_threshold(self):
        from recal_core.designer.rules import should_apply_woe_per_feature
        from recal_core.profiler.constants import SHAP_WOE_MINIMUM
        features = [
            FeatureProfile(
                name="stable_f", domain="preop",
                drift_type_v="STABLE",
                lbase_score=0.5, shap_importance=SHAP_WOE_MINIMUM + 0.01,
                combined_score=0.5, quadrant="A_core",
                univariate_concept_shift_beta3=0.0, univariate_concept_shift_qbh=1.0,
                flip_of_sign=False, cv_target=0.3, cv_source=0.3, var_ratio=1.0,
                near_constant_target=False, missing_rate_source=0.0, missing_rate_target=0.0,
            )
        ]
        profile = _make_profile(
            n_target_events=N_EVENTS_MINIMUM_WOE + 10,
            n_source_events=200,
            features=features,
        )
        decisions, _ = should_apply_woe_per_feature(profile)
        assert decisions["stable_f"] is True

    def test_concept_relational_excluded_from_woe(self):
        from recal_core.designer.rules import should_apply_woe_per_feature
        features = [
            FeatureProfile(
                name="cr_f", domain="preop",
                drift_type_v="CONCEPT_RELATIONAL",  # excluido
                lbase_score=0.5, shap_importance=0.05, combined_score=0.5,
                quadrant="A_core",
                univariate_concept_shift_beta3=0.5, univariate_concept_shift_qbh=0.01,
                flip_of_sign=True, cv_target=0.3, cv_source=0.3, var_ratio=1.0,
                near_constant_target=False, missing_rate_source=0.0, missing_rate_target=0.0,
            )
        ]
        profile = _make_profile(
            n_target_events=N_EVENTS_MINIMUM_WOE + 10,
            features=features,
        )
        decisions, _ = should_apply_woe_per_feature(profile)
        assert decisions["cr_f"] is False


# ── Regla 4: PCA-CORAL ────────────────────────────────────────────────────────

class TestPCACoralRule:

    def test_always_activated(self):
        from recal_core.designer.rules import should_apply_pca_coral
        profile = _make_profile()
        apply, _ = should_apply_pca_coral(profile)
        assert apply is True

    def test_k_respects_sqrt_cap(self):
        from recal_core.designer.rules import select_pca_coral_k
        from recal_core.profiler.constants import PCA_CORAL_K_RANGE_MIN
        # Con n_target pequeño, max_k = sqrt(50) ≈ 7
        profile = _make_profile(n_target_obs=50)
        k, _ = select_pca_coral_k(profile)
        max_k = max(PCA_CORAL_K_RANGE_MIN, int(np.sqrt(50)))
        assert k <= max_k

    def test_k_at_least_minimum(self):
        from recal_core.designer.rules import select_pca_coral_k
        from recal_core.profiler.constants import PCA_CORAL_K_RANGE_MIN
        profile = _make_profile(n_target_obs=20)
        k, _ = select_pca_coral_k(profile)
        assert k >= PCA_CORAL_K_RANGE_MIN

    def test_k_for_snuh_clinic(self):
        """Para n_target=105 → max_k=10, k esperado entre 2 y 10."""
        from recal_core.designer.rules import select_pca_coral_k
        profile = _make_profile(n_target_obs=105)
        k, _ = select_pca_coral_k(profile)
        assert 2 <= k <= 10, f"k={k} fuera de rango para n_target=105"


# ── Regla 5: Calibración ─────────────────────────────────────────────────────

class TestCalibrationRule:

    def test_activates_on_bad_slope(self):
        from recal_core.designer.rules import should_recalibrate
        profile = _make_profile(
            n_target_events=N_EVENTS_MINIMUM_CALIBRATION + 5,
            slope=9.0
        )
        apply, reason = should_recalibrate(profile)
        assert apply is True

    def test_deactivates_on_good_slope(self):
        from recal_core.designer.rules import should_recalibrate
        profile = _make_profile(
            n_target_events=N_EVENTS_MINIMUM_CALIBRATION + 5,
            slope=1.0 + CALIBRATION_SLOPE_RECAL_THRESHOLD * 0.9
        )
        apply, _ = should_recalibrate(profile)
        assert apply is False

    def test_deactivates_below_min_events(self):
        from recal_core.designer.rules import should_recalibrate
        profile = _make_profile(
            n_target_events=N_EVENTS_MINIMUM_CALIBRATION - 1,
            slope=9.0
        )
        apply, reason = should_recalibrate(profile)
        assert apply is False
        assert str(N_EVENTS_MINIMUM_CALIBRATION) in reason

    def test_platt_loo_default(self):
        from recal_core.designer.rules import select_calibration_method
        profile = _make_profile(n_target_events=50)
        method, _ = select_calibration_method(profile)
        assert method == "platt_loo"


# ── Tests del ComponentSelector ───────────────────────────────────────────────

class TestComponentSelector:

    def test_select_returns_adapter_config(self):
        from recal_core.designer.base import AdapterConfig
        from recal_core.designer.selector import ComponentSelector
        selector = ComponentSelector()
        profile = _make_profile()
        config = selector.select(profile)
        assert isinstance(config, AdapterConfig)

    def test_rationale_populated(self):
        from recal_core.designer.selector import ComponentSelector
        selector = ComponentSelector()
        profile = _make_profile()
        config = selector.select(profile)
        assert len(config.rationale) >= 5, "Se esperan al menos 5 entradas en rationale"

    def test_mask_features_length_matches_mask_n(self):
        from recal_core.designer.selector import ComponentSelector
        selector = ComponentSelector()
        profile = _make_profile(n_target_events=30)
        config = selector.select(profile)
        if config.apply_mask:
            assert len(config.mask_features) == config.mask_n
