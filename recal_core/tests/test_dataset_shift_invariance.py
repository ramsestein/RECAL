"""
recal_core/tests/test_dataset_shift_invariance.py
==============================================
Prueba la invarianza del Designer ante rotaciones/permutaciones de features
y cambios de escala (no deben cambiar las decisiones categóricas).

Estas son pruebas de coherencia del Designer, no de rendimiento.
"""

from __future__ import annotations

import numpy as np

from recal_core.designer.selector import ComponentSelector
from recal_core.profiler.base import DriftProfile, FeatureProfile


def _make_profile_with_events(n_target_events: int) -> DriftProfile:
    """Profile mínimo para test de invarianza."""
    n_features = 20
    features = [
        FeatureProfile(
            name=f"f{i:02d}", domain="preop",
            drift_type_v="STABLE" if i % 2 == 0 else "CONCEPT_RELATIONAL",
            lbase_score=float(i) / n_features,
            shap_importance=float(n_features - i) / n_features * 0.05,
            combined_score=float(i) / n_features,
            quadrant="A_core" if i > n_features // 2 else "D_ponzonous",
            univariate_concept_shift_beta3=0.1,
            univariate_concept_shift_qbh=0.5,
            flip_of_sign=False,
            cv_target=0.3, cv_source=0.5, var_ratio=1.0,
            near_constant_target=False,
            missing_rate_source=0.0, missing_rate_target=0.0,
        )
        for i in range(n_features)
    ]
    return DriftProfile(
        n_source_obs=5000, n_target_obs=120, n_source_events=1000,
        n_target_events=n_target_events,
        prevalence_source=0.2, prevalence_target=float(n_target_events) / 120,
        prevalence_shift_pvalue=0.3, p_n_ratio_target=float(n_features) / 120,
        mmd2_source_target=0.5, mmd2_pvalue=0.01,
        pca_variance_explained=[0.35, 0.55, 0.70, 0.82, 0.92],
        baseline_auroc=0.63, baseline_auroc_ci_low=0.52, baseline_auroc_ci_high=0.74,
        baseline_calibration_slope=8.0, baseline_ece=0.15, baseline_citl=0.02,
        features=features,
    )


class TestDesignerDecisionMonotonicity:
    """
    Las decisiones categóricas deben ser monótonas respecto al número de eventos.
    """

    def test_mask_threshold_monotone(self):
        """Por debajo del umbral: mask=False; por encima: mask=True."""
        from recal_core.designer.rules import should_mask_features
        from recal_core.profiler.constants import N_EVENTS_MINIMUM_MASK

        for n in range(1, 2 * N_EVENTS_MINIMUM_MASK):
            profile = _make_profile_with_events(n)
            apply, _ = should_mask_features(profile)
            if n < N_EVENTS_MINIMUM_MASK:
                assert apply is False, f"n_events={n}: mask no debería activarse"
            else:
                assert apply is True, f"n_events={n}: mask debería activarse"

    def test_woe_threshold_monotone(self):
        """Por debajo del umbral WOE: woe=False."""
        from recal_core.designer.rules import should_apply_woe_per_feature
        from recal_core.profiler.constants import N_EVENTS_MINIMUM_WOE

        for n in range(1, N_EVENTS_MINIMUM_WOE):
            profile = _make_profile_with_events(n)
            decisions, _ = should_apply_woe_per_feature(profile)
            assert all(not v for v in decisions.values()), (
                f"n_events={n}: WOE no debería activarse (mínimo={N_EVENTS_MINIMUM_WOE})"
            )

    def test_calibration_threshold_monotone(self):
        """Por debajo del umbral de calibración: apply=False."""
        from recal_core.designer.rules import should_recalibrate
        from recal_core.profiler.constants import N_EVENTS_MINIMUM_CALIBRATION

        for n in range(1, N_EVENTS_MINIMUM_CALIBRATION):
            profile = _make_profile_with_events(n)
            apply, _ = should_recalibrate(profile)
            assert apply is False, (
                f"n_events={n}: calibración no debería activarse "
                f"(mínimo={N_EVENTS_MINIMUM_CALIBRATION})"
            )


class TestDesignerDeterminism:
    """El Designer debe ser determinista: mismos inputs → mismos outputs."""

    def test_same_profile_same_config(self):
        selector = ComponentSelector()
        profile = _make_profile_with_events(30)
        config1 = selector.select(profile)
        config2 = selector.select(profile)

        assert config1.apply_mask == config2.apply_mask
        assert config1.mask_n == config2.mask_n
        assert config1.apply_pca_coral == config2.apply_pca_coral
        assert config1.pca_coral_k == config2.pca_coral_k
        assert config1.apply_calibration == config2.apply_calibration
        assert config1.calibration_method == config2.calibration_method

    def test_mask_features_set_consistent(self):
        """Las features enmascaradas deben ser un subconjunto del esquema."""
        selector = ComponentSelector()
        profile = _make_profile_with_events(30)
        config = selector.select(profile)

        schema_set = {f.name for f in profile.features}
        for feat in config.mask_features:
            assert feat in schema_set, f"Feature enmascarada {feat} no está en el schema"

    def test_mask_n_consistency(self):
        """mask_n debe coincidir con len(mask_features)."""
        selector = ComponentSelector()
        profile = _make_profile_with_events(30)
        config = selector.select(profile)
        if config.apply_mask:
            assert config.mask_n == len(config.mask_features)


class TestDesignerScaleInvariance:
    """
    Las decisiones categóricas no deben cambiar si escalamos los scores
    por un factor positivo constante (invarianza de rango).
    """

    def test_quadrant_assignment_scale_invariant(self):
        """assign_quadrants() debe dar el mismo resultado bajo escalado uniforme."""
        from recal_core.profiler.quadrant import assign_quadrants
        rng = np.random.default_rng(0)
        n = 50
        shap = np.abs(rng.normal(0, 1, n))
        lbase = np.abs(rng.normal(0, 1, n))

        q1 = assign_quadrants(shap, lbase)
        q2 = assign_quadrants(shap * 100, lbase * 100)

        assert q1 == q2, "assign_quadrants no es invariante al escalado uniforme"

    def test_quadrant_assignment_permutation_invariant(self):
        """assign_quadrants() debe preservar el orden relativo bajo permutaciones."""
        from recal_core.profiler.quadrant import assign_quadrants
        rng = np.random.default_rng(1)
        n = 30
        shap = np.abs(rng.normal(0, 1, n))
        lbase = np.abs(rng.normal(0, 1, n))
        perm = rng.permutation(n)

        q_orig = assign_quadrants(shap, lbase)
        q_perm = assign_quadrants(shap[perm], lbase[perm])

        for i, j in enumerate(perm):
            assert q_perm[i] == q_orig[j], (
                f"Cuadrante[{i}]={q_perm[i]} != original[{j}]={q_orig[j]} "
                "bajo permutación"
            )
