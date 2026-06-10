"""
Tests para IdentityAligner, AdaBNAligner, OTAligner, SelectiveAligner y WOEEncoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from recal.align.adabn import AdaBNAligner
from recal.align.identity import IdentityAligner
from recal.align.pca_coral import PCACoralAligner
from recal.align.selective import SelectiveAligner

# ── IdentityAligner ───────────────────────────────────────────────────────────

class TestIdentityAligner:
    def test_output_equal_to_input(self, synthetic_source, synthetic_target):
        aligner = IdentityAligner()
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        np.testing.assert_array_equal(out, synthetic_target)

    def test_nan_mask_restored(self, synthetic_source, synthetic_target, nan_mask_target):
        aligner = IdentityAligner()
        aligner.fit(synthetic_source, synthetic_target)
        out = aligner.transform(synthetic_target, nan_mask=nan_mask_target)
        assert np.all(np.isnan(out[nan_mask_target]))
        # valores no-NaN intactos
        np.testing.assert_array_equal(
            out[~nan_mask_target], synthetic_target[~nan_mask_target]
        )


# ── AdaBNAligner ──────────────────────────────────────────────────────────────

class TestAdaBNAligner:
    def test_output_shape(self, synthetic_source, synthetic_target):
        aligner = AdaBNAligner()
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert out.shape == synthetic_target.shape

    def test_aligned_mean_close_to_source(self, synthetic_source, synthetic_target):
        """Después de AdaBN la media del target debe ser ≈ media del source."""
        aligner = AdaBNAligner()
        aligned = aligner.fit_transform(synthetic_source, synthetic_target)
        mu_s = synthetic_source.mean(axis=0)
        mu_aligned = aligned.mean(axis=0)
        np.testing.assert_allclose(mu_aligned, mu_s, atol=1e-6)

    def test_aligned_std_close_to_source(self, synthetic_source, synthetic_target):
        """Después de AdaBN el std del target debe ser ≈ std del source."""
        aligner = AdaBNAligner()
        aligned = aligner.fit_transform(synthetic_source, synthetic_target)
        std_s = synthetic_source.std(axis=0)
        std_aligned = aligned.std(axis=0)
        np.testing.assert_allclose(std_aligned, std_s, atol=1e-6)

    def test_nan_mask_restored(self, synthetic_source, synthetic_target, nan_mask_target):
        aligner = AdaBNAligner()
        aligner.fit(synthetic_source, synthetic_target)
        out = aligner.transform(synthetic_target, nan_mask=nan_mask_target)
        assert np.all(np.isnan(out[nan_mask_target]))
        assert not np.any(np.isnan(out[~nan_mask_target]))

    def test_raises_if_not_fitted(self, synthetic_target):
        aligner = AdaBNAligner()
        with pytest.raises(RuntimeError, match="fitted"):
            aligner.transform(synthetic_target)


# ── SelectiveAligner ──────────────────────────────────────────────────────────

class TestSelectiveAligner:
    @pytest.fixture
    def bottom_5_idx(self):
        return np.arange(5)  # primeras 5 columnas

    def test_output_shape(self, synthetic_source, synthetic_target, bottom_5_idx):
        sel = SelectiveAligner(PCACoralAligner(k=3), feature_indices=bottom_5_idx)
        out = sel.fit_transform(synthetic_source, synthetic_target)
        assert out.shape == synthetic_target.shape

    def test_unselected_columns_unchanged(
        self, synthetic_source, synthetic_target, bottom_5_idx
    ):
        """Las columnas no seleccionadas deben ser idénticas al target original."""
        sel = SelectiveAligner(PCACoralAligner(k=3), feature_indices=bottom_5_idx)
        out = sel.fit_transform(synthetic_source, synthetic_target)

        unselected = np.setdiff1d(
            np.arange(synthetic_target.shape[1]), bottom_5_idx
        )
        np.testing.assert_array_equal(
            out[:, unselected], synthetic_target[:, unselected]
        )

    def test_selected_columns_changed(
        self, synthetic_source, synthetic_target, bottom_5_idx
    ):
        """Las columnas seleccionadas sí deben cambiar."""
        sel = SelectiveAligner(PCACoralAligner(k=3), feature_indices=bottom_5_idx)
        out = sel.fit_transform(synthetic_source, synthetic_target)
        # Al menos algún valor debe diferir
        assert not np.allclose(out[:, bottom_5_idx], synthetic_target[:, bottom_5_idx])

    def test_nan_mask_only_in_selected(
        self, synthetic_source, synthetic_target, nan_mask_target, bottom_5_idx
    ):
        """NaN se restaura en columnas seleccionadas Y no seleccionadas."""
        sel = SelectiveAligner(PCACoralAligner(k=3), feature_indices=bottom_5_idx)
        sel.fit(synthetic_source, synthetic_target)
        out = sel.transform(synthetic_target, nan_mask=nan_mask_target)
        # Todas las posiciones marcadas como NaN deben ser NaN
        assert np.all(np.isnan(out[nan_mask_target]))

    def test_raises_if_not_fitted(self, synthetic_target):
        sel = SelectiveAligner(PCACoralAligner(k=3), feature_indices=[0, 1, 2])
        with pytest.raises(RuntimeError, match="fitted"):
            sel.transform(synthetic_target)


# ── OTAligner (solo si POT está disponible) ───────────────────────────────────

ot_available = pytest.importorskip("ot", reason="POT not installed — skipping OT tests")


class TestOTAligner:
    def test_import_and_basic_shape(self, synthetic_source, synthetic_target):
        from recal.align.optimal_transport import OTAligner

        aligner = OTAligner(reg=0.5, max_src_samples=200)
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert out.shape == synthetic_target.shape

    def test_nan_mask_restored(self, synthetic_source, synthetic_target, nan_mask_target):
        from recal.align.optimal_transport import OTAligner

        aligner = OTAligner(reg=0.5, max_src_samples=200)
        aligner.fit(synthetic_source, synthetic_target)
        out = aligner.transform(synthetic_target, nan_mask=nan_mask_target)
        assert np.all(np.isnan(out[nan_mask_target]))
        assert not np.any(np.isnan(out[~nan_mask_target]))

    def test_ot_reduces_wasserstein1_by_half(self):
        """OTAligner con reg=0.05 debe reducir la distancia W1 media en ≥50%.

        Configuración: source~U[2,4], target~U[0,2] — desplazamiento claro.
        La distancia Wasserstein-1 por feature se mide con scipy antes y
        después de alinear; se exige una reducción de al menos la mitad.
        """
        from scipy.stats import wasserstein_distance

        from recal.align.optimal_transport import OTAligner

        rng = np.random.default_rng(0)
        p = 5
        n_s, n_t = 300, 80
        X_s = rng.uniform(2, 4, size=(n_s, p))
        X_t = rng.uniform(0, 2, size=(n_t, p))

        def mean_w1(A: np.ndarray, B: np.ndarray) -> float:
            return float(np.mean([wasserstein_distance(A[:, j], B[:, j]) for j in range(p)]))

        w1_before = mean_w1(X_s, X_t)
        aligner = OTAligner(reg=0.05, max_src_samples=n_s)
        X_t_aligned = aligner.fit_transform(X_s, X_t)
        w1_after = mean_w1(X_s, X_t_aligned)

        assert w1_after < 0.5 * w1_before, (
            f"W1 reduction insufficient: {w1_before:.4f} → {w1_after:.4f} "
            f"(ratio={w1_after / w1_before:.2f}, expected <0.50)"
        )


# ── WOEEncoder ────────────────────────────────────────────────────────────────


class TestWOEEncoder:
    """Tests del WOEEncoder con foco en suavizado Laplace y estabilidad numérica."""

    def _make_data(self, n: int = 200, p: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        y = (rng.standard_normal(n) > 0).astype(int)
        return X, y

    def test_zero_event_bins_no_infinite_woe(self):
        """Bins con 0 eventos (positivos o negativos) no producen WoE infinito."""
        from recal.select.woe_encoder import WOEEncoder

        rng = np.random.default_rng(7)
        # Create a feature whose values for positive class are all in the
        # upper tail, leaving some bins with 0 positive events.
        n = 100
        y = np.zeros(n, dtype=int)
        y[:5] = 1  # only 5 positive events out of 100
        # Feature: positives in [5, 6], negatives in [-3, 3]
        x = rng.standard_normal(n)
        x[y == 1] = rng.uniform(5, 6, size=5)
        X = x.reshape(-1, 1)

        enc = WOEEncoder(n_bins=10, smoothing=0.5)
        enc.fit_supervised(X, X, y)

        # All WoE values must be finite
        for woe_vals in enc._woe_values:
            assert np.all(np.isfinite(woe_vals)), (
                f"Non-finite WoE values found: {woe_vals}"
            )

    def test_zero_negative_bins_no_infinite_woe(self):
        """Bins con 0 negativos tampoco producen WoE infinito."""
        from recal.select.woe_encoder import WOEEncoder

        rng = np.random.default_rng(13)
        n = 100
        y = np.ones(n, dtype=int)
        y[:5] = 0  # only 5 negatives
        x = rng.standard_normal(n)
        X = x.reshape(-1, 1)

        enc = WOEEncoder(n_bins=10, smoothing=0.5)
        enc.fit_supervised(X, X, y)

        for woe_vals in enc._woe_values:
            assert np.all(np.isfinite(woe_vals)), (
                f"Non-finite WoE values found: {woe_vals}"
            )

    def test_transform_output_finite(self):
        """transform() devuelve valores finitos para todos los bins."""
        from recal.select.woe_encoder import WOEEncoder

        X, y = self._make_data()
        enc = WOEEncoder(n_bins=10, smoothing=0.5)
        enc.fit_supervised(X, X, y)
        X_woe = enc.transform(X)
        assert np.all(np.isfinite(X_woe)), "transform() returned non-finite values."

    def test_smoothing_zero_raises(self):
        """smoothing <= 0 debe lanzar ValueError en construcción."""
        from recal.select.woe_encoder import WOEEncoder

        with pytest.raises(ValueError, match="smoothing"):
            WOEEncoder(smoothing=0.0)

    def test_output_shape(self):
        """transform() preserva el shape."""
        from recal.select.woe_encoder import WOEEncoder

        X, y = self._make_data(n=150, p=4)
        enc = WOEEncoder(n_bins=8, smoothing=0.5)
        enc.fit_supervised(X, X, y)
        assert enc.transform(X).shape == X.shape

