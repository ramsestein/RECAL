"""
Tests para IdentityAligner, AdaBNAligner, OTAligner y SelectiveAligner.
"""

from __future__ import annotations

import numpy as np
import pytest

from domain_transfer.align.adabn import AdaBNAligner
from domain_transfer.align.identity import IdentityAligner
from domain_transfer.align.pca_coral import PCACoralAligner
from domain_transfer.align.selective import SelectiveAligner


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
        from domain_transfer.align.optimal_transport import OTAligner

        aligner = OTAligner(reg=0.5, max_src_samples=200)
        out = aligner.fit_transform(synthetic_source, synthetic_target)
        assert out.shape == synthetic_target.shape

    def test_nan_mask_restored(self, synthetic_source, synthetic_target, nan_mask_target):
        from domain_transfer.align.optimal_transport import OTAligner

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
        from domain_transfer.align.optimal_transport import OTAligner
        from scipy.stats import wasserstein_distance

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
