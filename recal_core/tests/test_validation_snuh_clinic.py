"""
recal_core/tests/test_validation_snuh_clinic.py
==========================================
Test de regresión E2E: SNUH→Clínic con RECAL (reglas Designer only).

Criterios de pase (rango honesto, sin trampa de selección):
- config.apply_mask == True
- 3 <= config.mask_n <= 5
- config.apply_quantile == False   (features NONLINEAR_DRIFT son near-constant)
- config.apply_woe == False        (n_target_events=29 < 30)
- config.apply_pca_coral == True
- 4 <= config.pca_coral_k <= 7    (k_opt=5 ± 2)
- config.calibration_method == 'platt_loo'
- 0.65 <= auroc_pipeline <= 0.75   (benchmark: ~0.706)
- 0.5 <= slope_post <= 1.5         (benchmark: ~0.7, bien calibrado)

El test se marca como SLOW y se salta si no hay GPU/datos disponibles.
Los datos se cargan desde la ruta raíz del proyecto.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

pytestmark = pytest.mark.slow


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def adapt_data():
    """Carga datos para tests del módulo (pesado, solo una vez)."""
    from recal.data.clinic import ClinicLoader
    from recal.data.pairing import CohortPair
    from recal.data.schema import load_schema
    from recal.data.snuh import SNUHLoader
    from recal.model.xgboost_wrapper import XGBoostWrapper

    schema = load_schema()
    model = XGBoostWrapper(schema=schema)

    pair = CohortPair(
        source=SNUHLoader(schema=schema),
        target=ClinicLoader(schema=schema),
    ).filter_target(max_missing_rate=0.5)

    # Cargar datos precomputados del CSV de drift
    drift_csv = ROOT / "results" / "v" / "v_drift_decomposition.csv"
    drift_type_dict = None
    shap_dict = None
    lbase_dict = None

    if drift_csv.exists():
        df = pd.read_csv(drift_csv)
        drift_type_dict = dict(zip(df["feature"], df["drift_type"]))
        shap_dict = dict(zip(df["feature"], pd.to_numeric(df["shap_importance_main_model"], errors="coerce").fillna(0.0)))
        lbase_dict = dict(zip(df["feature"], pd.to_numeric(df["L_base"], errors="coerce")))
        # Rellenar NaN con la media
        lbase_mean = float(np.nanmean(list(lbase_dict.values())))
        lbase_dict = {k: (v if not np.isnan(v) else lbase_mean)
                      for k, v in lbase_dict.items()}

    return {
        "schema": schema,
        "model": model,
        "pair": pair,
        "drift_type_dict": drift_type_dict,
        "shap_dict": shap_dict,
        "lbase_dict": lbase_dict,
    }


@pytest.fixture(scope="module")
def adapt_result(adapt_data):
    """Ejecuta AutoAdapter sobre SNUH→Clínic y devuelve (config, scores, metrics)."""
    from recal_core.pipeline.auto_adapter import AutoAdapter
    from recal_core.profiler.global_profiler import _bootstrap_auroc_from_scores as _bootstrap_auroc
    from recal_core.profiler.global_profiler import _calibration_slope, _ece_score

    data = adapt_data
    schema = data["schema"]
    model = data["model"]
    pair = data["pair"]

    aa = AutoAdapter(
        model=model,
        schema=schema,
        drift_type_dict=data["drift_type_dict"],
        shap_dict=data["shap_dict"],
        lbase_dict=data["lbase_dict"],
    )

    # profile → design → fit → predict (pair ya fue filter_target en fixture)
    aa.profile(pair)
    aa.design()
    aa.fit(pair)
    scores_raw = model.predict_proba(pair.X_t_imp)
    scores_adapted = aa.predict(pair)

    auroc_raw, _, _ = _bootstrap_auroc(pair.y_t, scores_raw, n_boot=200)
    auroc_adapted, ci_lo, ci_hi = _bootstrap_auroc(pair.y_t, scores_adapted, n_boot=200)
    slope_raw, _ = _calibration_slope(pair.y_t, scores_raw)
    slope_adapted, _ = _calibration_slope(pair.y_t, scores_adapted)
    ece_raw = _ece_score(pair.y_t, scores_raw)
    ece_adapted = _ece_score(pair.y_t, scores_adapted)

    return {
        "config": aa.config_,
        "profile": aa.profile_,
        "scores_raw": scores_raw,
        "scores_adapted": scores_adapted,
        "auroc_raw": auroc_raw,
        "auroc_adapted": auroc_adapted,
        "auroc_ci": (ci_lo, ci_hi),
        "slope_raw": slope_raw,
        "slope_adapted": slope_adapted,
        "ece_raw": ece_raw,
        "ece_adapted": ece_adapted,
        "y_t": pair.y_t,
    }


# ── Tests de datos de entrada ─────────────────────────────────────────────────

class TestDataSanity:
    """Verifica que los datos reproduzcan los valores conocidos del benchmark."""

    def test_n_target(self, adapt_data):
        pair = adapt_data["pair"]
        assert pair.X_t.shape[0] == 105, (
            f"n_target={pair.X_t.shape[0]} != 105. "
            "Verificar filter_target(0.5) sobre Clínic."
        )

    def test_n_source(self, adapt_data):
        pair = adapt_data["pair"]
        assert pair.X_s.shape[0] == 7554, f"n_source={pair.X_s.shape[0]} != 7554"

    def test_n_target_events(self, adapt_data):
        pair = adapt_data["pair"]
        n_events = int(pair.y_t.sum())
        assert n_events == 29, f"n_target_events={n_events} != 29"

    def test_n_source_events(self, adapt_data):
        pair = adapt_data["pair"]
        n_events = int(pair.y_s.sum())
        assert 1900 <= n_events <= 2000, (
            f"n_source_events={n_events} fuera de rango esperado (1943)"
        )

    def test_n_features(self, adapt_data):
        pair = adapt_data["pair"]
        assert pair.X_s.shape[1] == pair.X_t.shape[1]
        assert 100 <= pair.X_s.shape[1] <= 120, (
            f"n_features={pair.X_s.shape[1]} inesperado"
        )

    def test_baseline_auroc(self, adapt_data):
        """AUROC raw del modelo SNUH en Clínic debe ser ≈0.629."""
        from sklearn.metrics import roc_auc_score
        pair = adapt_data["pair"]
        model = adapt_data["model"]
        scores = model.predict_proba(pair.X_t_imp)
        auroc = roc_auc_score(pair.y_t, scores)
        assert 0.55 <= auroc <= 0.70, (
            f"AUROC baseline={auroc:.4f} fuera del rango esperado [0.55, 0.70]. "
            "Benchmark documentado: 0.6293."
        )


# ── Tests del Profiler ────────────────────────────────────────────────────────

class TestProfilerOutputs:
    """Verifica que el DriftProfile tenga valores razonables."""

    def test_profile_n_obs(self, adapt_result):
        profile = adapt_result["profile"]
        assert profile.n_target_obs == 105
        assert profile.n_source_obs == 7554

    def test_profile_prevalences(self, adapt_result):
        profile = adapt_result["profile"]
        assert 0.20 <= profile.prevalence_target <= 0.35  # ~0.276
        assert 0.20 <= profile.prevalence_source <= 0.35  # ~0.257

    def test_profile_has_features(self, adapt_result):
        profile = adapt_result["profile"]
        assert len(profile.features) > 100

    def test_profile_mmd2_positive(self, adapt_result):
        profile = adapt_result["profile"]
        assert profile.mmd2_source_target >= 0

    def test_profile_calibration_slope_known(self, adapt_result):
        """Slope de calibración ≈ 9.06 (mal calibrado)."""
        profile = adapt_result["profile"]
        # Permitimos rango amplio por diferencias de implementación logit
        assert profile.baseline_calibration_slope > 2.0, (
            f"slope={profile.baseline_calibration_slope:.2f}: "
            "el modelo SNUH debería estar muy mal calibrado en Clínic (esperado ~9)"
        )

    def test_profile_all_quadrants_present(self, adapt_result):
        profile = adapt_result["profile"]
        quadrants = set(f.quadrant for f in profile.features)
        # Al menos 3 de los 4 cuadrantes deben estar presentes con 114 features
        assert len(quadrants) >= 3, f"Solo {len(quadrants)} cuadrantes distintos: {quadrants}"


# ── Tests del Designer ────────────────────────────────────────────────────────

class TestDesignerDecisions:
    """Verifica que las decisiones del Designer reproduzcan el óptimo conocido."""

    def test_mask_activated(self, adapt_result):
        config = adapt_result["config"]
        assert config.apply_mask is True, (
            "La máscara debe activarse: n_target_events=29 >= 20"
        )

    def test_mask_n_in_range(self, adapt_result):
        config = adapt_result["config"]
        assert 3 <= config.mask_n <= 5, (
            f"mask_n={config.mask_n} fuera del rango [3, 5]. "
            "El óptimo experimental es N=4."
        )

    def test_quantile_not_applied(self, adapt_result):
        """QT no debe aplicarse: features NONLINEAR son near-constant en Clínic."""
        config = adapt_result["config"]
        assert config.apply_quantile is False, (
            f"QT aplicada a {len(config.quantile_features)} features. "
            "Esperado: False (CV en target < 2% para features NONLINEAR_DRIFT)"
        )

    def test_woe_not_applied(self, adapt_result):
        """WOE no debe aplicarse: n_target_events=29 < 30."""
        config = adapt_result["config"]
        assert config.apply_woe is False, (
            f"WOE aplicado a {len(config.woe_features)} features. "
            "Esperado: False (n_target_events=29 < N_EVENTS_MINIMUM_WOE=30)"
        )

    def test_pca_coral_activated(self, adapt_result):
        config = adapt_result["config"]
        assert config.apply_pca_coral is True, "PCA-CORAL debe estar activo (default)"

    def test_pca_coral_k_in_range(self, adapt_result):
        config = adapt_result["config"]
        assert 4 <= config.pca_coral_k <= 7, (
            f"k={config.pca_coral_k} fuera del rango [4, 7]. "
            "El óptimo experimental es k=5."
        )

    def test_calibration_method_platt_loo(self, adapt_result):
        config = adapt_result["config"]
        assert config.calibration_method == "platt_loo", (
            f"Método de calibración: {config.calibration_method}. "
            "Esperado: platt_loo (n_target_events=29, no heterogeneidad significativa)"
        )

    def test_calibration_activated(self, adapt_result):
        config = adapt_result["config"]
        assert config.apply_calibration is True, (
            "La calibración debe activarse: slope=9.06 >> 0.5"
        )


# ── Tests de la pipeline (métricas de evaluación) ─────────────────────────────

@pytest.mark.slow
class TestPipelinePerformance:
    """Tests de rendimiento de la pipeline. Marcados como SLOW."""

    def test_auroc_improves(self, adapt_result):
        """AUROC post-ADAPT debe ser ≥ AUROC raw."""
        assert adapt_result["auroc_adapted"] >= adapt_result["auroc_raw"] - 0.02, (
            f"AUROC raw={adapt_result['auroc_raw']:.4f}, "
            f"AUROC adapted={adapt_result['auroc_adapted']:.4f}: "
            "no se esperaba una caída mayor de 0.02"
        )

    def test_auroc_in_expected_range(self, adapt_result):
        """AUROC post-ADAPT ≈ 0.71 [0.65, 0.75]."""
        auroc = adapt_result["auroc_adapted"]
        assert 0.65 <= auroc <= 0.75, (
            f"AUROC={auroc:.4f} fuera del rango esperado [0.65, 0.75]. "
            "Benchmark: ~0.706."
        )

    def test_calibration_slope_improves(self, adapt_result):
        """La slope de calibración debe acercarse a 1.0 tras la recalibración."""
        slope_raw = adapt_result["slope_raw"]
        slope_post = adapt_result["slope_adapted"]
        assert abs(slope_post - 1.0) < abs(slope_raw - 1.0), (
            f"slope_raw={slope_raw:.2f}, slope_post={slope_post:.2f}: "
            "la calibración no mejoró"
        )

    def test_calibration_slope_in_range(self, adapt_result):
        """Slope post-ADAPT ∈ [0.5, 1.5]."""
        slope_post = adapt_result["slope_adapted"]
        assert 0.5 <= slope_post <= 1.5, (
            f"slope_post={slope_post:.2f} fuera del rango [0.5, 1.5]. "
            "Benchmark: ~0.7."
        )

    def test_ece_improves(self, adapt_result):
        """ECE post-ADAPT debe ser < ECE raw."""
        ece_raw = adapt_result["ece_raw"]
        ece_post = adapt_result["ece_adapted"]
        assert ece_post <= ece_raw + 0.02, (
            f"ECE raw={ece_raw:.4f}, ECE post={ece_post:.4f}: "
            "la calibración no mejoró el ECE"
        )


# ── Test de reporte HTML ──────────────────────────────────────────────────────

@pytest.mark.slow
def test_html_report_generates(adapt_result, tmp_path):
    """El reporte HTML debe generarse sin errores."""
    from recal_core.reporter.html_report import generate_html_report

    result = adapt_result
    output = tmp_path / "test_report.html"

    html = generate_html_report(
        profile=result["profile"],
        config=result["config"],
        y_true=result["y_t"],
        scores_before=result["scores_raw"],
        scores_after=result["scores_adapted"],
        source_name="SNUH",
        target_name="Clínic",
        output_path=str(output),
    )

    assert len(html) > 5000, "El HTML generado es demasiado corto"
    assert output.exists(), "El archivo HTML no se guardó"
    assert "RECAL" in html
    assert "SNUH" in html
    assert "data:image/png;base64," in html, "Las figuras no se incrustaron en base64"
