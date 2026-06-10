"""
Tests de reproducción de los números del pipeline.

Con datos reales: verifica valores exactos del benchmark SNUH→Clínic.
Con datos sintéticos: verifica rangos permisivos (pipeline funciona).
"""

from __future__ import annotations

from pathlib import Path

from sklearn.metrics import roc_auc_score

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_pipeline():
    """Load pipeline with real data if available, otherwise synthetic."""
    from recal.data.clinic import ClinicLoader
    from recal.data.pairing import CohortPair
    from recal.data.schema import load_schema
    from recal.data.snuh import SNUHLoader
    from recal.model.xgboost_wrapper import XGBoostWrapper

    repo_root = Path(__file__).resolve().parents[1]
    real_source = repo_root / "datasets" / "SNUH_AKI(SNUH_AKI).csv"
    is_real = real_source.exists()

    if is_real:
        schema = load_schema()
        model = XGBoostWrapper(schema=schema, model_path=repo_root / "model" / "aki_external.json")
    else:
        data_dir = repo_root / "recal_core" / "tests" / "data"
        schema = load_schema(data_dir / "synthetic_schema.json")
        model = XGBoostWrapper(schema=schema, model_path=data_dir / "synthetic_model.json")

    snuh = SNUHLoader(
        schema=schema,
        csv_path=(real_source if is_real else data_dir / "synthetic_source.csv"),
    )
    clinic = ClinicLoader(
        schema=schema,
        csv_path=(repo_root / "datasets" / "Clínic_AKI.csv" if is_real else data_dir / "synthetic_target.csv"),
    )

    pair_raw = CohortPair(source=snuh, target=clinic)
    pair = pair_raw.filter_target(max_missing_rate=0.5)
    return pair, model, is_real


# ── Números de referencia ─────────────────────────────────────────────────────

REF_N_TARGET_FILTERED = 105
REF_N_POS_TARGET = 29
REF_AUROC_RAW = 0.6293
REF_AUROC_PCA_CORAL_K5 = 0.7051
ATOL = 1e-3


# ── Test 1: filtrado ──────────────────────────────────────────────────────────

def test_filter_target_count():
    """n_target después del filtrado (< 50% missing) debe ser 105."""
    pair, _, _ = _load_pipeline()
    assert pair.X_t.shape[0] == REF_N_TARGET_FILTERED, (
        f"Expected {REF_N_TARGET_FILTERED} samples after filter, "
        f"got {pair.X_t.shape[0]}"
    )


def test_filter_target_positives():
    """n_pos_target después del filtrado debe ser 29."""
    pair, _, _ = _load_pipeline()
    assert int(pair.y_t.sum()) == REF_N_POS_TARGET


# ── Test 2: AUROC Raw (sin alineación) ───────────────────────────────────────

def test_auroc_raw():
    """AUROC target sin alineación debe ser discriminativo."""
    from recal.align.identity import IdentityAligner

    pair, model, is_real = _load_pipeline()
    X_aligned = pair.align(IdentityAligner())
    proba = model.predict_proba(X_aligned)
    auroc = roc_auc_score(pair.y_t, proba)
    if is_real:
        assert abs(auroc - REF_AUROC_RAW) <= ATOL, (
            f"Raw AUROC {auroc:.4f} deviates from reference {REF_AUROC_RAW} "
            f"by {abs(auroc - REF_AUROC_RAW):.4f} > {ATOL}"
        )
    else:
        assert 0.50 <= auroc <= 0.75, (
            f"Raw AUROC {auroc:.4f} fuera del rango esperado [0.50, 0.75]"
        )


# ── Test 3: AUROC PCA-CORAL k=5 ──────────────────────────────────────────────

def test_auroc_pca_coral_k5():
    """AUROC con PCA-CORAL k=5 debe mejorar o mantener discriminación."""
    from recal.align.pca_coral import PCACoralAligner

    pair, model, is_real = _load_pipeline()
    X_aligned = pair.align(PCACoralAligner(k=5, shrinkage=None))
    proba = model.predict_proba(X_aligned)
    auroc = roc_auc_score(pair.y_t, proba)
    if is_real:
        assert abs(auroc - REF_AUROC_PCA_CORAL_K5) <= ATOL, (
            f"PCA-CORAL k=5 AUROC {auroc:.4f} deviates from reference "
            f"{REF_AUROC_PCA_CORAL_K5} by {abs(auroc - REF_AUROC_PCA_CORAL_K5):.4f} > {ATOL}"
        )
    else:
        assert 0.50 <= auroc <= 0.80, (
            f"PCA-CORAL k=5 AUROC {auroc:.4f} fuera del rango esperado [0.50, 0.80]"
        )
