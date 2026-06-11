"""Apply RECAL to the synthetic domain-shift scenario.

Runs AutoAdapter (profile → design → fit → predict) on:
  source = synthetic_features.csv  (trained domain)
  target = synthetic_transfer.csv  (shifted domain)

Includes:
  - Oracle evaluation (XGBoost natively trained on target, k-fold OOF)
  - K-fold CV for honest ECE and AUROC CIs
  - Drift decomposition (recovery_ratio, optimism_gap)
  - Full RECAL HTML report
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from recal.data.pairing import CohortPair
from recal.model.xgboost_wrapper import XGBoostWrapper
from recal_cli.cross_validate import _bootstrap_auroc_ci
from recal_cli.data_loader import GenericCohortLoader
from recal_cli.drift_attribution import drift_decomposition
from recal_core.pipeline.auto_adapter import AutoAdapter
from recal_core.reporter.html_report import generate_html_report


class NNWrapper:
    """Simple sklearn MLPClassifier + scaler wrapper for RECAL."""

    def __init__(self, model, scaler, schema):
        self.model = model
        self.scaler = scaler
        self.schema = list(schema)
        self.n_features_in_ = len(schema)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_clean = np.where(np.isnan(X), 0.0, X)
        X_s = self.scaler.transform(X_clean)
        return self.model.predict_proba(X_s)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        # Use mean absolute weights of the first hidden layer as proxy importance
        w = self.model.coefs_[0]  # shape (n_features, n_hidden)
        imp = np.abs(w).mean(axis=1)
        imp = imp / imp.sum() if imp.sum() > 0 else imp
        return {name: float(v) for name, v in zip(self.schema, imp)}

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        # Fallback to feature_importance-based SHAP approximation
        fi = np.array(list(self.feature_importance().values()))
        return (X - X.mean(axis=0)) * fi


# Toggle: use 'xgb' or 'nn'
MODEL_KEY = "xgb"

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir = Path(__file__).parent
recal_dir = data_dir / "recal"
recal_dir.mkdir(exist_ok=True)

# ── 1. Load schema ───────────────────────────────────────────────────────────
schema = json.loads((data_dir / "models" / "feature_schema.json").read_text())

# ── 2. Create loaders ────────────────────────────────────────────────────────
source_loader = GenericCohortLoader(
    path=data_dir / "synthetic_features.csv",
    schema=schema,
    outcome_col="target",
)
target_loader = GenericCohortLoader(
    path=data_dir / "synthetic_transfer.csv",
    schema=schema,
    outcome_col="target",
)

# ── 3. Build pair ────────────────────────────────────────────────────────────
pair = CohortPair(source=source_loader, target=target_loader)
filtered_pair = pair.filter_target(max_missing_rate=0.5)

print(f"Source: {filtered_pair.X_s.shape[0]} rows, {filtered_pair.y_s.sum()} pos")
print(f"Target: {filtered_pair.X_t.shape[0]} rows, {filtered_pair.y_t.sum()} pos")

# ── 4. Load model (source-trained) ─────────────────────────────────────────
if MODEL_KEY == "xgb":
    model = XGBoostWrapper(
        schema=schema,
        model_path=data_dir / "models" / "xgb_model.json",
    )
    print("XGBoost model loaded.")
else:
    nn_data = joblib.load(data_dir / "models" / "nn_model.joblib")
    model = NNWrapper(
        model=nn_data["model"],
        scaler=nn_data["scaler"],
        schema=schema,
    )
    print("Neural Network model loaded.")

y_t = filtered_pair.y_t
X_t = filtered_pair.X_t

# ── 5. Evaluate RAW model on target ──────────────────────────────────────────
raw_proba = model.predict_proba(X_t)
raw_pred = (raw_proba >= 0.5).astype(int)

raw_metrics = {
    "auroc": roc_auc_score(y_t, raw_proba),
    "accuracy": accuracy_score(y_t, raw_pred),
    "precision": precision_score(y_t, raw_pred, zero_division=0),
    "recall": recall_score(y_t, raw_pred, zero_division=0),
    "f1": f1_score(y_t, raw_pred, zero_division=0),
}
raw_ci = _bootstrap_auroc_ci(y_t, raw_proba)
print(f"RAW   — AUROC: {raw_metrics['auroc']:.4f} [{raw_ci[0]:.4f}–{raw_ci[1]:.4f}], F1: {raw_metrics['f1']:.4f}")

# ── 6. Profile + Design + Fit (pipeline completa) ────────────────────────────
adapter = AutoAdapter(model=model, schema=schema)
adapter.profile(filtered_pair)
adapter.design(filtered_pair, pca_k=5, max_n_sweep=30)

# Mostrar configuración seleccionada por el Designer
if adapter._config:
    print(f"\n--- Designer decisions ---")
    print(adapter._config.summary())

# Fit y predict
adapter.fit(filtered_pair)
recal_proba = adapter.predict(filtered_pair)
recal_pred = (recal_proba >= 0.5).astype(int)

recal_metrics = {
    "auroc": roc_auc_score(y_t, recal_proba),
    "accuracy": accuracy_score(y_t, recal_pred),
    "precision": precision_score(y_t, recal_pred, zero_division=0),
    "recall": recall_score(y_t, recal_pred, zero_division=0),
    "f1": f1_score(y_t, recal_pred, zero_division=0),
}
recal_ci = _bootstrap_auroc_ci(y_t, recal_proba)
print(f"RECAL — AUROC: {recal_metrics['auroc']:.4f} [{recal_ci[0]:.4f}–{recal_ci[1]:.4f}], F1: {recal_metrics['f1']:.4f}")

# ── 7. K-fold CV honesto ─────────────────────────────────────────────────────
print("\nRunning 5-fold CV (honest OOF) ...")

def _make_pair_subset(pair, idx):
    """Crea un CohortPair con target restringido a idx."""
    new = object.__new__(CohortPair)
    new._source = pair._source
    new._target = pair._target
    new.schema = pair.schema
    new._X_s = pair._X_s
    new._y_s = pair._y_s
    new._X_t = pair._X_t[idx]
    new._y_t = pair._y_t[idx]
    new._p = pair._p
    return new

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_scores = np.full(len(y_t), np.nan)
cv_fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_t, y_t)):
    pair_tr = _make_pair_subset(filtered_pair, train_idx)
    pair_te = _make_pair_subset(filtered_pair, test_idx)

    # AutoAdapter completo (profile → design → fit) en train fold
    cv_adapter = AutoAdapter(model=model, schema=schema)
    cv_adapter.profile(pair_tr)
    cv_adapter.design(pair_tr, pca_k=5, max_n_sweep=15)
    cv_adapter.fit(pair_tr)

    scores_test = cv_adapter.predict(pair_te)
    oof_scores[test_idx] = scores_test

    cfg = cv_adapter._config
    m_fold = {
        "auroc": roc_auc_score(y_t[test_idx], scores_test),
        "f1": f1_score(y_t[test_idx], (scores_test >= 0.5).astype(int), zero_division=0),
    }
    cv_fold_metrics.append(m_fold)
    print(f"  fold {fold + 1}/5: mask={cfg.apply_mask}(N={cfg.mask_n}) PCA-CORAL k={cfg.pca_coral_k} WOE={cfg.apply_woe} QT={cfg.apply_quantile} cal={cfg.apply_calibration} → AUROC={m_fold['auroc']:.3f} F1={m_fold['f1']:.3f}")

cv_auroc = roc_auc_score(y_t[~np.isnan(oof_scores)], oof_scores[~np.isnan(oof_scores)])
cv_ci = _bootstrap_auroc_ci(y_t[~np.isnan(oof_scores)], oof_scores[~np.isnan(oof_scores)])
print(f"CV OOF — AUROC: {cv_auroc:.4f} [{cv_ci[0]:.4f}–{cv_ci[1]:.4f}], F1: {np.mean([m['f1'] for m in cv_fold_metrics]):.4f}")

# ── 8. Oracle: XGBoost nativo en target (k-fold OOF) ─────────────────────────
print("\nRunning Oracle (native XGBoost on target, 5-fold OOF) ...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oracle_scores = np.full(len(y_t), np.nan)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_t, y_t)):
    X_tr, X_te = X_t[train_idx], X_t[test_idx]
    y_tr, y_te = y_t[train_idx], y_t[test_idx]
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 5, "eta": 0.05, "seed": 42},
        dtrain,
        num_boost_round=200,
    )
    oracle_scores[test_idx] = booster.predict(xgb.DMatrix(X_te))

oracle_auroc = roc_auc_score(y_t, oracle_scores)
oracle_ci = _bootstrap_auroc_ci(y_t, oracle_scores)
print(f"ORACLE — AUROC: {oracle_auroc:.4f} [{oracle_ci[0]:.4f}–{oracle_ci[1]:.4f}]")

# ── 9. Drift decomposition ────────────────────────────────────────────────────
decomp = drift_decomposition(
    auroc_raw=raw_metrics["auroc"],
    auroc_adapted=recal_metrics["auroc"],
    auroc_oracle=oracle_auroc,
    ci_raw=raw_ci,
    ci_adapted=recal_ci,
    ci_oracle=oracle_ci,
)
print("\nDrift decomposition:")
print(f"  total_gap       = {decomp['total_gap']:+.4f}")
print(f"  recoverable_gap = {decomp['recoverable_gap']:+.4f}")
print(f"  irreducible_gap = {decomp['irreducible_gap']:+.4f}")
print(f"  recovery_ratio  = {decomp['recovery_ratio']:.3f} [{decomp['recovery_ratio_ci'][0]:.3f}–{decomp['recovery_ratio_ci'][1]:.3f}]" if decomp.get("recovery_ratio") else "  recovery_ratio  = N/A")

# ── 10. Save adapted model & config ──────────────────────────────────────────
joblib.dump(adapter, recal_dir / "adapted_model.joblib")
config_dict = adapter._config.__dict__ if adapter._config else {}
(recal_dir / "adapter_config.json").write_text(
    json.dumps({k: str(v) for k, v in config_dict.items()}, indent=2, default=str)
)

# ── 11. Save profile summary ─────────────────────────────────────────────────
if adapter._profile:
    profile_summary = {
        "n_source": int(filtered_pair.X_s.shape[0]),
        "n_target": int(filtered_pair.X_t.shape[0]),
        "n_pos_target": int(y_t.sum()),
        "prevalence_target": float(y_t.mean()),
    }
    (recal_dir / "drift_profile.json").write_text(json.dumps(profile_summary, indent=2, default=str))

# ── 12. Generate FULL RECAL HTML report ─────────────────────────────────────
in_sample = {
    "source": None,  # no tenemos métricas del source original
    "raw": {**raw_metrics, "auroc_ci": raw_ci},
    "adapted": {**recal_metrics, "auroc_ci": recal_ci},
    "n_source": int(filtered_pair.X_s.shape[0]),
    "n_source_events": int(filtered_pair.y_s.sum()),
}
cv_results_dict = {
    "oof_metrics": {"auroc": cv_auroc, "f1": float(np.mean([m["f1"] for m in cv_fold_metrics]))},
    "oof_auroc_ci": cv_ci,
    "per_fold": [],  # el reporte espera per_fold con keys específicas; simplificamos
    "n_splits": 5,
}

html = generate_html_report(
    profile=adapter._profile,
    config=adapter._config,
    y_true=y_t,
    scores_before=raw_proba,
    scores_after=recal_proba,
    source_name="SyntheticSource",
    target_name="SyntheticTransfer",
    output_path=str(recal_dir / "recal_report.html"),
    auroc_after=recal_metrics["auroc"],
    auroc_ci_after=recal_ci,
    slope_after=None,  # se calcula internamente
    ece_after=None,    # se calcula internamente
    cv_results=cv_results_dict,
    in_sample_metrics=in_sample,
    drift_decomp=decomp,
    oracle_results={"auroc": oracle_auroc, "auroc_ci": oracle_ci},
)
print(f"\nFull RECAL report saved: {recal_dir / 'recal_report.html'}")

# ── 13. Save comprehensive JSON summary ──────────────────────────────────────
summary = {
    "cohorts": {
        "n_source": int(filtered_pair.X_s.shape[0]),
        "n_target": int(filtered_pair.X_t.shape[0]),
        "n_pos_target": int(y_t.sum()),
        "prevalence_target": float(y_t.mean()),
    },
    "raw": {**raw_metrics, "auroc_ci": list(raw_ci)},
    "recal": {**recal_metrics, "auroc_ci": list(recal_ci)},
    "cv_oof": {
        "auroc": cv_auroc,
        "auroc_ci": list(cv_ci),
        "f1": float(np.mean([m["f1"] for m in cv_fold_metrics])) if cv_fold_metrics else np.nan,
    },
    "oracle": {"auroc": oracle_auroc, "auroc_ci": list(oracle_ci)},
    "drift_decomposition": {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in decomp.items()
    },
    "config": {k: str(v) for k, v in config_dict.items()},
}
(recal_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"Metrics summary: {recal_dir / 'metrics_summary.json'}")
print("Done.")
