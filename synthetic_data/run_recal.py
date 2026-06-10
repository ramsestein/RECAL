"""Apply RECAL to the synthetic domain-shift scenario.

Runs AutoAdapter (profile → design → fit → predict) on:
  source = synthetic_features.csv  (trained domain)
  target = synthetic_transfer.csv  (shifted domain)

Saves results, adapted model, and the FULL RECAL HTML report in synthetic_data/recal/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from recal.data.pairing import CohortPair
from recal.model.xgboost_wrapper import XGBoostWrapper
from recal_cli.data_loader import GenericCohortLoader
from recal_core.pipeline.auto_adapter import AutoAdapter
from recal_core.reporter.html_report import generate_html_report

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

# ── 4. Load XGBoost model ───────────────────────────────────────────────────
model = XGBoostWrapper(
    schema=schema,
    model_path=data_dir / "models" / "xgb_model.json",
)
print("XGBoost model loaded.")

# ── 5. Evaluate RAW model on target ──────────────────────────────────────────
raw_proba = model.predict_proba(filtered_pair.X_t)
raw_pred = (raw_proba >= 0.5).astype(int)
y_t = filtered_pair.y_t

raw_metrics = {
    "auroc": roc_auc_score(y_t, raw_proba),
    "accuracy": accuracy_score(y_t, raw_pred),
    "precision": precision_score(y_t, raw_pred, zero_division=0),
    "recall": recall_score(y_t, raw_pred, zero_division=0),
    "f1": f1_score(y_t, raw_pred, zero_division=0),
}
print(f"RAW   — AUROC: {raw_metrics['auroc']:.4f}, F1: {raw_metrics['f1']:.4f}")

# ── 6. Run RECAL ─────────────────────────────────────────────────────────────
adapter = AutoAdapter(model=model, schema=schema)
recal_proba = adapter.auto_adapt(filtered_pair)
recal_pred = (recal_proba >= 0.5).astype(int)

recal_metrics = {
    "auroc": roc_auc_score(y_t, recal_proba),
    "accuracy": accuracy_score(y_t, recal_pred),
    "precision": precision_score(y_t, recal_pred, zero_division=0),
    "recall": recall_score(y_t, recal_pred, zero_division=0),
    "f1": f1_score(y_t, recal_pred, zero_division=0),
}
print(f"RECAL — AUROC: {recal_metrics['auroc']:.4f}, F1: {recal_metrics['f1']:.4f}")

# ── 7. Save adapted model & config ───────────────────────────────────────────
joblib.dump(adapter, recal_dir / "adapted_model.joblib")
print(f"Saved adapted model: {recal_dir / 'adapted_model.joblib'}")

config_dict = adapter._config.__dict__ if adapter._config else {}
(recal_dir / "adapter_config.json").write_text(
    json.dumps({k: str(v) for k, v in config_dict.items()}, indent=2, default=str)
)
print(f"Saved config: {recal_dir / 'adapter_config.json'}")

# ── 8. Save profile summary ───────────────────────────────────────────────────
if adapter._profile:
    profile_summary = {
        "n_source": int(filtered_pair.X_s.shape[0]),
        "n_target": int(filtered_pair.X_t.shape[0]),
        "n_pos_target": int(y_t.sum()),
        "prevalence_target": float(y_t.mean()),
        "quadrants": adapter._profile.quadrants if hasattr(adapter._profile, "quadrants") else {},
    }
    (recal_dir / "drift_profile.json").write_text(json.dumps(profile_summary, indent=2, default=str))

# ── 9. Generate FULL RECAL HTML report ──────────────────────────────────────
html = generate_html_report(
    profile=adapter._profile,
    config=adapter._config,
    y_true=y_t,
    scores_before=raw_proba,
    scores_after=recal_proba,
    source_name="SyntheticSource",
    target_name="SyntheticTransfer",
    output_path=str(recal_dir / "recal_report.html"),
)
print(f"Full RECAL report saved: {recal_dir / 'recal_report.html'}")

# ── 10. Also save a concise JSON summary ─────────────────────────────────────
summary = {
    "raw": raw_metrics,
    "recal": recal_metrics,
    "delta": {
        k: round(recal_metrics[k] - raw_metrics[k], 6)
        for k in raw_metrics
    },
    "config": {k: str(v) for k, v in (adapter._config.__dict__ if adapter._config else {}).items()},
}
(recal_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
print(f"Metrics summary: {recal_dir / 'metrics_summary.json'}")
print("Done.")
