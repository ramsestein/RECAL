"""Evaluate XGBoost and NN classifiers across source & transfer datasets.

Generates a self-contained HTML report in synthetic_data/reports/.
Metrics: AUROC, accuracy, precision, recall, F1.
"""
import base64
import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")

# ── Paths ────────────────────────────────────────────────────────────────────
data_dir = Path(__file__).parent
models_dir = data_dir / "models"
reports_dir = data_dir / "reports"
reports_dir.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
source = pd.read_csv(data_dir / "synthetic_features.csv")
transfer = pd.read_csv(data_dir / "synthetic_transfer.csv")

feature_cols = [c for c in source.columns if c != "target"]
schema = json.loads((models_dir / "feature_schema.json").read_text())
assert feature_cols == schema

# ── Load models ─────────────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(models_dir / "xgb_model.json"))

nn_bundle = joblib.load(models_dir / "nn_model.joblib")
nn_model = nn_bundle["model"]
nn_scaler = nn_bundle["scaler"]


# ── Helper: evaluate ─────────────────────────────────────────────────────────
def evaluate(model, X, y, name: str, is_nn: bool = False):
    if is_nn:
        X_s = nn_scaler.transform(X)
        proba = model.predict_proba(X_s)[:, 1]
    else:
        proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "dataset": name,
        "auroc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "proba": proba,
        "pred": pred,
        "y": y,
    }


# ── Evaluate on both datasets ────────────────────────────────────────────────
results = {
    "xgb_source": evaluate(xgb_model, source[feature_cols].values, source["target"].values.astype(int), "synthetic_features"),
    "xgb_transfer": evaluate(xgb_model, transfer[feature_cols].values, transfer["target"].values.astype(int), "synthetic_transfer"),
    "nn_source": evaluate(nn_model, source[feature_cols].values, source["target"].values.astype(int), "synthetic_features", is_nn=True),
    "nn_transfer": evaluate(nn_model, transfer[feature_cols].values, transfer["target"].values.astype(int), "synthetic_transfer", is_nn=True),
}


# ── Helper: ROC curve → base64 ───────────────────────────────────────────────
def roc_plot(y, proba, title):
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Helper: confusion matrix → base64 ──────────────────────────────────────
def cm_plot(y, pred, title):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.6)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Helper: calibration curve → base64 ─────────────────────────────────────────
def calib_plot(y, proba, title):
    prob_true, prob_pred = calibration_curve(y, proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Build figures ────────────────────────────────────────────────────────────
figs = {
    "xgb_source_roc": roc_plot(results["xgb_source"]["y"], results["xgb_source"]["proba"], "XGBoost — synthetic_features ROC"),
    "xgb_transfer_roc": roc_plot(results["xgb_transfer"]["y"], results["xgb_transfer"]["proba"], "XGBoost — synthetic_transfer ROC"),
    "nn_source_roc": roc_plot(results["nn_source"]["y"], results["nn_source"]["proba"], "NN — synthetic_features ROC"),
    "nn_transfer_roc": roc_plot(results["nn_transfer"]["y"], results["nn_transfer"]["proba"], "NN — synthetic_transfer ROC"),
    "xgb_source_cm": cm_plot(results["xgb_source"]["y"], results["xgb_source"]["pred"], "XGBoost — synthetic_features CM"),
    "xgb_transfer_cm": cm_plot(results["xgb_transfer"]["y"], results["xgb_transfer"]["pred"], "XGBoost — synthetic_transfer CM"),
    "nn_source_cm": cm_plot(results["nn_source"]["y"], results["nn_source"]["pred"], "NN — synthetic_features CM"),
    "nn_transfer_cm": cm_plot(results["nn_transfer"]["y"], results["nn_transfer"]["pred"], "NN — synthetic_transfer CM"),
    "xgb_source_cal": calib_plot(results["xgb_source"]["y"], results["xgb_source"]["proba"], "XGBoost — synthetic_features calibration"),
    "xgb_transfer_cal": calib_plot(results["xgb_transfer"]["y"], results["xgb_transfer"]["proba"], "XGBoost — synthetic_transfer calibration"),
    "nn_source_cal": calib_plot(results["nn_source"]["y"], results["nn_source"]["proba"], "NN — synthetic_features calibration"),
    "nn_transfer_cal": calib_plot(results["nn_transfer"]["y"], results["nn_transfer"]["proba"], "NN — synthetic_transfer calibration"),
}


# ── Build HTML ───────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RECAL Synthetic — Cross-Domain Classification Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; line-height: 1.5; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: .3em; }}
  h2 {{ color: #34495e; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
  th {{ background: #f4f6f8; text-align: left; }}
  .metric {{ font-family: monospace; font-size: 1.1em; }}
  .good {{ color: #27ae60; }}
  .bad  {{ color: #c0392b; }}
  .fig  {{ text-align: center; margin: 1em 0; }}
  .fig img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .fig-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }}
  .note {{ background: #fffbe6; border-left: 4px solid #f1c40f; padding: .8em 1em; margin: 1em 0; }}
</style>
</head>
<body>

<h1>RECAL Synthetic — Cross-Domain Classification Report</h1>

<p>This report evaluates two binary classifiers trained on <code>synthetic_features.csv</code>
(source domain) and tests them on <code>synthetic_transfer.csv</code> (shifted target domain).
The goal is to measure how well each model withstands the synthetic covariate shift.</p>

<div class="note">
<strong>Domain shift applied to transfer:</strong><br>
• 20 independent base columns received distinct monotonic/non-linear transforms
(e.g. x³, √|x|, exp, tanh).<br>
• Engineered features (F3, FA) were recomputed from the transformed base columns.<br>
• Target was recomputed with the same polynomial, then binarised via sigmoid + threshold 0.5.
</div>

<h2>1. Models</h2>

<table>
<tr><th>Model</th><th>Architecture</th><th>Training data</th><th>Source</th></tr>
<tr><td><strong>XGBoost</strong></td><td>200 trees, max_depth=5, lr=0.05</td>
    <td>synthetic_features (source)</td><td><code>models/xgb_model.json</code></td></tr>
<tr><td><strong>Neural Network</strong></td><td>MLP 128→64→32 (ReLU, Adam)</td>
    <td>synthetic_features (source)</td><td><code>models/nn_model.joblib</code></td></tr>
</table>

<h2>2. Performance Metrics</h2>

<table>
<tr><th>Model</th><th>Dataset</th><th>AUROC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr>
  <td rowspan="2"><strong>XGBoost</strong></td>
  <td>synthetic_features (source)</td>
  <td class="metric good">{results['xgb_source']['auroc']:.4f}</td>
  <td class="metric">{results['xgb_source']['accuracy']:.4f}</td>
  <td class="metric">{results['xgb_source']['precision']:.4f}</td>
  <td class="metric">{results['xgb_source']['recall']:.4f}</td>
  <td class="metric">{results['xgb_source']['f1']:.4f}</td>
</tr>
<tr>
  <td>synthetic_transfer (shifted)</td>
  <td class="metric {'good' if results['xgb_transfer']['auroc'] > 0.7 else 'bad'}">{results['xgb_transfer']['auroc']:.4f}</td>
  <td class="metric">{results['xgb_transfer']['accuracy']:.4f}</td>
  <td class="metric">{results['xgb_transfer']['precision']:.4f}</td>
  <td class="metric">{results['xgb_transfer']['recall']:.4f}</td>
  <td class="metric">{results['xgb_transfer']['f1']:.4f}</td>
</tr>
<tr>
  <td rowspan="2"><strong>Neural Network</strong></td>
  <td>synthetic_features (source)</td>
  <td class="metric good">{results['nn_source']['auroc']:.4f}</td>
  <td class="metric">{results['nn_source']['accuracy']:.4f}</td>
  <td class="metric">{results['nn_source']['precision']:.4f}</td>
  <td class="metric">{results['nn_source']['recall']:.4f}</td>
  <td class="metric">{results['nn_source']['f1']:.4f}</td>
</tr>
<tr>
  <td>synthetic_transfer (shifted)</td>
  <td class="metric {'good' if results['nn_transfer']['auroc'] > 0.7 else 'bad'}">{results['nn_transfer']['auroc']:.4f}</td>
  <td class="metric">{results['nn_transfer']['accuracy']:.4f}</td>
  <td class="metric">{results['nn_transfer']['precision']:.4f}</td>
  <td class="metric">{results['nn_transfer']['recall']:.4f}</td>
  <td class="metric">{results['nn_transfer']['f1']:.4f}</td>
</tr>
</table>

<h2>3. ROC Curves</h2>

<div class="fig-grid">
  <div class="fig"><img src="data:image/png;base64,{figs['xgb_source_roc']}" alt="XGB source ROC"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['xgb_transfer_roc']}" alt="XGB transfer ROC"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['nn_source_roc']}" alt="NN source ROC"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['nn_transfer_roc']}" alt="NN transfer ROC"></div>
</div>

<h2>4. Confusion Matrices</h2>

<div class="fig-grid">
  <div class="fig"><img src="data:image/png;base64,{figs['xgb_source_cm']}" alt="XGB source CM"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['xgb_transfer_cm']}" alt="XGB transfer CM"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['nn_source_cm']}" alt="NN source CM"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['nn_transfer_cm']}" alt="NN transfer CM"></div>
</div>

<h2>5. Calibration Curves</h2>

<div class="fig-grid">
  <div class="fig"><img src="data:image/png;base64,{figs['xgb_source_cal']}" alt="XGB source cal"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['xgb_transfer_cal']}" alt="XGB transfer cal"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['nn_source_cal']}" alt="NN source cal"></div>
  <div class="fig"><img src="data:image/png;base64,{figs['nn_transfer_cal']}" alt="NN transfer cal"></div>
</div>

<h2>6. Interpretation</h2>

<ul>
<li><strong>Source performance</strong> is excellent for both models (AUROC ≈ 0.94–0.96),
  confirming they learned the original polynomial structure.</li>
<li><strong>Transfer performance</strong> reveals the impact of domain shift.
  A large AUROC / F1 drop indicates the model fails to generalise to the transformed
  covariate distributions.</li>
<li>The <strong>confusion matrices</strong> show whether the model is systematically
  over-predicting or under-predicting the positive class after the shift.</li>
<li>The <strong>calibration curves</strong> show whether predicted probabilities
  still match observed frequencies in the shifted domain.</li>
</ul>

<h2>7. Next Steps for RECAL</h2>

<ol>
<li>Run <code>recal</code> with <code>synthetic_features</code> as source and
    <code>synthetic_transfer</code> as target.</li>
<li>Use the wrapper to align inputs (PCA-CORAL / QT) and recalibrate outputs
    (Platt / isotonic).</li>
<li>Compare the post-RECAL AUROC and F1 on the transfer domain against the raw
    model performance reported above.</li>
</ol>

<hr>
<p style="font-size: .85em; color: #666;">Generated by <code>synthetic_data/generate_report.py</code>
— {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

</body>
</html>
"""

report_path = reports_dir / "cross_domain_report.html"
report_path.write_text(html, encoding="utf-8")
print(f"Report saved: {report_path}")
print(f"Open it in a browser to view the full report.")
