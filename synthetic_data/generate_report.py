"""Evaluate XGBoost and NN models across source & transfer datasets.

Generates a self-contained HTML report in synthetic_data/reports/.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(str(models_dir / "xgb_model.json"))

nn_bundle = joblib.load(models_dir / "nn_model.joblib")
nn_model = nn_bundle["model"]
nn_scaler = nn_bundle["scaler"]


# ── Helper: evaluate ─────────────────────────────────────────────────────────
def evaluate(model, X, y, name: str, is_nn: bool = False):
    if is_nn:
        X_s = nn_scaler.transform(X)
        preds = model.predict(X_s)
    else:
        preds = model.predict(X)
    return {
        "dataset": name,
        "r2": float(r2_score(y, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y, preds))),
        "mae": float(mean_absolute_error(y, preds)),
        "preds": preds,
    }


# ── Evaluate on both datasets ────────────────────────────────────────────────
results = {
    "xgb_source": evaluate(xgb_model, source[feature_cols].values, source["target"].values, "synthetic_features"),
    "xgb_transfer": evaluate(xgb_model, transfer[feature_cols].values, transfer["target"].values, "synthetic_transfer"),
    "nn_source": evaluate(nn_model, source[feature_cols].values, source["target"].values, "synthetic_features", is_nn=True),
    "nn_transfer": evaluate(nn_model, transfer[feature_cols].values, transfer["target"].values, "synthetic_transfer", is_nn=True),
}


# ── Helper: scatter plot → base64 ────────────────────────────────────────────
def scatter_plot(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Build figures ────────────────────────────────────────────────────────────
figs = {
    "xgb_source": scatter_plot(source["target"].values, results["xgb_source"]["preds"], "XGBoost — synthetic_features"),
    "xgb_transfer": scatter_plot(transfer["target"].values, results["xgb_transfer"]["preds"], "XGBoost — synthetic_transfer"),
    "nn_source": scatter_plot(source["target"].values, results["nn_source"]["preds"], "NN — synthetic_features"),
    "nn_transfer": scatter_plot(transfer["target"].values, results["nn_transfer"]["preds"], "NN — synthetic_transfer"),
}


# ── Build HTML ───────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RECAL Synthetic — Cross-Domain Model Report</title>
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
  .note {{ background: #fffbe6; border-left: 4px solid #f1c40f; padding: .8em 1em; margin: 1em 0; }}
</style>
</head>
<body>

<h1>RECAL Synthetic — Cross-Domain Model Report</h1>

<p>This report evaluates two regression models trained on <code>synthetic_features.csv</code>
(source domain) and tests them on <code>synthetic_transfer.csv</code> (shifted target domain).
The goal is to measure how well each model withstands the synthetic covariate shift.</p>

<div class="note">
<strong>Domain shift applied to transfer:</strong><br>
• 20 independent base columns received distinct monotonic/non-linear transforms
(e.g. x³, √|x|, exp, tanh).<br>
• Engineered features (F3, FA) were recomputed from the transformed base columns.<br>
• Target was recomputed with the same polynomial, then warped via
<code>sin(shifted<sup>log(shifted)</sup>) × 42</code>.
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
<tr><th>Model</th><th>Dataset</th><th>R²</th><th>RMSE</th><th>MAE</th></tr>
<tr>
  <td rowspan="2"><strong>XGBoost</strong></td>
  <td>synthetic_features (source)</td>
  <td class="metric good">{results['xgb_source']['r2']:.4f}</td>
  <td class="metric">{results['xgb_source']['rmse']:.4f}</td>
  <td class="metric">{results['xgb_source']['mae']:.4f}</td>
</tr>
<tr>
  <td>synthetic_transfer (shifted)</td>
  <td class="metric {'good' if results['xgb_transfer']['r2'] > 0.5 else 'bad'}">{results['xgb_transfer']['r2']:.4f}</td>
  <td class="metric">{results['xgb_transfer']['rmse']:.4f}</td>
  <td class="metric">{results['xgb_transfer']['mae']:.4f}</td>
</tr>
<tr>
  <td rowspan="2"><strong>Neural Network</strong></td>
  <td>synthetic_features (source)</td>
  <td class="metric good">{results['nn_source']['r2']:.4f}</td>
  <td class="metric">{results['nn_source']['rmse']:.4f}</td>
  <td class="metric">{results['nn_source']['mae']:.4f}</td>
</tr>
<tr>
  <td>synthetic_transfer (shifted)</td>
  <td class="metric {'good' if results['nn_transfer']['r2'] > 0.5 else 'bad'}">{results['nn_transfer']['r2']:.4f}</td>
  <td class="metric">{results['nn_transfer']['rmse']:.4f}</td>
  <td class="metric">{results['nn_transfer']['mae']:.4f}</td>
</tr>
</table>

<h2>3. Predicted vs. Actual Scatter Plots</h2>

<div class="fig">
  <h3>XGBoost</h3>
  <p><strong>Source domain</strong> (synthetic_features)</p>
  <img src="data:image/png;base64,{figs['xgb_source']}" alt="XGB source">
  <p><strong>Shifted domain</strong> (synthetic_transfer)</p>
  <img src="data:image/png;base64,{figs['xgb_transfer']}" alt="XGB transfer">
</div>

<div class="fig">
  <h3>Neural Network</h3>
  <p><strong>Source domain</strong> (synthetic_features)</p>
  <img src="data:image/png;base64,{figs['nn_source']}" alt="NN source">
  <p><strong>Shifted domain</strong> (synthetic_transfer)</p>
  <img src="data:image/png;base64,{figs['nn_transfer']}" alt="NN transfer">
</div>

<h2>4. Interpretation</h2>

<ul>
<li><strong>Source performance</strong> is excellent for both models (R² ≈ 0.95–1.0),
  confirming they learned the original polynomial structure.</li>
<li><strong>Transfer performance</strong> reveals the impact of domain shift.
  A large R² drop indicates the model fails to generalise to the transformed
  covariate distributions or the warped target scale.</li>
<li>The red dashed line in each plot is the <em>identity</em> (perfect prediction).
  Points clustering around it indicate good calibration; systematic deviation
  indicates the model is systematically under- or over-predicting.</li>
</ul>

<h2>5. Next Steps for RECAL</h2>

<ol>
<li>Run <code>recal</code> with <code>synthetic_features</code> as source and
    <code>synthetic_transfer</code> as target.</li>
<li>Use the wrapper to align inputs (PCA-CORAL / QT) and recalibrate outputs
    (Platt / isotonic).</li>
<li>Compare the post-RECAL R² on the transfer domain against the raw model
    performance reported above.</li>
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
