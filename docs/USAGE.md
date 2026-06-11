# RECAL — Usage walkthrough

End-to-end guide for building a domain-transfer wrapper with RECAL.
Assumes RECAL is installed (`pip install -e ".[dev]"`).

---

## Step 1 — Minimum file structure

```
inputs/
  models/
    your_model.json          # the frozen model (see MODEL_FORMAT.md)
  source/
    source.csv               # training/source cohort (no labels needed at runtime,
                             # but needed for covariance alignment)
  target/
    target.csv               # target validation cohort WITH outcome labels
  feature_schema.json        # list of numeric feature names (see below)
configs/
  my_run.yaml                # your run config (copy from example below)
```

**Feature schema** — minimal JSON format:

```json
["feature_a", "feature_b", "feature_c"]
```

Or a text file with one feature name per line — both are accepted by the
`schema:` key in the config.

---

## Step 2 — Annotated YAML configuration

Copy `configs/example_snuh_to_clinic.yaml` and edit.  Every parameter is
documented below.

```yaml
# ── Model ──────────────────────────────────────────────────────────────────
model:
  path: inputs/models/your_model.json   # required
  type: xgboost                         # optional: auto-detected from extension
  # pipeline: inputs/models/your_model_pipeline.json
  #   Path to a preprocessing pipeline JSON that transforms raw columns into
  #   the features expected by the model (e.g. PCA + low-VIF selection).
  #   If omitted, RECAL auto-detects any *_pipeline.json in the same folder
  #   as the model.  Set to null to disable auto-detection.
  # custom_loader: path/to/loader.py    # only for BYOM models (see MODEL_FORMAT.md)

# ── Source cohort ───────────────────────────────────────────────────────────
source:
  path: inputs/source/source.csv
  outcome_col: label          # column name of the binary outcome (0/1)
  schema: inputs/feature_schema.json

# ── Target validation cohort (must have outcome labels) ─────────────────────
target:
  path: inputs/target/target.csv
  outcome_col: label
  # unit_corrections: apply simple unit transforms before alignment
  #   div10 → value / 10   |  mul2 → value * 2   |  abs → abs(value)
  #   float → value * float
  unit_corrections: {}

# ── Adaptation parameters ───────────────────────────────────────────────────
recal_core:
  pca_k: 5              # heuristic starting k for PCA-CORAL alignment
                        # The Designer runs a mini-sweep comparing PCA-CORAL
                        # k = pca_k-1, pca_k, pca_k+1 vs CORAL pure on
                        # target AUROC, selecting the best strategy.
                        # Lower → faster, less overfitting risk
  apply_qt: true        # add quantile transform as a candidate per feature
  apply_woe: "auto"     # "auto" | true | false — WOE encoding for binary features
  max_missing_rate: 0.5 # drop target rows with > 50% NaN values
  drift_csv: outputs/cache/drift_decomposition.csv  # cache path
                        # comment out to disable caching

# ── Overfitting check (k-fold honest validation) ───────────────────────────
overfitting_check:
  enabled: true
  method: kfold         # "kfold" | "none"
  n_splits: 5
  random_state: 42

# ── Output paths ────────────────────────────────────────────────────────────
output:
  report: outputs/reports/my_run.html
  recal_model: outputs/recal_models/my_run.joblib
  metrics_json: outputs/reports/my_run.metrics.json
  source_name: "Source"   # label shown in the report
  target_name: "Target"
  timestamp: true         # true (default) → appends _YYYYMMDD_HHMMSS to every
                          # output file stem so concurrent runs never overwrite
                          # each other.  Set to false for fixed, predictable names.

# ── Joint drift thresholds ──────────────────────────────────────────────────
# Controls the VIF-based multicollinearity analysis (source cohort only).
# Target VIF is not computed because small target cohorts (n ≤ p) make OLS
# singular, producing unreliable values.
joint_drift:
  delta_vif_warn: 5           # absolute VIF source ≥ this → WATCH
  delta_vif_severe: 10        # absolute VIF source ≥ this → SEVERE
  compute_mi_matrix: false    # true → O(p²) MI matrix delta (slow, optional)
  severe_share_threshold: 0.20  # > 20% SEVERE features → retraining recommended

# ── Regularisation ──────────────────────────────────────────────────────────
regularization:
  shrinkage: "auto"   # covariance shrinkage for CORAL/PCA-CORAL
                      # "auto" = Ledoit-Wolf  |  float ∈ [0,1]  |  null = legacy
  calibration_C: 1.0  # Platt L2 inverse strength
                      # lower → stronger regularisation (safer with small n_target)
  woe_smoothing: 0.5  # Laplace smoothing for WOE bins
                      # 0.5 = Jeffreys prior; increase if target n is small

# ── Optional evaluations ────────────────────────────────────────────────────
evaluation:
  oracle_eval: true               # train oracle XGBoost on target (k-fold)
  oracle_cv: 5                    # folds for the oracle
  feature_attribution: true       # per-feature attribution of the recoverable gap
  feature_attribution_top_n: 10   # show top-N features in attribution chart
  counterfactual_alternatives: 3  # alternatives per key designer decision
  brier_decompose: true           # decompose Brier score (reliability/resolution)
```

**Skip expensive evaluations** when iterating quickly:

```yaml
evaluation:
  oracle_eval: false
  feature_attribution: false
  counterfactual_alternatives: 0
  brier_decompose: false
```

Or pass `--skip-expensive` on the CLI (equivalent shorthand).

---

## Step 3 — Run the CLI

```bash
# Standard run
python -m recal_cli.run --config configs/my_run.yaml

# Skip oracle + attribution (faster iteration)
python -m recal_cli.run --config configs/my_run.yaml --skip-expensive

# Override a parameter without editing the YAML
python -m recal_cli.run --config configs/my_run.yaml --override recal_core.pca_k=8

# Disable k-fold CV
python -m recal_cli.run --config configs/my_run.yaml --no-cv
```

Console output shows progress section by section.  Total wall time for ~100
features on a modern laptop: ~5 min cold (first drift cache build), ~90 s warm.

---

## Step 4 — Interpreting the HTML report

Open `outputs/reports/my_run.html` in any browser.  The sections appear in
this order; read them in the same order.

### 4.1 Executive summary

**The two numbers that matter first:**

| Key metric | Where to look | Good range |
|------------|---------------|------------|
| `recovery_ratio` | Top of summary | > 0.7 |
| `optimism_gap` | Below recovery_ratio | < 0.02 |

The summary shows a green / amber / red verdict box.  Green means the wrapper
is safe to use.  Amber means proceed with caution (read the details).  Red
means retraining is recommended — read the decision boundary table in
[ARCHITECTURE.md](ARCHITECTURE.md#decision-boundaries).

### 4.2 Per-feature drift

A table with every feature, its drift type (one of six categories), SHAP
importance in the original model, and the alignment method the designer
selected.  Features flagged SEVERE (ΔVIF > `delta_vif_severe`) are highlighted.

### 4.3 Joint drift

Covariance structure analysis: VIF per feature for the **source cohort only**
(target VIF is omitted — small target cohorts make OLS underdetermined), overall
condition number, and effective rank.  If `compute_mi_matrix: true`, a mutual
information delta matrix is also shown.

### 4.4 Drift attribution with oracle

Three AUROC bars:

1. **Raw model** — frozen model, no adaptation, on target.
2. **RECAL model** — after full wrapper, on target.
3. **Oracle** — XGBoost re-trained on target in k-fold (theoretical ceiling).

From these three numbers:

- `recoverable_gap` = Adapted − Raw (what the wrapper gains).
- `irreducible_gap` = Oracle − Adapted (what cannot be recovered without
  retraining).
- `recovery_ratio` = recoverable / (recoverable + irreducible).

All with 95 % DeLong confidence intervals.

### 4.5 Designer audit trail

Every decision the designer made, the alternatives it evaluated, and the
reason for the final choice.  Backed by the `outputs/audit/<run_id>.yaml`.

### 4.6 Counterfactuals

Performance under alternative configurations close to the chosen one.
Use this to check whether results are stable or highly sensitive to a
specific design choice.

### 4.7 Calibration decomposition

Brier score decomposed into **reliability** (calibration error),
**resolution** (sharpness), and **uncertainty** (irreducible noise).
Shown for source and target before/after calibration.

### 4.8 Per-feature log

Raw drift statistics for every feature: mean shift, variance ratio, KS
statistic, CORAL recovery, alignment method applied.

---

## Step 5 — Using the serialised wrapper for inference

After a successful run, the wrapper at `outputs/recal_models/my_run.joblib`
is ready for inference on **new data from the same target domain** (not a
third, unseen domain).

```python
import joblib
import pandas as pd

# Load the wrapper built in the previous run
wrapper = joblib.load("outputs/recal_models/my_run.joblib")

# Load new target-domain data (same feature schema, no labels needed)
X_new = pd.read_csv("inputs/target/new_batch.csv")[wrapper.schema]

# Get calibrated probabilities
proba = wrapper.predict_proba(X_new)   # shape (n,) — probability of positive class
print(proba[:5])
```

**Important caveats:**

- `wrapper.schema` is the list of feature names the pipeline was built on.
  New data must contain exactly those columns.
- The wrapper is valid only for data from the **same target distribution**.
  A deployment drift from the original target would require a new RECAL run.
- If the model was not re-trained, predictions remain limited by the frozen
  model's discriminative ability.

---

## Step 6 — Reproducible re-runs from the audit YAML

Every run writes `outputs/audit/<run_id>.yaml` with SHA-256 hashes of all
inputs and the full resolved config.  To reproduce a run exactly:

1. Verify that your input files still match the stored hashes.
2. Copy the `config` section from the YAML back into a new YAML file, or
   pass the audit YAML directly (replay flag planned for v0.3).
3. Run with the same Python environment (dependency versions are recorded in
   the audit YAML under `dependencies`).

---

*See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design and
[DRIFT_REPORT.md](DRIFT_REPORT.md) for metric definitions.*
