# ADAPT — Architecture

> **Quick orientation:** three diagrams, one table.
> Read the mental model (2 min) → skim the diagrams (3 min) → done.

---

## Mental model

**The original model is frozen and never retrained.**
Its weights, thresholds, and internal structure remain exactly as delivered by
the collaborator. ADAPT does not have access to the model's training set. The
source cohort is used only to characterise the distribution that the model
already expects.

**The wrapper is a double layer *around* the model — not inside it.**
Before the model sees any data, an *alignment layer* transforms each input row
from target-domain statistics into source-domain statistics (PCA-CORAL,
quantile transform, or WOE encoding, depending on what the designer selected
per feature). After the model produces its raw output, a *calibration layer*
re-maps the raw log-odds to calibrated probabilities that are reliable in the
target domain (Platt scaling with L2 regularisation and Brier decomposition
feedback). Nothing in the model itself is touched.

**The wrapper also produces an honest drift report that guides the
retraining decision.**
Alignment and calibration can recover *distributional* drift (covariate shift,
prevalence shift, miscalibration). They cannot recover *structural* drift
(different feature–outcome relationships). The report decomposes the
performance gap into a recoverable fraction and an irreducible residual with
confidence intervals, and flags when the irreducible residual is large enough
that retraining is the only viable path.

---

## Diagram 1 — Inference flow (using the already-built wrapper)

```mermaid
flowchart LR
    A[target_row\nraw feature values] --> B

    subgraph ALIGN["Alignment layer  (fitted on target validation cohort)"]
        direction TB
        B["PCA-CORAL / QT / WOE\nper-feature transform\n───────────────────\nparams learned from\ntarget ∩ source stats"]
    end

    ALIGN --> C

    subgraph MODEL["FROZEN ORIGINAL MODEL  (never retrained)"]
        direction TB
        C["XGBoost / sklearn / Keras / PyTorch / BYOM\n───────────────────────────────────────────\nweights unchanged — exactly as delivered"]
    end

    MODEL --> D

    subgraph CALIB["Calibration layer  (fitted on target validation cohort)"]
        direction TB
        D["Platt scaling with L2  (C = calibration_C)\nBrier decomposition feedback\n───────────────────────────\nparams learned from target labels only"]
    end

    CALIB --> E["calibrated_probability (0..1)"]

    style MODEL fill:#d9d9d9,stroke:#888,stroke-dasharray:4 4
```

**What is learned from the target validation cohort:**
- Alignment: feature-wise mean/covariance shift (CORAL), empirical quantile
  mapping (QT), or event-rate bins (WOE).
- Calibration: intercept + slope of Platt sigmoid fitted to target labels.

**What comes unchanged from the original model:**
- All internal weights, split thresholds, and feature importance rankings.

---

## Diagram 2 — Fit / Construction flow (building the wrapper)

```mermaid
flowchart TD
    I1[Source cohort\nsource.csv] --> DP
    I2[Target validation cohort\ntarget.csv  with labels] --> DP
    I3["Frozen model\n(any supported backend)"] --> DP
    I4[Config YAML] --> DP

    DP["Drift profiler\nPer-feature: drift_type taxonomy\nJoint: VIF source (only),\ncondition number, effective rank,\noptional MI matrix delta"] --> DES

    DES["Designer\nDecide per-feature:\n• mask or keep\n• alignment method (CORAL/PCA-CORAL/QT/WOE/identity)\n• PCA k (shrinkage Ledoit-Wolf)\n• calibration type & C\nEvery decision → audit trail with\nalternatives considered"] --> FIT

    FIT["Pipeline fit\n• PCA-CORAL with Ledoit-Wolf shrinkage\n• Quantile transform\n• WOE encoder with Laplace smoothing\n• Platt calibration with L2"] --> VAL

    VAL["Honest k-fold validation\noptimism_gap = in_sample_AUROC − OOF_AUROC"] --> ORA

    ORA["Target oracle\nk-fold XGBoost on target only\n→ achievable ceiling AUROC"] --> ATT

    ATT["Drift attribution\nraw model → adapted → oracle\nrecoverable_gap  with DeLong CI\nirreducible_gap  with DeLong CI\nrecovery_ratio = recoverable / total"] --> CF

    CF["Counterfactual sweep\nAlternative configs near each key decision\n→ robustness evidence"] --> OUT

    OUT["Outputs\n① wrapper .joblib\n② HTML report\n③ audit .yaml\n④ drift cache .csv"]
```

---

## Diagram 3 — Relationship with the original model

```mermaid
flowchart TB
    TGT["Target data\n(new row)"]

    subgraph WRAPPER["ADAPT wrapper"]
        direction TB
        AL["Alignment layer\nPCA-CORAL · QT · WOE\n(fitted on target cohort)"]
        FM["FROZEN MODEL\nnever retrained\nweights unchanged"]
        CL["Calibration layer\nPlatt L2\n(fitted on target cohort)"]
        AL --> FM --> CL
    end

    TGT --> AL
    CL --> PROB["calibrated_probability"]
    FM -.->|"parallel output"| DR["Drift Report\nrecovery_ratio\noptimism_gap\nrecoverable / irreducible gap\nBrier decomposition\ndesigner audit trail\ncounterfactuals"]

    style FM fill:#d9d9d9,stroke:#888,stroke-dasharray:4 4,color:#444
    style WRAPPER fill:#f0f4ff,stroke:#5577cc
```

The dashed border on **FROZEN MODEL** is intentional: the wrapper wraps around
it but never reaches inside.  The drift report is a side output produced during
fit — it does not affect inference at all.

---

## Anatomy of the final wrapper

Each run produces four artefacts:

### `outputs/adapted_models/<run_id>.joblib`

A serialised `AdaptedModelWrapper` object (via `joblib`).  Calling
`.predict_proba(X_target)` on it passes rows through the alignment layer,
through the frozen model, and through the calibration layer in one step.
This is the only artefact you need for production inference over **new
data from the same target domain**.

### `outputs/reports/<run_id>.html`

A self-contained HTML file (all CSS and JS inlined, no external dependencies).
Sections:

- Executive summary — `recovery_ratio`, `optimism_gap`, green/amber/red verdict.
- Per-feature drift — drift taxonomy, SHAP importance, alignment method chosen.
- Joint drift — VIF source (target omitted: small cohort makes OLS singular), condition number, effective rank.
- Drift attribution with oracle — raw / adapted / oracle AUROC, recoverable /
  irreducible gap with 95 % DeLong CIs.
- Designer audit trail — every decision with the alternatives considered.
- Counterfactuals — performance under nearby alternative configs.
- Calibration decomposition — Brier reliability / resolution / uncertainty.
- Per-feature log — raw drift stats for every feature in the schema.

### `outputs/audit/<run_id>.yaml`

Machine-readable audit trail containing:

- SHA-256 hashes of every input file (model, source CSV, target CSV, schema).
- Full `FullConfig` dump (all parameters, including defaults).
- Designer decision log with alternatives and selection rationale.
- Per-feature alignment method assigned and reason.
- Versions of critical dependencies (`xgboost`, `scikit-learn`, `scipy`, etc.).

Use this file to reproduce any run exactly (`--audit-replay` flag, planned).

### `outputs/cache/drift_decomposition.csv`

Per-feature drift decomposition cache (LASSO + XGBoost taxonomy, six
categories).  Computing it for ~100 features takes 2–3 minutes; subsequent
runs reuse it instantly.  Delete the file to force recomputation.

---

## Decision boundaries

Operational table for interpreting report metrics.  See [DRIFT_REPORT.md](DRIFT_REPORT.md)
for metric definitions and [OVERFITTING.md](OVERFITTING.md) for the optimism
gap in depth.

| Metric | Value | Interpretation | Suggested action |
|--------|-------|----------------|------------------|
| `optimism_gap` | < 0.02 | Robust | Deploy wrapper with confidence |
| `optimism_gap` | 0.02 – 0.05 | Moderate overfitting | Increase `n_target` or reduce `max_n_sweep` |
| `optimism_gap` | > 0.05 | Suspicious | Do not deploy; collect more target data |
| `recovery_ratio` | > 0.7 | Distributional drift dominates | Wrapper sufficient |
| `recovery_ratio` | 0.3 – 0.7 | Mixed drift | Wrapper useful; evaluate retraining |
| `recovery_ratio` | < 0.3 | Structural drift dominates | Retrain |
| % features with VIF source > `delta_vif_severe` threshold | > 20 % | High multicollinearity in source | Review feature redundancy; consider dimensionality reduction |
| Brier decomp: Δ Reliability (source → target) | large positive | Source miscalibration | Wrapper resolves this well |
| Brier decomp: Δ Resolution (source → target) | negative | Discrimination loss | Retrain (alignment does not recover this) |

---

*For the end-to-end walkthrough see [USAGE.md](USAGE.md).
For metric definitions see [DRIFT_REPORT.md](DRIFT_REPORT.md).*
