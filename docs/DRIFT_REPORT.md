# RECAL — Drift report metrics reference

Every metric in the HTML report and the `metrics.json` output is explained
here: what it is, how it is computed, how to interpret it, and what thresholds
to use.

Cross-reference: [ARCHITECTURE.md — Decision boundaries](ARCHITECTURE.md#decision-boundaries).

---

## Per-feature drift metrics

### Drift type taxonomy

RECAL classifies each feature into one of six categories by combining a
LASSO-based test for mean/variance shift and an XGBoost recovery score:

| Category | Meaning |
|---|---|
| `stable` | No significant drift; alignment unlikely to help |
| `covariate_shift` | Mean/scale shift only; CORAL / QT effective |
| `missing_pattern` | Missingness rate differs; may need imputation upstream |
| `concept_drift` | Feature–outcome relationship changed; alignment cannot help |
| `redundant` | Highly correlated with another drifted feature; mask candidate |
| `high_importance_drifted` | Drifted AND high SHAP importance; priority to address |

The drift type drives the designer's alignment decision for that feature.

### SHAP importance (`shap_importance_main_model`)

SHAP mean absolute value of the feature in the **original frozen model**,
computed on the source cohort.  Higher → the feature contributes more to the
model's predictions.

High-importance drifted features are the most dangerous: the model relies on
them heavily but their distribution is shifted.  These are flagged as
`high_importance_drifted`.

### Alignment method assigned

One of: `coral`, `pca_coral`, `quantile_transform`, `woe`, `identity`, `mask`.

- `mask` → the feature is excluded from the input to the model (zeroed /
  set to its source mean).
- `identity` → no transformation; the designer judged no alignment needed.
- All others → the feature is transformed before being passed to the model.

---

## Joint drift metrics

Joint drift measures whether the **covariance structure** has changed, beyond
what per-feature analysis captures.

### VIF — Variance Inflation Factor

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

where $R^2_j$ is the coefficient of determination when feature $j$ is
regressed on all other features.  Measures multicollinearity.

- **VIF source** — computed on the source cohort only.  Target VIF is omitted
  because small target cohorts ($n \le p$) make OLS singular, producing
  unreliable values that could be capped or infinite.

| VIF source | Label | Interpretation |
|---|---|---|
| < `delta_vif_warn` (default 5) | OK | Low multicollinearity |
| ≥ `delta_vif_warn`, < `delta_vif_severe` (default 10) | WATCH | Moderate collinearity; monitor |
| ≥ `delta_vif_severe` | SEVERE | High collinearity; features may be redundant |

If more than `severe_share_threshold` (default 20 %) of features are SEVERE,
RECAL emits a retraining recommendation.

### Condition number

$$\kappa = \frac{\sigma_{\max}}{\sigma_{\min}}$$

Ratio of the largest to smallest singular value of the feature matrix.
A high condition number means the covariance matrix is nearly singular —
small input perturbations can cause large prediction changes.

- Computed separately for source and target.
- A large increase from source to target suggests the alignment problem is
  ill-posed and PCA dimensionality reduction (low `pca_k`) is important.

### Effective rank

$$r_{\text{eff}} = \exp\!\left(-\sum_i p_i \log p_i\right), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$

Entropy-based rank of the singular value distribution.  Measures how many
effective independent dimensions the data has.

- Source effective rank >> target effective rank → the target cohort has lost
  variability (possibly due to smaller sample size or patient selection bias).
- Target effective rank >> source → the target covers more variance dimensions;
  the model may encounter feature combinations it has never seen.

### Shrinkage coefficient (Ledoit-Wolf)

When `shrinkage: "auto"` the Ledoit-Wolf estimator chooses an optimal
regularisation coefficient $\alpha \in [0, 1]$ for the covariance matrices
used in CORAL/PCA-CORAL:

$$\hat{\Sigma}_{\text{shrunk}} = (1 - \alpha)\,\hat{\Sigma} + \alpha\,\mu_{\text{trace}} \cdot I$$

Reported in the joint drift section.  Values near 1 indicate that the sample
covariance is unreliable (small $n$, high $p$).  In that case, using the raw
covariance for CORAL would produce unstable alignment — shrinkage is critical.

### MI matrix delta (optional)

When `compute_mi_matrix: true`, RECAL computes pairwise mutual information
between all feature pairs in source and target and reports the difference
matrix as a heatmap.  Detects non-linear dependency changes that VIF misses.

Expensive: $O(p^2)$ estimations.  Skip for $p > 50$ unless you suspect
non-linear structural shifts.

---

## Drift attribution metrics

These metrics answer: *how much of the performance gap can the wrapper
recover, and how much is irreducible without retraining?*

### Three reference points

| Name | Definition |
|---|---|
| `auroc_raw` | Frozen model on target, no adaptation |
| `auroc_adapted` | Wrapper (alignment + calibration) on target |
| `auroc_oracle` | Oracle XGBoost trained on target in k-fold (theoretical ceiling) |

The oracle represents the best possible performance achievable by any model
trained exclusively on target data.  It is not a target to reach — it is a
ceiling that quantifies what structural drift has cost.

### `recoverable_gap`

```
recoverable_gap = auroc_adapted − auroc_raw
```

The performance gain the wrapper actually delivers.  Positive means the
wrapper helped.  Reported with a 95 % DeLong confidence interval.

### `irreducible_gap`

```
irreducible_gap = auroc_oracle − auroc_adapted
```

The performance gap that remains even after optimal alignment and
calibration.  This is what only retraining can address.

### `recovery_ratio`

```
recovery_ratio = recoverable_gap / (recoverable_gap + irreducible_gap)
              = recoverable_gap / (auroc_oracle − auroc_raw)
```

The fraction of the total recoverable performance gap (oracle vs raw) that
the wrapper actually recovers.

| `recovery_ratio` | Interpretation | Action |
|---|---|---|
| > 0.7 | Drift is mainly distributional | Wrapper is sufficient |
| 0.3 – 0.7 | Mixed drift | Wrapper useful; evaluate retraining |
| < 0.3 | Drift is mainly structural | Retrain |

### DeLong confidence intervals

All three gap estimates are accompanied by 95 % confidence intervals computed
using the DeLong method for correlated AUROC pairs.  Overlapping CIs mean the
measured gap may not be statistically significant.

---

## Brier score decomposition

The Brier score decomposes into three orthogonal components (Murphy, 1973):

$$\text{BS} = \underbrace{\text{Reliability}}_{\text{calibration error}} - \underbrace{\text{Resolution}}_{\text{sharpness}} + \underbrace{\text{Uncertainty}}_{\text{irreducible noise}}$$

### Reliability (calibration error)

Average squared deviation between predicted probabilities and observed
frequencies across probability bins.  Lower is better.

- High reliability error → the model is systematically over- or under-
  confident.
- A large Δ Reliability (target > source) means the source model was
  miscalibrated for the target domain.  The calibration layer addresses this
  directly.

### Resolution (sharpness)

How much the predicted probabilities deviate from the prevalence.  Higher is
better — a model that always predicts the prevalence has zero resolution.

- Negative Δ Resolution (target < source) → the model has lost discriminative
  sharpness in the target domain.  This is due to structural drift or feature
  masking and **cannot** be recovered by calibration alone.

### Uncertainty

$p(1-p)$ where $p$ is the target prevalence.  Fixed for a given dataset;
irreducible.  Different between source and target when prevalence differs.

### How to read the decomposition

```
ΔReliability  large positive  → calibration was the main problem → wrapper helps
ΔResolution   negative        → discrimination was lost → consider retraining
ΔUncertainty  nonzero         → prevalence shifted → this is expected
```

---

## Feature recovery attribution

When `feature_attribution: true`, RECAL runs a Shapley attribution of the
`recoverable_gap` across individual features.  It answers: *which features,
when aligned, contributed most to the performance recovery?*

Shown as a bar chart of the top-N features (default 10) in the HTML report.
Features with high SHAP importance in the original model that also have high
recovery attribution are the "hero features" of the adaptation.

Features with negative recovery attribution made the adapted model *worse*
than the raw model after alignment — a signal that those features should be
masked or that their alignment method should be reconsidered.

---

*For operational decision guidance see
[ARCHITECTURE.md — Decision boundaries](ARCHITECTURE.md#decision-boundaries).
For overfitting control see [OVERFITTING.md](OVERFITTING.md).*
