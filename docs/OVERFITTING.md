# RECAL — Overfitting control and the optimism gap

The mask-size selector (the *sweep*) searches over subsets of features to
maximise adaptation performance on the target labels.  Because it optimises
against those same labels, it can memorise noise — the adapted AUROC looks
better than what you would actually get on truly new data. RECAL makes this
risk **visible and quantified**.

---

## Operational definition

```
optimism_gap = in_sample_AUROC − OOF_AUROC
```

- **`in_sample_AUROC`** — the sweep runs on the full target set, the pipeline
  is fit on the full target set, and AUROC is measured on the same full target
  set.  This is what a naive evaluation would report.
- **`OOF_AUROC`** — *Out-of-fold* AUROC from honest k-fold cross-validation.
  For each fold the sweep and fit use only the *training* split; the *test*
  split is held out.  Predictions from all test splits are concatenated and
  AUROC is computed once across all of them.  This is the honest estimate.

The gap tells you how much of the in-sample gain is real vs optimistic.

---

## Thresholds and interpretation

| `optimism_gap` | Label | Interpretation | Action |
|---|---|---|---|
| < 0.02 | Robust | Gap is within noise; the pipeline generalises | Deploy with confidence |
| 0.02 – 0.05 | Moderate | Mild overfitting; selection found features by chance | Increase `n_target`; reduce `max_n_sweep` |
| > 0.05 | Suspicious | Strong overfitting signal | Do not deploy; collect more target data |

These thresholds assume AUROC as the performance metric.  For smaller
datasets, even a gap of 0.01–0.02 warrants caution.

Cross-reference: [ARCHITECTURE.md — Decision boundaries](ARCHITECTURE.md#decision-boundaries).

---

## Numerical example

Suppose an adaptation run on 200 target patients reports:

```
in_sample_AUROC  = 0.81
OOF_AUROC        = 0.76
optimism_gap     = 0.05
```

Interpretation: 5 percentage points of the adaptation gain may be
illusory.  The honest performance is 0.76, not 0.81.  This is on the edge
of the "suspicious" zone.  You should:

1. Check `max_n_sweep` — is it set higher than needed?
2. Consider reducing it to 10–15 features.
3. If possible, add more target patients (200 is marginal for a 30-feature sweep).

After reducing `max_n_sweep` to 15:

```
in_sample_AUROC  = 0.79
OOF_AUROC        = 0.78
optimism_gap     = 0.01   ← now robust
```

The in-sample number dropped slightly, but you can trust it.

---

## Mitigation strategies

### 1. Reduce `max_n_sweep`

The sweep evaluates masks of increasing size up to `max_n_sweep`.  Fewer
candidates = less multiple-testing = less optimism.

```yaml
recal_core:
  max_n_sweep: 15   # down from default 30
```

### 2. Collect more target labels

The gap scales roughly as $O(p / n)$ where $p$ is the number of candidates
evaluated and $n$ is the target cohort size.  A rule of thumb:

$$n_{\text{target}} \geq 20 \times \text{max\_n\_sweep}$$

For `max_n_sweep = 30` you want at least 600 labelled target patients.

### 3. Increase k-fold splits

With small `n_target`, 5 folds leave very small test sets.  Increasing to
10 gives more stable OOF estimates:

```yaml
overfitting_check:
  n_splits: 10
```

### 4. Disable the sweep entirely

If you have strong domain knowledge about which features to align, list them
explicitly in the config and set `max_n_sweep: 0`.  The designer will then
use those features without any data-driven sweep, and the optimism gap will
be ~0.

### 5. Use `--skip-expensive` during development, but always run the full check before deploying

The `--skip-expensive` flag skips oracle + attribution + Brier decomposition
but **keeps** the honest k-fold check.  Never disable the k-fold for a
deployment decision.

---

## How k-fold is implemented

Each fold proceeds as follows:

```
for train_idx, test_idx in KFold(n_splits).split(target):
    X_train, y_train = target[train_idx], labels[train_idx]
    X_test,  y_test  = target[test_idx],  labels[test_idx]

    # Sweep runs only on X_train, y_train
    mask = sweep(source, X_train, y_train, model, max_n_sweep)

    # Pipeline is fit only on X_train
    pipeline = fit(source, X_train, y_train, mask, config)

    # Predictions on held-out test
    preds[test_idx] = pipeline.predict_proba(X_test)

OOF_AUROC = roc_auc_score(labels, preds)
```

Source data is used in full in every fold (it is not split) because the
source cohort plays the role of a fixed reference distribution.

---

*See also: [ARCHITECTURE.md — Decision boundaries](ARCHITECTURE.md#decision-boundaries),
[DRIFT_REPORT.md](DRIFT_REPORT.md) for the recovery_ratio context.*
