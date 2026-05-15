# ADAPT — Domain Transfer Wrapper

Adapt a binary predictive model (XGBoost / sklearn / Keras / PyTorch / BYOM)
trained on a **source** dataset so it can be used on a different **target**
dataset, **without re-training** the model. The only thing required is a
validation cohort from the target domain with known outcomes.

> Typical use case: a collaborator hands you a predictive model trained at
> hospital A. You want to validate it at hospital B (which has covariate
> drift, a different prevalence and a different missingness pattern). This
> wrapper gives you the "calibrated-adapted" version of the model plus a
> discrepancy report.

---

## Quick start

```bash
# 1. Drop your files in
cp your_model.json inputs/models/
cp source.csv      inputs/source/
cp target.csv      inputs/target/

# 2. Copy the example config and edit the paths
cp configs/example_snuh_to_clinic.yaml configs/my_run.yaml

# 3. Run
python -m adapt_cli.run --config configs/my_run.yaml
```

A self-contained HTML report is produced under `outputs/reports/` containing:

- **Original-model reference** — performance of the unmodified model on its
  source domain (upper bound for the transfer).
- **Raw vs adapted metrics** on the target — AUROC, precision, recall, F1.
- **Calibration curve** before / after the pipeline.
- **Designer decisions** — which features were masked, PCA-CORAL k, etc.
- **Section 3.3 — honest validation (k-fold CV)** with the *optimism gap*.

---

## Repository layout

```
adapt_cli/            Public CLI package
  ├── model_loader.py     Loads XGB / sklearn / keras / torch / BYOM
  ├── data_loader.py      CSV/parquet → CohortPair
  ├── config_schema.py    YAML validation
  ├── cross_validate.py   Honest k-fold
  ├── drift_compute.py    Self-contained per-feature drift decomposition
  └── run.py              End-to-end orchestrator

adapt/                Internals (profiler, designer, pipeline, reporter)
domain_transfer/      Alignment algorithms (PCA-CORAL, QT, WOE, calibration)

configs/              Your YAML run configurations
inputs/               Your models, datasets and feature schema
outputs/              Reports + serialised wrappers + drift cache
docs/                 Extended documentation
tests/                Regression tests
```

---

## Supported model formats

| Extension              | Backend             | Notes |
|------------------------|---------------------|-------|
| `.json` `.ubj` `.bin`  | XGBoost             | Native NaN handling. Recommended. |
| `.joblib` `.pkl`       | sklearn / anything with `.predict_proba()` | |
| `.h5` `.keras`         | Keras / TensorFlow  | NaNs are imputed to 0. |
| `.pt` `.pth`           | PyTorch (`torch.save(model)`) | Full model required, not a state_dict. |
| **BYOM**               | Anything            | Your own `.py` with `def load_model(path):` |

See [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md) for details.

---

## Overfitting control

The mask-size selector (sweep) optimises against the target labels and can
therefore overfit. To make this visible the CLI runs **two evaluations in
parallel**:

1. **In-sample** — sweep + fit + predict on the whole target (what you see
   when you adapt).
2. **Honest k-fold** — each fold redoes the sweep on its train split only
   and predicts on the held-out test split.

The difference `in_sample − OOF` is your **optimism gap**:

- `< 0.02` → no overfitting; the adapted pipeline is robust.
- `0.02–0.05` → moderate optimism.
- `> 0.05` → suspicious. Increase `n_target` or lower `max_n_sweep`.

See [docs/OVERFITTING.md](docs/OVERFITTING.md).

---

## Drift cache

The first run computes a per-feature drift decomposition (LASSO + XGBoost,
CORAL / PCA-CORAL recovery, six-category taxonomy) for every feature in the
schema. This takes a few minutes for ~100 features and is cached to
`outputs/cache/drift_decomposition.csv`. Subsequent runs reuse the cache
instantly.

---

## Limitations

- Binary outcomes only (label ∈ {0, 1}).
- Numeric features only (encode categoricals upstream).
- The model is **never** re-trained: the wrapper aligns covariates and
  recalibrates probabilities. If the model is structurally wrong for the
  target, ADAPT will not fix it.
- Requires labels in the target cohort (this is validation, not blind
  inference).

---

## Citation

If you use this wrapper in academic work, please cite the original paper of
the model you transferred and this repository.
