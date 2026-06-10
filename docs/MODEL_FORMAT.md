# RECAL — Model format guide

RECAL can load a frozen model from any of five backends.  The model is
**never** retrained — it must already be trained and serialised before
running RECAL.

---

## Supported backends at a glance

| File extension(s) | Backend | `type:` in config | Notes |
|---|---|---|---|
| `.json` `.ubj` `.bin` | XGBoost | `xgboost` | Recommended; native NaN handling |
| `.joblib` `.pkl` | sklearn (or any picklable object) | `sklearn` | Needs `.predict_proba()` |
| `.h5` `.keras` | Keras / TensorFlow | `keras` | NaNs imputed to 0 before inference |
| `.pt` `.pth` | PyTorch | `torch` | Must be a full model, not a `state_dict` |
| `<any>` + `custom_loader:` | BYOM | `byom` | You provide a Python loader |

The `type:` key is optional — RECAL auto-detects the backend from the file
extension.  Specify it explicitly to override or when using BYOM.

---

## XGBoost

### Saving correctly

```python
import xgboost as xgb
booster = xgb.train(params, dtrain)
booster.save_model("inputs/models/my_model.json")   # preferred
# booster.save_model("inputs/models/my_model.ubj")  # binary alternative
```

For sklearn-API wrappers:

```python
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, y_train)
clf.save_model("inputs/models/my_model.json")
```

### Config

```yaml
model:
  path: inputs/models/my_model.json
  type: xgboost   # optional
```

### NaN handling

XGBoost has native NaN support — missing values are routed through learned
default directions in each split.  RECAL passes the feature matrix as-is
without imputation.

### Caveats

- `.pkl` XGBoost models from old versions (< 1.6) may not restore correctly.
  Use `.json` instead.
- Multi-output models are not supported; the wrapper expects a single positive-
  class probability output.

---

## sklearn (and any object with `.predict_proba`)

### Saving correctly

```python
import joblib
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
joblib.dump(clf, "inputs/models/my_model.joblib")
```

Any Python object that implements `.predict_proba(X) -> (n, 2)` works —
pipeline, calibrated classifier, custom wrapper.

### Config

```yaml
model:
  path: inputs/models/my_model.joblib
  type: sklearn   # optional; also detected from .pkl extension
```

### NaN handling

sklearn models generally do not handle NaNs.  Impute upstream or use a
pipeline that includes an imputer step before loading.

---

## Keras / TensorFlow

### Saving correctly

```python
model.save("inputs/models/my_model.keras")   # recommended (SavedModel v3)
# model.save("inputs/models/my_model.h5")    # legacy HDF5 also accepted
```

### Config

```yaml
model:
  path: inputs/models/my_model.keras
  type: keras   # optional; detected from .h5 / .keras extension
```

### NaN handling

Keras models error on NaN inputs.  REACAL automatically imputes NaN values
to **0** before calling the model.  This is a fallback — if your model is
sensitive to imputation, handle missingness upstream.

### Caveats

- The model must output a single value per sample (sigmoid activation) or a
  two-column array where column index 1 is the positive-class probability.
- Requires `tensorflow >= 2.12` at runtime.

---

## PyTorch

### Saving correctly

```python
import torch
torch.save(model, "inputs/models/my_model.pt")   # full model, not state_dict
```

Do **not** save only the state dict (`torch.save(model.state_dict(), ...)`)
because RECAL cannot reconstruct the architecture without the class definition.

### Config

```yaml
model:
  path: inputs/models/my_model.pt
  type: torch   # optional; detected from .pt / .pth extension
```

### NaN handling

NaN values are imputed to **0** before inference (same as Keras).

### Caveats

- The model must expose a `.forward(x)` method that returns logits or
  probabilities for binary classification.
- RECAL calls `torch.sigmoid` on the output if the range is outside [0, 1].
- Requires `torch >= 2.0` at runtime.

---

## BYOM — Bring Your Own Model

Use BYOM when your model does not fit any standard backend (e.g., ensemble
of heterogeneous models, R model loaded via `rpy2`, remote API).

### Loader skeleton

Create a Python file (e.g., `inputs/models/my_loader.py`) with exactly this
function signature:

```python
# inputs/models/my_loader.py

def load_model(path: str):
    """
    Load and return any model object from `path`.

    The returned object MUST expose:
        .predict_proba(X: np.ndarray | pd.DataFrame) -> np.ndarray
    where the output is either shape (n,) (positive-class probability)
    or shape (n, 2) ([:,1] is the positive-class probability).
    """
    import joblib
    model = joblib.load(path)
    # ... any custom logic ...
    return model
```

### Config

```yaml
model:
  path: inputs/models/my_model.pkl
  type: byom
  custom_loader: inputs/models/my_loader.py
```

### Minimal runnable example

```python
# inputs/models/ensemble_loader.py

def load_model(path: str):
    import joblib, numpy as np

    class EnsembleWrapper:
        def __init__(self, models):
            self.models = models
        def predict_proba(self, X):
            preds = np.stack([m.predict_proba(X)[:, 1] for m in self.models])
            return preds.mean(axis=0)

    return EnsembleWrapper(joblib.load(path))
```

### NaN handling

BYOM: RECAL does **not** apply any automatic NaN imputation.  Your `load_model`
return object must handle NaNs however the original model expects.

---

## Choosing the right backend

```
Model was trained with XGBoost?          → XGBoost (.json preferred)
Model is a sklearn Pipeline or class?    → sklearn (.joblib preferred)
Model is a Keras/TF network?             → Keras (.keras preferred)
Model is a PyTorch Module?               → PyTorch (.pt)
Anything else?                           → BYOM
```

When in doubt, wrap the model in a sklearn-compatible class with
`.predict_proba()` and serialise with `joblib.dump` — that always works.

---

## Preprocessing pipeline (`_pipeline.json`)

Some externally-provided models require a fixed preprocessing step before the
model can receive any features (e.g., cluster-wise PCA + low-VIF feature
selection).  RECAL supports this via a companion JSON file called a
*preprocessing pipeline*.

### Auto-detection

If a file named `*_pipeline.json` exists in the **same directory** as the
model, RECAL detects and loads it automatically.  No config change is needed.

```
inputs/models/
  my_model.json            ← model
  my_model_pipeline.json   ← auto-detected preprocessing pipeline
```

### Explicit config

```yaml
model:
  path: inputs/models/my_model.json
  pipeline: inputs/models/my_model_pipeline.json   # explicit path (overrides auto-detect)
  # pipeline: null    # set to null to disable auto-detection
```

### Pipeline JSON format

```json
{
  "all_features": ["raw_feat_1", "raw_feat_2", ...],
  "train_medians": {"raw_feat_1": 1.23, "raw_feat_2": 4.56, ...},
  "clusters": [
    {
      "cluster_id": 1,
      "features": ["raw_feat_1", "raw_feat_2"],
      "scaler_mean": [0.1, 0.2],
      "scaler_std": [1.0, 1.1],
      "pca_components": [[0.7, 0.7]],
      "pca_mean": [0.0]
    }
  ],
  "low_vif_vars": ["raw_feat_3", "raw_feat_4"],
  "feature_names_out": ["C01_PC1", "C02_PC1", "raw_feat_3", "raw_feat_4", ...]
}
```

| Key | Description |
|-----|-------------|
| `all_features` | All raw input columns the pipeline expects (used for imputation) |
| `train_medians` | Median per raw feature, used to impute NaN values before PCA |
| `clusters` | List of feature groups — each produces one PCA component named `C{id:02d}_PC1` |
| `low_vif_vars` | Raw features passed through without transformation (low VIF, no PCA needed) |
| `feature_names_out` | Final ordered feature names that the model expects |

### What `PipelinePreprocessor` does

1. Extracts `all_features` columns from the raw DataFrame; imputes NaN → `train_medians`.
2. For each cluster: z-scores with `scaler_mean`/`scaler_std`, then projects with `pca_components`.  Output column: `C{id:02d}_PC1`.
3. Extracts `low_vif_vars` columns unchanged.
4. Concatenates PCA outputs + low-VIF columns and reorders to `feature_names_out`.
5. Preserves any `label` column if present.

The transform runs **after** `unit_corrections` and **before** schema alignment
inside `data_loader.py`.

---

*Back to [ARCHITECTURE.md](ARCHITECTURE.md) · [USAGE.md](USAGE.md)*
