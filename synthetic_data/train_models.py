"""Train XGBoost and Neural Network classifiers on synthetic_features.csv.

Saves:
- synthetic_data/models/xgb_model.json   (XGBoost classifier)
- synthetic_data/models/nn_model.joblib  (sklearn MLPClassifier + scaler)

Prints train/test AUROC / accuracy / F1 for sanity check.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# ── 1. Load data ──────────────────────────────────────────────────────────────
data_dir = Path(__file__).parent
df = pd.read_csv(data_dir / "synthetic_features.csv")

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values.astype(np.float32)
y = df["target"].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Train XGBoost ───────────────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
)
xgb_model.fit(X_train, y_train)

xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
xgb_test_proba  = xgb_model.predict_proba(X_test)[:, 1]
xgb_train_pred = (xgb_train_proba >= 0.5).astype(int)
xgb_test_pred  = (xgb_test_proba >= 0.5).astype(int)

print(
    f"XGBoost  — train AUROC: {roc_auc_score(y_train, xgb_train_proba):.4f}, "
    f"acc: {accuracy_score(y_train, xgb_train_pred):.4f}, "
    f"F1: {f1_score(y_train, xgb_train_pred):.4f}"
)
print(
    f"XGBoost  — test  AUROC: {roc_auc_score(y_test, xgb_test_proba):.4f}, "
    f"acc: {accuracy_score(y_test, xgb_test_pred):.4f}, "
    f"F1: {f1_score(y_test, xgb_test_pred):.4f}"
)

models_dir = data_dir / "models"
models_dir.mkdir(exist_ok=True)

xgb_path = models_dir / "xgb_model.json"
xgb_model.save_model(str(xgb_path))
print(f"Saved XGBoost model: {xgb_path}")

# ── 3. Train Neural Network (sklearn MLPClassifier) ───────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

nn_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)
nn_model.fit(X_train_s, y_train)

nn_train_proba = nn_model.predict_proba(X_train_s)[:, 1]
nn_test_proba  = nn_model.predict_proba(X_test_s)[:, 1]
nn_train_pred = (nn_train_proba >= 0.5).astype(int)
nn_test_pred  = (nn_test_proba >= 0.5).astype(int)

print(
    f"NN       — train AUROC: {roc_auc_score(y_train, nn_train_proba):.4f}, "
    f"acc: {accuracy_score(y_train, nn_train_pred):.4f}, "
    f"F1: {f1_score(y_train, nn_train_pred):.4f}"
)
print(
    f"NN       — test  AUROC: {roc_auc_score(y_test, nn_test_proba):.4f}, "
    f"acc: {accuracy_score(y_test, nn_test_pred):.4f}, "
    f"F1: {f1_score(y_test, nn_test_pred):.4f}"
)

nn_path = models_dir / "nn_model.joblib"
joblib.dump({"model": nn_model, "scaler": scaler}, nn_path)
print(f"Saved NN model + scaler: {nn_path}")

# ── 4. Save feature list (schema) for RECAL ──────────────────────────────────
schema_path = models_dir / "feature_schema.json"
schema_path.write_text(json.dumps(feature_cols))
print(f"Saved schema: {schema_path}")
