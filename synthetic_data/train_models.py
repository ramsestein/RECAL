"""Train XGBoost and Neural Network models on synthetic_features.csv.

Saves:
- synthetic_data/models/xgb_model.json   (XGBoost regressor)
- synthetic_data/models/nn_model.joblib  (sklearn MLPRegressor)

Prints train/test R² for sanity check.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# ── 1. Load data ──────────────────────────────────────────────────────────────
data_dir = Path(__file__).parent
df = pd.read_csv(data_dir / "synthetic_features.csv")

feature_cols = [c for c in df.columns if c != "target"]
X = df[feature_cols].values.astype(np.float32)
y = df["target"].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 2. Train XGBoost ───────────────────────────────────────────────────────
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)

xgb_train_r2 = r2_score(y_train, xgb_model.predict(X_train))
xgb_test_r2  = r2_score(y_test,  xgb_model.predict(X_test))
print(f"XGBoost  — train R²: {xgb_train_r2:.4f}, test R²: {xgb_test_r2:.4f}")

models_dir = data_dir / "models"
models_dir.mkdir(exist_ok=True)

xgb_path = models_dir / "xgb_model.json"
xgb_model.save_model(str(xgb_path))
print(f"Saved XGBoost model: {xgb_path}")

# ── 3. Train Neural Network (sklearn MLPRegressor) ───────────────────────────
# Normalise features for the NN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

nn_model = MLPRegressor(
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

nn_train_r2 = r2_score(y_train, nn_model.predict(X_train_s))
nn_test_r2  = r2_score(y_test,  nn_model.predict(X_test_s))
print(f"NN       — train R²: {nn_train_r2:.4f}, test R²: {nn_test_r2:.4f}")

nn_path = models_dir / "nn_model.joblib"
joblib.dump({"model": nn_model, "scaler": scaler}, nn_path)
print(f"Saved NN model + scaler: {nn_path}")

# ── 4. Save feature list (schema) for RECAL ──────────────────────────────────
schema_path = models_dir / "feature_schema.json"
import json
schema_path.write_text(json.dumps(feature_cols))
print(f"Saved schema: {schema_path}")
