"""recal.model.xgboost_wrapper
===============================
XGBoost model wrapper with schema-aware loading.

If ``model_path`` is provided, loads the booster from JSON/UBJ/BIN.
If omitted, trains a tiny dummy booster so that ``predict_proba`` is always
available (useful for CI / smoke tests).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class XGBoostWrapper:
    """Schema-aware wrapper around an XGBoost booster."""

    def __init__(
        self,
        schema: list[str],
        model_path: str | Path | None = None,
    ) -> None:
        self.schema = list(schema)
        self.n_features_in_ = len(schema)
        self._booster = None

        if model_path is not None:
            self._load_from_path(Path(model_path))
        else:
            self._build_dummy_booster()

    # ── Loading ─────────────────────────────────────────────────────────────

    def _load_from_path(self, path: Path) -> None:
        import xgboost as xgb

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))
        logger.info("XGBoostWrapper loaded model from %s", path)

    def _build_dummy_booster(self) -> None:
        """Train a tiny booster on random data so predict_proba works."""
        import xgboost as xgb

        rng = np.random.RandomState(42)
        n = max(100, self.n_features_in_ * 2)
        X = rng.randn(n, self.n_features_in_).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic",
            "max_depth": 2,
            "eta": 0.3,
            "eval_metric": "logloss",
            "seed": 42,
        }
        self._booster = xgb.train(params, dtrain, num_boost_round=5)
        logger.debug("XGBoostWrapper built dummy booster (n_features=%d)", self.n_features_in_)

    # ── Public API ──────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of the positive class."""
        import xgboost as xgb

        X_clean = np.where(np.isinf(X), np.nan, X)
        dmat = xgb.DMatrix(X_clean)
        return self._booster.predict(dmat)

    def feature_importance(self) -> dict:
        """Return feature importance by gain."""
        try:
            score = self._booster.get_score(importance_type="gain")
        except Exception:
            score = {}
        if not score:
            return {f"f{i}": 1.0 for i in range(self.n_features_in_)}
        return score

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP contributions (without bias column)."""
        import xgboost as xgb

        X_clean = np.where(np.isinf(X), np.nan, X)
        dmat = xgb.DMatrix(X_clean)
        contribs = self._booster.predict(dmat, pred_contribs=True)
        return contribs[:, :-1]
