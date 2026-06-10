"""recal.model.xgboost_wrapper
===============================
XGBoost model wrapper with schema-aware loading.

Loads a pre-trained booster from JSON/UBJ/BIN and exposes ``predict_proba``
with the expected schema.
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
        model_path: str | Path,
    ) -> None:
        self.schema = list(schema)
        self.n_features_in_ = len(schema)
        self._booster = None
        self._load_from_path(Path(model_path))

    # ── Loading ─────────────────────────────────────────────────────────────

    def _load_from_path(self, path: Path) -> None:
        import xgboost as xgb

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))
        logger.info("XGBoostWrapper loaded model from %s", path)

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
