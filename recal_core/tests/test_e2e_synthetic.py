"""
recal_core/tests/test_e2e_synthetic.py
======================================
End-to-end smoke test using fully synthetic data.

Generates a source cohort, a drifted target cohort, trains a tiny
XGBoost on source, saves it to disk, loads it via XGBoostWrapper,
and runs the full RECAL pipeline.  This test runs in CI (no real data
required) and verifies that the pipeline completes without errors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_synthetic_cohorts(
    n_features: int = 20,
    n_source: int = 500,
    n_target: int = 100,
    seed: int = 42,
):
    """Generate synthetic source/target with real drift.

    Returns
    -------
    X_s, y_s, X_t, y_t, schema, model_path
    """
    rng = np.random.RandomState(seed)
    schema = [f"feat_{i:02d}" for i in range(n_features)]

    # Source: logistic model with 3 strong predictors
    X_s = rng.randn(n_source, n_features).astype(np.float32)
    logit_s = (
        0.5 * X_s[:, 0]
        - 0.3 * X_s[:, 1]
        + 0.4 * X_s[:, 2]
        + 0.1 * rng.randn(n_source)
    )
    prob_s = 1.0 / (1.0 + np.exp(-logit_s))
    y_s = (rng.rand(n_source) < prob_s).astype(int)

    # Target: covariate shift (mean shift on feat_0, feat_1)
    # + concept shift (weaker signal)
    X_t = rng.randn(n_target, n_features).astype(np.float32)
    X_t[:, 0] += 1.2  # mean shift
    X_t[:, 1] -= 0.8  # mean shift
    logit_t = (
        0.3 * X_t[:, 0]
        - 0.2 * X_t[:, 1]
        + 0.3 * X_t[:, 2]
        + 0.1 * rng.randn(n_target)
    )
    prob_t = 1.0 / (1.0 + np.exp(-logit_t))
    y_t = (rng.rand(n_target) < prob_t).astype(int)

    return X_s, y_s, X_t, y_t, schema


# ── Tests ───────────────────────────────────────────────────────────────────


class TestE2ESynthetic:

    def test_train_xgboost_and_wrap(self, tmp_path: Path):
        """Train XGBoost on synthetic source, save/load via XGBoostWrapper."""
        import xgboost as xgb

        from recal.model.xgboost_wrapper import XGBoostWrapper

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohorts()
        dtrain = xgb.DMatrix(X_s, label=y_s)
        params = {
            "objective": "binary:logistic",
            "max_depth": 3,
            "eta": 0.1,
            "eval_metric": "logloss",
            "seed": 42,
        }
        booster = xgb.train(params, dtrain, num_boost_round=10)

        model_path = tmp_path / "synthetic_model.json"
        booster.save_model(str(model_path))

        wrapper = XGBoostWrapper(schema=schema, model_path=model_path)
        proba = wrapper.predict_proba(X_t)

        assert proba.shape == (len(X_t),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_full_recal_pipeline(self, tmp_path: Path):
        """Run the full RECAL pipeline on synthetic data."""
        import xgboost as xgb

        from recal.model.xgboost_wrapper import XGBoostWrapper
        from recal_core.pipeline.auto_adapter import AutoAdapter

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohorts()

        # Train + save XGBoost
        dtrain = xgb.DMatrix(X_s, label=y_s)
        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "seed": 42},
            dtrain,
            num_boost_round=10,
        )
        model_path = tmp_path / "synthetic_model.json"
        booster.save_model(str(model_path))
        model = XGBoostWrapper(schema=schema, model_path=model_path)

        # Build CohortPair-like from arrays
        class FakePair:
            def __init__(self):
                self.X_s = X_s
                self.y_s = y_s
                self.X_t = X_t
                self.y_t = y_t
                self.X_s_imp = np.nan_to_num(X_s, nan=0.0)
                self.X_t_imp = np.nan_to_num(X_t, nan=0.0)
                self.mu_s = np.nanmean(X_s, axis=0)
                self.nan_mask_t = np.isnan(X_t)
                self.idx_corr = list(range(X_t.shape[1]))
                self.schema = schema
                self.schema_list = schema

            def mask_features(self, features):
                return self

        pair = FakePair()

        aa = AutoAdapter(model=model, schema=schema)
        profile = aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        config = aa.design()
        aa.fit(pair)
        scores = aa.predict(pair)

        assert profile.n_source_obs == len(y_s)
        assert profile.n_target_obs == len(y_t)
        assert scores.shape == (len(y_t),)
        assert config.apply_pca_coral is True

    def test_html_report_generates(self, tmp_path: Path):
        """HTML report must generate without errors on synthetic data."""
        import xgboost as xgb

        from recal.model.xgboost_wrapper import XGBoostWrapper
        from recal_core.pipeline.auto_adapter import AutoAdapter
        from recal_core.reporter.html_report import generate_html_report

        X_s, y_s, X_t, y_t, schema = _make_synthetic_cohorts()
        dtrain = xgb.DMatrix(X_s, label=y_s)
        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "eta": 0.1, "seed": 42},
            dtrain,
            num_boost_round=10,
        )
        model_path = tmp_path / "synthetic_model.json"
        booster.save_model(str(model_path))
        model = XGBoostWrapper(schema=schema, model_path=model_path)

        class FakePair:
            def __init__(self):
                self.X_s = X_s
                self.y_s = y_s
                self.X_t = X_t
                self.y_t = y_t
                self.X_s_imp = np.nan_to_num(X_s, nan=0.0)
                self.X_t_imp = np.nan_to_num(X_t, nan=0.0)
                self.mu_s = np.nanmean(X_s, axis=0)
                self.nan_mask_t = np.isnan(X_t)
                self.idx_corr = list(range(X_t.shape[1]))
                self.schema = schema
                self.schema_list = schema

            def mask_features(self, features):
                return self

        pair = FakePair()
        aa = AutoAdapter(model=model, schema=schema)
        profile = aa.profile_from_arrays(X_s, y_s, X_t, y_t)
        config = aa.design()
        aa.fit(pair)

        scores_before = model.predict_proba(X_t)
        scores_after = aa.predict(pair)

        html = generate_html_report(
            profile=profile,
            config=config,
            y_true=y_t,
            scores_before=scores_before,
            scores_after=scores_after,
            source_name="SyntheticSource",
            target_name="SyntheticTarget",
            output_path=str(tmp_path / "report.html"),
        )
        assert len(html) > 3000
        assert "RECAL" in html
