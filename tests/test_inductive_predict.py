"""Prediction must not depend on the batch being predicted (v0.3.0 inductive fix)."""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "synthetic_data"


@pytest.mark.skipif(not (SYN / "models" / "xgb_model.json").exists(), reason="synthetic model missing")
def test_prediction_is_batch_independent():
    from recal.data.pairing import CohortPair
    from recal_cli.data_loader import GenericCohortLoader
    from recal_cli.model_loader import load_model
    from recal_core.pipeline.auto_adapter import AutoAdapter

    schema = json.loads((SYN / "models" / "feature_schema.json").read_text())
    model = load_model(SYN / "models" / "xgb_model.json", schema=schema)
    src = GenericCohortLoader(SYN / "synthetic_features.csv", schema, "target")
    tgt = GenericCohortLoader(SYN / "synthetic_transfer.csv", schema, "target")
    pair = CohortPair(src, tgt)

    # fit on the first 100 target patients, predict the remaining 50
    from recal_cli.cross_validate import _make_pair_with_target_subset
    fit_pair = _make_pair_with_target_subset(pair, np.arange(100))
    X_new = pair.X_t[100:]

    aa = AutoAdapter(model=model, schema=schema)
    aa.profile(fit_pair)
    aa.design(fit_pair, pca_k=5, max_n_sweep=5)
    aa.fit(fit_pair)

    batch = aa.predict_array(X_new)
    single = np.array([aa.predict_array(X_new[i:i + 1])[0] for i in range(5)])
    np.testing.assert_allclose(batch[:5], single, rtol=1e-6, atol=1e-9)

    # predicting a different batch must not change earlier predictions
    _ = aa.predict_array(X_new[::-1])
    np.testing.assert_allclose(aa.predict_array(X_new)[:5], batch[:5], rtol=1e-6, atol=1e-9)
