"""
recal_cli.adapt_select
======================
RECAL 0.4: choose, by cross-validation on the target cohort, the adaptation that best fits
it, and estimate how the adapted model performs in that population.

Candidates = input transformation × recalibration

    transformation : identity (inputs untouched)
                     | designed (RECAL Designer from the drift profile: mask, QT, PCA-CORAL/CORAL)
                     | designed_drift (Designer also informed by the per-feature drift
                       decomposition, which can add WOE and drift-type-specific choices)
                     Each transformation is chosen and fitted on the training part only.
    recalibration  : none | intercept (calibration-in-the-large) | logistic (Platt) | isotonic

Selection: K-fold CV on the cohort; criterion = mean held-out log-loss (a proper scoring
rule, so discrimination and calibration both count). One-standard-error rule: among the
candidates whose mean log-loss is within one SE of the best, the simplest is kept
(order: identity before designed; none < intercept < logistic < isotonic).

Estimate: nested CV — the whole selection is repeated inside each outer fold and the
chosen candidate predicts the outer test fold. The pooled out-of-fold predictions give the
expected performance in the target population (AUROC as fold mean; Brier, log-loss,
calibration slope/intercept and ECE pooled), with bootstrap CIs resampled within folds.
"""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from recal.data.pairing import CohortPair
from recal_cli.cross_validate import _make_pair_with_target_subset
from recal_cli.verdict import calibration_slope_intercept, fold_mean_auroc

logger = logging.getLogger(__name__)
EPS = 1e-6
TRANSFORMS = ["identity", "designed", "designed_drift"]
CALIBRATORS = ["none", "intercept", "logistic", "isotonic"]
SIMPLICITY = [(t, c) for c in ["none", "intercept", "logistic"] for t in TRANSFORMS] + \
             [(t, "isotonic") for t in TRANSFORMS]


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ── recalibrators ─────────────────────────────────────────────────────────────

class Calibrator:
    """Picklable recalibration map fitted on (y, scores)."""

    def __init__(self, kind: str):
        self.kind = kind
        self.a = 0.0
        self.lr = None
        self.iso = None

    def fit(self, y, s):
        y = np.asarray(y, int)
        if self.kind == "none" or y.min() == y.max():
            self.kind_effective = "none"
            return self
        self.kind_effective = self.kind
        lp = _logit(s)
        if self.kind == "intercept":
            a = 0.0
            for _ in range(200):
                mu = 1 / (1 + np.exp(-(a + lp)))
                h = np.sum(mu * (1 - mu))
                if h <= 1e-12:
                    break
                step = float(np.clip(np.sum(y - mu) / h, -5, 5))
                a += step
                if abs(step) < 1e-10:
                    break
            self.a = a
        elif self.kind == "logistic":
            self.lr = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), y)
        elif self.kind == "isotonic":
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(s, y)
        else:
            raise ValueError(self.kind)
        return self

    def __call__(self, q):
        q = np.asarray(q, float)
        k = getattr(self, "kind_effective", "none")
        if k == "none":
            return np.clip(q, EPS, 1 - EPS)
        if k == "intercept":
            return 1 / (1 + np.exp(-(self.a + _logit(q))))
        if k == "logistic":
            return self.lr.predict_proba(_logit(q).reshape(-1, 1))[:, 1]
        return self.iso.predict(q)


def fit_calibrator(kind: str, y: np.ndarray, s: np.ndarray) -> Calibrator:
    return Calibrator(kind).fit(y, s)


# ── transformations ──────────────────────────────────────────────────────────

@dataclass
class FittedTransform:
    kind: str
    adapter: object | None = None   # AutoAdapter for "designed"
    model: object | None = None

    def scores(self, pair: CohortPair) -> np.ndarray:
        if self.kind == "identity":
            return np.asarray(self.model.predict_proba(pair.X_t), float)
        return np.asarray(self.adapter.predict(pair), float)

    def scores_array(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "identity":
            return np.asarray(self.model.predict_proba(X), float)
        return np.asarray(self.adapter.predict_array(X), float)


def fit_transform(kind: str, pair: CohortPair, model, schema, drift_fn=None,
                  pca_k: int = 5, max_n_sweep: int = 30) -> FittedTransform:
    if kind == "identity":
        return FittedTransform("identity", model=model)
    from recal_core.pipeline.auto_adapter import AutoAdapter
    if kind == "designed_drift" and drift_fn is None:
        raise ValueError("designed_drift needs drift_fn")
    drift = drift_fn(pair) if kind == "designed_drift" else {}
    aa = AutoAdapter(model=model, schema=schema, **drift)
    with contextlib.redirect_stdout(io.StringIO()):
        aa.profile(pair)
        cfg = aa.design(pair, pca_k=pca_k, max_n_sweep=max_n_sweep, consider_identity=False)
        cfg.apply_calibration = False
        aa.fit(pair)
    return FittedTransform(kind, adapter=aa, model=model)


# ── selection ────────────────────────────────────────────────────────────────

def _splits(y, n_splits, seed):
    k = int(min(n_splits, y.sum(), (1 - y).sum()))
    if k < 2:
        return None
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def select_adaptation(pair: CohortPair, model, schema, drift_fn=None, n_splits: int = 5,
                      seed: int = 42, pca_k: int = 5, max_n_sweep: int = 30) -> dict:
    """CV selection of (transformation, recalibration) by held-out log-loss, 1-SE rule."""
    y = np.asarray(pair.y_t, int)
    splits = _splits(y, n_splits, seed)
    if splits is None:
        return {"chosen": ("identity", "none"), "table": [], "reason": "too few events for CV"}
    transforms = [t for t in TRANSFORMS if t != "designed_drift" or drift_fn is not None]
    losses = {c: [] for c in SIMPLICITY if c[0] in transforms}
    for tr, te in splits:
        p_tr, p_te = _make_pair_with_target_subset(pair, tr), _make_pair_with_target_subset(pair, te)
        for t in transforms:
            try:
                ft = fit_transform(t, p_tr, model, schema, drift_fn, pca_k, max_n_sweep)
                s_tr, s_te = ft.scores(p_tr), ft.scores(p_te)
            except Exception as e:  # a failed candidate is simply not available
                logger.warning("transform %s failed in a fold: %s", t, e)
                for c in CALIBRATORS:
                    losses[(t, c)].append(np.nan)
                continue
            for c in CALIBRATORS:
                try:
                    p = fit_calibrator(c, y[tr], s_tr)(s_te)
                    losses[(t, c)].append(_logloss(y[te], p))
                except Exception:
                    losses[(t, c)].append(np.nan)
    table = []
    for cand in [c for c in SIMPLICITY if c in losses]:
        v = np.asarray(losses[cand], float)
        ok = np.isfinite(v).all()
        table.append({"transform": cand[0], "calibration": cand[1],
                      "cv_logloss": float(np.mean(v)) if ok else np.nan,
                      "se": float(np.std(v, ddof=1) / np.sqrt(len(v))) if ok and len(v) > 1 else np.nan})
    valid = [r for r in table if np.isfinite(r["cv_logloss"])]
    best = min(valid, key=lambda r: r["cv_logloss"])
    thr = best["cv_logloss"] + (best["se"] if np.isfinite(best["se"]) else 0.0)
    chosen = next(r for r in table if np.isfinite(r["cv_logloss"]) and r["cv_logloss"] <= thr)
    for r in table:
        r["selected"] = (r is chosen)
        r["best"] = (r is best)
    return {"chosen": (chosen["transform"], chosen["calibration"]), "table": table,
            "reason": f"lowest CV log-loss {best['transform']}+{best['calibration']} "
                      f"({best['cv_logloss']:.4f}); 1-SE rule → {chosen['transform']}+{chosen['calibration']}"}


@dataclass
class SelectedAdapter:
    transform: FittedTransform
    calibrator: object
    choice: tuple
    schema: list | None = None
    metadata: dict | None = None

    def save(self, path) -> None:
        from pathlib import Path

        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> SelectedAdapter:
        import joblib
        return joblib.load(path)

    def predict_pair(self, pair: CohortPair) -> np.ndarray:
        return self.calibrator(self.transform.scores(pair))

    def predict_proba(self, X) -> np.ndarray:
        """X: DataFrame (columns by name) or array in schema order. Nothing is refitted."""
        if hasattr(X, "loc") and self.schema is not None:
            X = X.reindex(columns=self.schema).to_numpy(dtype=float)
        return self.calibrator(self.transform.scores_array(np.asarray(X, float)))


def fit_selected(choice: tuple, pair: CohortPair, model, schema, drift_fn=None,
                 pca_k: int = 5, max_n_sweep: int = 30) -> SelectedAdapter:
    ft = fit_transform(choice[0], pair, model, schema, drift_fn, pca_k, max_n_sweep)
    cal = fit_calibrator(choice[1], pair.y_t, ft.scores(pair))
    return SelectedAdapter(ft, cal, choice, schema=list(schema))


# ── nested estimate of performance in the target population ─────────────────

def _metrics(y, p, fold_ids):
    slope, citl = calibration_slope_intercept(y, p)
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    ece = float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))
    return {"auroc": fold_mean_auroc(y, p, fold_ids), "brier": float(brier_score_loss(y, p)),
            "logloss": _logloss(y, p), "slope": slope, "citl": citl, "ece10": ece}


def nested_estimate(pair: CohortPair, model, schema, drift_fn=None, n_splits: int = 5,
                    seed: int = 42, n_boot: int = 500, pca_k: int = 5, max_n_sweep: int = 30) -> dict:
    """Out-of-fold predictions of the WHOLE procedure (selection + fit) and their metrics."""
    y = np.asarray(pair.y_t, int)
    outer = _splits(y, n_splits, seed + 1)
    if outer is None:
        return {"available": False, "reason": "too few events for nested CV"}
    oof = np.full(len(y), np.nan)
    fids = np.full(len(y), -1)
    choices = []
    for k, (tr, te) in enumerate(outer):
        p_tr, p_te = _make_pair_with_target_subset(pair, tr), _make_pair_with_target_subset(pair, te)
        sel = select_adaptation(p_tr, model, schema, drift_fn, n_splits, seed + 10 + k, pca_k, max_n_sweep)
        fitted = fit_selected(sel["chosen"], p_tr, model, schema, drift_fn, pca_k, max_n_sweep)
        oof[te] = fitted.predict_pair(p_te)
        fids[te] = k
        choices.append("+".join(sel["chosen"]))
    point = _metrics(y, oof, fids)
    rng = np.random.default_rng(seed)
    strata = [np.where((fids == k) & (y == v))[0] for k in np.unique(fids) for v in (0, 1)]
    strata = [s for s in strata if len(s)]
    boots = {m: [] for m in point}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        try:
            mb = _metrics(y[idx], oof[idx], fids[idx])
        except Exception:
            continue
        for m, v in mb.items():
            boots[m].append(v)
    ci = {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if v else [np.nan, np.nan]
          for m, v in boots.items()}
    raw = np.asarray(model.predict_proba(pair.X_t), float)
    return {"available": True, "estimate": point, "ci95": ci, "outer_choices": choices,
            "raw_on_cohort": _metrics(y, raw, fids), "n_outer": len(outer), "oof_scores": oof,
            "fold_ids": fids}


def adapt(pair: CohortPair, model, schema, drift_fn=None, n_splits: int = 5, seed: int = 42,
          estimate: bool = True, pca_k: int = 5, max_n_sweep: int = 30) -> dict:
    """Full RECAL 0.4 procedure: selection on the whole cohort, final fit, nested estimate."""
    sel = select_adaptation(pair, model, schema, drift_fn, n_splits, seed, pca_k, max_n_sweep)
    fitted = fit_selected(sel["chosen"], pair, model, schema, drift_fn, pca_k, max_n_sweep)
    est = nested_estimate(pair, model, schema, drift_fn, n_splits, seed, pca_k=pca_k,
                          max_n_sweep=max_n_sweep) if estimate else {"available": False}
    return {"selection": sel, "adapter": fitted, "estimate": est}
