"""Baselines B1–B8 (protocol v1.0, §6). Each is fitted on the target cohort only."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from recal.align.coral import CoralAligner
from recal.align.quantile_transform import QuantileTransformAligner

EPS = 1e-7


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _loglik(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _unpenalised_lr():
    return LogisticRegression(C=1e12, max_iter=2000)


# ── B1, B2, B3 ───────────────────────────────────────────────────────────────

def fit_citl(y, p):
    """Recalibration-in-the-large: intercept only, LP as offset (Newton)."""
    lp, a = logit(p), 0.0
    for _ in range(100):
        mu = sigmoid(a + lp)
        h = np.sum(mu * (1 - mu))
        if h <= 0:
            break
        step = np.sum(y - mu) / h
        a += step
        if abs(step) < 1e-10:
            break
    return lambda q: sigmoid(a + logit(q))


def fit_platt(y, p):
    lr = _unpenalised_lr().fit(logit(p).reshape(-1, 1), y)
    return lambda q: lr.predict_proba(logit(q).reshape(-1, 1))[:, 1]


def fit_isotonic(y, p):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
    return lambda q: iso.predict(q)


# ── B4: closed testing procedure (Vergouwe et al., 2017), black-box adaptation ─

def fit_closed_testing(y, p, X, alpha=0.05, min_epv=10):
    """Returns (predict_fn(p, X), info). M3 = M2 + main effects of the predictors; used as
    the most extensive model only when E / (added parameters) ≥ min_epv, else M2."""
    y = np.asarray(y, dtype=int)
    lp = logit(p)
    E = int(y.sum())
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def imp(Z):
        return np.where(np.isnan(Z), med, Z)

    m1 = fit_citl(y, p)
    lr2 = _unpenalised_lr().fit(lp.reshape(-1, 1), y)
    fits = {
        "M0": (lambda q, Z: np.clip(q, EPS, 1 - EPS), 0),
        "M1": (lambda q, Z: m1(q), 1),
        "M2": (lambda q, Z: lr2.predict_proba(logit(q).reshape(-1, 1))[:, 1], 2),
    }
    use_m3 = E / X.shape[1] >= min_epv
    if use_m3:
        lr3 = _unpenalised_lr().fit(np.column_stack([lp, imp(X)]), y)
        fits["M3"] = (lambda q, Z: lr3.predict_proba(np.column_stack([logit(q), imp(Z)]))[:, 1],
                      2 + X.shape[1])
    full = "M3" if use_m3 else "M2"
    ll = {k: _loglik(y, f(p, X)) for k, (f, _) in fits.items()}
    k_full = fits[full][1]
    chosen, tests = full, {}
    for k in ["M0", "M1", "M2", "M3"]:
        if k == full:
            break
        df = k_full - fits[k][1]
        stat = 2 * (ll[full] - ll[k])
        pval = float(chi2.sf(stat, df))
        tests[f"{k}_vs_{full}"] = {"lr": stat, "df": df, "p": pval}
        if pval >= alpha:
            chosen = k
            break
    info = {"most_extensive": full, "chosen": chosen, "tests": tests,
            "decision": "revise" if chosen == "M3" else "deploy_updated"}
    return fits[chosen][0], info


# ── B5, B6, B7: alignment-only and SKADA CORAL ───────────────────────────────

def _impute(X, med):
    return np.where(np.isnan(X), med, X)


def fit_coral_only(X_source, X_target):
    med = np.nanmedian(X_source, axis=0)
    al = CoralAligner(reg=1e-4, shrinkage="auto").fit(_impute(X_source, med), _impute(X_target, med))

    def transform(X):
        mask = np.isnan(X)
        return al.transform(_impute(X, med), nan_mask=mask)
    return transform


def fit_qt_only(X_source, X_target):
    med = np.nanmedian(X_source, axis=0)
    al = QuantileTransformAligner().fit(_impute(X_source, med), _impute(X_target, med))

    def transform(X):
        mask = np.isnan(X)
        return al.transform(_impute(X, med), nan_mask=mask)
    return transform


def fit_skada_coral(X_source, X_target):
    """SKADA CORALAdapter mapping TARGET onto SOURCE statistics: the target is given the
    'source' domain role so that SKADA re-colours it to the source covariance."""
    from skada import CORALAdapter
    med = np.nanmedian(X_source, axis=0)
    Xs, Xt = _impute(X_source, med), _impute(X_target, med)
    # SKADA convention: positive sample_domain = source, negative = target
    X = np.vstack([Xt, Xs])
    dom = np.concatenate([np.ones(len(Xt), int), -np.ones(len(Xs), int)])
    ad = CORALAdapter().fit(X, sample_domain=dom)

    def transform(Z):
        mask = np.isnan(Z)
        out = ad.transform(_impute(Z, med), sample_domain=np.ones(len(Z), int), allow_source=True)
        out = np.asarray(out, dtype=float)
        out[mask] = np.nan
        return out
    return transform
