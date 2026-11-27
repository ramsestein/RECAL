"""
RECAL 0.4 — analysis of the eICU-CRD evaluation (protocol v2.0, §5–§6; Amendment B1).

Writes aggregate tables (no patient-level data) to <results>/analysis/:
  primary.csv      Δ log-loss vs frozen model (C1: superiority, harm rate) and vs Platt (C2:
                   non-inferiority, margin +0.002), by source × E, two-stage bootstrap
  comparators.csv  mean Δ (RECAL − comparator) for log-loss, Brier, AUROC, slope, CITL, ECE, ICI
  regret.csv       regret vs the best candidate in hindsight
  estimate.csv     accuracy and 95 % CI coverage of RECAL's nested estimate (C3)
  choices.csv      chosen candidate frequencies
  summary.md

Usage:  python analyse_v2.py --results RESULTS_V2
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261125
B = 2000
NI_MARGIN = 0.002
HARM = 0.005
COMPARATORS = ["F_frozen", "P_platt", "I_citl", "T_isotonic", "V_closed_testing"]
METRICS = ["logloss", "brier", "auroc", "slope", "citl", "ece10", "ici"]


def load(results: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(results / "runs/*.json")):
        r = json.loads(Path(f).read_text())
        base = {k: r.get(k) for k in ("task", "hospital", "source", "E", "replicate", "region")}
        if not r.get("feasible"):
            rows.append({**base, "feasible": False})
            continue
        row = {**base, "feasible": True, "ref_auroc": r["reference_holdout_auroc"]}
        rec = r["info"].get("recal", {})
        row["recal_error"] = rec.get("error")
        sel = rec.get("selection") or {}
        row["chosen"] = "+".join(sel.get("chosen", [])) if sel else None
        est = (rec.get("estimate") or {})
        for m in ("auroc", "brier", "slope", "citl"):
            row[f"est_{m}"] = (est.get("estimate") or {}).get(m)
            ci = (est.get("ci95") or {}).get(m)
            row[f"est_{m}_lo"], row[f"est_{m}_hi"] = (ci or [np.nan, np.nan])
        for name, m in r["metrics"].items():
            if "error" in m:
                continue
            for k in METRICS:
                row[f"{name}__{k}"] = m.get(k)
        cand = [c for c in r["metrics"] if c.startswith("K_") and "error" not in r["metrics"][c]]
        row["best_candidate_logloss"] = min(r["metrics"][c]["logloss"] for c in cand) if cand else np.nan
        row["best_candidate"] = min(cand, key=lambda c: r["metrics"][c]["logloss"]) if cand else None
        rows.append(row)
    return pd.DataFrame(rows)


def two_stage(df: pd.DataFrame, col: str, rng, fn=np.mean) -> tuple[float, float, float]:
    """Point estimate (mean of hospital means) and two-stage bootstrap CI: hospitals, then runs."""
    g = {h: d[col].dropna().to_numpy() for h, d in df.groupby("hospital")}
    g = {h: v for h, v in g.items() if len(v)}
    if not g:
        return (np.nan, np.nan, np.nan)
    hs = list(g)
    point = float(np.mean([fn(g[h]) for h in hs]))
    boots = []
    for _ in range(B):
        sample = rng.choice(hs, len(hs))
        boots.append(np.mean([fn(rng.choice(g[h], len(g[h]))) for h in sample]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _md(d):
    try:
        return d.to_markdown(index=False, floatfmt=".4f")
    except ImportError:
        return d.to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args(argv)
    out = args.results / "analysis"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    all_ = load(args.results)
    df = all_[all_.feasible].copy()
    df["d_F"] = df["recal__logloss"] - df["F_frozen__logloss"]
    df["d_P"] = df["recal__logloss"] - df["P_platt__logloss"]
    df["harm"] = (df["d_F"] > HARM).astype(float)
    df["regret"] = df["recal__logloss"] - df["best_candidate_logloss"]

    prim, comp, reg, est = [], [], [], []
    for (s, E), g in df.groupby(["source", "E"]):
        dF, dP, hr = two_stage(g, "d_F", rng), two_stage(g, "d_P", rng), two_stage(g, "harm", rng)
        prim.append({"source": s, "E": E, "runs": len(g), "hospitals": g.hospital.nunique(),
                     "dF_mean": dF[0], "dF_lo": dF[1], "dF_hi": dF[2], "C1_superior": dF[2] < 0,
                     "harm_rate": hr[0], "harm_lo": hr[1], "harm_hi": hr[2],
                     "dP_mean": dP[0], "dP_lo": dP[1], "dP_hi": dP[2], "C2_noninferior": dP[2] < NI_MARGIN})
        for c in COMPARATORS:
            for m in METRICS:
                col = f"{c}__{m}"
                if col not in g:
                    continue
                g2 = g.assign(_d=g[f"recal__{m}"] - g[col])
                p = two_stage(g2, "_d", rng)
                comp.append({"source": s, "E": E, "comparator": c, "metric": m,
                             "mean_diff_recal_minus": p[0], "lo": p[1], "hi": p[2]})
        reg.append({"source": s, "E": E, "median_regret": g.regret.median(), "p90_regret": g.regret.quantile(0.9),
                    "share_chose_best": float((g.chosen.apply(lambda x: f"K_{x}") == g.best_candidate).mean())})
        for m in ("auroc", "brier", "slope", "citl"):
            e = g[f"est_{m}"] - g[f"recal__{m}"]
            cov = ((g[f"est_{m}_lo"] <= g[f"recal__{m}"]) & (g[f"recal__{m}"] <= g[f"est_{m}_hi"])).mean()
            est.append({"source": s, "E": E, "metric": m, "median_error_est_minus_holdout": e.median(),
                        "median_abs_error": e.abs().median(), "ci95_coverage": cov})
    prim, comp, reg, est = map(pd.DataFrame, (prim, comp, reg, est))
    ch = df.groupby(["source", "E"]).chosen.value_counts(normalize=True).rename("share").reset_index()
    for name, d in [("primary", prim), ("comparators", comp), ("regret", reg), ("estimate", est), ("choices", ch)]:
        d.to_csv(out / f"{name}.csv", index=False)

    means = df.groupby(["source", "E"])[[f"{n}__{m}" for n in ["recal"] + COMPARATORS for m in ("logloss", "auroc", "slope", "ece10")]].mean()
    L = ["# RECAL 0.4 on eICU-CRD — analysis summary (protocol v2.0)", "",
         f"Runs: {len(all_)} ({len(df)} feasible, {int((~all_.feasible).sum())} not feasible). "
         f"RECAL errors: {int(df.recal_error.notna().sum())}. Hospitals: {df.hospital.nunique()}.", "",
         "## Primary endpoints (two-stage bootstrap over hospitals and runs)", "", _md(prim), "",
         "## Regret vs best candidate in hindsight", "", _md(reg), "",
         "## Accuracy of RECAL's nested estimate (C3)", "", _md(est), "",
         "## Comparators: mean difference RECAL − comparator", "",
         _md(comp[comp.metric.isin(["logloss", "brier", "auroc", "slope", "ece10"])]), "",
         "## Mean hold-out metrics", "", _md(means.reset_index()), "",
         "## Choices", "", _md(ch)]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
