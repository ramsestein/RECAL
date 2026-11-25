"""
RECAL validation — analysis (protocol v1.0, §8; Amendment A1).

Reads <results>/runs/*.json and <results>/setup/*/meta.json and writes aggregate tables
(no patient-level data) to <results>/analysis/:

  verdict_by_cell.csv        per transfer × model × E: verdict counts and error rates (V2, V1)
  primary_false_deploy.csv   §8.1 false-deploy rate (V2, XGBoost), Wilson CI, pooled CI
  acceptance.csv             §8.3 acceptance criterion per E and E_min
  gt_sensitivity.csv         §5.6 ground-truth sensitivity analyses
  b4_decisions.csv           B4 (closed testing) decisions scored like the verdict
  baselines_paired.csv       §8.2 paired differences vs RECAL on the Target-Holdout
  method_summary.csv         mean hold-out metrics and deployability per method
  oracle_bias.csv            RECAL internal oracle vs Reference model
  design_choices.csv         Designer choices (identity, alignment, calibration)
  transfers.csv              cohort sizes and source-model performance per transfer
  summary.md                 compact report of the above

Usage:  python analyse.py --results RESULTS
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261124
REAL = ["mimic__nwicu", "mimic__inspire", "nwicu__mimic", "nwicu__inspire", "inspire__mimic", "inspire__nwicu"]
METHODS = ["recal", "B0_raw", "B1_citl", "B2_platt", "B3_isotonic", "B4_closed_testing", "B5_coral",
           "B6_qt", "B7_skada_coral_platt", "B8_retrain"]
METRICS = ["auroc", "auprc", "brier", "citl", "slope", "ece10", "ici", "sensitivity", "specificity", "ppv", "npv"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def deployable(m, ref, delta=0.02, slope=(0.7, 1.3), citl=0.2):
    return bool(m["auroc"] >= ref - delta and slope[0] <= m["slope"] <= slope[1] and abs(m["citl"]) <= citl)


def load(results: Path) -> tuple[pd.DataFrame, dict]:
    metas = {Path(f).parent.name: json.loads(Path(f).read_text()) for f in glob.glob(str(results / "setup/*/meta.json"))}
    rows = []
    for f in glob.glob(str(results / "runs/*.json")):
        r = json.loads(Path(f).read_text())
        if not r.get("feasible"):
            rows.append({"task": r["task"], "transfer": r.get("transfer"), "model": r.get("model"),
                         "E": r.get("E"), "replicate": r.get("replicate"), "feasible": False})
            continue
        rec = r["info"].get("recal", {})
        v = rec.get("verdict") or {}
        row = {"task": r["task"], "transfer": r["transfer"], "model": r["model"], "E": r["E"],
               "replicate": r["replicate"], "feasible": True, "real": r["real_transfer"],
               "low_precision": r["holdout_low_precision"], "n_cohort": r["n_cohort"],
               "ref_auroc": r["reference_auroc_holdout"], "recal_error": rec.get("error"),
               "v2": (v.get("v2") or {}).get("verdict"), "v1": (v.get("v1") or {}).get("verdict"),
               "v2_reason": (v.get("v2") or {}).get("reason"),
               "rr": v.get("recovery_ratio"), "rr_lo": (v.get("recovery_ratio_ci") or [None, None])[0],
               "rr_hi": (v.get("recovery_ratio_ci") or [None, None])[1],
               "optimism": v.get("optimism_gap"), "total_gap": v.get("total_gap"),
               "slope_oof": v.get("slope_oof"), "auroc_oracle_cohort": rec.get("auroc_oracle"),
               "b4_decision": (r["info"].get("B4_closed_testing") or {}).get("decision"),
               "b4_chosen": (r["info"].get("B4_closed_testing") or {}).get("chosen")}
        for k, val in (rec.get("design") or {}).items():
            row[f"design_{k}"] = val
        for meth in METHODS:
            m = r["metrics"].get(meth)
            if not m or "error" in m:
                continue
            for k in METRICS:
                row[f"{meth}__{k}"] = m.get(k)
            for t in ("0.10", "0.20"):
                row[f"{meth}__nb{t}"] = m["net_benefit"].get(t)
            row[f"{meth}__dep"] = m["deployable_primary"]
            row[f"{meth}__m"] = m
        rows.append(row)
    return pd.DataFrame(rows), metas


def classify(df: pd.DataFrame, col: str, gt: str) -> pd.DataFrame:
    g = df[gt].astype(bool)
    v = df[col]
    return pd.DataFrame({
        "green": v == "GREEN", "gtd": g,
        "false_deploy": (v == "GREEN") & ~g, "correct_deploy": (v == "GREEN") & g,
        "missed_deploy": (v != "GREEN") & g, "correct_nodeploy": (v != "GREEN") & ~g,
        "unnecessary_retrain": (v == "RED") & g,
        "indeterminate": v == "INDETERMINATE", "amber": v == "AMBER", "red": v == "RED",
    })


def cell_table(df, verdict_col, gt_col):
    out = []
    for (tr, mo, E), g in df.groupby(["transfer", "model", "E"]):
        c = classify(g, verdict_col, gt_col)
        n0, n1 = int((~c.gtd).sum()), int(c.gtd.sum())
        fd = wilson(int(c.false_deploy.sum()), n0)
        sens = wilson(int(c.correct_deploy.sum()), n1)
        out.append({"transfer": tr, "model": mo, "E": E, "runs": len(g), "gt_deployable": n1, "gt_not": n0,
                    "green": int(c.green.sum()), "amber": int(c.amber.sum()), "red": int(c.red.sum()),
                    "indeterminate": int(c.indeterminate.sum()),
                    "false_deploy": int(c.false_deploy.sum()), "false_deploy_rate": fd[0],
                    "fd_lo": fd[1], "fd_hi": fd[2], "fd_unconditional": c.false_deploy.mean(),
                    "sensitivity": sens[0], "accuracy": float(((c.green & c.gtd) | (~c.green & ~c.gtd)).mean()),
                    "unnecessary_retrain": int(c.unnecessary_retrain.sum()),
                    "real": tr in REAL, "low_precision": bool(g.low_precision.iloc[0])})
    return pd.DataFrame(out)


def pooled_fd(df, verdict_col, gt_col, rng, B=2000):
    """False-deploy rate pooled over the six real transfers; cluster bootstrap over transfers."""
    out = []
    for (mo, E), g in df[df.real].groupby(["model", "E"]):
        trs = sorted(g.transfer.unique())
        per = {t: classify(g[g.transfer == t], verdict_col, gt_col) for t in trs}

        def rate(sample):
            fd = sum(int(per[t].false_deploy.sum()) for t in sample)
            n0 = sum(int((~per[t].gtd).sum()) for t in sample)
            return fd / n0 if n0 else np.nan
        point = rate(trs)
        boots = [rate(list(rng.choice(trs, len(trs)))) for _ in range(B)]
        boots = [b for b in boots if np.isfinite(b)]
        lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
        out.append({"model": mo, "E": E, "transfers": len(trs), "pooled_false_deploy": point, "lo": lo, "hi": hi})
    return pd.DataFrame(out)


def _md(d: pd.DataFrame) -> str:
    try:
        return d.to_markdown(index=False, floatfmt=".3f")
    except ImportError:
        return d.to_string(index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args(argv)
    out = args.results / "analysis"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    df_all, metas = load(args.results)
    df = df_all[df_all.feasible].copy()
    df["gt"] = df["recal__dep"].astype(bool)

    # transfers
    tt = []
    for name, m in sorted(metas.items()):
        for kind, mm in m["models"].items():
            tt.append({"transfer": name, "model": kind, "n_source_dev": m["n_source_dev"],
                       "events_source_dev": m["events_source_dev"], "n_pool": m["n_pool"],
                       "events_pool": m["events_pool"], "n_holdout": m["n_holdout"],
                       "events_holdout": m["events_holdout"], "low_precision": m["holdout_low_precision"],
                       "source_test_auroc": mm["source_test_auroc"],
                       "reference_holdout_auroc": mm["reference_holdout"]["auroc"]})
    pd.DataFrame(tt).to_csv(out / "transfers.csv", index=False)
    infeasible = df_all[~df_all.feasible].groupby(["transfer", "model", "E"]).size().rename("runs").reset_index()

    # verdict tables
    cells_v2 = cell_table(df, "v2", "gt").assign(rule="V2")
    cells_v1 = cell_table(df, "v1", "gt").assign(rule="V1")
    cells = pd.concat([cells_v2, cells_v1])
    cells.to_csv(out / "verdict_by_cell.csv", index=False)

    prim = cells_v2[cells_v2.model == "xgb"].copy()
    pooled = pooled_fd(df, "v2", "gt", rng)
    prim.to_csv(out / "primary_false_deploy.csv", index=False)
    pooled.to_csv(out / "primary_pooled.csv", index=False)

    # acceptance criterion (§8.3): every evaluable real transfer, upper CI ≤ 0.10
    acc = []
    for E, g in prim[prim.real].groupby("E"):
        ok = (~g.low_precision) & (g.gt_not >= 10)
        ev, ne = g[ok], g[~ok]
        acc.append({"E": E, "evaluable_transfers": len(ev), "max_fd_hi": ev.fd_hi.max() if len(ev) else np.nan,
                    "meets": bool(len(ev) and (ev.fd_hi <= 0.10).all()),
                    "non_evaluable": "; ".join(f"{r.transfer} (gt_not={r.gt_not}, low_precision={r.low_precision})"
                                               for r in ne.itertuples())})
    acc = pd.DataFrame(acc)
    e_min = acc[acc.meets].E.min() if acc.meets.any() else None
    acc.to_csv(out / "acceptance.csv", index=False)

    # GT sensitivity (§5.6)
    sens_rows = []
    variants = {"primary": {}, "delta0.01": {"delta": 0.01}, "delta0.03": {"delta": 0.03}, "delta0.05": {"delta": 0.05},
                "slope0.8-1.2": {"slope": (0.8, 1.2)}, "citl0.1": {"citl": 0.1}}
    for name, kw in variants.items():
        d = df.copy()
        d["gt_v"] = [deployable(m, ref, **kw) if isinstance(m, dict) else False
                     for m, ref in zip(d["recal__m"], d["ref_auroc"])]
        c = cell_table(d, "v2", "gt_v")
        c = c[(c.model == "xgb") & c.real]
        for E, g in c.groupby("E"):
            n0 = int(g.gt_not.sum())
            k = int(g.false_deploy.sum())
            p, lo, hi = wilson(k, n0)
            sens_rows.append({"variant": name, "E": E, "gt_not": n0, "false_deploy": k, "rate": p, "lo": lo, "hi": hi,
                              "share_gt_deployable": float(g.gt_deployable.sum() / g.runs.sum())})
    pd.DataFrame(sens_rows).to_csv(out / "gt_sensitivity.csv", index=False)

    # B4 decisions scored against its own ground truth
    d4 = df[df["B4_closed_testing__dep"].notna()].copy()
    d4["b4_verdict"] = np.where(d4.b4_decision == "deploy_updated", "GREEN", "RED")
    d4["gt4"] = d4["B4_closed_testing__dep"].astype(bool)
    b4 = cell_table(d4, "b4_verdict", "gt4")
    b4 = b4.merge(d4.groupby(["transfer", "model", "E"]).b4_chosen.agg(lambda s: s.value_counts().to_dict())
                  .rename("chosen_levels").reset_index())
    b4.to_csv(out / "b4_decisions.csv", index=False)

    # baselines: method summary and paired differences vs RECAL
    ms = []
    for (mo, E), g in df.groupby(["model", "E"]):
        for meth in METHODS:
            if f"{meth}__auroc" not in g:
                continue
            ms.append({"model": mo, "E": E, "method": meth, "runs": int(g[f"{meth}__auroc"].notna().sum()),
                       **{k: g[f"{meth}__{k}"].mean() for k in ["auroc", "brier", "citl", "slope", "ece10", "ici",
                                                                 "nb0.10", "nb0.20"]},
                       "deployable_share": g[f"{meth}__dep"].mean()})
    pd.DataFrame(ms).to_csv(out / "method_summary.csv", index=False)

    pr = []
    for (tr, mo, E), g in df.groupby(["transfer", "model", "E"]):
        for meth in METHODS[1:]:
            for k in ["auroc", "brier", "citl", "slope", "ece10", "ici", "nb0.10", "nb0.20"]:
                a, b = g[f"{meth}__{k}"], g[f"recal__{k}"]
                diff = (a - b).dropna().to_numpy()
                if not len(diff):
                    continue
                boots = [rng.choice(diff, len(diff)).mean() for _ in range(1000)]
                pr.append({"transfer": tr, "model": mo, "E": E, "method": meth, "metric": k,
                           "mean_diff_vs_recal": diff.mean(), "lo": np.percentile(boots, 2.5),
                           "hi": np.percentile(boots, 97.5), "n": len(diff)})
    pd.DataFrame(pr).to_csv(out / "baselines_paired.csv", index=False)

    # oracle bias and design choices
    df["oracle_bias"] = df.auroc_oracle_cohort - df.ref_auroc
    ob = df.groupby(["transfer", "model", "E"]).oracle_bias.agg(["mean", "std", "count"]).reset_index()
    ob.to_csv(out / "oracle_bias.csv", index=False)
    dc = df.assign(identity=~(df.get("design_apply_pca_coral", False).astype(bool)
                              | (df.get("design_mask_n", 0) > 0) | df.get("design_apply_woe", False).astype(bool)
                              | df.get("design_apply_qt", False).astype(bool)))
    des = dc.groupby(["transfer", "model", "E"]).agg(
        identity_share=("identity", "mean"), calibration_share=("design_apply_calibration", "mean"),
        mask_n_mean=("design_mask_n", "mean"), woe_share=("design_apply_woe", "mean"),
        qt_share=("design_apply_qt", "mean"), coral_share=("design_apply_pca_coral", "mean")).reset_index()
    des.to_csv(out / "design_choices.csv", index=False)

    # summary
    L = ["# RECAL validation — analysis summary", "",
         f"Runs: {len(df_all)} ({len(df)} feasible, {int((~df_all.feasible).sum())} not feasible). "
         f"RECAL errors: {int(df.recal_error.notna().sum())}.", "",
         "## Primary endpoint — false-deploy rate, V2, XGBoost (§8.1)", "",
         "| Transfer | E | runs | GT not deployable | GREEN | false deploy | rate [95% CI] |", "|---|---|---|---|---|---|---|"]
    for r in prim.sort_values(["transfer", "E"]).itertuples():
        flag = " (low-precision GT)" if r.low_precision else ""
        L.append(f"| {r.transfer}{flag} | {r.E} | {r.runs} | {r.gt_not} | {r.green} | {r.false_deploy} | "
                 f"{r.false_deploy_rate:.2f} [{r.fd_lo:.2f}, {r.fd_hi:.2f}] |")
    L += ["", "Pooled over real transfers (cluster bootstrap over transfers):", "",
          "| model | E | pooled rate [95% CI] |", "|---|---|---|"]
    for r in pooled.itertuples():
        L.append(f"| {r.model} | {r.E} | {r.pooled_false_deploy:.3f} [{r.lo:.3f}, {r.hi:.3f}] |")
    L += ["", "## Acceptance criterion (§8.3)", "", _md(acc), "",
          f"E_min = {e_min}", "", "## Ground-truth sensitivity (§5.6)", "",
          _md(pd.DataFrame(sens_rows)), "",
          "## Mean hold-out metrics by method", "",
          _md(pd.DataFrame(ms)), "",
          "## Not feasible cells (A1.5)", "", _md(infeasible)]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
