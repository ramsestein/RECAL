"""
RECAL 0.4 — independent evaluation on eICU-CRD (protocol v2.0, Amendment B1).

Stages
------
  setup : source models S1 (MIMIC-IV) and S2 (eICU leave-region-out), eligible target
          hospitals, Target-Pool / Target-Holdout split per hospital, Reference models.
  run   : per (hospital, source, E, replicate): draw the cohort, run RECAL 0.4 (CLI pipeline,
          selection: cv), the comparators and every candidate of RECAL's set; score all on the
          Target-Holdout; store RECAL's nested estimate. One JSON per task, resumable.

Usage
-----
    python run_validation_v2.py --derived DERIVED --out RESULTS_V2 --stage all --n-jobs 6
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as B  # noqa: E402, N812
from metrics import holdout_metrics  # noqa: E402
from run_validation import FEATURES, OUTCOME, XGB_PARAMS, proba  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

MASTER_SEED = 20261125
E_GRID = [25, 50, 100]
REPLICATES = 10
MIN_DEATHS, MIN_DEATHS_FALLBACK, MIN_HOSPITALS = 200, 100, 8
MAX_SOURCE_FOR_RECAL = 20000
PURPOSE = {"s1": 1, "s2": 2, "split": 3, "reference": 4, "cohort": 5, "recal": 6, "subsample": 7}
SOURCES = ["S1_mimic", "S2_leave_region_out"]


def seed(*key: int) -> int:
    return int(np.random.SeedSequence(MASTER_SEED, spawn_key=tuple(int(k) for k in key)).generate_state(1)[0])


def xgb(rs):
    return XGBClassifier(**XGB_PARAMS, random_state=rs)


def _save_xgb(m, path: Path) -> Path:
    m.get_booster().save_model(str(path))
    return path


# ── setup ──────────────────────────────────────────────────────────────────────

def setup(derived: Path, out: Path) -> dict:
    meta_f = out / "setup" / "meta.json"
    if meta_f.exists():
        return json.loads(meta_f.read_text())
    d = out / "setup"
    d.mkdir(parents=True, exist_ok=True)
    cols = ["key", "hospital_id", "region", OUTCOME] + FEATURES
    e = pd.read_parquet(derived / "eicu_features.parquet")[cols]
    e["region"] = e["region"].fillna("Unknown")
    m = pd.read_parquet(derived / "mimic_features.parquet")[["key", OUTCOME] + FEATURES]

    deaths = e.groupby("hospital_id")[OUTCOME].sum()
    thr = MIN_DEATHS if (deaths >= MIN_DEATHS).sum() >= MIN_HOSPITALS else MIN_DEATHS_FALLBACK
    hospitals = sorted(deaths[deaths >= thr].index.tolist(), key=lambda h: -deaths[h])
    meta = {"death_threshold": thr, "hospitals": {}, "sources": {}}

    # S1: MIMIC-IV whole cohort
    s1 = xgb(seed(PURPOSE["s1"])).fit(m[FEATURES].to_numpy(float), m[OUTCOME].to_numpy())
    _save_xgb(s1, d / "S1_mimic.json")
    joblib.dump(s1, d / "S1_mimic.pkl")
    sub = m.sample(n=min(MAX_SOURCE_FOR_RECAL, len(m)), random_state=seed(PURPOSE["subsample"], 1))
    sub[FEATURES + [OUTCOME]].to_parquet(d / "S1_mimic_source.parquet")
    meta["sources"]["S1_mimic"] = {"n_train": len(m), "events_train": int(m[OUTCOME].sum()), "n_recal_source": len(sub)}

    # S2: leave-region-out, one model per region of the eligible hospitals
    regions = sorted({e.loc[e.hospital_id == h, "region"].iloc[0] for h in hospitals})
    for ri, reg in enumerate(regions):
        tr = e[e.region != reg]
        mdl = xgb(seed(PURPOSE["s2"], ri)).fit(tr[FEATURES].to_numpy(float), tr[OUTCOME].to_numpy())
        _save_xgb(mdl, d / f"S2_{reg}.json")
        joblib.dump(mdl, d / f"S2_{reg}.pkl")
        sub = tr.sample(n=min(MAX_SOURCE_FOR_RECAL, len(tr)), random_state=seed(PURPOSE["subsample"], 2, ri))
        sub[FEATURES + [OUTCOME]].to_parquet(d / f"S2_{reg}_source.parquet")
        meta["sources"][f"S2_{reg}"] = {"n_train": len(tr), "events_train": int(tr[OUTCOME].sum()),
                                        "n_recal_source": len(sub), "held_out_region": reg}

    for hi, h in enumerate(hospitals):
        g = e[e.hospital_id == h].reset_index(drop=True)
        pool, hold = train_test_split(g, test_size=0.5, stratify=g[OUTCOME], random_state=seed(PURPOSE["split"], hi))
        hd = d / f"hosp_{h}"
        hd.mkdir(exist_ok=True)
        pool.reset_index(drop=True).to_parquet(hd / "pool.parquet")
        hold.reset_index(drop=True).to_parquet(hd / "holdout.parquet")
        ref = xgb(seed(PURPOSE["reference"], hi)).fit(pool[FEATURES].to_numpy(float), pool[OUTCOME].to_numpy())
        reg = g.region.iloc[0]
        info = {"region": reg, "n": len(g), "deaths": int(g[OUTCOME].sum()),
                "n_pool": len(pool), "events_pool": int(pool[OUTCOME].sum()),
                "n_holdout": len(hold), "events_holdout": int(hold[OUTCOME].sum()),
                "reference_holdout_auroc": float(roc_auc_score(hold[OUTCOME], proba(ref, hold[FEATURES].to_numpy(float)))),
                "source_files": {"S1_mimic": "S1_mimic", "S2_leave_region_out": f"S2_{reg}"}}
        for s, stem in info["source_files"].items():
            mdl = joblib.load(d / f"{stem}.pkl")
            info[f"{s}_frozen_holdout_auroc"] = float(roc_auc_score(hold[OUTCOME], proba(mdl, hold[FEATURES].to_numpy(float))))
        meta["hospitals"][str(h)] = info
    meta_f.write_text(json.dumps(meta, indent=2))
    return meta


def all_tasks(meta):
    tasks = []
    hs = list(meta["hospitals"])
    for r in range(REPLICATES):
        for hi, h in enumerate(hs):
            for si, s in enumerate(SOURCES):
                for E in E_GRID:
                    tasks.append({"hospital": h, "hi": hi, "source": s, "si": si, "E": E, "r": r,
                                  "id": f"h{h}__{s}__E{E:03d}__r{r:02d}"})
    return tasks


def _recal_yaml(tmp: Path, model_path: Path, rs: int) -> Path:
    cfg = {
        "model": {"path": str(model_path)},
        "source": {"path": str(tmp / "source.parquet"), "outcome_col": OUTCOME, "schema": str(tmp / "schema.json")},
        "target": {"path": str(tmp / "target.parquet"), "outcome_col": OUTCOME},
        "recal_core": {"selection": "cv", "pca_k": 5, "max_n_sweep": 30, "apply_qt": True,
                       "apply_woe": "auto", "max_missing_rate": 0.5},
        "overfitting_check": {"enabled": False, "method": "kfold", "n_splits": 5, "random_state": rs},
        "evaluation": {"oracle_eval": False, "feature_attribution": False, "counterfactual_alternatives": 0,
                       "brier_decompose": False},
        "output": {"report": None, "recal_model": str(tmp / "recal.joblib"), "metrics_json": str(tmp / "metrics.json"),
                   "timestamp": False},
    }
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run_task(task, out: Path, tmp_root: Path) -> dict:
    t0 = time.time()
    meta = json.loads((out / "setup" / "meta.json").read_text())
    h, s, E, r = task["hospital"], task["source"], task["E"], task["r"]
    info_h = meta["hospitals"][h]
    d = out / "setup"
    res = {"task": task["id"], "hospital": h, "source": s, "E": E, "replicate": r, "region": info_h["region"]}
    pool = pd.read_parquet(d / f"hosp_{h}" / "pool.parquet")
    pi = pool[OUTCOME].mean()
    n_neg = int(round(E * (1 - pi) / pi))
    pos, neg = np.where(pool[OUTCOME] == 1)[0], np.where(pool[OUTCOME] == 0)[0]
    if len(pos) < E or len(neg) < n_neg:
        res.update(feasible=False, reason=f"Target-Pool has {len(pos)} events < E={E}")
        return res
    rng = np.random.default_rng(seed(PURPOSE["cohort"], task["hi"], task["si"], E, r))
    cohort = pool.iloc[np.concatenate([rng.choice(pos, E, replace=False), rng.choice(neg, n_neg, replace=False)])].reset_index(drop=True)
    hold = pd.read_parquet(d / f"hosp_{h}" / "holdout.parquet")
    stem = info_h["source_files"][s]
    src = joblib.load(d / f"{stem}.pkl")
    Xc, yc = cohort[FEATURES].to_numpy(float), cohort[OUTCOME].to_numpy(int)
    Xh, yh = hold[FEATURES].to_numpy(float), hold[OUTCOME].to_numpy(int)
    thr = float(yc.mean())
    res.update(feasible=True, n_cohort=len(cohort), events_cohort=int(yc.sum()),
               reference_holdout_auroc=info_h["reference_holdout_auroc"])
    preds, info = {}, {}

    rs = seed(PURPOSE["recal"], task["hi"], task["si"], E, r)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_s:
        tmp = Path(tmp_s)
        pd.read_parquet(d / f"{stem}_source.parquet").to_parquet(tmp / "source.parquet")
        cohort[FEATURES + [OUTCOME]].to_parquet(tmp / "target.parquet")
        (tmp / "schema.json").write_text(json.dumps(FEATURES))
        cfg_path = _recal_yaml(tmp, d / f"{stem}.json", rs)
        cwd = os.getcwd()
        try:
            from recal_cli.adapt_select import CALIBRATORS, SelectedAdapter, fit_selected
            from recal_cli.config_schema import load_config
            from recal_cli.run import run
            os.chdir(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                rout = run(load_config(cfg_path), run_cv=False, skip_expensive=False)
            sel = SelectedAdapter.load(tmp / "recal.joblib")
            preds["recal"] = sel.predict_proba(Xh)
            info["recal"] = {"selection": rout.get("selection"), "estimate": rout.get("estimate")}
            # every candidate of RECAL's set, fitted on the whole cohort (for regret)
            from recal.data.pairing import CohortPair
            from recal_cli.data_loader import GenericCohortLoader
            from recal_cli.drift_compute import compute_drift_decomposition
            from recal_cli.model_loader import load_model
            model = load_model(d / f"{stem}.json", schema=FEATURES)
            pair = CohortPair(GenericCohortLoader(tmp / "source.parquet", FEATURES, OUTCOME),
                              GenericCohortLoader(tmp / "target.parquet", FEATURES, OUTCOME)).filter_target(0.5)

            def _drift(pt):
                with contextlib.redirect_stdout(io.StringIO()):
                    rr = compute_drift_decomposition(X_s=pt.X_s, X_t=pt.X_t, schema=FEATURES, model=model,
                                                     cache_path=None, verbose=False)
                return {k: rr[k] for k in ("drift_type_dict", "shap_dict", "lbase_dict")}
            for t in ["identity", "designed", "designed_drift"]:
                for c in CALIBRATORS:
                    try:
                        fa = fit_selected((t, c), pair, model, FEATURES, drift_fn=_drift)
                        preds[f"K_{t}+{c}"] = fa.predict_proba(Xh)
                    except Exception as e:
                        info[f"K_{t}+{c}"] = {"error": repr(e)}
        except Exception as e:
            info["recal"] = {"error": repr(e), "trace": traceback.format_exc()[-2000:]}
        finally:
            os.chdir(cwd)

    pc, ph = proba(src, Xc), proba(src, Xh)
    preds["F_frozen"] = ph
    for name, fn in [("I_citl", B.fit_citl), ("P_platt", B.fit_platt), ("T_isotonic", B.fit_isotonic)]:
        try:
            preds[name] = fn(yc, pc)(ph)
        except Exception as e:
            info[name] = {"error": repr(e)}
    try:
        f4, info["V_closed_testing"] = B.fit_closed_testing(yc, pc, Xc)
        preds["V_closed_testing"] = f4(ph, Xh)
    except Exception as e:
        info["V_closed_testing"] = {"error": repr(e)}

    res["metrics"] = {}
    for name, p in preds.items():
        try:
            res["metrics"][name] = holdout_metrics(yh, p, thr)
        except Exception as e:
            res["metrics"][name] = {"error": repr(e)}
    res["info"] = info
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def _worker(args):
    task, out, tmp_root = args
    logging.disable(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")
    try:
        res = run_task(task, Path(out), Path(tmp_root))
    except Exception as e:
        res = {"task": task["id"], "fatal_error": repr(e), "trace": traceback.format_exc()[-3000:]}
    (Path(out) / "runs" / f"{task['id']}.json").write_text(
        json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return task["id"], res.get("runtime_s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stage", choices=["setup", "run", "all"], default="all")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=None)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmp", type=Path, default=Path(tempfile.gettempdir()))
    args = ap.parse_args(argv)
    t0 = time.time()
    os.environ.setdefault("RECAL_DRIFT_CACHE_DIR", str(args.out / "drift_cache"))
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    meta = setup(args.derived, args.out)
    (args.out / "seeds.json").write_text(json.dumps({"master_seed": MASTER_SEED, "purpose_codes": PURPOSE,
                                                     "scheme": "SeedSequence(master, spawn_key=key)"}, indent=2))
    print(f"setup: {len(meta['hospitals'])} target hospitals (≥{meta['death_threshold']} deaths)", flush=True)
    if args.stage == "setup":
        return
    tasks = [t for t in all_tasks(meta) if not args.only or args.only in t["id"]]
    todo = [t for t in tasks if not (args.out / "runs" / f"{t['id']}.json").exists()]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done", flush=True)
    if args.max_tasks:
        todo = todo[: args.max_tasks]
    jobs = [(t, str(args.out), str(args.tmp)) for t in todo]
    done = 0
    if args.n_jobs == 1:
        for j in jobs:
            if args.budget_s and time.time() - t0 > args.budget_s:
                break
            tid, rt = _worker(j)
            done += 1
            print(f"done {tid} ({rt}s)", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for f in as_completed(futs):
                tid, rt = f.result()
                done += 1
                print(f"done {tid} ({rt}s) [{done}/{len(jobs)}]", flush=True)
    left = len(todo) - done
    print("DONE" if left == 0 else f"CONTINUE ({left} left)")


if __name__ == "__main__":
    main()
