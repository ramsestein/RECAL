"""
RECAL 0.4 on the DEVELOPMENT databases (MIMIC-IV, NWICU, INSPIRE) — protocol v2.0 §7.

Not confirmatory: these databases informed the redesign of RECAL 0.4. The design reuses the
v1.0 setup (<results_v1>/setup: source/target splits, XGBoost source models, Target-Pool and
Target-Holdout) and applies the v2.0 procedure and endpoints (RECAL 0.4 selection: cv,
comparators F/P/I/T/V and every candidate K) to the six real transfers and the no-shift
control, E ∈ {25, 50, 100}, 10 replicates.

Usage
-----
    python run_validation_dev04.py --setup RESULTS_V1 --out RESULTS_DEV04 --n-jobs 6
    python analyse_v2.py --results RESULTS_DEV04   (clusters = transfers; "source" = real | control)
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as B  # noqa: E402, N812
from metrics import holdout_metrics  # noqa: E402
from run_validation import FEATURES, OUTCOME, TRANSFERS, proba, tname  # noqa: E402
from run_validation_v2 import MAX_SOURCE_FOR_RECAL, _recal_yaml  # noqa: E402

MASTER_SEED = 20261127
E_GRID = [25, 50, 100]
REPLICATES = 10


def seed(*key: int) -> int:
    return int(np.random.SeedSequence(MASTER_SEED, spawn_key=tuple(int(k) for k in key)).generate_state(1)[0])


def all_tasks():
    return [{"transfer": tr, "ti": ti, "E": E, "r": r, "id": f"{tname(tr)}__xgb__E{E:03d}__r{r:02d}"}
            for r in range(REPLICATES) for ti, tr in enumerate(TRANSFERS) for E in E_GRID]


def run_task(task, setup_root: Path, out: Path, tmp_root: Path) -> dict:
    t0 = time.time()
    tr, ti, E, r = task["transfer"], task["ti"], task["E"], task["r"]
    d = setup_root / "setup" / tname(tr)
    meta = json.loads((d / "meta.json").read_text())
    # v2 runner reports by "hospital" (cluster) and "source"; here cluster = transfer
    res = {"task": task["id"], "hospital": tname(tr), "source": "real" if tr[0] != "mimicA" else "control",
           "E": E, "replicate": r, "region": None}
    pool = pd.read_parquet(d / "target_pool.parquet")
    pi = pool[OUTCOME].mean()
    n_neg = int(round(E * (1 - pi) / pi))
    pos, neg = np.where(pool[OUTCOME] == 1)[0], np.where(pool[OUTCOME] == 0)[0]
    if len(pos) < E or len(neg) < n_neg:
        res.update(feasible=False, reason=f"Target-Pool has {len(pos)} events < E={E}")
        return res
    rng = np.random.default_rng(seed(ti, E, r, 1))
    cohort = pool.iloc[np.concatenate([rng.choice(pos, E, replace=False), rng.choice(neg, n_neg, replace=False)])].reset_index(drop=True)
    hold = pd.read_parquet(d / "target_holdout.parquet")
    dev = pd.read_parquet(d / "source_dev.parquet")
    src = joblib.load(d / "source_xgb.pkl")
    Xc, yc = cohort[FEATURES].to_numpy(float), cohort[OUTCOME].to_numpy(int)
    Xh, yh = hold[FEATURES].to_numpy(float), hold[OUTCOME].to_numpy(int)
    thr = float(yc.mean())
    res.update(feasible=True, n_cohort=len(cohort), events_cohort=int(yc.sum()),
               reference_holdout_auroc=meta["models"]["xgb"]["reference_holdout"]["auroc"])
    preds, info = {}, {}
    rs = seed(ti, E, r, 2)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_s:
        tmp = Path(tmp_s)
        dev.sample(n=min(MAX_SOURCE_FOR_RECAL, len(dev)), random_state=seed(ti, 3))[FEATURES + [OUTCOME]].to_parquet(tmp / "source.parquet")
        cohort[FEATURES + [OUTCOME]].to_parquet(tmp / "target.parquet")
        (tmp / "schema.json").write_text(json.dumps(FEATURES))
        cfg_path = _recal_yaml(tmp, d / meta["models"]["xgb"]["source_model_file"], rs)
        cwd = os.getcwd()
        try:
            from recal.data.pairing import CohortPair
            from recal_cli.adapt_select import CALIBRATORS, SelectedAdapter, fit_selected
            from recal_cli.config_schema import load_config
            from recal_cli.data_loader import GenericCohortLoader
            from recal_cli.drift_compute import compute_drift_decomposition
            from recal_cli.model_loader import load_model
            from recal_cli.run import run
            os.chdir(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                rout = run(load_config(cfg_path), run_cv=False, skip_expensive=False)
            preds["recal"] = SelectedAdapter.load(tmp / "recal.joblib").predict_proba(Xh)
            info["recal"] = {"selection": rout.get("selection"), "estimate": rout.get("estimate")}
            model = load_model(d / meta["models"]["xgb"]["source_model_file"], schema=FEATURES)
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
                        preds[f"K_{t}+{c}"] = fit_selected((t, c), pair, model, FEATURES, drift_fn=_drift).predict_proba(Xh)
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
    task, setup_root, out, tmp_root = args
    logging.disable(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")
    try:
        res = run_task(task, Path(setup_root), Path(out), Path(tmp_root))
    except Exception as e:
        res = {"task": task["id"], "fatal_error": repr(e), "trace": traceback.format_exc()[-3000:]}
    (Path(out) / "runs" / f"{task['id']}.json").write_text(
        json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return task["id"], res.get("runtime_s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", required=True, type=Path, help="results folder of protocol v1.0 (contains setup/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmp", type=Path, default=Path(tempfile.gettempdir()))
    args = ap.parse_args(argv)
    os.environ.setdefault("RECAL_DRIFT_CACHE_DIR", str(args.out / "drift_cache"))
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    (args.out / "seeds.json").write_text(json.dumps({"master_seed": MASTER_SEED,
                                                     "scheme": "SeedSequence(master, spawn_key=key)"}, indent=2))
    tasks = [t for t in all_tasks() if not args.only or args.only in t["id"]]
    todo = [t for t in tasks if not (args.out / "runs" / f"{t['id']}.json").exists()]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done", flush=True)
    if args.max_tasks:
        todo = todo[: args.max_tasks]
    jobs = [(t, str(args.setup), str(args.out), str(args.tmp)) for t in todo]
    if args.n_jobs == 1:
        for j in jobs:
            print("done", *_worker(j), flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            for i, f in enumerate(as_completed([ex.submit(_worker, j) for j in jobs]), 1):
                print("done", *f.result(), f"[{i}/{len(jobs)}]", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
