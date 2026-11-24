"""
RECAL validation — main experiment (protocol v1.0, §4–§7; Amendment A1).

Stages
------
  setup : per transfer, split source (80/20) and target (Target-Pool/Target-Holdout 50/50),
          train the source models and the Reference models; cached under <out>/setup/.
  run   : per (transfer, model, E, replicate), draw the target cohort, run RECAL (CLI
          pipeline, protocol configuration) and baselines B0–B8, score everything on the
          Target-Holdout and write one JSON per task under <out>/runs/. Resumable.

Usage
-----
    python run_validation.py --derived DERIVED --out RESULTS --stage all --n-jobs 8
    python run_validation.py --derived DERIVED --out RESULTS --stage run --budget-s 150   # time-boxed chunk
    python run_validation.py --derived DERIVED --out RESULTS --list                       # task list only

<out> holds patient-level splits and must stay inside the credentialed environment.
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
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baselines as B  # noqa: E402, N812
from metrics import deployable, holdout_metrics  # noqa: E402

MASTER_SEED = 20261124
OUTCOME = "icu_death"
DBS = ["mimic", "nwicu", "inspire"]
REAL = [(s, t) for s in DBS for t in DBS if s != t]
CONTROL = [("mimicA", "mimicB")]
TRANSFERS = REAL + CONTROL
E_GRID = [10, 25, 50, 100, 200]
REPLICATES = 20
MODELS = {"xgb": E_GRID, "mlp": [25, 100]}
XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", n_jobs=1)
FEATURES = json.loads((HERE / "harmonisation" / "final_features.json").read_text())["kept"]


# ── seeds (documented scheme: SeedSequence(MASTER_SEED, spawn_key=key)) ────────

def seed(*key: int) -> int:
    return int(np.random.SeedSequence(MASTER_SEED, spawn_key=tuple(int(k) for k in key)).generate_state(1)[0])


PURPOSE = {"mimic_ab": 1, "source_split": 2, "target_split": 3, "source_model": 4,
           "reference_model": 5, "cohort": 6, "recal": 7, "retrain": 8}


def tname(tr):
    return f"{tr[0]}__{tr[1]}"


# ── models ─────────────────────────────────────────────────────────────────────

def make_model(kind: str, rs: int):
    if kind == "xgb":
        return XGBClassifier(**XGB_PARAMS, random_state=rs)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                       early_stopping=True, validation_fraction=0.1, max_iter=500,
                                       random_state=rs))


def save_model(model, kind: str, path_stem: Path) -> Path:
    if kind == "xgb":
        p = path_stem.with_suffix(".json")
        model.get_booster().save_model(str(p))
    else:
        p = path_stem.with_suffix(".joblib")
        joblib.dump(model, p)
    return p


def proba(model, X):
    return model.predict_proba(X)[:, 1]


# ── data ───────────────────────────────────────────────────────────────────────

def load_db(derived: Path, db: str) -> pd.DataFrame:
    df = pd.read_parquet(derived / f"{db}_features.parquet")
    return df[["key", "year_group", OUTCOME] + FEATURES].reset_index(drop=True)


def get_db(derived: Path, name: str, cache: dict) -> pd.DataFrame:
    if name in cache:
        return cache[name]
    if name in ("mimicA", "mimicB"):
        m = load_db(derived, "mimic")
        a, b = train_test_split(m, test_size=0.5, stratify=m[OUTCOME], random_state=seed(PURPOSE["mimic_ab"]))
        cache["mimicA"], cache["mimicB"] = a.reset_index(drop=True), b.reset_index(drop=True)
    else:
        cache[name] = load_db(derived, name)
    return cache[name]


# ── setup ──────────────────────────────────────────────────────────────────────

def setup_transfer(derived: Path, out: Path, tr, cache: dict) -> dict:
    ti = TRANSFERS.index(tr)
    d = out / "setup" / tname(tr)
    meta_f = d / "meta.json"
    if meta_f.exists():
        return json.loads(meta_f.read_text())
    d.mkdir(parents=True, exist_ok=True)
    src, tgt = get_db(derived, tr[0], cache), get_db(derived, tr[1], cache)
    dev, test = train_test_split(src, test_size=0.2, stratify=src[OUTCOME], random_state=seed(ti, PURPOSE["source_split"]))
    pool, hold = train_test_split(tgt, test_size=0.5, stratify=tgt[OUTCOME], random_state=seed(ti, PURPOSE["target_split"]))
    for name, df in [("source_dev", dev), ("source_test", test), ("target_pool", pool), ("target_holdout", hold)]:
        df.reset_index(drop=True).to_parquet(d / f"{name}.parquet")
    meta = {"transfer": tname(tr), "n_source_dev": len(dev), "events_source_dev": int(dev[OUTCOME].sum()),
            "n_pool": len(pool), "events_pool": int(pool[OUTCOME].sum()),
            "n_holdout": len(hold), "events_holdout": int(hold[OUTCOME].sum()),
            "holdout_low_precision": bool(hold[OUTCOME].sum() < 200), "models": {}}
    for mi, kind in enumerate(MODELS):
        m = make_model(kind, seed(ti, mi, PURPOSE["source_model"])).fit(dev[FEATURES].to_numpy(float), dev[OUTCOME].to_numpy())
        path = save_model(m, kind, d / f"source_{kind}")
        joblib.dump(m, d / f"source_{kind}.pkl")
        ref = make_model(kind, seed(ti, mi, PURPOSE["reference_model"])).fit(pool[FEATURES].to_numpy(float), pool[OUTCOME].to_numpy())
        joblib.dump(ref, d / f"reference_{kind}.pkl")
        y_h = hold[OUTCOME].to_numpy()
        p_ref = proba(ref, hold[FEATURES].to_numpy(float))
        from sklearn.metrics import roc_auc_score
        meta["models"][kind] = {
            "source_model_file": path.name,
            "source_test_auroc": float(roc_auc_score(test[OUTCOME], proba(m, test[FEATURES].to_numpy(float)))),
            "reference_holdout": holdout_metrics(y_h, p_ref, float(pool[OUTCOME].mean())),
        }
    meta_f.write_text(json.dumps(meta, indent=2))
    return meta


# ── tasks ──────────────────────────────────────────────────────────────────────

def all_tasks():
    tasks = []
    for r in range(REPLICATES):                      # replicate-major order: balanced partial runs
        for tr in TRANSFERS:
            for kind, grid in MODELS.items():
                for E in grid:
                    tasks.append({"transfer": tr, "model": kind, "E": E, "r": r,
                                  "id": f"{tname(tr)}__{kind}__E{E:03d}__r{r:02d}"})
    return tasks


def _recal_yaml(tmp: Path, model_path: Path, rs: int) -> Path:
    cfg = {
        "model": {"path": str(model_path)},
        "source": {"path": str(tmp / "source.parquet"), "outcome_col": OUTCOME, "schema": str(tmp / "schema.json")},
        "target": {"path": str(tmp / "target.parquet"), "outcome_col": OUTCOME},
        "recal_core": {"pca_k": 5, "max_n_sweep": 30, "apply_qt": True, "apply_woe": "auto", "max_missing_rate": 0.5},
        "overfitting_check": {"enabled": True, "method": "kfold", "n_splits": 5, "random_state": rs},
        "evaluation": {"oracle_eval": True, "oracle_cv": 5, "feature_attribution": False,
                       "counterfactual_alternatives": 0, "brier_decompose": True},
        "output": {"report": None, "recal_model": str(tmp / "recal.joblib"), "metrics_json": str(tmp / "metrics.json"),
                   "timestamp": False},
    }
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def run_task(task: dict, derived: Path, out: Path, tmp_root: Path) -> dict:
    t0 = time.time()
    tr, kind, E, r = task["transfer"], task["model"], task["E"], task["r"]
    ti, mi = TRANSFERS.index(tr), list(MODELS).index(kind)
    d = out / "setup" / tname(tr)
    meta = json.loads((d / "meta.json").read_text())
    res = {"task": task["id"], "transfer": tname(tr), "model": kind, "E": E, "replicate": r,
           "real_transfer": tr in REAL, "holdout_low_precision": meta["holdout_low_precision"]}

    pool = pd.read_parquet(d / "target_pool.parquet")
    pi = pool[OUTCOME].mean()
    n_neg = int(round(E * (1 - pi) / pi))
    pos, neg = np.where(pool[OUTCOME] == 1)[0], np.where(pool[OUTCOME] == 0)[0]
    if len(pos) < E or len(neg) < n_neg:
        res.update(feasible=False, reason=f"Target-Pool has {len(pos)} events < E={E}")
        return res
    rng = np.random.default_rng(seed(ti, mi, E, r, PURPOSE["cohort"]))
    idx = np.concatenate([rng.choice(pos, E, replace=False), rng.choice(neg, n_neg, replace=False)])
    cohort = pool.iloc[idx].reset_index(drop=True)
    dev = pd.read_parquet(d / "source_dev.parquet")
    hold = pd.read_parquet(d / "target_holdout.parquet")
    Xc, yc = cohort[FEATURES].to_numpy(float), cohort[OUTCOME].to_numpy(int)
    Xh, yh = hold[FEATURES].to_numpy(float), hold[OUTCOME].to_numpy(int)
    Xs = dev[FEATURES].to_numpy(float)
    thr = float(yc.mean())
    src = joblib.load(d / f"source_{kind}.pkl")
    ref_auc = meta["models"][kind]["reference_holdout"]["auroc"]
    res.update(feasible=True, n_cohort=len(cohort), events_cohort=int(yc.sum()))

    preds, info = {}, {}
    # ── RECAL (CLI pipeline with the protocol configuration) ────────────────
    rs = seed(ti, mi, E, r, PURPOSE["recal"])
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_s:
        tmp = Path(tmp_s)
        dev[FEATURES + [OUTCOME]].to_parquet(tmp / "source.parquet")
        cohort[FEATURES + [OUTCOME]].to_parquet(tmp / "target.parquet")
        (tmp / "schema.json").write_text(json.dumps(FEATURES))
        cfg_path = _recal_yaml(tmp, d / meta["models"][kind]["source_model_file"], rs)
        cwd = os.getcwd()
        try:
            from recal_cli.config_schema import load_config
            from recal_cli.run import AdaptedModelWrapper, run
            os.chdir(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                rout = run(load_config(cfg_path), run_cv=True, skip_expensive=False)
            wrapper = AdaptedModelWrapper.load(tmp / "recal.joblib")
            preds["recal"] = wrapper.predict_proba(Xh)
            aa = wrapper.auto_adapter
            c = aa._config
            info["recal"] = {
                "verdict": rout.get("verdict"),
                "auroc_raw_cohort": rout["in_sample"]["raw"]["auroc"],
                "auroc_in_sample": rout["in_sample"]["adapted"]["auroc"],
                "auroc_oof": (rout.get("cv") or {}).get("oof_metrics", {}).get("auroc"),
                "auroc_oracle": (rout.get("oracle") or {}).get("auroc"),
                "n_target_after_filter": rout["in_sample"]["n_target"],
                "design": {"mask_n": int(c.mask_n), "apply_qt": bool(c.apply_quantile),
                           "apply_woe": bool(c.apply_woe), "apply_pca_coral": bool(c.apply_pca_coral),
                           "pca_coral_k": int(c.pca_coral_k), "apply_calibration": bool(c.apply_calibration),
                           "calibration_method": c.calibration_method},
            }
        except Exception as e:  # recorded, counted and reported (§8.4)
            info["recal"] = {"error": repr(e), "trace": traceback.format_exc()[-2000:]}
        finally:
            os.chdir(cwd)

    # ── Baselines (§6) ──────────────────────────────────────────────────────
    pc, ph = proba(src, Xc), proba(src, Xh)
    preds["B0_raw"] = ph
    for name, fn in [("B1_citl", B.fit_citl), ("B2_platt", B.fit_platt), ("B3_isotonic", B.fit_isotonic)]:
        try:
            preds[name] = fn(yc, pc)(ph)
        except Exception as e:
            info[name] = {"error": repr(e)}
    try:
        f4, info["B4_closed_testing"] = B.fit_closed_testing(yc, pc, Xc)
        preds["B4_closed_testing"] = f4(ph, Xh)
    except Exception as e:
        info["B4_closed_testing"] = {"error": repr(e)}
    for name, fn in [("B5_coral", B.fit_coral_only), ("B6_qt", B.fit_qt_only)]:
        try:
            T = fn(Xs, Xc)
            preds[name] = proba(src, T(Xh))
        except Exception as e:
            info[name] = {"error": repr(e)}
    try:
        T = B.fit_skada_coral(Xs, Xc)
        preds["B7_skada_coral_platt"] = B.fit_platt(yc, proba(src, T(Xc)))(proba(src, T(Xh)))
    except Exception as e:
        info["B7_skada_coral_platt"] = {"error": repr(e)}
    try:
        m8 = make_model(kind, seed(ti, mi, E, r, PURPOSE["retrain"])).fit(Xc, yc)
        preds["B8_retrain"] = proba(m8, Xh)
    except Exception as e:
        info["B8_retrain"] = {"error": repr(e)}

    # ── Hold-out metrics and ground truth (§5.6, §7) ───────────────────────
    res["metrics"] = {}
    for name, p in preds.items():
        try:
            m = holdout_metrics(yh, p, thr)
            m["deployable_primary"] = deployable(m, ref_auc)
            res["metrics"][name] = m
        except Exception as e:
            res["metrics"][name] = {"error": repr(e)}
    res["reference_auroc_holdout"] = ref_auc
    res["info"] = info
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def _worker(args):
    task, derived, out, tmp_root = args
    logging.disable(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")
    try:
        res = run_task(task, Path(derived), Path(out), Path(tmp_root))
    except Exception as e:
        res = {"task": task["id"], "fatal_error": repr(e), "trace": traceback.format_exc()[-3000:]}
    f = Path(out) / "runs" / f"{task['id']}.json"
    f.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return task["id"], res.get("runtime_s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--stage", choices=["setup", "run", "all"], default="all")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=None, help="stop starting new tasks after this many seconds")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--tmp", type=Path, default=Path(tempfile.gettempdir()))
    ap.add_argument("--only", default=None, help="substring filter on task id")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    # per-feature source models of the drift decomposition depend only on the source cohort
    os.environ.setdefault("RECAL_DRIFT_CACHE_DIR", str(args.out / "drift_cache"))
    tasks = [t for t in all_tasks() if not args.only or args.only in t["id"]]
    if args.list:
        print(len(tasks), "tasks")
        for t in tasks[:10]:
            print(t["id"])
        return
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    (args.out / "seeds.json").write_text(json.dumps({"master_seed": MASTER_SEED, "scheme":
        "numpy SeedSequence(master, spawn_key=key).generate_state(1)[0]", "purpose_codes": PURPOSE,
        "transfers": [tname(t) for t in TRANSFERS], "models": list(MODELS)}, indent=2))

    if args.stage in ("setup", "all"):
        cache = {}
        for tr in TRANSFERS:
            if args.budget_s and time.time() - t0 > args.budget_s:
                print("BUDGET reached during setup")
                return
            m = setup_transfer(args.derived, args.out, tr, cache)
            print("setup", m["transfer"], "pool events", m["events_pool"], "holdout events", m["events_holdout"], flush=True)
    if args.stage == "setup":
        return

    todo = [t for t in tasks if not (args.out / "runs" / f"{t['id']}.json").exists()]
    if args.max_tasks:
        todo = todo[: args.max_tasks]
    print(f"{len(todo)} tasks to run ({len(tasks) - len(todo)} done)", flush=True)
    jobs = [(t, str(args.derived), str(args.out), str(args.tmp)) for t in todo]
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
            futs = {ex.submit(_worker, j): j for j in jobs}
            for f in as_completed(futs):
                tid, rt = f.result()
                done += 1
                print(f"done {tid} ({rt}s) [{done}/{len(jobs)}]", flush=True)
    left = len(todo) - done
    print("DONE" if left == 0 else f"CONTINUE ({left} left)")


if __name__ == "__main__":
    main()
