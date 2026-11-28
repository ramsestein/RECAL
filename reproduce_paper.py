"""Regenerate Table 2 and Fig. 3 of the RECAL SoftwareX paper from the shipped synthetic data.

Usage (repository root, package installed with `pip install -e .`):
    python reproduce_paper.py

Runs RECAL on the frozen XGBoost and MLP synthetic models (configs/synthetic.yaml and
configs/synthetic_mlp.yaml), then writes outputs/paper/table2.csv, table2.md and figure3.png,
and the wall-clock time of each run (outputs/paper/runtime.json).
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "paper"
RUNS = {"XGBoost": ("configs/synthetic.yaml", "outputs/synthetic/metrics.json"),
        "MLP": ("configs/synthetic_mlp.yaml", "outputs/synthetic_mlp/metrics.json")}
METRICS = [("auroc", "AUROC"), ("brier", "Brier score"), ("logloss", "Log-loss"),
           ("slope", "Calibration slope"), ("citl", "Calibration intercept"), ("ece10", "ECE (10 bins)")]


def make_mlp_pipeline() -> None:
    """The MLP is stored as {'model', 'scaler'}; RECAL needs one object with predict_proba."""
    dst = ROOT / "synthetic_data" / "models" / "nn_pipeline.joblib"
    if dst.exists():
        return
    import joblib
    from sklearn.pipeline import Pipeline
    obj = joblib.load(ROOT / "synthetic_data" / "models" / "nn_model.joblib")
    joblib.dump(Pipeline([("scaler", obj["scaler"]), ("mlp", obj["model"])]), dst)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_mlp_pipeline()
    runtime, results = {}, {}
    for name, (cfg, mjson) in RUNS.items():
        t0 = time.time()
        subprocess.run([sys.executable, "-m", "recal_cli.run", "--config", cfg], cwd=ROOT, check=True)
        runtime[name] = round(time.time() - t0, 1)
        results[name] = json.loads((ROOT / mjson).read_text(encoding="utf-8"))
    runtime["platform"] = f"{platform.platform()} | {platform.processor()} | Python {platform.python_version()}"
    (OUT / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    rows = ["model,metric,frozen,recal_estimate,ci_low,ci_high,chosen"]
    md = ["| Model | Metric | Frozen model | RECAL (nested CV) [95% CI] |", "|---|---|---|---|"]
    for name, m in results.items():
        est, chosen = m["estimate"], "+".join(m["selection"]["chosen"])
        for key, label in METRICS:
            f, e, (lo, hi) = est["raw_on_cohort"][key], est["estimate"][key], est["ci95"][key]
            rows.append(f"{name},{label},{f:.4f},{e:.4f},{lo:.4f},{hi:.4f},{chosen}")
            md.append(f"| {name} | {label} | {f:.3f} | {e:.3f} [{lo:.3f}, {hi:.3f}] |")
        md.append(f"| {name} | Chosen adaptation | | {chosen} |")
    (OUT / "table2.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (OUT / "table2.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    est = results["XGBoost"]["estimate"]
    panels = [("auroc", "AUROC (higher is better)"), ("brier", "Brier score (lower is better)"),
              ("ece10", "ECE (lower is better)"), ("slope", "Calibration slope (ideal 1)")]
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (k, title) in zip(axs, panels):
        ax.plot(est["raw_on_cohort"][k], 1, "s", ms=9, color="#f47230", mec="white", mew=2)
        ax.plot(est["ci95"][k], [0, 0], color="#257de0", lw=2)
        ax.plot(est["estimate"][k], 0, "o", ms=9, color="#257de0", mec="white", mew=2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["RECAL\n(nested CV)", "Frozen\nmodel"] if k == "auroc" else ["", ""])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(title, fontsize=10.5, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure3.png", dpi=200, bbox_inches="tight")
    print((OUT / "table2.md").read_text(encoding="utf-8"))
    print("runtime (s):", runtime)


if __name__ == "__main__":
    main()
