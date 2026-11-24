"""
RECAL validation — harmonisation audit (protocol v1.0, §3.3–3.4).

Reads <derived>/<db>_features.parquet and <db>_flow.json for every database and writes:

  <derived>/harmonisation_audit.md   cohort flow, missing rates, per-database distributions
  <derived>/final_features.json      predictors kept by the missing-rate rule (frozen in A1)

Only descriptive statistics of predictors and cohort counts are produced. No
predictor–outcome association is computed.

Usage
-----
    python -I audit.py --derived /path/to/derived
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
SPEC = yaml.safe_load((HERE / "spec.yaml").read_text(encoding="utf-8"))
DBS = ["mimic", "nwicu", "inspire"]
META = ["db", "key", "subject_id", "year_group", "los_h", "icu_death"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived", required=True, type=Path)
    args = ap.parse_args(argv)

    frames, flows = {}, {}
    for db in DBS:
        frames[db] = pd.read_parquet(args.derived / f"{db}_features.parquet")
        flows[db] = json.loads((args.derived / f"{db}_flow.json").read_text(encoding="utf-8"))

    features = [c for c in frames[DBS[0]].columns if c not in META]
    miss = pd.DataFrame({db: frames[db][features].isna().mean() for db in DBS})
    thr = SPEC["max_missing_rate"]
    keep = miss.index[(miss <= thr).all(axis=1)].tolist()
    drop = miss.index[~(miss <= thr).all(axis=1)].tolist()

    lines = ["# Harmonisation audit", "",
             f"Spec version {SPEC['version']}. Missing-rate rule: keep a feature if missing ≤ {thr:.0%} "
             "in every database (protocol §3.3).", "", "## Cohort flow", ""]
    steps = ["patients_total", "icu_stays_total", "first_icu_stay", "age_ge_18",
             "icu_los_ge_24h", "alive_at_24h", "final_cohort", "events_icu_death"]
    lines.append("| Step | " + " | ".join(flows[db]["database"] for db in DBS) + " |")
    lines.append("|---|" + "---|" * len(DBS))
    for s in steps:
        lines.append(f"| {s} | " + " | ".join(str(flows[db]["cohort_flow"].get(s, "—")) for db in DBS) + " |")
    prev = {db: frames[db]["icu_death"].mean() for db in DBS}
    lines.append("| ICU mortality | " + " | ".join(f"{prev[db]:.1%}" for db in DBS) + " |")

    lines += ["", "## Records dropped by plausibility bounds", "",
              "| Database | Vital records | Dropped | Lab records | Dropped |", "|---|---|---|---|---|"]
    for db in DBS:
        a = flows[db]["extraction_audit"]
        lines.append(f"| {db} | {a['vitals_long_records']} | {a['vitals_long_dropped_out_of_bounds']} | "
                     f"{a['labs_long_records']} | {a['labs_long_dropped_out_of_bounds']} |")

    lines += ["", f"## Missing rate by feature (kept: {len(keep)}, dropped: {len(drop)})", "",
              "| Feature | " + " | ".join(DBS) + " | Kept |", "|---|" + "---|" * (len(DBS) + 1)]
    for f in features:
        lines.append(f"| {f} | " + " | ".join(f"{miss.loc[f, db]:.1%}" for db in DBS) +
                     f" | {'yes' if f in keep else 'no'} |")

    lines += ["", "## Distributions (median [IQR]) of kept features", "",
              "| Feature | " + " | ".join(DBS) + " |", "|---|" + "---|" * len(DBS)]
    for f in keep:
        cells = []
        for db in DBS:
            s = frames[db][f].dropna()
            cells.append(f"{s.median():.3g} [{s.quantile(.25):.3g}–{s.quantile(.75):.3g}]" if len(s) else "—")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")

    (args.derived / "harmonisation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.derived / "final_features.json").write_text(
        json.dumps({"spec_version": SPEC["version"], "max_missing_rate": thr,
                    "kept": keep, "dropped": drop}, indent=2), encoding="utf-8")
    print(f"kept {len(keep)} / {len(features)} features")


if __name__ == "__main__":
    main()
