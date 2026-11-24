"""
RECAL validation — harmonised extraction (protocol v1.0, §3).

Builds one flat table per database (one row per patient: index ICU stay) with the
harmonised predictors, the outcome and audit metadata, following spec.yaml.

Usage
-----
    python extract.py --data-root /path/to/data --out /path/to/derived --db mimic
    python extract.py --data-root /path/to/data --out /path/to/derived --db nwicu
    python extract.py --data-root /path/to/data --out /path/to/derived --db inspire

`--data-root` contains the PhysioNet copies laid out as in spec.yaml (databases.*.root).
Outputs (never committed): <out>/<db>_features.parquet, <out>/<db>_flow.json and the
intermediate event files <out>/<db>_vitals_long.parquet, <out>/<db>_labs_long.parquet.

Only predictors, the outcome and cohort counts are produced here. No model is fitted and
no predictor–outcome association is computed (protocol §3.4).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import duckdb
import yaml

HERE = Path(__file__).resolve().parent
SPEC = yaml.safe_load((HERE / "spec.yaml").read_text(encoding="utf-8"))

F_TO_C = "((v - 32.0) * 5.0 / 9.0)"


def _con(out: Path, memory_limit: str, threads: int) -> duckdb.DuckDBPyConnection:
    tmp = Path(tempfile.gettempdir()) / "recal_duckdb_tmp"   # spill files outside the output folder
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def _csv(path: Path) -> str:
    """read_csv expression with every column as VARCHAR (robust to sniffing on huge gz files)."""
    return f"read_csv('{path.as_posix()}', header=true, all_varchar=true)"


PREFILTERED: Path | None = None   # set by --prefiltered (see prefilter_gz.py)


def _events(root: Path, rel: str, db: str, table: str) -> str:
    """Event table source: the gzip CSV, or its itemid-prefiltered parquet parts (same columns)."""
    if PREFILTERED is not None:
        parts = PREFILTERED / f"{db}_{table}"
        state = json.loads((parts / "_state.json").read_text())
        if not state.get("done"):
            raise RuntimeError(f"prefilter of {db}/{table} is not complete")
        return f"read_parquet('{(parts / 'part_*.parquet').as_posix()}')"
    return _csv(root / rel)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Item maps from spec
# ─────────────────────────────────────────────────────────────────────────────

def _vital_item_rows(db: str) -> list[tuple]:
    """(item, var, modality, conversion) rows for one database."""
    rows = []
    for var, cfg in SPEC["vitals"].items():
        if "derived" in cfg:
            continue
        items = cfg[db]["items"]
        if isinstance(items, list):
            for it in items:
                rows.append((str(it), var, "any", "none"))
        else:
            for mod, its in items.items():
                conv = "f_to_c" if mod == "f" else "none"
                modality = mod if mod in ("art", "nbp") else "any"
                for it in its:
                    rows.append((str(it), var, modality, conv))
    return rows


def _lab_item_rows(db: str) -> list[tuple]:
    return [(str(it), var) for var, cfg in SPEC["labs"].items() for it in cfg[db]]


def _register_maps(con, db: str) -> None:
    con.execute("CREATE OR REPLACE TABLE vmap(item VARCHAR, var VARCHAR, modality VARCHAR, conv VARCHAR)")
    con.executemany("INSERT INTO vmap VALUES (?, ?, ?, ?)", _vital_item_rows(db))
    con.execute("CREATE OR REPLACE TABLE lmap(item VARCHAR, var VARCHAR)")
    con.executemany("INSERT INTO lmap VALUES (?, ?)", _lab_item_rows(db))
    bounds = [(v, c["bounds"][0], c["bounds"][1]) for v, c in SPEC["vitals"].items()]
    bounds += [(v, c["bounds"][0], c["bounds"][1]) for v, c in SPEC["labs"].items()]
    con.execute("CREATE OR REPLACE TABLE bounds(var VARCHAR, lo DOUBLE, hi DOUBLE)")
    con.executemany("INSERT INTO bounds VALUES (?, ?, ?)", bounds)


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-IV and NWICU (same schema family)
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT = {
    "mimic": {"hosp": "hosp", "icu": "icu"},
    "nwicu": {"hosp": "nw_hosp", "icu": "nw_icu"},
}


def cohort_mimic_like(con, root: Path, db: str) -> dict:
    lay = LAYOUT[db]
    c = SPEC["cohort"]
    flow = {}
    con.execute(f"""
        CREATE OR REPLACE TABLE stays_all AS
        SELECT subject_id, hadm_id, stay_id, first_careunit,
               TRY_CAST(intime AS TIMESTAMP) AS intime,
               TRY_CAST(outtime AS TIMESTAMP) AS outtime
        FROM {_csv(root / lay['icu'] / 'icustays.csv.gz')}
    """)
    flow["icu_stays_total"] = con.execute("SELECT count(*) FROM stays_all").fetchone()[0]
    flow["patients_total"] = con.execute("SELECT count(DISTINCT subject_id) FROM stays_all").fetchone()[0]
    flow["careunits_excluded"] = {
        u: int(n) for u, n in con.execute(f"""
            SELECT first_careunit, count(*) FROM stays_all
            WHERE NOT (regexp_matches(coalesce(first_careunit, ''), '{c['icu_careunit_include_regex']}')
                       AND NOT regexp_matches(coalesce(first_careunit, ''), '{c['icu_careunit_exclude_regex']}'))
            GROUP BY 1 ORDER BY 2 DESC""").fetchall()}
    con.execute(f"""
        DELETE FROM stays_all
        WHERE NOT (regexp_matches(coalesce(first_careunit, ''), '{c['icu_careunit_include_regex']}')
                   AND NOT regexp_matches(coalesce(first_careunit, ''), '{c['icu_careunit_exclude_regex']}'))
    """)
    flow["adult_icu_unit_stays"] = con.execute("SELECT count(*) FROM stays_all").fetchone()[0]

    con.execute("""
        CREATE OR REPLACE TABLE stays AS
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) AS rn
            FROM stays_all WHERE intime IS NOT NULL
        ) WHERE rn = 1
    """)
    flow["first_icu_stay"] = con.execute("SELECT count(*) FROM stays").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TABLE base AS
        SELECT s.subject_id, s.hadm_id, s.stay_id, s.intime, s.outtime,
               TRY_CAST(p.anchor_age AS INTEGER)
                 + (year(s.intime) - TRY_CAST(p.anchor_year AS INTEGER)) AS age,
               CASE WHEN p.gender = 'M' THEN 1 WHEN p.gender = 'F' THEN 0 END AS male,
               p.anchor_year_group AS year_group,
               TRY_CAST(a.deathtime AS TIMESTAMP) AS deathtime,
               date_diff('minute', s.intime, s.outtime) / 60.0 AS los_h
        FROM stays s
        LEFT JOIN {_csv(root / lay['hosp'] / 'patients.csv.gz')} p USING (subject_id)
        LEFT JOIN {_csv(root / lay['hosp'] / 'admissions.csv.gz')} a
               ON a.hadm_id = s.hadm_id
    """)
    _apply_common_cohort_rules(con, flow, c)
    return flow


def _apply_common_cohort_rules(con, flow: dict, c: dict) -> None:
    con.execute(f"CREATE OR REPLACE TABLE b1 AS SELECT * FROM base WHERE age >= {c['min_age']}")
    flow["age_ge_18"] = con.execute("SELECT count(*) FROM b1").fetchone()[0]
    con.execute(f"CREATE OR REPLACE TABLE b2 AS SELECT * FROM b1 WHERE los_h >= {c['min_icu_los_hours']}")
    flow["icu_los_ge_24h"] = con.execute("SELECT count(*) FROM b2").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE b3 AS SELECT * FROM b2
        WHERE deathtime IS NULL
           OR deathtime >= intime + INTERVAL {c['exclude_death_before_hours']} HOUR
    """)
    flow["alive_at_24h"] = con.execute("SELECT count(*) FROM b3").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT subject_id, hadm_id, stay_id AS key, intime, outtime, age, male, year_group, los_h,
               CASE WHEN deathtime IS NOT NULL
                         AND deathtime <= outtime + INTERVAL {c['death_tolerance_hours']} HOUR
                    THEN 1 ELSE 0 END AS icu_death
        FROM b3
    """)
    flow["final_cohort"] = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
    flow["events_icu_death"] = con.execute("SELECT sum(icu_death) FROM cohort").fetchone()[0]


def events_mimic_like(con, root: Path, db: str, out: Path) -> None:
    lay = LAYOUT[db]
    w_v, w_l, w_w = SPEC["windows"]["vitals"], SPEC["windows"]["labs"], SPEC["windows"]["weight"]
    wcfg = SPEC["demographics"]["weight"][db]
    w_items = ",".join(f"'{i}'" for i in wcfg["items"])
    v_items = ",".join(f"'{r[0]}'" for r in _vital_item_rows(db))
    l_items = ",".join(f"'{r[0]}'" for r in _lab_item_rows(db))

    _log(f"{db}: scanning chartevents")
    con.execute(f"""
        CREATE OR REPLACE TABLE chart_raw AS
        SELECT TRY_CAST(stay_id AS BIGINT) AS key, itemid AS item,
               TRY_CAST(charttime AS TIMESTAMP) AS t, TRY_CAST(valuenum AS DOUBLE) AS v
        FROM {_events(root, lay['icu'] + '/chartevents.csv.gz', db, 'chartevents')}
        WHERE itemid IN ({v_items},{w_items})
          AND TRY_CAST(stay_id AS BIGINT) IN (SELECT CAST(key AS BIGINT) FROM cohort)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE vitals_long AS
        SELECT c.key, m.var, m.modality,
               date_diff('second', k.intime, c.t) / 3600.0 AS t_h,
               CASE WHEN m.conv = 'f_to_c' THEN {F_TO_C.replace('v', 'c.v')} ELSE c.v END AS v
        FROM chart_raw c
        JOIN vmap m ON m.item = c.item
        JOIN cohort k ON CAST(k.key AS BIGINT) = c.key
        WHERE c.v IS NOT NULL
          AND c.t >= k.intime + INTERVAL {w_v[0]} HOUR AND c.t < k.intime + INTERVAL {w_v[1]} HOUR
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE weight AS
        SELECT key, arg_min(v, t) AS weight FROM (
            SELECT c.key, c.t, c.v * {wcfg['factor']} AS v
            FROM chart_raw c JOIN cohort k ON CAST(k.key AS BIGINT) = c.key
            WHERE c.item IN ({w_items}) AND c.v IS NOT NULL
              AND c.t >= k.intime + INTERVAL ({w_w[0]}) HOUR AND c.t < k.intime + INTERVAL {w_w[1]} HOUR
        ) GROUP BY key
    """)

    _log(f"{db}: scanning labevents")
    con.execute(f"""
        CREATE OR REPLACE TABLE labs_long AS
        SELECT CAST(k.key AS BIGINT) AS key, m.var,
               date_diff('second', k.intime, l.t) / 3600.0 AS t_h, l.v
        FROM (
            SELECT subject_id, itemid AS item, TRY_CAST(charttime AS TIMESTAMP) AS t,
                   TRY_CAST(valuenum AS DOUBLE) AS v
            FROM {_events(root, lay['hosp'] + '/labevents.csv.gz', db, 'labevents')}
            WHERE itemid IN ({l_items})
              AND subject_id IN (SELECT subject_id FROM cohort)
        ) l
        JOIN lmap m ON m.item = l.item
        JOIN cohort k ON k.subject_id = l.subject_id
        WHERE l.v IS NOT NULL
          AND l.t >= k.intime + INTERVAL ({w_l[0]}) HOUR AND l.t < k.intime + INTERVAL {w_l[1]} HOUR
    """)


# ─────────────────────────────────────────────────────────────────────────────
# INSPIRE (relative times in minutes, per-patient reference)
# ─────────────────────────────────────────────────────────────────────────────

def cohort_inspire(con, root: Path) -> dict:
    c = SPEC["cohort"]
    flow = {}
    con.execute(f"""
        CREATE OR REPLACE TABLE ops AS
        SELECT subject_id, hadm_id, op_id,
               TRY_CAST(age AS DOUBLE) AS age, sex, TRY_CAST(weight AS DOUBLE) AS weight_op,
               TRY_CAST(icuin_time AS DOUBLE) AS icuin, TRY_CAST(icuout_time AS DOUBLE) AS icuout,
               TRY_CAST(inhosp_death_time AS DOUBLE) AS death
        FROM {_csv(root / 'operations.csv.gz')}
    """)
    flow["operations_total"] = con.execute("SELECT count(*) FROM ops").fetchone()[0]
    con.execute("CREATE OR REPLACE TABLE stays_all AS SELECT * FROM ops WHERE icuin IS NOT NULL")
    flow["icu_stays_total"] = con.execute("SELECT count(*) FROM stays_all").fetchone()[0]
    flow["patients_total"] = con.execute("SELECT count(DISTINCT subject_id) FROM stays_all").fetchone()[0]
    # Several operations can share one ICU episode: the index episode is the patient's first icuin.
    con.execute("""
        CREATE OR REPLACE TABLE stays AS
        SELECT * FROM (
            SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY icuin, op_id) AS rn
            FROM stays_all
        ) WHERE rn = 1
    """)
    flow["first_icu_stay"] = con.execute("SELECT count(*) FROM stays").fetchone()[0]
    con.execute("""
        CREATE OR REPLACE TABLE base AS
        SELECT subject_id, hadm_id, op_id AS stay_id, icuin, icuout,
               age, CASE WHEN sex = 'M' THEN 1 WHEN sex = 'F' THEN 0 END AS male,
               NULL::VARCHAR AS year_group, weight_op, death,
               (icuout - icuin) / 60.0 AS los_h
        FROM stays
    """)
    con.execute(f"CREATE OR REPLACE TABLE b1 AS SELECT * FROM base WHERE age >= {c['min_age']}")
    flow["age_ge_18"] = con.execute("SELECT count(*) FROM b1").fetchone()[0]
    con.execute(f"CREATE OR REPLACE TABLE b2 AS SELECT * FROM b1 WHERE los_h >= {c['min_icu_los_hours']}")
    flow["icu_los_ge_24h"] = con.execute("SELECT count(*) FROM b2").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE b3 AS SELECT * FROM b2
        WHERE death IS NULL OR death >= icuin + {c['exclude_death_before_hours']} * 60
    """)
    flow["alive_at_24h"] = con.execute("SELECT count(*) FROM b3").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        SELECT subject_id, hadm_id, stay_id AS key, icuin, icuout, age, male, year_group, los_h,
               weight_op,
               CASE WHEN death IS NOT NULL AND death <= icuout + {c['death_tolerance_hours']} * 60
                    THEN 1 ELSE 0 END AS icu_death
        FROM b3
    """)
    flow["final_cohort"] = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
    flow["events_icu_death"] = con.execute("SELECT sum(icu_death) FROM cohort").fetchone()[0]
    return flow


def events_inspire(con, root: Path) -> None:
    w_v, w_l = SPEC["windows"]["vitals"], SPEC["windows"]["labs"]
    wb = SPEC["demographics"]["weight"]["bounds"]
    _log("inspire: scanning ward_vitals")
    con.execute(f"""
        CREATE OR REPLACE TABLE vitals_long AS
        SELECT k.key, m.var, m.modality, (w.t - k.icuin) / 60.0 AS t_h,
               CASE WHEN m.conv = 'f_to_c' THEN {F_TO_C.replace('v', 'w.v')} ELSE w.v END AS v
        FROM (
            SELECT subject_id, item_name AS item, TRY_CAST(chart_time AS DOUBLE) AS t,
                   TRY_CAST(value AS DOUBLE) AS v
            FROM {_csv(root / 'ward_vitals.csv.gz')}
            WHERE subject_id IN (SELECT subject_id FROM cohort)
        ) w
        JOIN vmap m ON m.item = w.item
        JOIN cohort k ON k.subject_id = w.subject_id
        WHERE w.v IS NOT NULL
          AND w.t >= k.icuin + {w_v[0]} * 60 AND w.t < k.icuin + {w_v[1]} * 60
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE weight AS
        SELECT key, CASE WHEN weight_op BETWEEN {wb[0]} AND {wb[1]} THEN weight_op END AS weight
        FROM cohort
    """)
    _log("inspire: scanning labs")
    con.execute(f"""
        CREATE OR REPLACE TABLE labs_long AS
        SELECT k.key, m.var, (l.t - k.icuin) / 60.0 AS t_h, l.v
        FROM (
            SELECT subject_id, item_name AS item, TRY_CAST(chart_time AS DOUBLE) AS t,
                   TRY_CAST(value AS DOUBLE) AS v
            FROM {_csv(root / 'labs.csv.gz')}
            WHERE subject_id IN (SELECT subject_id FROM cohort)
        ) l
        JOIN lmap m ON m.item = l.item
        JOIN cohort k ON k.subject_id = l.subject_id
        WHERE l.v IS NOT NULL
          AND l.t >= k.icuin + ({w_l[0]}) * 60 AND l.t < k.icuin + {w_l[1]} * 60
    """)


# ─────────────────────────────────────────────────────────────────────────────
# Common: bounds, derived MAP, aggregation
# ─────────────────────────────────────────────────────────────────────────────

def build_features(con, db: str, out: Path) -> dict:
    audit = {}
    # Plausibility bounds → out-of-range values dropped (counted for the audit)
    for tbl in ("vitals_long", "labs_long"):
        n0 = con.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0]
        con.execute(f"""
            CREATE OR REPLACE TABLE {tbl}_ok AS
            SELECT t.* FROM {tbl} t JOIN bounds b ON b.var = t.var
            WHERE t.v BETWEEN b.lo AND b.hi
        """)
        n1 = con.execute(f"SELECT count(*) FROM {tbl}_ok").fetchone()[0]
        audit[f"{tbl}_records"] = n0
        audit[f"{tbl}_dropped_out_of_bounds"] = n0 - n1
        per_var = con.execute(f"""
            SELECT t.var, count(*) AS n, sum(CASE WHEN t.v BETWEEN b.lo AND b.hi THEN 0 ELSE 1 END) AS dropped
            FROM {tbl} t JOIN bounds b ON b.var = t.var GROUP BY t.var ORDER BY t.var
        """).fetchall()
        audit[f"{tbl}_by_var"] = {v: {"n": int(n), "dropped": int(d)} for v, n, d in per_var}

    # Derived MAP: SBP and DBP of the same modality at the same time
    mlo, mhi = SPEC["vitals"]["map"]["bounds"]
    con.execute(f"""
        CREATE OR REPLACE TABLE map_long AS
        SELECT key, 'map' AS var, modality, t_h, (sbp + 2 * dbp) / 3.0 AS v FROM (
            SELECT key, modality, t_h,
                   avg(CASE WHEN var = 'sbp' THEN v END) AS sbp,
                   avg(CASE WHEN var = 'dbp' THEN v END) AS dbp
            FROM vitals_long_ok WHERE var IN ('sbp', 'dbp') AND modality IN ('art', 'nbp')
            GROUP BY key, modality, t_h
        ) WHERE sbp IS NOT NULL AND dbp IS NOT NULL AND sbp > dbp
          AND (sbp + 2 * dbp) / 3.0 BETWEEN {mlo} AND {mhi}
    """)
    con.execute("""
        CREATE OR REPLACE TABLE vit_all AS
        SELECT key, var, t_h, v FROM vitals_long_ok UNION ALL SELECT key, var, t_h, v FROM map_long
    """)

    v_aggs, l_aggs = SPEC["aggregations"]["vitals"], SPEC["aggregations"]["labs"]
    fn = {"min": "min", "max": "max", "mean": "avg"}
    vit_cols = [f"{fn[a]}(CASE WHEN var = '{var}' THEN v END) AS {var}_{a}"
                for var in SPEC["vitals"] for a in v_aggs]
    lab_cols = [f"{fn[a]}(CASE WHEN var = '{var}' THEN v END) AS {var}_{a}"
                for var in SPEC["labs"] for a in l_aggs]
    con.execute(f"CREATE OR REPLACE TABLE fv AS SELECT key, {', '.join(vit_cols)} FROM vit_all GROUP BY key")
    con.execute(f"CREATE OR REPLACE TABLE fl AS SELECT key, {', '.join(lab_cols)} FROM labs_long_ok GROUP BY key")

    wb = SPEC["demographics"]["weight"]["bounds"]
    con.execute(f"""
        CREATE OR REPLACE TABLE features AS
        SELECT '{db}' AS db, CAST(k.key AS VARCHAR) AS key, CAST(k.subject_id AS VARCHAR) AS subject_id,
               k.year_group, k.los_h, k.icu_death,
               CAST(k.age AS DOUBLE) AS age, CAST(k.male AS DOUBLE) AS male,
               CASE WHEN w.weight BETWEEN {wb[0]} AND {wb[1]} THEN w.weight END AS weight,
               fv.* EXCLUDE (key), fl.* EXCLUDE (key)
        FROM cohort k
        LEFT JOIN weight w ON CAST(w.key AS VARCHAR) = CAST(k.key AS VARCHAR)
        LEFT JOIN fv ON CAST(fv.key AS VARCHAR) = CAST(k.key AS VARCHAR)
        LEFT JOIN fl ON CAST(fl.key AS VARCHAR) = CAST(k.key AS VARCHAR)
    """)
    con.execute(f"COPY features TO '{(out / f'{db}_features.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY vit_all TO '{(out / f'{db}_vitals_long.parquet').as_posix()}' (FORMAT parquet)")
    con.execute(f"COPY labs_long_ok TO '{(out / f'{db}_labs_long.parquet').as_posix()}' (FORMAT parquet)")
    return audit


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--db", required=True, choices=["mimic", "nwicu", "inspire"])
    ap.add_argument("--memory-limit", default="2GB")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--prefiltered", action="store_true",
                    help="read chartevents/labevents from <out>/prefiltered (prefilter_gz.py)")
    args = ap.parse_args(argv)
    global PREFILTERED
    PREFILTERED = (args.out / "prefiltered") if args.prefiltered else None

    args.out.mkdir(parents=True, exist_ok=True)
    root = args.data_root / SPEC["databases"][args.db]["root"]
    con = _con(args.out, args.memory_limit, args.threads)
    _register_maps(con, args.db)

    t0 = time.time()
    _log(f"{args.db}: cohort")
    if args.db == "inspire":
        flow = cohort_inspire(con, root)
        events_inspire(con, root)
    else:
        flow = cohort_mimic_like(con, root, args.db)
        events_mimic_like(con, root, args.db, args.out)
    _log(f"{args.db}: features")
    audit = build_features(con, args.db, args.out)

    report = {
        "database": SPEC["databases"][args.db]["version"],
        "spec_version": SPEC["version"],
        "cohort_flow": {k: (v if isinstance(v, dict) else int(v)) for k, v in flow.items()},
        "extraction_audit": audit,
        "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (args.out / f"{args.db}_flow.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _log(f"{args.db}: done in {report['runtime_min']} min — {flow['final_cohort']} patients")


if __name__ == "__main__":
    main()
