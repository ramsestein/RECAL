# Harmonisation (protocol v1.0, §3)

Builds one row per patient (index adult ICU stay ≥ 24 h) with 24-h predictors and ICU
mortality for MIMIC-IV 3.1, NWICU 0.1.0 and INSPIRE 1.4.2, from the PhysioNet files.

| File | Role |
|---|---|
| `spec.yaml` | Cohort rules, windows, item mappings, units, bounds, excluded variables |
| `extract.py` | Extraction per database → `<db>_features.parquet`, `<db>_flow.json` |
| `prefilter_gz.py` | Optional resumable itemid pre-filter for MIMIC/NWICU event tables (environments with per-command time limits) |
| `audit.py` | Cohort flow, missing rates, ≤ 50 % rule → `harmonisation_audit.md`, `final_features.json` |

```bash
pip install duckdb pyarrow pandas pyyaml indexed_gzip
DATA=/path/to/physionet   # mimic4/, nwicu/northwestern-icu-nwicu-database-0.1.0/, inspire/
OUT=/path/to/derived      # never committed

python extract.py --data-root $DATA --out $OUT --db inspire
python extract.py --data-root $DATA --out $OUT --db nwicu
python extract.py --data-root $DATA --out $OUT --db mimic
python audit.py --derived $OUT
```

If long processes are killed in your environment, pre-filter MIMIC first (repeat each
command until it prints `DONE`) and then pass `--prefiltered`:

```bash
python prefilter_gz.py --data-root $DATA --out $OUT --db mimic --table chartevents
python prefilter_gz.py --data-root $DATA --out $OUT --db mimic --table labevents
python extract.py --data-root $DATA --out $OUT --db mimic --prefiltered
```

Outputs contain patient-level data and must stay inside the credentialed environment.
