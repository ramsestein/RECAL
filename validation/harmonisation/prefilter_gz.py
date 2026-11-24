"""
Resumable, time-budgeted pre-filter of a large gzip CSV event table by itemid.

Some environments kill long-running processes (e.g. a sandboxed VM with per-command
time limits), which makes a single pass over MIMIC-IV `chartevents` (~3.5 GB gz) or
`labevents` impossible. This script reads the gzip stream in blocks, keeps only rows whose
`itemid` is one of the items in spec.yaml for that table, and writes them as parquet
parts. Progress (uncompressed offset + a seek index) is saved after every call, so the
same command can be repeated until it reports DONE.

The output contains the SAME columns as the CSV (all as strings), restricted to the
spec's itemids, so extract.py produces identical results from it (`--prefiltered`).
In an environment without time limits this step is unnecessary.

Usage
-----
    python prefilter_gz.py --data-root DATA --out DERIVED --db mimic --table chartevents --budget-s 150
    (repeat until it prints DONE; then the same for --table labevents)
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import indexed_gzip as igz
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import extract as ex

TABLE_PATH = {
    ("mimic", "chartevents"): "icu/chartevents.csv.gz",
    ("mimic", "labevents"): "hosp/labevents.csv.gz",
    ("nwicu", "chartevents"): "nw_icu/chartevents.csv.gz",
    ("nwicu", "labevents"): "nw_hosp/labevents.csv.gz",
}
BLOCK = 128 * 1024 * 1024


def itemids(db: str, table: str) -> list[str]:
    if table == "chartevents":
        ids = [r[0] for r in ex._vital_item_rows(db)]
        ids += [str(i) for i in ex.SPEC["demographics"]["weight"][db]["items"]]
    else:
        ids = [r[0] for r in ex._lab_item_rows(db)]
    return sorted(set(ids))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--db", required=True, choices=["mimic", "nwicu"])
    ap.add_argument("--table", required=True, choices=["chartevents", "labevents"])
    ap.add_argument("--budget-s", type=float, default=150)
    args = ap.parse_args(argv)

    src = args.data_root / ex.SPEC["databases"][args.db]["root"] / TABLE_PATH[(args.db, args.table)]
    work = args.out / "prefiltered" / f"{args.db}_{args.table}"
    work.mkdir(parents=True, exist_ok=True)
    state_f, index_f = work / "_state.json", work / "_index.gzidx"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"pos": 0, "part": 0, "header": None,
                                                                         "rows_in": 0, "rows_kept": 0, "done": False}
    if state["done"]:
        print("DONE (already complete)", json.dumps(state))
        return
    keep = pa.array(itemids(args.db, args.table), type=pa.string())

    f = igz.IndexedGzipFile(str(src), spacing=16 * 1024 * 1024)
    if index_f.exists():
        f.import_index(str(index_f))
    f.seek(state["pos"])
    t0 = time.time()
    while time.time() - t0 < args.budget_s:
        buf = f.read(BLOCK)
        at_eof = len(buf) < BLOCK
        if not buf:
            state["done"] = True
            break
        if state["header"] is None:
            nl = buf.index(b"\n")
            state["header"] = buf[:nl].decode().strip().split(",")
            state["pos"] += nl + 1
            buf = buf[nl + 1:]
        if at_eof and not buf.endswith(b"\n"):
            buf += b"\n"  # last line of the file without a trailing newline
        # cut at the last newline that is not inside a quoted field (quoted values may
        # contain newlines); every block starts at a record boundary, so quote parity holds
        last = buf.rfind(b"\n")
        while last >= 0 and buf.count(b'"', 0, last) % 2:
            last = buf.rfind(b"\n", 0, last)
        if last < 0:
            raise RuntimeError("record longer than block size")
        chunk = buf[: last + 1]
        cols = state["header"]
        tbl = pacsv.read_csv(
            io.BytesIO(chunk),
            read_options=pacsv.ReadOptions(column_names=cols, block_size=16 * 1024 * 1024),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                                 strings_can_be_null=True),
        )
        sub = tbl.filter(pc.is_in(tbl["itemid"], value_set=keep))
        if sub.num_rows:
            pq.write_table(sub, work / f"part_{state['part']:05d}.parquet")
            state["part"] += 1
        state["rows_in"] += tbl.num_rows
        state["rows_kept"] += sub.num_rows
        state["pos"] += last + 1
        f.seek(state["pos"])
    f.export_index(str(index_f))
    f.close()
    state_f.write_text(json.dumps(state))
    print(("DONE " if state["done"] else "CONTINUE ") +
          f"pos={state['pos'] / 1e9:.2f} GB rows_in={state['rows_in']:,} rows_kept={state['rows_kept']:,}")


if __name__ == "__main__":
    main()
