#!/usr/bin/env python3
"""Convert the IF-RLVR parquet (allenai/IF_multi_constraints_upto5) to the JSONL rows ``nightshift.lab.ifdata`` reads.

    python3 scripts/lab_convert_parquet.py train-00000-of-00001.parquet rows.jsonl [--expect-sha256 HEX]

This is the only place ``pyarrow`` is imported (lazily, inside ``main``). Output rows: ``key``, ``prompt`` (the single user message),
``ground_truth`` (kept as the dataset stores it), ``instruction_ids`` (parsed from ``ground_truth``), ``constraint`` (the tab-separated
constraint sentences, used to get the base prompt), and ``constraint_type`` / ``dataset`` when the parquet has them. A row with other than
exactly one user message, or an unparsable ``ground_truth``, is skipped and counted. The parquet's sha256 is printed (record it as
``ifrlvr_parquet_sha256`` in the manifest; scripts/lab_pins.json holds the pinned value).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from typing import Any, Iterable, Iterator, Mapping

PASS_THROUGH = ("constraint_type", "dataset")


def parse_instruction_ids(ground_truth: Any) -> list:
    obj = ast.literal_eval(ground_truth) if isinstance(ground_truth, str) else ground_truth
    obj = obj[0]
    if isinstance(obj, str):
        obj = json.loads(obj)
    return [str(i) for i in obj["instruction_id"]]


def convert_rows(records: Iterable[Mapping[str, Any]], counts: dict | None = None) -> Iterator[dict]:
    """Pure conversion of parquet records (dicts) to data rows. ``counts`` collects ``skipped_turns`` and ``skipped_ground_truth``."""
    counts = counts if counts is not None else {}
    for rec in records:
        msgs = rec.get("messages") or []
        if len(msgs) != 1 or msgs[0].get("role") != "user" or not isinstance(msgs[0].get("content"), str):
            counts["skipped_turns"] = counts.get("skipped_turns", 0) + 1
            continue
        try:
            ids = parse_instruction_ids(rec["ground_truth"])
        except Exception:
            counts["skipped_ground_truth"] = counts.get("skipped_ground_truth", 0) + 1
            continue
        row = {"key": str(rec["key"]), "prompt": msgs[0]["content"], "ground_truth": rec["ground_truth"], "instruction_ids": ids,
               "constraint": rec.get("constraint") or ""}
        for k in PASS_THROUGH:
            if rec.get(k) is not None:
                row[k] = rec[k]
        yield row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet")
    ap.add_argument("out_jsonl")
    ap.add_argument("--expect-sha256", help="refuse to convert a parquet whose sha256 differs")
    a = ap.parse_args(argv)
    h = hashlib.sha256()
    with open(a.parquet, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if a.expect_sha256 and digest != a.expect_sha256:
        print(f"parquet sha256 {digest} != expected {a.expect_sha256}", file=sys.stderr)
        return 1
    try:
        import pyarrow.parquet as pq  # the only pyarrow import in the lab
    except ImportError:
        print("pyarrow is required for this one-off conversion: python3 -m pip install pyarrow", file=sys.stderr)
        return 2
    counts: dict = {}
    n = 0
    with open(a.out_jsonl, "w", encoding="utf-8", newline="\n") as out:
        for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=4096):
            for row in convert_rows(batch.to_pylist(), counts):
                out.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                n += 1
    print(json.dumps({"parquet_sha256": digest, "rows_written": n, **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
