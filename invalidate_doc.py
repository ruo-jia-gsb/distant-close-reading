#!/usr/bin/env python3
"""
invalidate_doc.py - After replacing a document's PDF, set aside every run record that depended on it,
so `run_dcr.py run all` redoes only that work.

Moves (never deletes) these records from <runs>/outputs/*.jsonl into <runs>/outputs/superseded.jsonl:
  read and extract records for the document(s)
  edit records for every corpus condition containing the document(s)
Each moved record is tagged with the reason and time. Commit before running so history keeps the originals.

Usage:
  python invalidate_doc.py W2124927217 --reason "PDF was first-page preview; replaced with full text"
  python invalidate_doc.py W2124927217 --dry-run
"""

import argparse
import csv
import datetime
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", nargs="+", help="OpenAlex IDs of replaced documents")
    ap.add_argument("--runs", default="runs_cc")
    ap.add_argument("--reason", default="document text replaced")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus_of = {}
    for c in ("legitimacy", "tce", "splice"):
        with open(Path("corpus") / f"{c}_selected.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                corpus_of[r["openalex_id"]] = c
    exp = yaml.safe_load(open("prompts/experiment.yaml", encoding="utf-8"))
    cond_corpora = {k: {Path(p).stem.replace("_selected", "") for p in v["documents"]}
                    for k, v in exp["corpora"].items()}
    docs = set(args.docs)
    unknown = docs - set(corpus_of)
    if unknown:
        raise SystemExit(f"Not in the corpus: {sorted(unknown)}")
    conditions = {k for k, cs in cond_corpora.items() if any(corpus_of[d] in cs for d in docs)}
    print(f"Documents: {sorted(docs)}  ->  editor conditions to redo: {sorted(conditions)}")

    out = Path(args.runs) / "outputs"
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    moved_all = []
    for stage in ("read", "extract", "edit"):
        path = out / f"{stage}.jsonl"
        if not path.exists():
            continue
        keep, moved = [], []
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            parts = rec["custom_id"].split("__")
            hit = (parts[2] in docs) if stage in ("read", "extract") else (parts[2] in conditions)
            (moved if hit else keep).append(line if not hit else rec)
        print(f"  {stage:<8} {len(moved):>4} records to set aside, {len(keep):>5} kept")
        if not args.dry_run and moved:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
            for rec in moved:
                rec["superseded"] = {"reason": args.reason, "utc": stamp, "stage": stage}
            moved_all += moved
    if args.dry_run:
        print("Dry run: nothing changed.")
        return
    with open(out / "superseded.jsonl", "a", encoding="utf-8") as f:
        for rec in moved_all:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Moved {len(moved_all)} records to {out / 'superseded.jsonl'}. Now run: python run_dcr.py run all --backend claude-code")


if __name__ == "__main__":
    main()
