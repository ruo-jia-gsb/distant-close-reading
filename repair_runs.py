#!/usr/bin/env python3
"""
repair_runs.py - Clean a runs folder after duplicate or failed calls, so the runner redoes only what
is needed. Moves (never deletes) records into outputs/superseded.jsonl, tagged with the reason.

  1. Readings: for each custom_id keep the latest reading that parsed as JSON; set aside duplicates.
     If no reading for an ID parsed, set all of them aside so the reading is redone.
  2. Extractions: set aside every extraction for a reading that had any unparsed record (it may have
     been extracted from error text), plus duplicates.
  3. Editors: if any extraction was set aside, set aside editor runs for the conditions containing
     those documents.

Usage:
  python repair_runs.py --runs runs_sens --dry-run      # show what would change
  python repair_runs.py --runs runs_sens
  python repair_runs.py --runs runs_cc --report          # list unparsed readings and their text, change nothing
"""
import argparse, csv, datetime, json
from collections import defaultdict
from pathlib import Path
import yaml


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--experiment", default="prompts/experiment.yaml")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true", help="only list unparsed readings")
    args = ap.parse_args()
    out = Path(args.runs) / "outputs"
    reads, extracts, edits = load(out / "read.jsonl"), load(out / "extract.jsonl"), load(out / "edit.jsonl")

    by_id = defaultdict(list)
    for i, r in enumerate(reads):
        by_id[r["custom_id"]].append((i, r))
    unparsed_ids = {cid for cid, rs in by_id.items() if any(r["status"] == "succeeded" and not r.get("parse_ok") for _, r in rs)}

    if args.report:
        print(f"{len(reads)} reading records, {len(by_id)} unique IDs, {len(unparsed_ids)} IDs with an unparsed record")
        for cid in sorted(unparsed_ids):
            for _, r in by_id[cid]:
                if not r.get("parse_ok"):
                    print(f"  {cid}: stop={r.get('stop_reason')} out_tokens={r.get('usage', {}).get('output_tokens')} "
                          f"text={r.get('text', '')[:160]!r}")
        return

    move_read, keep_read = [], []
    for cid, rs in by_id.items():
        good = [(i, r) for i, r in rs if r["status"] == "succeeded" and r.get("parse_ok")]
        keep = good[-1][0] if good else None
        for i, r in rs:
            (keep_read if i == keep else move_read).append((i, r))
    keep_idx = {i for i, _ in keep_read}

    ex_by = defaultdict(list)
    for i, r in enumerate(extracts):
        ex_by[r["custom_id"].replace("extract__", "read__", 1)].append((i, r))
    move_ex, keep_ex_idx, affected_docs = [], set(), set()
    for rid, rs in ex_by.items():
        if rid in unparsed_ids or rid not in {cid for cid, _ in [(r["custom_id"], 0) for _, r in keep_read]}:
            move_ex += rs
            affected_docs.add(rid.split("__")[2])
        else:
            good = [(i, r) for i, r in rs if r["status"] == "succeeded" and r.get("parse_ok")]
            keep = good[-1][0] if good else None
            for i, r in rs:
                if i == keep:
                    keep_ex_idx.add(i)
                else:
                    move_ex.append((i, r))

    move_ed = []
    if affected_docs and edits:
        corpus_of = {}
        for c in ("legitimacy", "tce", "splice"):
            with open(Path("corpus") / f"{c}_selected.csv", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    corpus_of[row["openalex_id"]] = c
        exp = yaml.safe_load(open(args.experiment, encoding="utf-8"))
        cond = {k: {Path(p).stem.replace("_selected", "") for p in v["documents"]} for k, v in exp["corpora"].items()}
        hit = {k for k, cs in cond.items() if any(corpus_of[d] in cs for d in affected_docs)}
        move_ed = [(i, r) for i, r in enumerate(edits) if r["custom_id"].split("__")[2] in hit]

    print(f"Readings:    {len(reads)} records -> keep {len(keep_read)}, set aside {len(move_read)} "
          f"({len(unparsed_ids)} IDs had an unparsed record)")
    print(f"Extractions: {len(extracts)} records -> set aside {len(move_ex)}")
    print(f"Editors:     {len(edits)} records -> set aside {len(move_ed)}")
    redo = sum(1 for cid, rs in by_id.items() if not any(r.get("parse_ok") for _, r in rs))
    print(f"The runner will redo {redo} readings, {len(set(r['custom_id'] for _, r in move_ex))} extractions"
          f" and {len(set(r['custom_id'] for _, r in move_ed))} editor runs.")
    if args.dry_run:
        print("Dry run: nothing changed.")
        return
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    sup = open(out / "superseded.jsonl", "a", encoding="utf-8")
    for stage, recs, moved in (("read", reads, move_read), ("extract", extracts, move_ex), ("edit", edits, move_ed)):
        mi = {i for i, _ in moved}
        for i, r in moved:
            r["superseded"] = {"reason": "repair: duplicate or unparsed record, or derived from one", "utc": stamp, "stage": stage}
            sup.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / f"{stage}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(recs):
                if i not in mi:
                    r.pop("superseded", None)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sup.close()
    print("Done. Now rerun the same run_dcr.py command for this folder.")


if __name__ == "__main__":
    main()
