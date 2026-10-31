#!/usr/bin/env python3
"""
nli_diagnostic.py - Measure how often the NLI model labels compatible-but-different claims as
contradictions (the bias suspected from the noise floor). Uses the same model and scoring as
check_gold_pairs.py (both directions, averaged).

Run AFTER the diagnostic pairs are vetted and committed.

Usage:  python nli_diagnostic.py            (reads prompts/diagnostic_pairs.yaml)
Output: results/nli_diagnostic.csv and results/nli_diagnostic.json
"""
import argparse, csv, json, datetime, sys
from pathlib import Path
import yaml
from check_gold_pairs import nli_scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="prompts/diagnostic_pairs.yaml")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    pairs = yaml.safe_load(open(args.pairs, encoding="utf-8"))["pairs"]
    scores = nli_scores(pairs, args.nli_model)
    rows = []
    for p, n in zip(pairs, scores):
        label = max(n, key=n.get)
        rows.append({"id": p["id"], "construct": p.get("construct", ""), "p_entail": round(n["entail"], 4),
                     "p_neutral": round(n["neutral"], 4), "p_contradict": round(n["contradict"], 4),
                     "label": label, "contradiction_over_half": n["contradict"] > 0.5})
    k = len(rows)
    share = sum(r["label"] == "contradict" for r in rows) / k
    over_half = sum(r["contradiction_over_half"] for r in rows) / k
    mean_c = sum(r["p_contradict"] for r in rows) / k
    print(f"{'id':<5}{'entail':>8}{'neutral':>9}{'contra':>8}  label")
    for r in rows:
        print(f"{r['id']:<5}{r['p_entail']:>8.3f}{r['p_neutral']:>9.3f}{r['p_contradict']:>8.3f}  {r['label']}")
    print(f"\n{k} compatible pairs: labeled contradiction {share:.0%}; P(contradiction) > 0.5 in {over_half:.0%}; "
          f"mean P(contradiction) {mean_c:.3f}")
    print("A well-calibrated model would label these neutral or entailment.")

    out = Path(args.outdir); out.mkdir(exist_ok=True)
    with open(out / "nli_diagnostic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (out / "nli_diagnostic.json").write_text(json.dumps({
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pairs_file": args.pairs, "nli_model": args.nli_model, "n_pairs": k,
        "share_labeled_contradiction": share, "share_contradiction_over_half": over_half,
        "mean_p_contradiction": mean_c}, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'nli_diagnostic.csv'} and {out / 'nli_diagnostic.json'}")


if __name__ == "__main__":
    sys.exit(main())
