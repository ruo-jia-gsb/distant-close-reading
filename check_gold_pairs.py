#!/usr/bin/env python3
"""
check_gold_pairs.py - Go/no-go test of the measures on expert-labeled claim pairs.

Asks whether each measure separates theoretically OPPOSED pairs (similar wording) from
theoretically EQUIVALENT pairs (different wording):

  cosine  similarity of sentence embeddings. Should be higher for equivalent pairs.
  NLI     entailment vs contradiction, scored in both directions and averaged.
          Should label opposed pairs "contradiction" and equivalent pairs "entailment".

Separation is reported as AUC: the probability that a random equivalent pair scores higher than
a random opposed pair (1.0 = perfect, 0.5 = chance). A measure "separates" at AUC >= 0.80.

Run AFTER the gold pairs are vetted and committed. Editing pairs after seeing results is tuning.

Usage:
  pip install sentence-transformers transformers torch pyyaml
  python check_gold_pairs.py
  python check_gold_pairs.py --embed-model BAAI/bge-large-en-v1.5 --nli-model microsoft/deberta-large-mnli

Outputs: results/gold_check.csv (per pair), results/gold_check.json (summary, models, versions)
The first run downloads the models (about 2 GB); later runs reuse them.
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import yaml

THRESHOLD = 0.80


def auc(pos, neg):
    """P(random pos score > random neg score), ties counted as half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def load_pairs(path):
    pairs = yaml.safe_load(open(path, encoding="utf-8"))["pairs"]
    bad = [p["id"] for p in pairs if p.get("type") not in ("opposed", "equivalent")]
    if bad:
        sys.exit(f"Pairs with unknown type: {bad}")
    return pairs


def cosine_scores(pairs, model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    a = model.encode([p["a"] for p in pairs], normalize_embeddings=True)
    b = model.encode([p["b"] for p in pairs], normalize_embeddings=True)
    return [float((x * y).sum()) for x, y in zip(a, b)]


def nli_scores(pairs, model_name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    labels = {i: l.lower() for i, l in model.config.id2label.items()}
    idx = {name: next(i for i, l in labels.items() if l.startswith(name))
           for name in ("entail", "neutral", "contradict")}

    def probs(premise, hypothesis):
        enc = tok(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            return torch.softmax(model(**enc).logits[0], dim=-1).tolist()

    out = []
    for p in pairs:
        both = [probs(p["a"], p["b"]), probs(p["b"], p["a"])]
        avg = {k: sum(x[i] for x in both) / 2 for k, i in idx.items()}
        out.append(avg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="prompts/gold_pairs.yaml")
    ap.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    n_opp = sum(p["type"] == "opposed" for p in pairs)
    n_eq = len(pairs) - n_opp
    print(f"{len(pairs)} pairs ({n_opp} opposed, {n_eq} equivalent)")
    print(f"Embeddings: {args.embed_model}\nNLI:        {args.nli_model}\n")

    cos = cosine_scores(pairs, args.embed_model)
    nli = nli_scores(pairs, args.nli_model)

    rows = []
    for p, c, n in zip(pairs, cos, nli):
        nli_label = max(n, key=n.get)
        expected = "contradict" if p["type"] == "opposed" else "entail"
        rows.append({
            "id": p["id"], "type": p["type"], "construct": p.get("construct", ""),
            "cosine": round(c, 4),
            "p_entail": round(n["entail"], 4), "p_neutral": round(n["neutral"], 4),
            "p_contradict": round(n["contradict"], 4),
            "nli_score": round(n["entail"] - n["contradict"], 4),
            "nli_label": nli_label, "nli_correct": nli_label == expected,
        })

    eq = [r for r in rows if r["type"] == "equivalent"]
    op = [r for r in rows if r["type"] == "opposed"]
    cos_auc = auc([r["cosine"] for r in eq], [r["cosine"] for r in op])
    nli_auc = auc([r["nli_score"] for r in eq], [r["nli_score"] for r in op])
    nli_acc = sum(r["nli_correct"] for r in rows) / len(rows)
    mean = lambda xs: sum(xs) / len(xs)

    print(f"{'id':<5}{'type':<12}{'cosine':>8}{'entail':>8}{'contra':>8}  nli label")
    for r in rows:
        mark = "" if r["nli_correct"] else "  <-- NLI wrong"
        print(f"{r['id']:<5}{r['type']:<12}{r['cosine']:>8.3f}{r['p_entail']:>8.3f}"
              f"{r['p_contradict']:>8.3f}  {r['nli_label']}{mark}")

    print(f"\nCosine: mean equivalent {mean([r['cosine'] for r in eq]):.3f}, "
          f"mean opposed {mean([r['cosine'] for r in op]):.3f}, AUC {cos_auc:.2f}")
    print(f"NLI:    AUC {nli_auc:.2f}, label accuracy {nli_acc:.0%}")

    cos_ok, nli_ok = cos_auc >= THRESHOLD, nli_auc >= THRESHOLD
    if cos_ok and nli_ok:
        verdict = "BOTH SEPARATE: both measures are fit for the main runs."
    elif nli_ok:
        verdict = ("NLI SEPARATES, COSINE DOES NOT: use NLI as the primary measure; report cosine's "
                   "failure as evidence that word overlap does not track theoretical agreement.")
    elif cos_ok:
        verdict = "COSINE SEPARATES, NLI DOES NOT: unexpected; inspect the NLI errors before the main runs."
    else:
        verdict = ("NEITHER SEPARATES: switch to the boundary-setting framing before spending on full runs.")
    print(f"\nVerdict (AUC threshold {THRESHOLD}): {verdict}")

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    with open(out / "gold_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    import sentence_transformers, transformers, torch
    summary = {
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pairs_file": args.pairs, "n_pairs": len(pairs), "n_opposed": n_opp, "n_equivalent": n_eq,
        "embed_model": args.embed_model, "nli_model": args.nli_model,
        "versions": {"sentence_transformers": sentence_transformers.__version__,
                     "transformers": transformers.__version__, "torch": torch.__version__},
        "threshold": THRESHOLD, "cosine_auc": round(cos_auc, 3), "nli_auc": round(nli_auc, 3),
        "nli_label_accuracy": round(nli_acc, 3), "verdict": verdict,
    }
    (out / "gold_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'gold_check.csv'} and {out / 'gold_check.json'}")


if __name__ == "__main__":
    sys.exit(main())
