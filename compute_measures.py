#!/usr/bin/env python3
"""
compute_measures.py - Turn DCR outputs into the measures reported in the proposal.

Unit of comparison: characterization claims from claim extraction (stance is analyzed separately and
never counted as incoherence).

Similarity of two claim sets A and B (e.g., two readings of one document, or two documents):
  NLI score of a claim pair  m(a,b) = (P_entail(a->b) + P_entail(b->a))/2 - (P_contra(a->b) + P_contra(b->a))/2
  Agreement  S(A,B) = 1/2 [ mean_a max_b m(a,b) + mean_b max_a m(a,b) ]        (range -1..1)
  Dispersion D = 1 - S
  Conflict ratio = contradiction pairs / (contradiction + entailment pairs), counting a pair when its
                   averaged probability exceeds 0.5. Neutral pairs (different topics) are ignored.
For document-to-document comparisons, each claim is compared with its K nearest claims in the other
document by embedding similarity (cosine retrieves topically related claims; NLI judges them).

Measures:
  1. Noise floor: same reader, same document, different repetitions.
  2. Real-reader divergence: different real readers, same document, same repetition.
  3. Sham control: sham vs sham, and sham vs structural (same method, different wording).
  4. Corpus coherence: agreement between documents, pooling real readers, per repetition and condition
     (legitimacy, tce, tce_spliced), with the spliced pairs broken out (manipulation check).
  5. Stance distribution (genealogical reader) per corpus.
  6. Editors: overall judgments, number of positions, and agreement across repeated runs on fixed input.
  7. Co-citation baseline: bibliographic coupling between documents vs NLI agreement (Spearman).
  Cosine versions of 1-4 are reported alongside as the secondary measure.

Usage (in ddr_venv; uses the GPU if available):
  python compute_measures.py                    # reads runs_cc/
  python compute_measures.py --runs runs --k 3

Outputs in results/: measures_summary.json, report.md, reading_pairs.csv, doc_pairs.csv, editors.csv,
and nli_cache.jsonl (reused on later runs).
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORPUS = Path("corpus")
GROUPS = {"legitimacy": ["legitimacy"], "tce_all": ["tce", "splice"]}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def latest_ok(path):
    best = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] == "succeeded" and r.get("parsed"):
                best[r["custom_id"]] = r
    return best


def load(runs):
    corpus_of = {}
    for c in ("legitimacy", "tce", "splice"):
        with open(CORPUS / f"{c}_selected.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                corpus_of[r["openalex_id"]] = c
    readings = {}   # (reader, doc, rep) -> list of claim texts
    stances = {}    # (reader, doc, rep) -> stance dict
    for cid, rec in latest_ok(Path(runs) / "outputs" / "extract.jsonl").items():
        _, reader, doc, rep = cid.split("__")
        claims = [k.get("text", "").strip() for k in rec["parsed"].get("characterization", [])]
        claims = [c for c in claims if c]
        if claims:
            readings[(reader, doc, int(rep[1:]))] = claims
        stances[(reader, doc, int(rep[1:]))] = rec["parsed"].get("stance", {}) or {}
    edits = latest_ok(Path(runs) / "outputs" / "edit.jsonl")
    refs = {}
    for c in ("legitimacy", "tce", "splice"):
        p = CORPUS / f"{c}_references.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                refs[r["openalex_id"]] = set(r["referenced_works"])
    return corpus_of, readings, stances, edits, refs


# ---------------------------------------------------------------------------
# Models (embedding + NLI), with an on-disk NLI cache
# ---------------------------------------------------------------------------


class Models:
    def __init__(self, embed_model, nli_model, cache_path, batch=64):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else ""))
        self.embedder = SentenceTransformer(embed_model, device=self.device)
        self.tok = AutoTokenizer.from_pretrained(nli_model)
        self.nli = AutoModelForSequenceClassification.from_pretrained(nli_model).to(self.device).eval()
        if self.device == "cuda":
            self.nli = self.nli.half()
        labels = {i: l.lower() for i, l in self.nli.config.id2label.items()}
        self.idx = {k: next(i for i, l in labels.items() if l.startswith(k)) for k in ("entail", "contradict")}
        self.batch = batch
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for line in open(self.cache_path, encoding="utf-8"):
                k, e, c = json.loads(line)
                self.cache[k] = (e, c)

    def embed(self, texts):
        return self.embedder.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)

    @staticmethod
    def key(p, h):
        return hashlib.sha1(f"{p}\x1f{h}".encode("utf-8")).hexdigest()

    def ensure(self, pairs):
        """Compute NLI (entail, contradict) for (premise, hypothesis) text pairs not yet cached."""
        todo = list({self.key(p, h): (p, h) for p, h in pairs if self.key(p, h) not in self.cache}.items())
        if not todo:
            return
        todo.sort(key=lambda kv: len(kv[1][0]) + len(kv[1][1]))
        print(f"  NLI: {len(todo)} new pairs")
        out = open(self.cache_path, "a", encoding="utf-8")
        for start in range(0, len(todo), self.batch):
            chunk = todo[start:start + self.batch]
            enc = self.tok([p for _, (p, _) in chunk], [h for _, (_, h) in chunk], return_tensors="pt",
                           padding=True, truncation=True, max_length=256).to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.nli(**enc).logits.float(), dim=-1).cpu().numpy()
            for (k, _), pr in zip(chunk, probs):
                e, c = float(pr[self.idx["entail"]]), float(pr[self.idx["contradict"]])
                self.cache[k] = (e, c)
                out.write(json.dumps([k, round(e, 5), round(c, 5)]) + "\n")
            if (start // self.batch) % 200 == 0 and start:
                print(f"    {start}/{len(todo)}")
        out.close()

    def pair(self, a, b):
        e1, c1 = self.cache[self.key(a, b)]
        e2, c2 = self.cache[self.key(b, a)]
        return (e1 + e2) / 2, (c1 + c2) / 2


# ---------------------------------------------------------------------------
# Set comparison
# ---------------------------------------------------------------------------


def candidate_pairs(A, B, emb, k):
    """Claim pairs to evaluate: all pairs if k is None, else each claim's k nearest by cosine."""
    if k is None:
        return [(a, b) for a in A for b in B]
    ea, eb = np.array([emb[a] for a in A]), np.array([emb[b] for b in B])
    sim = ea @ eb.T
    pairs = set()
    for i in range(len(A)):
        for j in np.argsort(-sim[i])[:k]:
            pairs.add((A[i], B[j]))
    for j in range(len(B)):
        for i in np.argsort(-sim[:, j])[:k]:
            pairs.add((A[i], B[j]))
    return sorted(pairs)


def compare(A, B, models, emb, k=None):
    pairs = candidate_pairs(A, B, emb, k)
    m, ent, con = {}, 0, 0
    for a, b in pairs:
        e, c = models.pair(a, b)
        m[(a, b)] = e - c
        ent += e > 0.5
        con += c > 0.5
    best_a = [max(v for (x, _), v in m.items() if x == a) for a in A if any(x == a for x, _ in m)]
    best_b = [max(v for (_, y), v in m.items() if y == b) for b in B if any(y == b for _, y in m)]
    s = 0.5 * (np.mean(best_a) + np.mean(best_b))
    ea, eb = np.array([emb[a] for a in A]), np.array([emb[b] for b in B])
    sim = ea @ eb.T
    cos = 0.5 * (sim.max(axis=1).mean() + sim.max(axis=0).mean())
    conflict = con / (con + ent) if (con + ent) else float("nan")
    return float(s), float(cos), conflict, int(ent), int(con)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def cluster_boot(rows, value, cluster="doc", n=1000, seed=7):
    """Mean and 95% CI, resampling clusters (documents) with replacement."""
    by = defaultdict(list)
    for r in rows:
        by[r[cluster]].append(r[value])
    keys = list(by)
    if not keys:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        vals = [v for i in pick for v in by[keys[i]]]
        means.append(np.mean(vals))
    allv = [v for k in keys for v in by[k]]
    return float(np.mean(allv)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def fmt(mean, lo, hi, d=3):
    return f"{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_cc")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--k", type=int, default=3, help="nearest claims per claim for document comparisons")
    ap.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--experiment", default="prompts/experiment.yaml")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    corpus_of, readings, stances, edits, refs = load(args.runs)
    if not readings:
        sys.exit(f"No extracted claims found in {args.runs}/outputs/extract.jsonl")
    import yaml
    exp = yaml.safe_load(open(args.experiment, encoding="utf-8"))
    real, sham = exp["readers"]["real"], exp["readers"]["sham"]
    sham_base = "structural"
    print(f"{len(readings)} readings with claims; {len(edits)} editor outputs")

    models = Models(args.embed_model, args.nli_model, out / "nli_cache.jsonl", args.batch)
    texts = sorted({c for cl in readings.values() for c in cl})
    print(f"Embedding {len(texts)} unique claims")
    emb = dict(zip(texts, models.embed(texts)))

    # ---- reading pairs (within document) ------------------------------------
    by_doc = defaultdict(dict)
    for (reader, doc, rep), cl in readings.items():
        by_doc[doc][(reader, rep)] = cl
    comparisons = []  # (category, doc, reader_a, rep_a, reader_b, rep_b)
    for doc, rd in by_doc.items():
        readers = sorted({r for r, _ in rd})
        reps = sorted({p for _, p in rd})
        for r in readers:
            for p1, p2 in itertools.combinations([p for p in reps if (r, p) in rd], 2):
                comparisons.append(("floor_" + ("sham" if r in sham else "real"), doc, r, p1, r, p2))
        for p in reps:
            for r1, r2 in itertools.combinations([r for r in readers if (r, p) in rd], 2):
                if r1 in real and r2 in real:
                    cat = "real_vs_real"
                elif r1 in sham and r2 in sham:
                    cat = "sham_vs_sham"
                elif sham_base in (r1, r2) and (r1 in sham or r2 in sham):
                    cat = "sham_vs_structural"
                else:
                    cat = "sham_vs_other_real"
                comparisons.append((cat, doc, r1, p, r2, p))
    print(f"Within-document comparisons: {len(comparisons)}")
    need = []
    for cat, doc, r1, p1, r2, p2 in comparisons:
        for a, b in candidate_pairs(by_doc[doc][(r1, p1)], by_doc[doc][(r2, p2)], emb, None):
            need += [(a, b), (b, a)]
    models.ensure(need)
    rp_rows = []
    for cat, doc, r1, p1, r2, p2 in comparisons:
        s, cos, conflict, ent, con = compare(by_doc[doc][(r1, p1)], by_doc[doc][(r2, p2)], models, emb)
        rp_rows.append({"category": cat, "doc": doc, "corpus": corpus_of[doc], "reader_a": r1, "rep_a": p1,
                        "reader_b": r2, "rep_b": p2, "S": round(s, 4), "D": round(1 - s, 4),
                        "cosine": round(cos, 4), "conflict_ratio": conflict, "n_entail": ent, "n_contra": con})
    with open(out / "reading_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rp_rows[0].keys()))
        w.writeheader()
        w.writerows(rp_rows)

    # ---- document pairs (corpus coherence) ----------------------------------
    reps_all = sorted({p for (_, _, p) in readings})
    doc_claims = defaultdict(list)  # (doc, rep) -> pooled real-reader claims
    for (reader, doc, rep), cl in readings.items():
        if reader in real:
            doc_claims[(doc, rep)] += cl
    dp_jobs = []
    for group, corpora in GROUPS.items():
        docs = sorted({d for (d, _) in doc_claims if corpus_of[d] in corpora})
        for rep in reps_all:
            for d1, d2 in itertools.combinations(docs, 2):
                if (d1, rep) in doc_claims and (d2, rep) in doc_claims:
                    dp_jobs.append((group, rep, d1, d2))
    print(f"Document-pair comparisons: {len(dp_jobs)} (k={args.k})")
    need = []
    for group, rep, d1, d2 in dp_jobs:
        for a, b in candidate_pairs(doc_claims[(d1, rep)], doc_claims[(d2, rep)], emb, args.k):
            need += [(a, b), (b, a)]
    models.ensure(need)
    dp_rows = []
    for group, rep, d1, d2 in dp_jobs:
        s, cos, conflict, ent, con = compare(doc_claims[(d1, rep)], doc_claims[(d2, rep)], models, emb, args.k)
        c1, c2 = corpus_of[d1], corpus_of[d2]
        kind = "+".join(sorted([c1, c2]))
        r1, r2 = refs.get(d1, set()), refs.get(d2, set())
        coupling = len(r1 & r2) / len(r1 | r2) if (r1 | r2) else float("nan")
        dp_rows.append({"group": group, "rep": rep, "doc_a": d1, "doc_b": d2, "pair_kind": kind,
                        "S": round(s, 4), "cosine": round(cos, 4), "conflict_ratio": conflict,
                        "n_entail": ent, "n_contra": con, "coupling": coupling})
    with open(out / "doc_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dp_rows[0].keys()))
        w.writeheader()
        w.writerows(dp_rows)

    # ---- summaries ----------------------------------------------------------
    summary = {"run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               "runs": args.runs, "embed_model": args.embed_model, "nli_model": args.nli_model, "k": args.k,
               "n_readings": len(readings), "n_unique_claims": len(texts)}
    lines = ["# DCR measures\n", f"Runs: `{args.runs}` | NLI: `{args.nli_model}` | embeddings: `{args.embed_model}`\n"]

    # 1-3: reading-level
    lines.append("## Reader agreement within documents (NLI agreement S; dispersion D = 1 - S)\n")
    lines.append("| Comparison | n | S mean [95% CI, docs resampled] | cosine | conflict ratio |")
    lines.append("|---|---|---|---|---|")
    rl = {}
    for cat in ["floor_real", "floor_sham", "real_vs_real", "sham_vs_sham", "sham_vs_structural", "sham_vs_other_real"]:
        rows = [r for r in rp_rows if r["category"] == cat]
        if not rows:
            continue
        m, lo, hi = cluster_boot(rows, "S")
        cosm = np.mean([r["cosine"] for r in rows])
        cr = [r["conflict_ratio"] for r in rows if not math.isnan(r["conflict_ratio"])]
        rl[cat] = {"n": len(rows), "S": m, "S_lo": lo, "S_hi": hi, "cosine": float(cosm),
                   "conflict_ratio": float(np.mean(cr)) if cr else None}
        lines.append(f"| {cat} | {len(rows)} | {fmt(m, lo, hi)} | {cosm:.3f} | "
                     f"{np.mean(cr):.3f} |" if cr else f"| {cat} | {len(rows)} | {fmt(m, lo, hi)} | {cosm:.3f} | n/a |")
    if "floor_real" in rl and "real_vs_real" in rl:
        floor_d, real_d = 1 - rl["floor_real"]["S"], 1 - rl["real_vs_real"]["S"]
        ratio = real_d / floor_d if floor_d > 0 else float("inf")
        rl["dispersion_ratio_real"] = ratio
        lines.append(f"\n**Real readers disperse {ratio:.2f}x the noise floor** "
                     f"(D = {real_d:.3f} vs {floor_d:.3f}).")
    if "floor_real" in rl and "sham_vs_sham" in rl:
        ratio_s = (1 - rl["sham_vs_sham"]["S"]) / (1 - rl["floor_real"]["S"])
        rl["dispersion_ratio_sham"] = ratio_s
        lines.append(f"**Sham readers disperse {ratio_s:.2f}x the noise floor.** "
                     "If epistemic distance drives divergence, this should be near 1 and well below the real ratio.")
    # per real-reader pair
    lines.append("\n| Real reader pair | S mean [95% CI] |")
    lines.append("|---|---|")
    for r1, r2 in itertools.combinations(sorted(real), 2):
        rows = [r for r in rp_rows if r["category"] == "real_vs_real" and {r["reader_a"], r["reader_b"]} == {r1, r2}]
        if rows:
            m, lo, hi = cluster_boot(rows, "S")
            rl[f"pair_{r1}_{r2}"] = m
            lines.append(f"| {r1} vs {r2} | {fmt(m, lo, hi)} |")
    # by corpus
    lines.append("\n| Corpus | noise floor S | real vs real S |")
    lines.append("|---|---|---|")
    for c in ("legitimacy", "tce", "splice"):
        f_ = [r for r in rp_rows if r["category"] == "floor_real" and r["corpus"] == c]
        x_ = [r for r in rp_rows if r["category"] == "real_vs_real" and r["corpus"] == c]
        if f_ and x_:
            fm, xm = cluster_boot(f_, "S")[0], cluster_boot(x_, "S")[0]
            rl[f"{c}_floor"], rl[f"{c}_real"] = fm, xm
            lines.append(f"| {c} | {fm:.3f} | {xm:.3f} |")
    summary["reading_level"] = rl

    # 4: corpus coherence
    lines.append("\n## Corpus coherence: agreement between documents (real readers pooled)\n")
    lines.append("Each value is the mean over document pairs; spread is across the 5 repetitions.\n")
    lines.append("| Condition | doc pairs | S mean (sd over reps) | cosine | conflict ratio |")
    lines.append("|---|---|---|---|---|")
    cc = {}
    conds = {"legitimacy": lambda r: r["group"] == "legitimacy",
             "tce": lambda r: r["group"] == "tce_all" and r["pair_kind"] == "tce+tce",
             "tce_spliced": lambda r: r["group"] == "tce_all",
             "spliced pairs only (tce+splice)": lambda r: r["group"] == "tce_all" and r["pair_kind"] == "splice+tce",
             "splice+splice": lambda r: r["group"] == "tce_all" and r["pair_kind"] == "splice+splice"}
    for name, pick in conds.items():
        rows = [r for r in dp_rows if pick(r)]
        if not rows:
            continue
        per_rep = [np.mean([r["S"] for r in rows if r["rep"] == p]) for p in reps_all
                   if any(r["rep"] == p for r in rows)]
        cr = [r["conflict_ratio"] for r in rows if not math.isnan(r["conflict_ratio"])]
        cc[name] = {"pairs_per_rep": len(rows) // max(1, len(per_rep)), "S": float(np.mean(per_rep)),
                    "S_sd_reps": float(np.std(per_rep)), "cosine": float(np.mean([r["cosine"] for r in rows])),
                    "conflict_ratio": float(np.mean(cr)) if cr else None}
        lines.append(f"| {name} | {cc[name]['pairs_per_rep']} | {cc[name]['S']:.3f} ({cc[name]['S_sd_reps']:.3f}) | "
                     f"{cc[name]['cosine']:.3f} | " + (f"{cc[name]['conflict_ratio']:.3f} |" if cr else "n/a |"))
    if "tce" in cc and "tce_spliced" in cc:
        lines.append(f"\n**Manipulation check:** coherence {cc['tce']['S']:.3f} (TCE) -> {cc['tce_spliced']['S']:.3f} "
                     f"(spliced). It should drop.")
    summary["corpus_coherence"] = cc

    # 5: stance
    lines.append("\n## Stance (genealogical reader; tracked separately from characterization)\n")
    lines.append("| Corpus | documents | stance distribution (modal label per document) |")
    lines.append("|---|---|---|")
    st_sum = {}
    for c in ("legitimacy", "tce", "splice"):
        modal = {}
        for doc in {d for (_, d, _) in stances if corpus_of.get(d) == c}:
            labs = [s.get("label", "unclear") for (r, d, _), s in stances.items() if d == doc and r == "genealogical"]
            if labs:
                modal[doc] = Counter(labs).most_common(1)[0][0]
        dist = Counter(modal.values())
        st_sum[c] = dict(dist)
        lines.append(f"| {c} | {len(modal)} | " + ", ".join(f"{k} {v}" for k, v in dist.most_common()) + " |")
    summary["stance"] = st_sum

    # 6: editors
    ed_rows = []
    for cid, rec in edits.items():
        parts = cid.split("__")
        p = rec["parsed"]
        ov = p.get("overall", {}) or {}
        drivers = Counter(d.get("driver", "?") for d in p.get("contested", []) or [])
        ed_rows.append({"custom_id": cid, "editor": parts[1], "condition": parts[2], "rep": parts[3],
                        "floor_run": parts[4] if len(parts) > 4 else "", "judgment": ov.get("judgment"),
                        "confidence": ov.get("confidence"), "n_positions": len(p.get("positions", []) or []),
                        "n_stable": len(p.get("stable", []) or []), "n_contested": len(p.get("contested", []) or []),
                        **{f"driver_{k}": drivers.get(k, 0) for k in
                           ("substantive", "terminological", "method", "scope", "unclear")}})
    ed_sum = {}
    if ed_rows:
        with open(out / "editors.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(ed_rows[0].keys()))
            w.writeheader()
            w.writerows(ed_rows)
        lines.append("\n## Editors\n")
        lines.append("| Editor | Condition | judgments across reader repetitions | positions mean (sd) | contested drivers |")
        lines.append("|---|---|---|---|---|")
        grouped = defaultdict(list)
        for r in ed_rows:
            if not r["floor_run"]:
                grouped[(r["editor"], r["condition"])].append(r)
        for (e, c), rows in sorted(grouped.items()):
            j = Counter(r["judgment"] for r in rows)
            pos = [r["n_positions"] for r in rows]
            drv = Counter()
            for r in rows:
                for k in ("substantive", "terminological", "method", "scope", "unclear"):
                    drv[k] += r[f"driver_{k}"]
            ed_sum[f"{e}|{c}"] = {"judgments": dict(j), "positions_mean": float(np.mean(pos)),
                                  "positions_sd": float(np.std(pos)), "drivers": dict(drv)}
            lines.append(f"| {e} | {c} | " + ", ".join(f"{k} {v}" for k, v in j.most_common()) +
                         f" | {np.mean(pos):.1f} ({np.std(pos):.1f}) | " +
                         ", ".join(f"{k} {v}" for k, v in drv.most_common() if v) + " |")
        lines.append("\n| Editor noise floor (fixed input) | runs | judgment agreement | positions (sd) |")
        lines.append("|---|---|---|---|")
        for e in sorted({r["editor"] for r in ed_rows}):
            for c in sorted({r["condition"] for r in ed_rows}):
                rows = [r for r in ed_rows if r["editor"] == e and r["condition"] == c and r["rep"] == "r1"]
                if len(rows) < 2 or not any(r["floor_run"] for r in rows):
                    continue
                j = Counter(r["judgment"] for r in rows)
                agree = j.most_common(1)[0][1] / len(rows)
                pos = [r["n_positions"] for r in rows]
                ed_sum[f"floor|{e}|{c}"] = {"runs": len(rows), "agreement": agree, "positions_sd": float(np.std(pos))}
                lines.append(f"| {e} / {c} | {len(rows)} | {agree:.0%} ({j.most_common(1)[0][0]}) | {np.std(pos):.2f} |")
    summary["editors"] = ed_sum

    # 7: co-citation baseline
    lines.append("\n## Co-citation baseline (bibliographic coupling vs NLI agreement, document pairs)\n")
    lines.append("| Group | doc pairs | Spearman rho | mean coupling |")
    lines.append("|---|---|---|---|")
    cb = {}
    for group in GROUPS:
        agg = defaultdict(list)
        coup = {}
        for r in dp_rows:
            if r["group"] == group and not math.isnan(r["coupling"]):
                agg[(r["doc_a"], r["doc_b"])].append(r["S"])
                coup[(r["doc_a"], r["doc_b"])] = r["coupling"]
        keys = list(agg)
        rho = spearman([coup[k] for k in keys], [np.mean(agg[k]) for k in keys])
        cb[group] = {"pairs": len(keys), "spearman": rho, "mean_coupling": float(np.mean(list(coup.values()))) if coup else None}
        lines.append(f"| {group} | {len(keys)} | {rho:.3f} | {cb[group]['mean_coupling']:.3f} |" if coup else f"| {group} | 0 | n/a | n/a |")
    lines.append("\nA high correlation would mean interpretive agreement adds little beyond citation structure.")
    summary["cocitation"] = cb

    (out / "measures_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}/report.md, measures_summary.json, reading_pairs.csv, doc_pairs.csv, editors.csv")


if __name__ == "__main__":
    sys.exit(main())
