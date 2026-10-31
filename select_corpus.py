#!/usr/bin/env python3
"""
select_corpus.py - Mechanical, reproducible corpus selection for Distant Close Reading (DCR).

Selects papers from OpenAlex using a fixed rule, written down before results are seen:

    A paper is a candidate if it
      (1) cites at least one anchor work for its corpus,
      (2) was published as an article in one of six target journals,
      (3) contains at least one include-term in its title or abstract, and
      (4) contains none of the exclude-terms.
    Candidates are ranked by citation count (total or per-year; chosen before running),
    and the top N are selected. Anchor works are added where specified.

Outputs (in --outdir):
    <corpus>_candidates.csv    every paper passing the rule, ranked (for transparency)
    <corpus>_selected.csv      the frozen corpus (top N + anchors)
    <corpus>_references.jsonl  reference lists of selected papers (for the co-citation baseline)
    manifest.json              rule text, parameters, anchor resolutions, counts, script hash, run date

Usage:
    pip install requests
    python select_corpus.py --api-key YOUR_KEY --anchors-only    # step 1: verify anchors
    python select_corpus.py --api-key YOUR_KEY                   # step 2: build all corpora

A free OpenAlex API key (openalex.org/settings/api) is required for a full build; keyless use has a very
small daily budget. The key and any --email are sent only to OpenAlex and are never written to outputs.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

API = "https://api.openalex.org"

# ---------------------------------------------------------------------------
# Configuration. Edit this BEFORE the first full run, then commit and freeze it.
# ---------------------------------------------------------------------------

JOURNAL_ISSNS = {
    "Academy of Management Review": "0363-7425",
    "Academy of Management Journal": "0001-4273",
    "Administrative Science Quarterly": "0001-8392",
    "Organization Science": "1047-7039",
    "Strategic Management Journal": "0143-2095",
    "Journal of Management Studies": "0022-2380",
}

# Anchors resolve by OpenAlex ID if given, else by DOI, else by title search within +/-1 year.
# After running --anchors-only, paste confirmed OpenAlex IDs into "openalex_id" to lock them.
# "extra_ids": duplicate OpenAlex records of the same work (common for classic books). They are
# used only to find citing papers; the anchor itself is added to the corpus once.
# "in_corpus": False keeps an anchor as a citation-lookup anchor only (e.g., book-length works).
# "match_fields": "title" (term must appear in the title) or "title_abstract".
#
# RULE VERSION LOG
#   v1 (2026-10-21): match_fields = title_abstract; all anchors in corpus.
#   v2 (2026-10-21, considered and NOT adopted): title-only matching; Williamson books lookup-only.
#       Prompted by the v1 TCE selection being dominated by critics and rival theories.
#       Rejected before any reading or stability results, because contestation is not incoherence:
#       critics who characterize TCE the same way while disputing it are evidence of a coherent
#       literature, and invoking papers show how a construct is used. DCR therefore separates a
#       paper's characterization of a construct (on which stability is computed) from its stance
#       toward it (tracked separately). Book-length anchors stay in the corpus; per-document passage
#       caps in the extraction step prevent long works from dominating.
#   Frozen corpus = v1 output, committed 2026-10-21.
RULE_VERSION = "v1"
CORPORA = {
    "legitimacy": {
        "description": "Organizational legitimacy (institutional theory)",
        "anchors": [
            {"label": "Suchman 1995", "openalex_id": "W2162412465"},
            {"label": "Deephouse & Suchman 2008", "openalex_id": "W2496218594"},
            {"label": "Suddaby, Bitektine & Haack 2017", "openalex_id": "W4240547422"},
        ],
        "include_terms": ["legitimacy"],
        "exclude_terms": [],
        "match_fields": "title_abstract",
        "n_select": 20,
        "add_anchors": True,
    },
    "tce": {
        "description": "Transaction cost economics (coherence contrast)",
        "anchors": [
            # OpenAlex types these books as "article" (legacy metadata); IDs confirmed by citation counts.
            {"label": "Williamson 1975", "openalex_id": "W1973308529", "extra_ids": []},
            {"label": "Williamson 1981", "openalex_id": "W1972698480"},
            {"label": "Williamson 1985", "openalex_id": "W1684540946", "extra_ids": []},
            {"label": "Williamson 1991", "openalex_id": "W2109280325"},
        ],
        "include_terms": ["transaction cost", "make-or-buy", "make or buy"],
        "exclude_terms": [],
        "match_fields": "title_abstract",
        "n_select": 15,
        "add_anchors": True,
    },
    "splice": {
        "description": "Splice set: 'governance' in the corporate-governance / agency sense (manipulation check)",
        "anchors": [
            {"label": "Jensen & Meckling 1976", "openalex_id": "W2752617332"},
        ],
        "include_terms": ["corporate governance", "board of directors", "boards of directors"],
        "exclude_terms": ["transaction cost"],  # keeps the different sense of 'governance' clean
        "match_fields": "title_abstract",
        "n_select": 5,
        "add_anchors": False,
    },
}

SELECT_FIELDS = ",".join([
    "id", "doi", "display_name", "publication_year", "type", "cited_by_count",
    "primary_location", "authorships", "abstract_inverted_index", "referenced_works",
])

# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------


API_KEY = None  # set in main() from --api-key or the OPENALEX_API_KEY environment variable


def api_get(session, path, params, email, retries=6):
    params = dict(params)
    if email:
        params["mailto"] = email
    if API_KEY:
        params["api_key"] = API_KEY
    for attempt in range(retries):
        resp = session.get(f"{API}{path}", params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 401:
            sys.exit("\nOpenAlex returned 401 Unauthorized. Keyless use now has a very small daily budget.\n"
                     "Get a free API key at https://openalex.org/settings/api, then rerun with\n"
                     "  --api-key YOUR_KEY   (or set the OPENALEX_API_KEY environment variable).\n"
                     "If you already used a key, check it is correct or wait for the daily budget to reset.")
        if resp.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt, 30))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"OpenAlex request failed after {retries} retries: {path} {params}")


def short_id(openalex_id):
    return openalex_id.rsplit("/", 1)[-1] if openalex_id else ""


def resolve_anchor(session, anchor, email):
    """Return (work, method). Order of preference: openalex_id, doi, title search."""
    if anchor.get("openalex_id"):
        return api_get(session, f"/works/{anchor['openalex_id']}", {}, email), "openalex_id"
    if anchor.get("doi"):
        try:
            return api_get(session, f"/works/doi:{anchor['doi']}", {}, email), "doi"
        except requests.HTTPError:
            pass
    year = anchor["year"]
    data = api_get(session, "/works", {
        "search": anchor["search"],
        "filter": f"from_publication_date:{year - 1}-01-01,to_publication_date:{year + 1}-12-31",
        "sort": "cited_by_count:desc",
        "per-page": 5,
    }, email)
    results = data.get("results", [])
    return (results[0], "title_search") if results else (None, "not_found")


def fetch_citing(session, anchor_id, email, max_results):
    """All articles in the target journals citing anchor_id, highest-cited first."""
    filt = ",".join([
        f"cites:{short_id(anchor_id)}",
        "primary_location.source.issn:" + "|".join(JOURNAL_ISSNS.values()),
        "type:article",
        "is_retracted:false",
    ])
    out, cursor = [], "*"
    while cursor and len(out) < max_results:
        data = api_get(session, "/works", {
            "filter": filt, "sort": "cited_by_count:desc",
            "per-page": 200, "cursor": cursor, "select": SELECT_FIELDS,
        }, email)
        batch = data.get("results", [])
        if not batch:
            break
        out.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(0.1)
    return out[:max_results]


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def term_match(work, include_terms, exclude_terms, match_fields="title_abstract"):
    """Return 'title', 'abstract', or None (fails rule). Exclude-terms are checked in title and abstract."""
    title = (work.get("display_name") or "").lower()
    abstract = reconstruct_abstract(work.get("abstract_inverted_index")).lower()
    full = f"{title} {abstract}"
    if any(t.lower() in full for t in exclude_terms):
        return None
    if any(t.lower() in title for t in include_terms):
        return "title"
    if match_fields == "title":
        return None
    if any(t.lower() in abstract for t in include_terms):
        return "abstract"
    return None


def score(work, rank_by, current_year):
    cites = work.get("cited_by_count") or 0
    if rank_by == "per_year":
        age = max(1, current_year - (work.get("publication_year") or current_year) + 1)
        return cites / age
    return cites


def journal_name(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name", "")


def authors(work, k=3):
    names = [a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])]
    return "; ".join(names[:k]) + ("; et al." if len(names) > k else "")


def rule_text(name, cfg, rank_by):
    anchors = ", ".join(a["label"] for a in cfg["anchors"])
    inc = " OR ".join(f'"{t}"' for t in cfg["include_terms"])
    exc = (" and none of " + " OR ".join(f'"{t}"' for t in cfg["exclude_terms"])) if cfg["exclude_terms"] else ""
    rank = "citations per year since publication" if rank_by == "per_year" else "total citation count"
    where = "the title" if cfg.get("match_fields", "title_abstract") == "title" else "the title or abstract"
    in_corpus = [a["label"] for a in cfg["anchors"] if a.get("in_corpus", True)]
    lookup_only = [a["label"] for a in cfg["anchors"] if not a.get("in_corpus", True)]
    add = ""
    if cfg["add_anchors"] and in_corpus:
        add = f" Anchor works ({', '.join(in_corpus)}) are added to the corpus."
        if lookup_only:
            add += f" {', '.join(lookup_only)} (book-length) serve as citation anchors only."
    return (f"[{name}] ({RULE_VERSION}) Articles in {', '.join(JOURNAL_ISSNS)} that cite {anchors}, contain "
            f"{inc} in {where}{exc}, ranked by {rank} in OpenAlex; the top {cfg['n_select']} are "
            f"selected.{add}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ROW_FIELDS = ["corpus", "rank", "selected", "is_anchor", "openalex_id", "doi", "title", "authors",
              "year", "journal", "cited_by_count", "score", "term_matched_in", "via_anchors"]


def row(corpus, rank, selected, is_anchor, work, sc, matched, via):
    return {
        "corpus": corpus, "rank": rank, "selected": selected, "is_anchor": is_anchor,
        "openalex_id": short_id(work.get("id")), "doi": work.get("doi") or "",
        "title": work.get("display_name") or "", "authors": authors(work),
        "year": work.get("publication_year"), "journal": journal_name(work),
        "cited_by_count": work.get("cited_by_count"), "score": round(sc, 2),
        "term_matched_in": matched, "via_anchors": "; ".join(via),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)


def build_corpus(session, name, cfg, args, outdir, manifest):
    print(f"\n=== {name}: {cfg['description']} ===")
    current_year = datetime.date.today().year
    anchor_works, anchor_log = [], []
    lookup_only_labels = set()
    for a in cfg["anchors"]:
        work, method = resolve_anchor(session, a, args.email)
        entry = {"label": a["label"], "method": method,
                 "openalex_id": short_id(work.get("id")) if work else None,
                 "matched_title": work.get("display_name") if work else None,
                 "year": work.get("publication_year") if work else None,
                 "cited_by_count": work.get("cited_by_count") if work else None}
        anchor_log.append(entry)
        flag = "" if method in ("openalex_id", "doi") else "   <-- VERIFY (title search)"
        print(f"  anchor {a['label']}: {entry['openalex_id']} | {entry['matched_title']} "
              f"({entry['year']}, {entry['cited_by_count']} cites) via {method}{flag}")
        if work:
            extras = ["https://openalex.org/" + short_id(x) for x in a.get("extra_ids", [])]
            if extras:
                print(f"    + {len(extras)} duplicate record(s) used for citing lookup: "
                      + ", ".join(short_id(x) for x in extras))
            entry["extra_ids"] = [short_id(x) for x in extras]
            anchor_works.append((a["label"], work, extras))
            if not a.get("in_corpus", True):
                lookup_only_labels.add(a["label"])
    manifest["corpora"][name] = {"rule": rule_text(name, cfg, args.rank_by), "anchors": anchor_log}
    if args.anchors_only:
        return

    candidates = {}
    for label, aw, extras in anchor_works:
        seen = set()
        for lookup_id in [aw["id"], *extras]:
            for w in fetch_citing(session, lookup_id, args.email, args.max_per_anchor):
                seen.add(w["id"])
                entry = candidates.setdefault(w["id"], {"work": w, "via": []})
                if label not in entry["via"]:
                    entry["via"].append(label)
        print(f"  {label}: {len(seen)} citing articles in target journals")

    anchor_ids = {aw["id"] for _, aw, _ in anchor_works}
    anchor_ids |= {x for _, _, extras in anchor_works for x in extras}
    passing = []
    for wid, entry in candidates.items():
        if wid in anchor_ids:
            continue
        matched = term_match(entry["work"], cfg["include_terms"], cfg["exclude_terms"],
                             cfg.get("match_fields", "title_abstract"))
        if matched:
            passing.append((score(entry["work"], args.rank_by, current_year), matched, entry))
    passing.sort(key=lambda x: (-x[0], x[2]["work"].get("publication_year") or 0))
    print(f"  {len(candidates)} unique citing articles; {len(passing)} pass term rule")

    cand_rows, selected_rows, selected_works = [], [], []
    for i, (sc, matched, entry) in enumerate(passing, start=1):
        sel = i <= cfg["n_select"]
        r = row(name, i, sel, False, entry["work"], sc, matched, entry["via"])
        cand_rows.append(r)
        if sel:
            selected_rows.append(r)
            selected_works.append(entry["work"])
    if cfg["add_anchors"]:
        for label, aw, _ in anchor_works:
            if label in lookup_only_labels:
                continue
            selected_rows.append(row(name, 0, True, True, aw, score(aw, args.rank_by, current_year),
                                     "anchor", [label]))
            selected_works.append(aw)

    write_csv(outdir / f"{name}_candidates.csv", cand_rows)
    write_csv(outdir / f"{name}_selected.csv", selected_rows)
    with open(outdir / f"{name}_references.jsonl", "w", encoding="utf-8") as f:
        for w in selected_works:
            f.write(json.dumps({"openalex_id": short_id(w["id"]),
                                "referenced_works": [short_id(x) for x in (w.get("referenced_works") or [])]}) + "\n")

    missing_abs = sum(1 for w in selected_works if not w.get("abstract_inverted_index"))
    manifest["corpora"][name].update({
        "unique_citing": len(candidates), "passing_rule": len(passing),
        "selected": len(selected_rows), "selected_missing_abstract": missing_abs,
    })
    print(f"  selected {len(selected_rows)} -> {name}_selected.csv"
          + (f"  ({missing_abs} lack abstracts in OpenAlex; matched on title only)" if missing_abs else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--email", default=None, help="Contact email sent to OpenAlex (optional; not saved).")
    p.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"),
                   help="OpenAlex API key (free at openalex.org/settings/api). Defaults to $OPENALEX_API_KEY. Not saved.")
    p.add_argument("--corpus", choices=["all", *CORPORA], default="all")
    p.add_argument("--rank-by", choices=["total", "per_year"], default="total",
                   help="Ranking rule. Choose BEFORE the first full run and do not change it afterward.")
    p.add_argument("--max-per-anchor", type=int, default=1000,
                   help="Max citing articles fetched per anchor (highest-cited first).")
    p.add_argument("--outdir", default="corpus")
    p.add_argument("--anchors-only", action="store_true", help="Only resolve and print anchors.")
    args = p.parse_args()
    global API_KEY
    API_KEY = args.api_key
    if not API_KEY:
        print("Warning: no OpenAlex API key; keyless budget is very small and the run may stop with a 401.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "source": "OpenAlex API", "rule_version": RULE_VERSION, "script_sha256": script_hash,
        "rank_by": args.rank_by, "max_per_anchor": args.max_per_anchor,
        "journals": JOURNAL_ISSNS, "anchors_only": args.anchors_only, "corpora": {},
    }

    session = requests.Session()
    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for name in names:
        build_corpus(session, name, CORPORA[name], args, outdir, manifest)

    mpath = outdir / ("manifest_anchors.json" if args.anchors_only else "manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {mpath}")
    print("\nInclusion rules (paste into the technical appendix):")
    for name in names:
        print("  " + manifest["corpora"][name]["rule"])


if __name__ == "__main__":
    sys.exit(main())
