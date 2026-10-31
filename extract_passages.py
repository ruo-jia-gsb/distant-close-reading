#!/usr/bin/env python3
"""
extract_passages.py - Turn corpus PDFs into text and select passages under a fixed, documented rule.

Rule (frozen before any reading runs; recorded in corpus/manifest_extraction.json):
  1. Extract text with PyMuPDF, block by block; join hyphenated line breaks; drop blocks under
     MIN_WORDS words (headers, footers, captions, page numbers).
  2. Drop everything from the first "References" / "Bibliography" heading in the second half onward.
  3. A paragraph is a candidate if it matches at least one construct term for its corpus.
     Splice documents use the TCE terms, because they are injected into the TCE corpus.
  4. Rank candidates by number of term matches (ties: earlier position first) and keep the top
     MAX_PASSAGES per document, so long works (e.g., book excerpts) cannot dominate.

Sources: corpus/pdfs/<id>.pdf, or corpus/pdfs/<id>.txt where only a text version exists.

Book-length anchors are trimmed to the preface/introduction plus Chapter 1 (fixed rule) using the
marker phrases in BOOK_EXCERPTS: the excerpt runs from the first match of "start" up to the first
match of "end" after it. Markers are matched case-insensitively, ignoring line breaks and spacing.

Outputs:
  corpus/text/<id>.txt             full extracted text (local only; copyrighted; gitignored)
  corpus/passages.jsonl            passages with text (local only; gitignored) - input to the readers
  corpus/passage_index.csv         passage IDs, pages, character offsets, hashes (safe to commit)
  corpus/manifest_extraction.json  rule, terms, versions, per-document text hashes and counts (commit)

Anyone with the same PDFs and this script regenerates identical text; the text hashes let them check.

Usage:
  pip install pymupdf
  python extract_passages.py
"""

import csv
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf  # older releases
    except ImportError:
        sys.exit("pip install pymupdf")

CORPUS_DIR = Path("corpus")
PDF_DIR = CORPUS_DIR / "pdfs"
TEXT_DIR = CORPUS_DIR / "text"
CORPORA = ["legitimacy", "tce", "splice"]

MIN_WORDS = 40
MAX_PASSAGES = 8
LOW_TEXT_CHARS = 2000  # below this, the PDF is probably a scan without a text layer

TERMS = {
    "legitimacy": [r"legitim\w*"],
    "tce": [r"transaction[- ]cost\w*", r"asset specificit\w*", r"specific assets?", r"opportunis\w*",
            r"bounded(ly)? rational\w*", r"governance", r"hierarch\w*", r"make[- ]or[- ]buy",
            r"vertical(ly)? integrat\w*", r"hybrid\w*"],
}
TERMS["splice"] = TERMS["tce"]

# Book excerpt markers. "start": a distinctive phrase opening the preface or introduction.
# "end": the opening phrase of Chapter 2 (its first sentence is safest; a bare "Chapter 2" can match
# the table of contents or a cross-reference). Leave both empty only if the file is already an excerpt.
BOOK_EXCERPTS = {
    "W1973308529": {"title": "Markets and Hierarchies (1975)",
                    "start": "This book is concerned with the organization of economic activity",
                    "end": "A more complete statement of the organizational failures framework",
                    "already_excerpt": False},
    "W1684540946": {"title": "The Economic Institutions of Capitalism (1985)",
                    "start": "What is referred to herein as transaction cost economics",
                    "end": "Complex systems are usefully studied from several points of view",
                    "already_excerpt": False},
}

PAGE_MARKER = re.compile(r"=+\s*PAGE\s+0*(\d+)\s*(?:\([^)]*\))?\s*=+", re.I)

REF_HEADING = re.compile(r"^\s*(references|bibliography|literature cited|works cited)\s*$", re.I)


def clean_block(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # join hyphenated line breaks
    text = re.sub(r"\s*\n\s*", " ", text)          # lines within a block -> spaces
    return re.sub(r"[ \t]+", " ", text).strip()


def extract(pdf_path):
    """Return (full_text, blocks) where blocks = [(page, start, end, text)] with offsets into full_text."""
    parts, blocks, pos = [], [], 0
    with pymupdf.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            for b in page.get_text("blocks", sort=True):
                if b[6] != 0:  # skip image blocks
                    continue
                t = clean_block(b[4])
                if not t:
                    continue
                if parts:
                    parts.append("\n\n")
                    pos += 2
                blocks.append((pno, pos, pos + len(t), t))
                parts.append(t)
                pos += len(t)
    return "".join(parts), blocks


def extract_txt(txt_path):
    """Plain-text source: paragraphs split on blank lines (or on single lines if the file has none)."""
    raw = txt_path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    paras = re.split(r"\n\s*\n", raw)
    if len(paras) < 5 or max(len(x.split()) for x in paras) > 1500:
        paras = raw.split("\n")  # no blank-line paragraphing; treat each line as a paragraph
    parts, blocks, pos, page = [], [], 0, 0
    for para in paras:
        t = clean_block(para)
        # Scanned-book text files (e.g., Internet Archive) mark pages as "===== PAGE 00000011 (...) =====".
        # Use the marker as the page number and strip it from the text.
        m = PAGE_MARKER.search(t)
        if m:
            page = int(m.group(1))
            t = PAGE_MARKER.sub(" ", t).strip()
        if not t:
            continue
        if parts:
            parts.append("\n\n")
            pos += 2
        blocks.append((page, pos, pos + len(t), t))  # page 0 = unknown
        parts.append(t)
        pos += len(t)
    return "".join(parts), blocks


def phrase_regex(phrase):
    words = re.findall(r"\w+", phrase.lower())
    return re.compile(r"\W+".join(map(re.escape, words)), re.I) if words else None


def trim_book(oid, full, blocks):
    """Return (blocks within the excerpt, problem message or None)."""
    cfg = BOOK_EXCERPTS[oid]
    if cfg.get("already_excerpt"):
        return blocks, None
    start_re, end_re = phrase_regex(cfg["start"]), phrase_regex(cfg["end"])
    if not start_re or not end_re:
        return [], "BOOK NOT TRIMMED: set start/end markers in BOOK_EXCERPTS (or already_excerpt: True)"
    m1 = start_re.search(full)
    if not m1:
        return [], "BOOK START MARKER NOT FOUND: check the phrase against the text"
    m2 = end_re.search(full, m1.end())
    if not m2:
        return [], "BOOK END MARKER NOT FOUND after the start marker"
    # Keep blocks that reach into the excerpt and end before Chapter 2 begins; a block containing
    # the end marker (e.g., the Chapter 2 heading merged with its first sentence) is excluded.
    kept = [b for b in blocks if b[2] > m1.start() and b[2] <= m2.start()]
    return kept, None


def body_blocks(blocks):
    """Drop the reference list: everything from the first reference heading in the second half."""
    half = len(blocks) // 2
    for i, (_, _, _, t) in enumerate(blocks):
        if i >= half and REF_HEADING.match(t):
            return blocks[:i]
    return blocks


def load_rows():
    rows = []
    for name in CORPORA:
        with open(CORPUS_DIR / f"{name}_selected.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["corpus"] = name
                rows.append(r)
    return rows


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    patterns = {c: re.compile("|".join(TERMS[c]), re.I) for c in CORPORA}

    index_rows, passages, docs, problems = [], [], {}, []
    for r in rows:
        oid, corpus = r["openalex_id"], r["corpus"]
        pdf, txt = PDF_DIR / f"{oid}.pdf", PDF_DIR / f"{oid}.txt"
        if pdf.exists():
            full, blocks = extract(pdf)
            source = "pdf"
        elif txt.exists():
            full, blocks = extract_txt(txt)
            source = "txt"
        else:
            problems.append(f"MISSING  {corpus:<10} {oid}  {r['title'][:60]}")
            continue
        (TEXT_DIR / f"{oid}.txt").write_text(full, encoding="utf-8")
        if oid in BOOK_EXCERPTS:
            blocks, why = trim_book(oid, full, blocks)
            if why:
                problems.append(f"{why}  ({oid}, {BOOK_EXCERPTS[oid]['title']})")
            body = blocks  # excerpt ends before any reference list
        else:
            body = body_blocks(blocks)

        cands = []
        for order, (page, start, end, t) in enumerate(body):
            if len(t.split()) < MIN_WORDS:
                continue
            hits = len(patterns[corpus].findall(t))
            if hits:
                cands.append((hits, order, page, start, end, t))
        chosen = sorted(cands, key=lambda c: (-c[0], c[1]))[:MAX_PASSAGES]
        chosen.sort(key=lambda c: c[1])  # restore reading order

        for k, (hits, order, page, start, end, t) in enumerate(chosen, start=1):
            pid = f"{oid}_p{k:02d}"
            index_rows.append({
                "passage_id": pid, "corpus": corpus, "openalex_id": oid, "page": page,
                "char_start": start, "char_end": end, "n_words": len(t.split()), "term_hits": hits,
                "text_sha256": hashlib.sha256(t.encode("utf-8")).hexdigest()[:16],
            })
            passages.append({"passage_id": pid, "corpus": corpus, "openalex_id": oid, "page": page,
                             "char_start": start, "char_end": end, "text": t})

        docs[oid] = {
            "corpus": corpus, "source": source, "chars": len(full), "blocks": len(blocks), "body_blocks": len(body),
            "candidates": len(cands), "selected": len(chosen),
            "text_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
        }
        if oid in BOOK_EXCERPTS:
            docs[oid]["excerpt_chars"] = sum(b[2] - b[1] for b in body)
            docs[oid]["excerpt_markers"] = {k: BOOK_EXCERPTS[oid][k] for k in ("start", "end", "already_excerpt")}
        if len(full) < LOW_TEXT_CHARS:
            problems.append(f"NO TEXT  {corpus:<10} {oid}  ({len(full)} chars; likely a scan, needs OCR)")
        elif not chosen and oid not in BOOK_EXCERPTS:
            problems.append(f"NO HITS  {corpus:<10} {oid}  (no paragraph matched the construct terms)")

    with open(CORPUS_DIR / "passage_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()) if index_rows else ["passage_id"])
        w.writeheader()
        w.writerows(index_rows)
    with open(CORPUS_DIR / "passages.jsonl", "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    manifest = {
        "run_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pymupdf_version": getattr(pymupdf, "VersionBind", getattr(pymupdf, "__version__", "unknown")),
        "rule": {"min_words": MIN_WORDS, "max_passages_per_doc": MAX_PASSAGES, "terms": TERMS,
                 "reference_heading": REF_HEADING.pattern,
                 "book_rule": "Book-length anchors: preface/introduction plus Chapter 1 only.",
                 "book_excerpts": BOOK_EXCERPTS},
        "documents": docs,
    }
    (CORPUS_DIR / "manifest_extraction.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Passages per corpus:")
    for c in CORPORA:
        n_docs = sum(1 for d in docs.values() if d["corpus"] == c)
        n_pass = sum(1 for p in index_rows if p["corpus"] == c)
        print(f"  {c:<11} {n_pass:>4} passages from {n_docs} documents")
    if problems:
        print("\nCheck these:")
        for p in problems:
            print("  " + p)
    print("\nCommit passage_index.csv and manifest_extraction.json. Never commit text/ or passages.jsonl.")


if __name__ == "__main__":
    sys.exit(main())