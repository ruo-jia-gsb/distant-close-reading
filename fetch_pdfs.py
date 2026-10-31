#!/usr/bin/env python3
"""
fetch_pdfs.py - Gather the DCR corpus PDFs.

Step 1  python fetch_pdfs.py
        Tries legal open-access copies listed by OpenAlex for every paper not yet in corpus/pdfs/,
        then writes corpus/download_checklist.html (clickable DOI links) for the rest.

Step 2  Download the remaining PDFs through your library, with any filenames, into corpus/pdfs/inbox/.

Step 3  python fetch_pdfs.py --sort-inbox
        Matches each inbox PDF to the corpus by DOI or title on its first pages and renames it to
        <OpenAlexID>.pdf in corpus/pdfs/. Unmatched files are left in the inbox and listed.

        python fetch_pdfs.py --status      just report what is present and missing

Requires: pip install requests pymupdf
Uses the OPENALEX_API_KEY environment variable if set (recommended).
"""

import argparse
import csv
import html
import os
import re
import shutil
import sys
import time
from pathlib import Path

import requests

API = "https://api.openalex.org"
CORPORA = ["legitimacy", "tce", "splice"]

# Book-length anchors: save the preface/introduction plus Chapter 1 as the PDF (fixed rule).
BOOKS = {
    "W1973308529": "Williamson, Markets and Hierarchies (1975)",
    "W1684540946": "Williamson, The Economic Institutions of Capitalism (1985)",
}

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>,;]+", re.I)


def norm_doi(doi):
    doi = (doi or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi.rstrip(".)]")


def norm_text(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def load_corpus(corpus_dir):
    rows = []
    for name in CORPORA:
        path = corpus_dir / f"{name}_selected.csv"
        if not path.exists():
            sys.exit(f"Missing {path}. Run select_corpus.py first.")
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["corpus"] = name
                rows.append(r)
    return rows


def have(pdf_dir, oid):
    return (pdf_dir / f"{oid}.pdf").exists() or (pdf_dir / f"{oid}.txt").exists()


# ---------------------------------------------------------------------------
# Step 1: open-access download
# ---------------------------------------------------------------------------


def oa_pdf_urls(session, oid):
    params = {"select": "id,open_access,best_oa_location,locations"}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    resp = session.get(f"{API}/works/{oid}", params=params, timeout=60)
    if resp.status_code != 200:
        return []
    w = resp.json()
    urls = []
    best = w.get("best_oa_location") or {}
    if best.get("pdf_url"):
        urls.append(best["pdf_url"])
    for loc in w.get("locations") or []:
        u = (loc or {}).get("pdf_url")
        if u and u not in urls:
            urls.append(u)
    return urls


def try_download(session, url, target):
    try:
        resp = session.get(url, timeout=60, allow_redirects=True)
    except requests.RequestException:
        return False
    if resp.status_code == 200 and resp.content[:5] == b"%PDF-":
        target.write_bytes(resp.content)
        return True
    return False


def fetch_open_access(rows, pdf_dir):
    session = requests.Session()
    session.headers["User-Agent"] = "DCR-corpus-builder (academic research; open-access copies only)"
    got = 0
    for r in rows:
        oid = r["openalex_id"]
        if have(pdf_dir, oid) or oid in BOOKS:
            continue
        for url in oa_pdf_urls(session, oid):
            if try_download(session, url, pdf_dir / f"{oid}.pdf"):
                print(f"  OA  {oid}  {r['title'][:70]}")
                got += 1
                break
            time.sleep(1)
        time.sleep(0.2)
    print(f"Downloaded {got} open-access PDF(s).")


# ---------------------------------------------------------------------------
# Checklist
# ---------------------------------------------------------------------------


def write_checklist(rows, pdf_dir, corpus_dir):
    missing = [r for r in rows if not have(pdf_dir, r["openalex_id"])]
    with open(corpus_dir / "download_checklist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["corpus", "save_as", "title", "authors", "year", "journal", "doi_link", "note"])
        for r in missing:
            note = "BOOK: save preface/introduction + Chapter 1 only" if r["openalex_id"] in BOOKS else ""
            w.writerow([r["corpus"], f"{r['openalex_id']}.pdf", r["title"], r["authors"], r["year"],
                        r["journal"], r["doi"], note])

    trs = []
    for r in missing:
        oid = r["openalex_id"]
        if oid in BOOKS:
            link = f"<strong>Book:</strong> {html.escape(BOOKS[oid])}. Save preface/introduction + Chapter 1 only."
        elif r["doi"]:
            link = f'<a href="{html.escape(r["doi"])}" target="_blank">{html.escape(r["doi"])}</a>'
        else:
            link = "(no DOI; search by title)"
        trs.append(f"<tr><td>{r['corpus']}</td><td><code>{oid}.pdf</code></td>"
                   f"<td>{html.escape(r['title'])}<br><small>{html.escape(r['authors'])} "
                   f"({r['year']}), {html.escape(r['journal'])}</small></td><td>{link}</td></tr>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>DCR download checklist</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#ddd; --accent:#0b57d0; }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; --bg:#141414; --fg:#eee; --muted:#aaa; --line:#333; --accent:#8ab4f8; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; margin:2rem; }}
table {{ border-collapse:collapse; width:100%; }} td, th {{ border-bottom:1px solid var(--line); padding:.5rem; text-align:left; vertical-align:top; }}
small {{ color:var(--muted); }} a {{ color:var(--accent); }} code {{ font-size:13px; }}
</style></head><body>
<h1>DCR download checklist: {len(missing)} remaining</h1>
<p>Download each PDF through your library into <code>corpus/pdfs/inbox/</code> with any filename, then run
<code>python fetch_pdfs.py --sort-inbox</code>. The script renames files to the <em>Save as</em> names.
Rerun the script to refresh this list.</p>
<table><tr><th>Corpus</th><th>Save as</th><th>Paper</th><th>Link</th></tr>
{''.join(trs)}
</table></body></html>"""
    (corpus_dir / "download_checklist.html").write_text(page, encoding="utf-8")
    return missing


# ---------------------------------------------------------------------------
# Step 3: sort inbox
# ---------------------------------------------------------------------------


def _pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz  # older pymupdf releases
        return fitz


def first_pages_text(path, pages=2):
    with _pymupdf().open(path) as doc:
        return " ".join(doc[i].get_text() for i in range(min(pages, len(doc))))


def match_pdf(text, rows):
    """Return the matching row, or None. DOI first; title to break ties or as fallback."""
    found = {norm_doi(d) for d in DOI_RE.findall(text)}
    text_n = norm_text(text)

    def title_hit(r):
        t = norm_text(r["title"])[:50]
        return len(t) >= 15 and t in text_n

    by_doi = [r for r in rows if r["doi"] and norm_doi(r["doi"]) in found]
    if len(by_doi) == 1:
        return by_doi[0]
    if len(by_doi) > 1:
        narrowed = [r for r in by_doi if title_hit(r)]
        return narrowed[0] if len(narrowed) == 1 else None
    by_title = [r for r in rows if title_hit(r)]
    if len(by_title) > 1:
        by_title.sort(key=lambda r: -len(r["title"]))
        if len(norm_text(by_title[0]["title"])) > len(norm_text(by_title[1]["title"])):
            return by_title[0]
        return None
    return by_title[0] if by_title else None


def sort_inbox(rows, pdf_dir):
    try:
        _pymupdf()
    except ImportError:
        sys.exit("pymupdf is needed for --sort-inbox:  pip install pymupdf")
    inbox = pdf_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    files = sorted(list(inbox.glob("*.pdf")) + list(inbox.glob("*.txt")))
    if not files:
        print(f"No PDF or TXT files in {inbox}")
        return
    moved, unmatched = 0, []
    for f in files:
        try:
            text = (f.read_text(encoding="utf-8", errors="replace")[:20000] if f.suffix.lower() == ".txt"
                    else first_pages_text(f))
            r = match_pdf(text, rows)
        except Exception as e:  # damaged or encrypted PDF
            unmatched.append((f.name, f"could not read ({e.__class__.__name__})"))
            continue
        if not r:
            unmatched.append((f.name, "no DOI or title match"))
            continue
        target = pdf_dir / f"{r['openalex_id']}{f.suffix.lower()}"
        if have(pdf_dir, r["openalex_id"]):
            unmatched.append((f.name, f"duplicate of {target.name}"))
            continue
        shutil.move(str(f), str(target))
        print(f"  {f.name}  ->  {target.name}  ({r['corpus']}: {r['title'][:60]})")
        moved += 1
    print(f"Renamed {moved} file(s).")
    for name, why in unmatched:
        print(f"  LEFT IN INBOX: {name}  ({why}). Rename it by hand from the checklist.")


# ---------------------------------------------------------------------------


def report(rows, pdf_dir):
    print("\nStatus:")
    for name in CORPORA:
        sub = [r for r in rows if r["corpus"] == name]
        n = sum(have(pdf_dir, r["openalex_id"]) for r in sub)
        print(f"  {name:<11} {n}/{len(sub)} present")
    total = sum(have(pdf_dir, r["openalex_id"]) for r in rows)
    print(f"  {'total':<11} {total}/{len(rows)} present")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus-dir", default="corpus")
    p.add_argument("--sort-inbox", action="store_true", help="Rename manually downloaded PDFs in pdfs/inbox/.")
    p.add_argument("--status", action="store_true", help="Only report what is present and missing.")
    p.add_argument("--no-oa", action="store_true", help="Skip the open-access download attempt.")
    args = p.parse_args()

    corpus_dir = Path(args.corpus_dir)
    pdf_dir = corpus_dir / "pdfs"
    (pdf_dir / "inbox").mkdir(parents=True, exist_ok=True)
    rows = load_corpus(corpus_dir)

    if args.sort_inbox:
        sort_inbox(rows, pdf_dir)
    elif not args.status and not args.no_oa:
        fetch_open_access(rows, pdf_dir)

    missing = write_checklist(rows, pdf_dir, corpus_dir)
    report(rows, pdf_dir)
    if missing:
        print(f"\n{len(missing)} still needed. Open {corpus_dir / 'download_checklist.html'} in a browser.")
    else:
        print("\nAll documents present.")


if __name__ == "__main__":
    sys.exit(main())
