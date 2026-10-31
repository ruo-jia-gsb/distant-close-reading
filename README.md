# Distant Close Reading (DCR)

**Measuring whether a scholarly literature holds together.**

DCR instantiates language-model readers that are independent, repeatable and specified: each starts fresh, each can be rerun on the same text, and each follows a written method anyone can inspect. Three readers with epistemically distant methods (structural, operational, genealogical) read the same passages; their readings are reduced to short claims about what each document takes a construct to be (*characterization*) and whether it endorses or contests the account it engages (*stance*). Natural language inference (NLI) then measures agreement between readers and between documents, always against a noise floor: the same reader on the same document, repeated.

This repository is the Stage 1 artifact for the *Organization Science* special issue “AI-Assisted Frontiers in Organizational Science.” It runs the method on two organizational literatures, legitimacy and transaction cost economics (TCE), and validates it with a sham control, a manipulation check, a gold-pair test, a citation baseline and two pre-registered robustness checks.

## Key results

| Result | Value |
|---|---|
| NLI separates opposed from equivalent claims (15 expert-vetted pairs) | AUC 1.00 (embedding cosine: 0.18, inverted) |
| Noise floor: same reader, repeated | S = 0.325 [0.305, 0.343] |
| Different methods, same document | S = 0.256 [0.236, 0.277] |
| Same method, different wording (shams) | S = 0.309 and 0.319, at the noise floor |
| Between-document coherence, legitimacy vs. TCE | 0.072 vs. 0.017 (sd over repetitions ≤ 0.004) |
| Manipulation check: TCE with off-sense “governance” papers spliced in | 0.017 → −0.004 |
| TCE with construct term “transaction cost economics” instead of “governance” | 0.025 (split not an artifact of the term) |
| NLI diagnostic: compatible-but-different pairs read as contradiction | 2 of 10 |

S is NLI agreement between claim sets, from −1 to 1; intervals resample documents. Full tables are in `results/report.md` and `results_sens/report.md`.

## Repository map

| Path | Contents |
|---|---|
| `select_corpus.py` | Mechanical corpus selection from OpenAlex (rule, anchors, version log in the header) |
| `fetch_pdfs.py` | Open-access download, a manual-download checklist, and auto-renaming of PDFs by DOI or title |
| `extract_passages.py` | PDF/text extraction and the fixed passage rule (offsets, hashes, book trimming, text checks) |
| `check_gold_pairs.py`, `nli_diagnostic.py` | Measurement sanity checks |
| `run_dcr.py` | Runs reading, claim extraction and editing (Claude Code CLI or Anthropic API), resumable, fully logged |
| `invalidate_doc.py`, `repair_runs.py` | Targeted reruns after a document is replaced; cleanup after interrupted runs |
| `compute_measures.py` | All measures and the report |
| `prompts/` | Reader, sham, extraction and editor prompts; `experiment.yaml`; `experiment_sensitivity.yaml`; gold and diagnostic pairs; `CHANGELOG.md` |
| `corpus/` | Selected and candidate lists (DOIs), reference lists, `passage_index.csv` (IDs, pages, character offsets, hashes), selection and extraction manifests |
| `runs_cc/`, `runs_sens/` | Request logs (prompt hashes, settings, commits), extracted claims and stances, editor judgments |
| `results/`, `results_sens/` | Reports, per-pair CSVs, gold and diagnostic results, NLI cache |
| `requirements.txt` | Pinned environment |

**Withheld:** article PDFs, extracted text and passage text (copyrighted), and full reader outputs, which can quote short passages. Every reported measure is computed from the claims and judgments included here; the full outputs and the complete commit history are available to the editors on request.

## Reproducing the study

Python 3.11. A GPU is optional (measures ran on an 8 GB laptop GPU; CPU works, slowly).

```bash
python -m venv ddr_venv && source ddr_venv/bin/activate      # Windows: .\ddr_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**1. Recompute the measures from the included claims** (no PDFs or model calls needed):

```bash
python compute_measures.py                                     # main study -> results/
python compute_measures.py --runs runs_sens --outdir results_sens --experiment prompts/experiment_sensitivity.yaml
python check_gold_pairs.py && python nli_diagnostic.py
```

**2. Rebuild the corpus and passages.** Selection needs a free OpenAlex API key (`OPENALEX_API_KEY`). PDFs must be obtained legally by DOI; the checklist lists them, and the inbox sorter names them.

```bash
python select_corpus.py --email you@example.org               # reproduces corpus/*_selected.csv
python fetch_pdfs.py                                           # open-access copies + corpus/download_checklist.html
python fetch_pdfs.py --sort-inbox                              # after downloading the rest into corpus/pdfs/inbox/
python extract_passages.py                                     # compare text hashes with corpus/manifest_extraction.json
```

Copies of the same article from different sources can differ (cover pages, scans, preprints); the text hashes in the manifest show whether yours match. For the two Williamson books, the rule uses the original preface or introduction plus Chapter 1, cut at the marker phrases in `extract_passages.py`.

**3. Rerun the model stages.**

```bash
python run_dcr.py pilot --backend claude-code                  # ~15 calls, checks prompts and parsing
python run_dcr.py run all --backend claude-code --workers 4    # 1,125 readings, 1,125 extractions, 42 editor runs
```

The study ran through the Claude Code CLI (`claude -p`) with Claude Sonnet 5.5, which uses default sampling settings. With an Anthropic API key, `run_dcr.py submit|status|collect read|extract|edit` uses the Message Batches API with the temperatures set in `prompts/experiment.yaml`. Runs resume after interruption; if one is interrupted mid-call, run `repair_runs.py --runs <folder> --report` and then `repair_runs.py --runs <folder>` before resuming.

## Extending DCR

- **A new reader** is a YAML file in `prompts/readers/` with a method block and an output schema; add its name to `experiment.yaml`.
- **A new literature** is a new entry in `select_corpus.py` (anchors, journals, terms) plus a construct term in `experiment.yaml`.
- **A different model** is a change to `models` in `experiment.yaml` (API backend) or `--model` (Claude Code backend).
- **A robustness check** is a copy of `experiment.yaml` passed with `--experiment`; `experiment_sensitivity.yaml` is a worked example.

## Integrity of the record

Every rule was committed to Git before the results it governs: the corpus rule (Oct 22), vetted gold pairs before scoring, frozen prompts before the runner, and each later check with its predictions before it ran. Post-freeze changes are logged with reasons in `prompts/CHANGELOG.md` and in script headers, including one rule change considered and rejected. Replaced or repaired records are moved to `superseded.jsonl`, never deleted.

## Known limitations

One model family under default sampling; NLI occasionally reads contrasting sibling categories as contradiction; claim extraction normalizes voice; a small, curated corpus (47 documents); editors are reliable but rarely discriminate between corpora. Details and the Stage 2 response to each are in the technical appendix.

## License

Code: MIT (see LICENSE). Derived data (claims, judgments and measures): CC BY 4.0.
