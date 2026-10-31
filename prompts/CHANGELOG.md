# Changes after freeze

Prompts, schemas and experiment.yaml were frozen on 2026-10-25. Nothing in prompts/readers, prompts/editors
or claim_extraction.yaml has changed since. The changes below concern how the runs were executed, one
corpus document, and two checks added after the main run, each committed before it ran.

## 2026-10-25 - Execution backend (departs from experiment.yaml)
All model calls ran through the Claude Code CLI (claude -p) rather than the Anthropic API, because no API
key was available. Model: claude-sonnet-5-5, passed with --model and confirmed by a test call whose usage
report named that model. Claude Code does not expose sampling parameters, so the temperatures in
experiment.yaml (1.0 for readers and editors, 0 for extraction) were NOT applied; all calls used Claude
Code's defaults. Our system prompt replaced Claude Code's own (--system-prompt), each call was limited to
one turn, and calls ran from an empty temporary folder. Calls ran 4 at a time.

## 2026-10-26 - Corpus document replaced; targeted rerun
W2124927217 (Zaheer & Venkatraman, 1995) was a first-page preview (4,675 characters; one 107-word passage).
It was found after the main run because its claims scored as unusually coherent. Replaced with the full
article (JSTOR copy, cover page removed); re-extraction changed no other document's passages. Its 25
readings and 25 extractions, and the 26 editor runs for the tce and tce_spliced conditions, were set aside
(runs_cc/outputs/superseded.jsonl) and redone: 76 calls. extract_passages.py now also flags PDFs averaging
under 1,000 characters per page.

## 2026-10-26 - NLI diagnostic added (committed before scoring)
prompts/diagnostic_pairs.yaml and nli_diagnostic.py: 10 compatible-but-different claim pairs, to measure
whether NLI reads difference of aspect as contradiction. Result: 2 of 10 labeled contradiction (both
contrasting sibling categories), 8 neutral, none entailment.

## 2026-10-26 - Construct-sensitivity check added (committed before running)
prompts/experiment_sensitivity.yaml: TCE documents reread with construct term "transaction cost economics"
instead of "governance" (real readers, 3 repetitions; prompts and models otherwise unchanged). run_dcr.py
and compute_measures.py gained an --experiment option. Predictions were written in the file before the run.
Result: TCE coherence 0.025 (vs 0.017 under "governance"); the split is not an artifact of the term.

## 2026-10-26 - Runner fix and repair of the sensitivity run
The sensitivity run was interrupted and restarted at higher parallelism; Claude Code then returned
rate-limit errors inside otherwise successful responses, which the runner saved as readings. run_dcr.py now
treats such responses as failures and retries them. repair_runs.py set aside 110 duplicate or empty records
(runs_sens/outputs/superseded.jsonl) and the affected calls were redone at 4 workers. An audit of the main
run (repair_runs.py --report) found no such records: its 9 unparsed readings are complete readings with
malformed JSON.
