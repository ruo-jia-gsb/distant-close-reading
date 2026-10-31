# DCR measures

Runs: `runs_sens` | NLI: `microsoft/deberta-large-mnli` | embeddings: `BAAI/bge-large-en-v1.5`

## Reader agreement within documents (NLI agreement S; dispersion D = 1 - S)

| Comparison | n | S mean [95% CI, docs resampled] | cosine | conflict ratio |
|---|---|---|---|---|
| floor_real | 171 | 0.332 [0.302, 0.368] | 0.866 | 0.608 |
| real_vs_real | 171 | 0.229 [0.201, 0.259] | 0.835 | 0.732 |

**Real readers disperse 1.15x the noise floor** (D = 0.771 vs 0.668).

| Real reader pair | S mean [95% CI] |
|---|---|
| genealogical vs operational | 0.215 [0.187, 0.246] |
| genealogical vs structural | 0.211 [0.178, 0.243] |
| operational vs structural | 0.262 [0.221, 0.307] |

| Corpus | noise floor S | real vs real S |
|---|---|---|
| tce | 0.332 | 0.229 |

## Corpus coherence: agreement between documents (real readers pooled)

Each value is the mean over document pairs; spread is across the 5 repetitions.

| Condition | doc pairs | S mean (sd over reps) | cosine | conflict ratio |
|---|---|---|---|---|
| tce | 171 | 0.025 (0.004) | 0.762 | 0.972 |
| tce_spliced | 171 | 0.025 (0.004) | 0.762 | 0.972 |

**Manipulation check:** coherence 0.025 (TCE) -> 0.025 (spliced). It should drop.

## Stance (genealogical reader; tracked separately from characterization)

| Corpus | documents | stance distribution (modal label per document) |
|---|---|---|
| legitimacy | 0 |  |
| tce | 19 | modify 9, extend 8, contest 1, endorse 1 |
| splice | 0 |  |

## Co-citation baseline (bibliographic coupling vs NLI agreement, document pairs)

| Group | doc pairs | Spearman rho | mean coupling |
|---|---|---|---|
| legitimacy | 0 | n/a | n/a |
| tce_all | 170 | -0.060 | 0.029 |

A high correlation would mean interpretive agreement adds little beyond citation structure.
