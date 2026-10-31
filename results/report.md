# DCR measures

Runs: `runs_cc` | NLI: `microsoft/deberta-large-mnli` | embeddings: `BAAI/bge-large-en-v1.5`

## Reader agreement within documents (NLI agreement S; dispersion D = 1 - S)

| Comparison | n | S mean [95% CI, docs resampled] | cosine | conflict ratio |
|---|---|---|---|---|
| floor_real | 1394 | 0.325 [0.305, 0.343] | 0.843 | 0.648 |
| floor_sham | 840 | 0.323 [0.300, 0.344] | 0.845 | 0.627 |
| real_vs_real | 697 | 0.256 [0.236, 0.277] | 0.811 | 0.743 |
| sham_vs_sham | 210 | 0.309 [0.281, 0.337] | 0.840 | 0.659 |
| sham_vs_structural | 420 | 0.319 [0.293, 0.342] | 0.843 | 0.644 |
| sham_vs_other_real | 832 | 0.257 [0.236, 0.278] | 0.813 | 0.737 |

**Real readers disperse 1.10x the noise floor** (D = 0.744 vs 0.675).
**Sham readers disperse 1.02x the noise floor.** If epistemic distance drives divergence, this should be near 1 and well below the real ratio.

| Real reader pair | S mean [95% CI] |
|---|---|
| genealogical vs operational | 0.238 [0.215, 0.260] |
| genealogical vs structural | 0.251 [0.231, 0.271] |
| operational vs structural | 0.281 [0.258, 0.303] |

| Corpus | noise floor S | real vs real S |
|---|---|---|
| legitimacy | 0.338 | 0.270 |
| tce | 0.301 | 0.230 |
| splice | 0.352 | 0.292 |

## Corpus coherence: agreement between documents (real readers pooled)

Each value is the mean over document pairs; spread is across the 5 repetitions.

| Condition | doc pairs | S mean (sd over reps) | cosine | conflict ratio |
|---|---|---|---|---|
| legitimacy | 253 | 0.072 (0.003) | 0.720 | 0.928 |
| tce | 171 | 0.017 (0.004) | 0.725 | 0.974 |
| tce_spliced | 276 | -0.004 (0.004) | 0.714 | 0.983 |
| spliced pairs only (tce+splice) | 95 | -0.045 (0.005) | 0.692 | 0.998 |
| splice+splice | 10 | 0.042 (0.009) | 0.722 | 0.978 |

**Manipulation check:** coherence 0.017 (TCE) -> -0.004 (spliced). It should drop.

## Stance (genealogical reader; tracked separately from characterization)

| Corpus | documents | stance distribution (modal label per document) |
|---|---|---|
| legitimacy | 23 | modify 18, extend 5 |
| tce | 19 | extend 9, modify 7, contest 2, endorse 1 |
| splice | 5 | modify 3, extend 2 |

## Editors

| Editor | Condition | judgments across reader repetitions | positions mean (sd) | contested drivers |
|---|---|---|---|---|
| adversarial | legitimacy | coherent_contestation 5 | 5.4 (0.5) | substantive 18, scope 10 |
| adversarial | tce | coherent_contestation 5 | 5.0 (0.0) | substantive 20, scope 5, terminological 3 |
| adversarial | tce_spliced | coherent_contestation 5 | 6.2 (0.7) | substantive 18, scope 8, terminological 2, method 1 |
| synthesizing | legitimacy | coherent_contestation 4, settled 1 | 3.2 (1.8) | substantive 8, scope 4, unclear 2, terminological 1 |
| synthesizing | tce | coherent_contestation 5 | 5.2 (0.4) | substantive 15, scope 7, terminological 4 |
| synthesizing | tce_spliced | coherent_contestation 5 | 5.6 (0.5) | substantive 17, scope 8, terminological 1 |

| Editor noise floor (fixed input) | runs | judgment agreement | positions (sd) |
|---|---|---|---|
| adversarial / legitimacy | 4 | 100% (coherent_contestation) | 0.43 |
| adversarial / tce | 4 | 100% (coherent_contestation) | 0.43 |
| synthesizing / legitimacy | 4 | 100% (coherent_contestation) | 0.43 |
| synthesizing / tce | 4 | 100% (coherent_contestation) | 0.43 |

## Co-citation baseline (bibliographic coupling vs NLI agreement, document pairs)

| Group | doc pairs | Spearman rho | mean coupling |
|---|---|---|---|
| legitimacy | 253 | 0.432 | 0.045 |
| tce_all | 275 | 0.259 | 0.023 |

A high correlation would mean interpretive agreement adds little beyond citation structure.
