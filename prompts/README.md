# DCR prompts and configuration

Everything that shapes a model's output lives here. **Commit this folder before any reading run.**
The commit timestamp is the evidence that prompts and schemas were fixed before results existed.
After the freeze, any change goes in `CHANGELOG.md` with its date and reason.

| File | Role |
|---|---|
| `experiment.yaml` | Models, constructs, readers, editors, repetitions, measures |
| `readers/_template.yaml` | Instructions shared by every reader; holds everything constant except method |
| `readers/structural.yaml`, `operational.yaml`, `genealogical.yaml` | The three real methods |
| `readers/structural_sham_plain.yaml`, `structural_sham_formal.yaml` | Shams: the structural method in two different voices, same schema |
| `claim_extraction.yaml` | Reduces each reading to characterization claims plus a stance; same for all readers |
| `editors/_template.yaml` | Editor instructions and the frozen output schema |
| `editors/synthesizing.yaml`, `adversarial.yaml` | The two editor roles |
| `gold_pairs.yaml` | Expert-vetted claim pairs for the measurement sanity check |

## Placeholders

Prompts use `<<name>>` placeholders filled by plain string replacement (not `str.format`), so JSON
braces in schemas are safe. A reader prompt is `_template.yaml` with `<<persona_block>>` and
`<<output_schema>>` taken from the persona file; `<<construct>>` is then filled everywhere, including
inside the persona block.

## Design notes for the appendix

- **One difference per comparison.** Real readers differ only in method. Shams differ from
  `structural` only in wording. Editors differ only in role. Every other instruction is shared.
- **Characterization vs. stance.** Claim extraction separates them, and stability is computed on
  characterization only, so a contested literature is not mistaken for an incoherent one.
- **Extraction normalizes voice.** Extraction strips method vocabulary so that measures compare
  content rather than style. This could also make readings converge artificially; the sham control
  and the noise floor are how we check that.
- **Construct terms.** Legitimacy corpus: "legitimacy". TCE corpora: "governance", the term at the
  center of TCE and the one the splice set uses in a different (corporate-governance) sense.
- **No hypotheses in prompts.** Readers and editors are never told which corpus is expected to be
  coherent. The four overall judgments are defined neutrally, and the number of positions is left open.
