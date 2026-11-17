# Possession research and verification

This branch contains completed Rivanna experiments and audits. The prepared
competition release, final document, and guarded reproduction tools are on the
[`submission` branch](https://github.com/PiShrestha/more-than-a-hack-2027/tree/submission),
which is now the repository's default branch. Use that branch for upload files.

The model and original prediction remain frozen at
`602b6139b8fe82d0256515102444534bf94b32d7` on `possession-model`. No research
variant replaces the fixed submission. No form receipt or private score has
been recorded here.

## What was verified

Repeated Linux runs regenerated the frozen CSV exactly, including a fresh-clone
check. Its SHA-256 is in [reproduction-hashes.txt](reproduction-hashes.txt).
All 17 staged raw inputs were checked by size and hash; 16 are needed by the
final tracking-only pipeline. Smart Tagging is staged but unused.

The fixed external validation reconstruction gave mean spellF1 0.6383 versus
0.1925 for the shipped smooth50 baseline under the same noise. These seven
matches were already used for model selection. The confidence intervals are
conditional on that choice, not estimates from an untouched test set.

## Completed experiments

- Five observation patterns across seven matches and three corruption seeds:
  105 cases, with 840 saved member probability matrices.
- Twenty-six predefined ensemble and decoder settings across those cases:
  2,730 CPU evaluations. No replacement was selected.
- Four sequence-context conditions across seven matches: 28 retained cases.
  Distant context changed predictions through whole-sequence GroupNorm.
- A same-capacity per-frame normalization ablation: three trained folds and
  28 retained cases. It removed distant-context dependence, but one seed did
  not establish a general accuracy gain.
- Period-boundary, short-possession failure, saved-model, CPU replay, and
  private-backup integrity audits.

See [followup-results.json](followup-results.json) for aggregate comparisons and
[WORK_LOG.md](WORK_LOG.md) for decisions, failures, job IDs, and limitations.
Corruption seeds repeat the same seven games; they are not independent matches.
No new candidate was evaluated against BAR-ATH public labels.

## Retained artifacts and later CPU work

Private, gitignored `artifacts/` contains all raw tracking/metadata inputs,
18 stock CNN fold checkpoints, three per-frame CNN fold checkpoints, three
LightGBM fold models, the final full-data tree, and cached predictions. The
half-isolated research candidate is separate and unsubmitted. Raw assets and
frame-level arrays are not distributed by this branch.

`replay.py` can change decoder settings or ensemble weights from the caches
without installing Torch or LightGBM. The tested dependencies are in
[requirements-replay.txt](requirements-replay.txt). A complete replay of 105
cases took about 15 seconds on the Mac. Choosing settings against these caches
remains model selection.

`verify_backup.py` checks input and output hashes, probability alignment,
normalization, replay agreement, and the private named submission package.
`collect.py` retrieves existing cluster outputs; it submits no jobs.
`accounting.py` includes failed and cancelled work in allocated CPU/GPU hours.
The full Linux environment is recorded in [requirements-linux-lock.txt](requirements-linux-lock.txt).

The public records contain source pointers, hashes and aggregate evidence.
Reproduction of the experiments needs the permitted inputs described by their
protocol files. See [idsse-source.json](idsse-source.json) for the open source
records and [verified-inputs.json](verified-inputs.json) for the staged hashes.

## Submission boundary

The CSV, final technical document and canonical upload command live on
`submission`. `package_submission.py` here is an earlier local staging helper;
it is not the release entry point. The root `SOLUTION.md` is a preserved team
draft, not the final named technical document.

Keep model experiments and outputs on this branch. In particular, the optional
`possession.model ens` command can overwrite the decoder and defaults to one
CNN. Do not run it in the fixed submission checkout.
