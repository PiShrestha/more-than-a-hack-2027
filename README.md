# encara_messi: football possession from automatic tracking

[![Submission integrity](https://github.com/PiShrestha/more-than-a-hack-2027/actions/workflows/submission.yml/badge.svg?branch=submission)](https://github.com/PiShrestha/more-than-a-hack-2027/actions/workflows/submission.yml)

Our Phase 1 entry for More than a Hack 2027 labels all 199,319 frames of the
Barcelona vs Athletic Club tactical video as home possession, away possession,
dead ball, or outside the published halves.

We train LightGBM and six temporal CNNs on seven external Bundesliga matches,
with tracking degraded to match BAR-ATH's missing observations. A Viterbi
decoder joins frame predictions into possessions. No BAR-ATH labels are used
for training, tuning, or model selection.

## The submission

- [Final technical document](submission/encara_messi.md), uploaded as `encara_messi.md`.
- [Fixed prediction CSV](submission/submission.csv), uploaded as `encara_messi.csv`.
- [File hashes and model record](submission/manifest.json).
- [Reproduction instructions](REPRODUCING.md).

These are the exact files prepared in Downloads. A form receipt and private
full-match score have not been recorded in this repository.

The model is frozen at commit `602b6139b8fe82d0256515102444534bf94b32d7` on
[`possession-model`](https://github.com/PiShrestha/more-than-a-hack-2027/tree/possession-model).
This branch adds the final document, release tools, and checks. It does not
change the classifier, weights, features, decoder, or predictions.

## Verify or prepare upload files

These commands use Python's standard library and need no match data or ML packages:

```bash
python3 tools/submission.py verify
python3 tools/submission.py package --output-dir dist/encara_messi
```

The second command creates `encara_messi.csv` and `encara_messi.md`. The CSV
SHA-256 is:

```text
557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c
```

## Evidence and limits

Independent reconstruction gave mean external-match spellF1 **0.6383**, versus
**0.1925** for the shipped smooth50 baseline under the same noise. The seven
matches were also used for model selection. This is validation, not an untouched
test. The small public BAR-ATH window is a pipeline check, not a tuning set or an
estimate of the private score.

Repeated Linux runs reproduced the CSV exactly, including a fresh-clone check.
CI checks artifact integrity and packaging on each release-branch change. It
neither trains models nor evaluates private match labels.

## Project layout

`possession/` contains the frozen model and six saved CNNs. `evaluation/` contains
the official validator and scorer. `submission/` contains the exact release
record and final document. `tools/submission.py` verifies, packages, and safely
runs the original reproduction pipeline.

The completed experiments and audits remain on the separate
[`cluster-verification-20261116`](https://github.com/PiShrestha/more-than-a-hack-2027/tree/cluster-verification-20261116)
branch. The earlier heuristic model remains on
[`possession-pipeline`](https://github.com/PiShrestha/more-than-a-hack-2027/tree/possession-pipeline).
Neither is a replacement submission.

The [organizer README](CHALLENGE_README.md), [challenge](CHALLENGE.md),
[rules](RULES.md), [data guide](DATA.md), [evaluation guide](EVALUATION.md), and
[submission instructions](SUBMISSION.md) are retained. The original
[SOLUTION.md](SOLUTION.md) is preserved as a historical draft; use the final
technical document linked above.

## Data and license

Licensed BAR-ATH tracking and video are not distributed here. Supply the
permitted tactical tracking and metadata locally. The open external data comes
from [IDSSE](https://doi.org/10.1038/s41597-025-04505-y), by Bassek, Rein, Weber,
and Memmert, under CC BY 4.0. See [LICENSE](LICENSE) for the repository license
and its separate treatment of competition assets.
