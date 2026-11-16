# Possession sequence model

This is an initial, untrained H/A/D sequence model. It is implemented using only
the Python standard library. It does not read a ground-truth file.

## Reproduction

Place the provided tactical assets in `data/tactical/`, or set `MTAH_DATA_DIR` to
the directory containing `tactical/`. The required files are tracking metadata,
raw tracking, and Smart Tagging. Video and homography are not used yet.

From the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m solution.predict --out evaluation/results/sequence.csv
python3 -m solution.predict --no-tags --out evaluation/results/tracking_only.csv
```

The existing official validator and evaluation require the repository's NumPy
and pandas dependencies:

```bash
python3 -m pip install -r requirements-evaluation-lock.txt
python3 evaluation/validate_submission.py evaluation/results/sequence.csv
python3 evaluation/run_evaluation.py
```

The public five minutes are a pipeline check only, not a tuning or selection set.
The second command above is an ablation for inspection of model behaviour; it is
not a parameter search against that window.

## Method

1. Stream the tactical ATD. Require exactly the published video frame count in
   one-based sequential order and the metadata's number of track slots.
2. Convert coordinates to metres. Use relative team distance only when a player
   is within 2.5 metres of the ball. Ignore player identities, which are unstable.
3. A missing ball supplies neutral evidence across H/A/D. A visible ball outside
   the pitch supports D. A ball staying within 0.75 metres of an anchor for at
   least two seconds supplies weak stoppage evidence. This is a heuristic, not a
   referee model.
4. Add capped weak team priors from automatic attacking and defending tags.
   Defending tags support the opposing team. Conflicting tags cancel; N/A and
   unsupported codes are ignored. Restart windows are not dead-ball intervals.
5. Decode the minimum-cost path with three-state Viterbi. Team changes cost 12,
   entry/exit from D costs 8. A brief opposition touch therefore need not split
   a possession. Each published half block is decoded independently; the rest
   of the video is X. Ties favour persistence within a block.

Distances, durations and costs are declared in `Config` as initial heuristics.
No thresholds were trained or tuned against the public labels. Offline decoding
uses future observations and is intended for batch submission, not live play.
Time and memory are linear in frame count with three states.

Each prediction writes a JSON sidecar with configuration, input/output SHA-256
hashes, observation counts and output label counts. CSV output is deterministic,
UTF-8, LF terminated, with one row per tactical video frame.

## Current validation and limitations

Synthetic tests check occlusions, deflections, sustained handovers, stoppages,
half resets, tag interpretation, malformed inputs, and exact optimality against
an exhaustive enumeration. They do not establish football accuracy.

Validation performed: all 13 unit tests passed. A separate full-length synthetic
CLI run produced 199,319 contiguous rows, the expected independent half labels
and X gaps, and byte-identical CSV output on a second run. No real match labels
were used in these checks.

The three required tactical assets have now been downloaded from the supplied
Drive folder. Full-match inference produced 199,319 rows and passed the official
submission validator. The CSV has 84,502 H, 64,334 A, 10,021 D and 40,462 X frames;
all X frames are outside the published half blocks. A second real-data run
produced the same CSV hash.

The official evaluator was run once against the public five-minute window:

| Method | macroF1 | spellF1 |
|---|---:|---:|
| Sequence model | 0.5615 | 0.3263 |
| Published smooth50 baseline | 0.6207 | 0.3256 |
| Published smooth125 baseline | 0.5812 | 0.4242 |

These results do not establish improvement over the best baseline or estimate
the private full-match score. No model parameters were changed after observing
these public results. Full-match ground truth is unavailable.

The inference itself needs no dependencies. `requirements-evaluation-lock.txt`
records the packages used for the official validator and evaluator under
Python 3.11 on macOS arm64; NumPy 2.3.5 requires Python 3.11 or later.

Stationary balls can be live, outside coordinates can be tracking errors,
missing-ball intervals can contain unseen turnovers, and tactical tags can be
wrong. Dead-ball inference is the principal unfinished component. The next
engineering step is the unlabelled tracking/stability pilot described in
[research/README.md](../research/README.md), followed by external-match reference
label audits before model selection. Improve restart evidence without
hand-labelling or encoding frame-specific answers.

No labelled training data, external match events, or private ground truth are
used. Keep the licensed match assets out of git.
