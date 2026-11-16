# Football tracking robustness pilot

Research question: how much does a frozen possession model change when tracking
observations disappear, particularly in sustained gaps? This first engineering
pilot is deliberately **unlabelled**. Agreement with the original prediction is
stability, not accuracy: an always-H predictor can be perfectly stable and wrong.
Do not select submission parameters or model families using this experiment.

## Stage 0: fixed-model missingness pilot

Frozen model: `solution.predict.Config()` at base commit `b2d04f3`, including its
automatic tactical priors. No inference or submission parameters are changed.

Design fixed before collecting results:

- One clean run; 18 corruptions: IID and Markov burst missingness × marginal mask
  rates 0.1/0.3/0.5 × seeds 11/22/33. Mean burst duration is two seconds.
- Masks are sampled on the half timelines independently of observed/missing ball
  positions, reset at half boundaries. Existing missing data remains missing.
  The rate is a timeline marginal, not a promised percentage of available balls.
  Report the actual removed fraction among originally observed ball frames.
- Recompute emissions from perturbed raw input so stationary-ball state resets
  correctly; do not just zero an already-computed emission array.
- Preserve players, metadata and automatic tags. This tests ball observation
  failure conditional on unchanged tag evidence. It does not simulate failure of
  the shared upstream video/tracking/tagging system. Add a no-tag stratum in the
  later benchmark to quantify this limitation.
- Measure label-change fraction over published halves, separately in masked and
  unmasked frames, H/A/D counts, possession durations and transition counts.
  Unmasked-frame changes expose propagation through offline Viterbi decoding.
- Profile natural missing-run counts and nearest-rank duration quantiles by half.
  No public/private labels are opened, and no manual annotations are created.

Outputs: `tracking_profile.json`, incrementally checkpointed `summary.json`, and
`runtime.json`. Source/input/config hashes and corruption seeds identify each run.
The summary is deterministic in a fixed code/Python environment; runtime is not.
The clean label-string hash is distinct from the submission CSV hash. There are
no frame-level prediction/trajectory exports in this experiment. Transient
corrupted ATD files are created privately inside the output folder and removed
after each run; an interrupted process may leave one that must stay private.
Only one ~196 MB temporary ATD exists at a time. Allow 500 MB free storage.

From the repository root, Python 3.11+ with no third-party packages:

```bash
python -m research.robustness --data-dir /private/path/to/data --out research/results/pilot-001
python -m research.robustness --data-dir /private/path/to/data --out research/results/profile-001 --profile-only
python -m unittest discover -s tests -v
```

Use a new output directory per invocation. A failed run has `complete: false`;
checkpointed runs are evidence of partial completion, not a full experiment.
The current runner does not resume: rerun into a new directory.

## Stage 1: external data and reference-label audit

Candidate: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data).
The publisher supplies synchronized tracking and events for other anonymized
games, including EPTS-format Sample Game 3. These are candidates, **not yet an
approved benchmark with complete H/A/D labels**. Audit rights, exact source
revision, assets, timing, ball visibility, and event semantics before downloading
or training. Event-to-possession interpolation is a proxy and may not reproduce
controlled possession or referee-defined dead ball. Unlabelled intervals must
stay unknown; do not silently fill them as reference truth. Existing incomplete
events alone cannot establish dead-ball recall. A larger appropriate labelled
multi-match source is needed before claiming generalization or training a neural
model responsibly. Do not retrieve human-produced data about BAR-ATH.

Create a match manifest with dataset/revision/hash, licence or permission URL,
label provenance (human/mechanical/proxy), time base, FPS/pitch, half boundaries,
coverage and documented exclusions. Refuse duplicate source match IDs or hashes
across train/validation/test. Keep all augmented versions and windows of a match
in its original split. With only three matches, use leave-one-match-out results
as exploratory evidence and state the small-sample limitation.

## Stage 2: benchmark proposal, not yet implemented or authorized to run

Compare nearest-player majority smoothing (fixed windows 25/50/125), the current
Viterbi decoder, an explicit-duration model, and a small temporal convolutional
model, with feature and automatic-tag availability controlled across methods.
Select parameters on external validation matches only; lock the choice before
testing external held-out matches or generating the competition submission.

Corruptions: IID and burst ball missingness, player-dropout, metre-scale position
noise, and track-slot permutations within team. Add spatially dependent missingness
to challenge the independence assumption. Slot permutations should be a negative
control for collective methods. Duration priors and normalization must be fitted
on training matches only. Predeclare rates, seeds, split and candidate grids.

Primary accuracy: official spellF1; secondary macroF1, dead-ball precision/recall,
transition timing, fragmentation, calibration and selective risk/coverage. Apply
the official abstention filling semantics to temporal metrics. Use paired,
match-level bootstrap intervals across test matches; corruption seeds are repeats
of a match, not independent matches. If too few matches, show every match result
and report uncertainty as exploratory rather than a narrow frame-wise CI.

Reusable deliverables: dataset manifests and adapters, corruption generator,
frozen split/config files, per-run hashes/seeds/environment/accounting records,
aggregate failure tables and paired confidence intervals. Publish code and
permitted aggregate summaries only; distribute source pointers rather than the
licensed match assets or reconstructive exports. Inspect any result for allowed
publication before committing it.
