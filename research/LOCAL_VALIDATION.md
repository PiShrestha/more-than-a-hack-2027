# Local validation — 2026-11-16

Source: a new writable checkout copied directly from the active working copy at
`b2d04f36556b10389d23b516a393846583bfaee0`. The original clone under
`Documents/GitHub` was not used. The active working copy, its local-only commit,
downloaded data and Python environment were not modified during validation.
Publication status is separate from these experiment results.

## Engineering checks

- All 20 unit tests pass: the existing 13 plus seven robustness checks covering
  mask determinism, Markov rates/durations, missing-data preservation, half
  boundaries, aggregation denominators and predictor integration.
- Regenerated the existing 199,319-row prediction and passed the official
  submission validator. CSV SHA-256 remains
  `9d00cb8db58bd259c655e3af50d21090f32435a67c1fc354385718d3df951944`.
- No public-window evaluation was repeated and no parameters were changed.
- Repeated the two-corruption full-match experiment into a fresh directory:
  `summary.json` and `tracking_profile.json` are byte-identical. The clean
  research prediction's label-string hash exactly matches the regenerated
  submission's labels. Both runs removed all temporary corrupted tracking.

## Unlabelled tracking profile

| Half | Frames | Ball missing | Missing runs | Median gap | Longest gap |
|---|---:|---:|---:|---:|---:|
| 1 | 79,467 | 40.65% | 69 | 11.56 s | 138.64 s |
| 2 | 79,390 | 40.73% | 81 | 9.80 s | 221.48 s |

Only published half boundaries define these aggregates. Missing ball coordinates
are observations about input quality, not dead-ball labels. These long gaps
motivate later empirical-duration stress tests; two-second synthetic bursts
alone do not represent the longest natural outages.

## Full-match smoke experiment

Run configuration: fixed predictor and tags, `--rates 0.3 --seeds 11`, both modes,
two-second expected burst duration. This is one clean decode plus two corrupted
decodes, not the proposed 19-run pilot. Wall time was 25.19 seconds under Python
3.11.15 on this Mac, with a concurrent submission regeneration; this is an
engineering timing observation, not a controlled performance benchmark.

| Input | Observed ball frames removed in halves | Labels changed vs clean | Predicted possessions | State transitions |
|---|---:|---:|---:|---:|
| Clean | 0 | 0 | 354 | 422 |
| IID dropout | 30.23% | 3.435% | 279 | 332 |
| Burst dropout | 28.20% | 3.645% | 283 | 340 |

Despite changing only about 3.5% of frame labels, both perturbations reduce the
predicted possession count by about one fifth. That suggests sequence structure
is sensitive to missing observations, even when frame agreement looks high.
It does **not** identify which predictions are correct or prove that either
corruption is harder: one seed was tested and realised removed rates differ.
It does not demonstrate an improvement over any baseline.

Aggregate machine-readable records remain private under
`research/results/local-smoke/`; no corrupted tracking remains after success.
The summary identifies input/source hashes, frozen configuration, mask hashes
and seeds. Rerunning into a new output directory checks determinism; runtime is
recorded separately and will differ.
