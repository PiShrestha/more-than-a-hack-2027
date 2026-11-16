# Compute-first continuation

Research checkpoint: November 16, 2026. All required tracking/metadata inputs, models,
105 stress evaluations and audits are verified and backed up privately. The
frozen submission reproduced twice exactly. CPU replay of all 105 cached cases
took 14.6 seconds on the Mac, with no Torch or LightGBM installed. See the final
entry in WORK_LOG.md for results, resource totals and remaining submission tasks.

See [WORK_LOG.md](WORK_LOG.md) for the frozen source, data provenance, budget,
job IDs, results and failures. This directory is added on a separate research
branch. Nothing here changes the submitted possession-model pipeline.

The expensive deliverables are the environment/data caches, two Linux
reproduction checks, 18 match-held-out CNN fold models and 49 per-match
probability files (7 LightGBM, 42 CNN). Once those exist, ensemble/decoder
analysis and match-bootstrap summaries run on CPU without training again.

`aggregate.py` evaluates only the other-match data with the frozen submitted
ensemble configuration. It does not tune or select any settings. Its confidence
intervals quantify variation across seven matches conditional on this chosen
pipeline, not uncertainty from the prior model-selection process. No target
public ground truth is opened by these research scripts.

Inputs required for this stage:

- tactical ATD TXT and metadata XML (licensed, already staged);
- all 14 IDSSE position/metadata XMLs (open, checksummed by prepare_data.py);
- six submitted network checkpoints and the submitted decoder (in source).

Smart Tagging is staged but unused. Video, broadcast and homography are not
inputs to this tracking-only experiment. The public BAR-ATH labels are not
needed for the queued jobs.

The cluster's Python 3.14 module and pinned requirements are used. The complete
resolved Linux environment is recorded in results/environment-lock.txt. GPU
smoke testing gates all CNN training. Per-fold retained checkpoints store the
training and validation match IDs, settings and normalization statistics.

Do not execute the original optional `possession.model ens` command on the
submission copy: it defaults to a single CNN and overwrites decoder.json.
Research LightGBM CV sets POSS_TAG=_audit to preserve the submitted decoder.
Any future grid search belongs in a separate output configuration, and its
score must be described as validation/model-selection performance.

After completion, back up scratch outputs before cleanup. Private backups live
under research/artifacts/ locally and are gitignored. Keep source, aggregate
records, logs and dependency locks separately from licensed raw inputs. Do not
publish BAR-ATH-derived frame arrays or any licensed match assets.

## Completed expensive artifacts and later CPU work

The latest submitted CSV reproduced byte-for-byte twice on Linux. The frozen
hash is in reproduction-hashes.txt. All three LightGBM fold models and 18 CNN
fold checkpoints are retained privately under artifacts/results/folds/.
Original held-out predictions live under artifacts/data/external/idsse/.

The fixed stress benchmark adds five observation patterns x seven matches x
three corruption seeds. It retains per-member probabilities, the frozen
ensemble, labels/periods for the open matches, and scoring reports. These are
private experiment artifacts, not files to add to the submission branch.

After the jobs complete, `replay.py` can evaluate cached probabilities with
alternative decoder penalties or ensemble weights on CPU. It needs neither
Torch nor LightGBM nor raw tracking; requirements-replay.txt gives the tested
CPU dependencies. Defaults replay the frozen ensemble and do not change it.
The only initial float32-only ensemble caches are from the first pilot; later
caches retain float64 ensemble probabilities and decoded labels as well.

Keep alternative settings in separate output files. Changing settings against
these seven matches remains model selection, not independent test evaluation.
No competition public labels are needed for these tasks.

## November 16 follow-up artifacts

The CPU sensitivity study evaluated 26 predefined configurations on all 105
stress cases. Controlled context probes retained 28 external-match prediction
archives, and the matched-capacity per-frame normalization experiment retained
three additional fold checkpoints and 28 prediction archives. These are research
diagnostics on the same seven selected validation matches. No alternative model
was selected for submission. See WORK_LOG.md for results and limitations.

The final original LightGBM model is now saved under
`artifacts/results/final-tree/model.txt`. It reproduces the original submission
with the committed six CNNs and avoids retraining for later inference. A
separately named half-isolated research candidate is retained in that directory;
it is not the official submission and has not been evaluated against BAR-ATH
labels. Its generating scripts are research code on this local branch.

The registered team name supplied by the user is `encara_messi`. Exact frozen
submission copies are staged privately in
`artifacts/submission-packages/encara_messi/`, alongside a proposed document for
team review. `package_submission.py --team-name encara_messi` validates and
stages them without overwriting edits. The submission deadline is November 17,
2026, 23:59 CET / 17:59 America/New_York. One designated team member still needs
to complete the Google Form after reviewing the document. Nothing has been
pushed or submitted.
