# Compute-first continuation

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
