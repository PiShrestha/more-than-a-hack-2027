# Possession research work log

## 2026-11-16: scope and frozen inputs

Authorised: use Rivanna and queue time/compute-heavy work now, retain data and
experiment records so later analysis is inexpensive. No push to possession-model.

Frozen source: `602b6139b8fe82d0256515102444534bf94b32d7`.
Frozen CSV SHA-256: `557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c`.
Research branch: `codex/cluster-verification-20261116`, separate local clone.
The submission branch and original checkout remain unchanged.

Local: `/Users/pratikshrestha/Documents/Codex/2026-11-15/new-chat-2/possession-research`.
Cluster code/logs: `/home/jpj8rf/Documents/github/possession-research-20261116`.
Cluster data/environment/results: `/scratch/jpj8rf/possession-research-20261116`.
Scratch outputs must be backed up to local/durable storage after jobs finish.

Scope: (1) Linux reproduction of the latest six-network submission, twice;
(2) reconstruct all other-match cross-validation probabilities for LightGBM and
six CNN variants, retaining CNN fold checkpoints; (3) frozen-decoder metrics,
per-match failure summaries and match-bootstrap uncertainty. Do not train,
select or tune against the BAR-ATH public labels. These CV matches were already
used for model selection; reconstruction and bootstrap do not turn them into
an untouched test set. No claim of independent full-match BAR-ATH accuracy.

Working cap pending any user adjustment: 8 GPU-hours, 32 CPU core-hours,
20 GiB new storage. Planned job wall-time reservations stay below these caps;
failed work and retries count toward the same budget. Account `nssac_students`
is explicit because the user's default account is a class account.

## Inputs

Three tactical assets staged from the active pipeline checkout. Sizes and hashes
are recorded in `tactical-manifest.json`. The model requires tactical tracking
TXT/XML, not video. Smart Tagging is staged for completeness but unused by the
frozen model. Videos, broadcast and homography are not required for these jobs
and have not been downloaded.

The 14 external files (tracking + metadata for all seven IDSSE matches) are
pinned by Figshare file ID in `prepare_data.py`. `idsse-source.json` is the
publisher's API record, including CC-BY 4.0, sizes and MD5 checksums. Downloads
use temporary files and atomic renames; every file is checked against the
published size and MD5 and recorded with SHA-256 in `verified-inputs.json`.
The data preparation job also checks staged tactical SHA-256 hashes, converts
all seven matches, and records class counts and the Linux dependency lock.

## Prepared jobs

- setup: 4 CPU, 96 GiB RAM, 45 min, standard; dependency install, verified downloads,
  input audit, conversions and BAR-ATH cache.
- reproduce: 8 CPU, 32 GiB RAM, 40 min, standard; frozen prediction twice and hash check.
- cv-lgb: 8 CPU, 32 GiB RAM, 60 min, standard; original LightGBM CV, separate audit decoder.
- cv-tcn: six tasks, 2 CPU/one A100/96 GiB RAM/60 min each; three folds per
  submitted variant, no changes to submitted weights. Retains fold models and OOF probabilities.
- aggregate: 2 CPU, 8 GiB RAM, 30 min; frozen ensemble and official smooth50
  on identical noisy other-match inputs, per-match metrics, period-boundary
  audit and 10,000 paired match-bootstrap draws. No competition public labels.

Submission scripts are saved in research/. GPU readiness will be checked before
submitting the training array. No new model is substituted into the submission.

## Queued Slurm jobs

```json
{
  "setup": "23038805",
  "reproduce": "23038817",
  "cv_lgb": "23038818",
  "gpu_smoke": "23038819",
  "cv_tcn_array": "23038820",
  "aggregate": "23038821"
}
```

All downstream jobs depend on successful input preparation. GPU training additionally
depends on a CUDA smoke test. The array allows two concurrent GPU tasks, six
GPU-hours maximum in aggregate. Invalid dependencies cancel rather than linger.

## Monitoring interruption (2026-11-16, approximately 12:12 America/New_York)

SSH previously succeeded and setup job 23038805 was observed RUNNING with pinned
packages installed and the first three external matches checksum-verified.
Later monitoring timed out connecting to port 22, before authentication. DNS
still resolves login.hpc.virginia.edu. macOS network inspection showed only en0,
no configured scutil VPN service, and the UVA address routed through the default
Wi-Fi gateway. No VPN/network settings were changed. A request to restore the
UVA route was sent to the user. This is a new connectivity interruption, not the
earlier sandbox DNS problem. Queued jobs and dependencies remain on Slurm;
completion, GPU readiness and the full external-data manifest are not yet verified.

Next monitoring must inspect sacct for ALL recorded job IDs, including failed
or cancelled dependent jobs; retrieve setup logs, data manifests, environment
lock and any reproduction-hash report before claiming success. Do not blindly
resubmit jobs or overwrite partial results. Include failed jobs and any retry
reservations in the budget. Remaining queued wall-time limits bound resource use
without this chat connection.

A local read-only collection helper, `research/collect.py`, is prepared. It
records sacct/squeue state and backs up cluster logs, results and all staged
input/cached data into gitignored research/artifacts/. Source inspection confirms
the 14 required external XML files total 2,622,686,298 bytes; availability of all
14 on the cluster remains unconfirmed until monitoring is restored. The local
frozen submission hash and all original model/source files remain unchanged.

## Access restored; verified results and recovery

User authorised up to **24 GPU-hours**. CPU cap remains 32 core-hours and
storage cap 20 GiB.

- Setup 23038805 completed in 10m07s. All 17 staged input files (three tactical,
  14 external) passed checksums; all seven IDSSE matches converted.
- Reproduction 23038817 completed in 6m34s. BOTH regenerated CSVs match the
  frozen SHA-256 exactly. This verifies latest six-CNN inference on Linux.
- CUDA smoke 23038819 completed in six seconds on A100-SXM4-80GB.
- LGB CV 23038818 timed out at 1h00m24s. Only J03WMX and J03WN1 predictions
  persisted. Their expensive computation will be reused.
- Original aggregate 23038821 was cancelled due to the failed dependency.
- Original GPU array 23038820 had not started; its A100-only estimate was
  November 18, after the deadline. Cancelled that pending array, without GPU use.
- Resumable LGB array 23041361: fold zero reuses saved probabilities; remaining
  folds run separately, retain models, and omit the unnecessary decoder grid.
  Four CPUs, 30 min per task (two actual training tasks plus a quick reuse task).
- Replacement GPU array 23041362: one CPU, 48 GiB RAM, one GPU, 2h50m per task;
  six tasks with up to three concurrent. Compatible A100/A40/A6000/B200/RTX Pro
  6000 nodes are eligible; V100 is excluded by feature constraints. Maximum
  GPU reservation 17 hours, within the revised 24-hour cap. CPU wall-time
  reservations plus prior allocated usage stay approximately 31.1 core-hours.

A private local backup of all inputs, caches, results and logs is in progress.
No submitted model or decoder was changed. Reconstructed CV scores may differ
from the original GPU training because its kernels were not made bit-deterministic.

### Shorter fold jobs for backfill

Pending array 23041362 and dependent aggregate 23041369 were cancelled before
starting. Replaced by **23041383**, 18 independent variant/fold tasks, each one
GPU + one CPU, 48 GiB RAM and 50 minutes, at most six concurrent. Maximum
reservation 15 GPU-hours / 15 CPU core-hours. Shorter jobs are easier to backfill
than three folds in one 2h50m allocation. Retained checkpoints and predictions
are written atomically. Aggregate **23041384** depends on the new array and
LGB recovery 23041361. Revised reservations including spent CPU time total
approximately 29.1 CPU core-hours, leaving retry margin below 32.

Private local collection completed. All three tactical and 14 external inputs
were independently rechecked against their manifests after transfer to the Mac.
The two reproduction CSVs and environment lock are now backed up locally.

## Reconstruction completed; fixed stress protocol

All 18 CNN fold jobs in 23041383 completed successfully (approximately 2 to 9.5
minutes each). Missing LightGBM folds 23041439 completed. The recovery runner
explicitly matches LightGBM thread count to allocated CPUs, avoiding eight
threads on a four-CPU allocation. Earlier recovery tasks were cancelled after
about four minutes to correct this mismatch; their usage is included. First-fold
LightGBM retention job 23041457 also completed, so all three tree models and
all 18 CNN fold models are now retained. Final aggregate 23041458 completed.

Reconstructed frozen ensemble mean spellF1 is **0.6382927** (original reported
0.637). Paired official smooth50 on identical saved noise is **0.1924802**,
not the 0.182 reported in SOLUTION.md. Original baseline benchmark driver was
not committed, so the discrepancy cannot be resolved by replaying that driver.
Our comparison uses the shipped starter.baseline.smooth implementation and
asserts identical observation masks across model probability files.
Match-bootstrap 95% interval for ensemble mean: [0.6060711, 0.6662003]; paired
improvement: [0.4143942, 0.4739778]. These intervals do not correct model-selection
optimism. No competition public labels were opened for this experiment.

### Stress benchmark protocol, fixed before execution

Reuse the three held-out tree models and 18 held-out CNN models; freeze decoder
and ensemble weights. Seven matches x three new corruption seeds (101,202,303),
with five cases: no added ball dropout, rotated target observation mask, 40% IID
dropout, 40% Markov dropout with 2-second mean gaps, and 40% Markov dropout with
10-second mean gaps. Keep player dropout and augmented dead-ball trajectories
identical across the five cases within each match/seed. The no-dropout case
still has player noise and augmented dead-ball carries; it is not raw DFL data.

Use fold models that never trained on the evaluated match. New seeds are noise
replications, not new independent matches. This is sensitivity analysis of the
selected pipeline, not an independent test or another hyperparameter search.
Match labels never enter the mask generators. Retain per-member and ensemble
probabilities for future CPU-only analyses; record realised missingness, gaps,
D precision/recall, spellF1, macroF1, Brier score and 15-bin calibration error.
Check CPU versus GPU probabilities on a full sequence per task; fail if maximum
absolute discrepancy exceeds 0.002. Disable cuDNN TF32 for this comparison.

Planned GPU cap for stress: 21 tasks x 20 minutes = 7 GPU-hours (and 7 CPU
core-hours), with an initial one-task smoke test before submitting the remaining
20 tasks. Prior actual accounting and storage are checked before submission.

Stress smoke 23042202 started on RTX A6000. The first full-match CPU/GPU
comparison passed and the visible-ball case completed in 232 seconds. Future
stress tasks request 30 minutes rather than 20 (21 x 30 min = 10.5 GPU/core
hours maximum; total remains below both caps). Slurm denied extending the
already-running smoke job; its existing 20-minute limit remains. Per-case
probabilities are atomic; recovery now reuses completed cases and rescoring
records, rather than repeating inference. The protocol, models, masks and
selection rules are unchanged. Ensemble probabilities retain float64 and decoded
labels in new caches; the initial smoke uses float32 probability caches.

Pilot timing: first two cases each required approximately 226-232 seconds,
mostly CPU tree inference. To reduce GPU idle time, replaced the still-pending
20-task one-CPU array with four-CPU, 12-minute tasks. Worst-case reservation
20 x 4 x 12 min = 16 CPU core-hours and 4 GPU-hours. Together with prior actual
12.50 core-hours, the pilot/resume limits and final summary, reservation remains
under 32 core-hours and 24 GPU-hours. No experimental choices changed.

The first two stress cases passed the full CPU/GPU comparison, all model calls
and scoring. Released remaining array 23042378 from the completed CUDA gate
23038819 to overlap it with the pilot's remaining cases. Each task performs
its own CPU/GPU check. Summary 23042379 still waits for all 21 complete reports,
including the resumable pilot 23042339. CPU-only cache replay 23042401 follows
the summary and explicitly asserts that Torch was not imported.

Slurm rejected replacement dependency `afterok:23038819` with a job-dependency
error. After manually confirming all input, CUDA and full-sequence pilot gates,
cleared the pending array dependency directly (`Dependency=0`). Readback showed
no dependency, and tasks 23042378_1 and _2 were observed RUNNING. This corrects
the earlier log wording: the completed job dependency itself was not accepted.

Normalization note for future work: the TCN's convolutions have about 80 seconds
of nominal receptive field, but GroupNorm also computes sequence-wide statistics
for input shape (N,C,T). Training on 4096-frame chunks and inference on whole
matches therefore use different normalization scopes. Stress inference keeps
whole-match input shape and checks CPU/GPU agreement; it does not chunk the
sequence for speed. This is a documentation/transfer limitation, not a change
to the frozen model. PyTorch reference:
https://docs.pytorch.org/docs/2.15/generated/torch.nn.GroupNorm.html

Artifact audit 23042558 passed: 18 CNN fold checkpoints, three retained tree
models, 49 original validation probability files; all training/validation match
sets disjoint and all probabilities finite, normalized and correctly aligned.
Period-boundary audit 23042638 is queued after CPU replay. It reconstructs the
paired baseline independently and compares its scores against saved reports,
then inserts an X separator between compact IDSSE halves and reports metric
differences alongside the fixed original-scoring summary.

The ungraded BAR-ATH intervals also enter whole-sequence CNN normalization,
whereas the converted other-match records are compact halves. The current
missingness experiment does not estimate that normalization-domain mismatch.
This limitation remains for future investigation; it is not a reason to alter
the frozen submission without separate validation.

Failure analysis 23042752 completed in four seconds. Duration-stratified GT
possession detection recall, averaged across IoU .3-.7: under 2s 0.2503
(919 spells), 2-5s 0.5229 (934), 5-10s 0.7315 (569), 10-20s 0.8444 (369),
and at least 20s 0.9422 (270). DFL definitions include shorter spells than the
competition; these diagnostics do not justify increasing target-game switches
without separate validation. Cached probabilities allow decoder trade-off
analysis without further GPU training.

Two monitoring attempts later received SSH key-exchange connection resets
from login.hpc.virginia.edu. Jobs are unaffected; retry monitoring after a
brief backoff rather than resubmitting work or changing credentials.
