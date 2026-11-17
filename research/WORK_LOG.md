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

Monitoring recovered through **128.143.223.181**, another address returned by
DNS for login.hpc.virginia.edu. The .182 frontend reset key exchange and .180
timed out; .181 authenticated with the existing user/config and normal host-key
verification via HostKeyAlias=login.hpc.virginia.edu. No credentials, SSH config
or host-key verification rules were changed. Collection/accounting helpers now
accept an optional UVA_SSH_FRONTEND for this legitimate frontend selection.
Fifteen complete stress reports were confirmed after recovery.

## Completed compute phase — 2026-11-16

All submitted experiment and audit jobs have finished. No active jobs remain.

Frozen source branch possession-model remains at 602b613; original model, decoder,
requirements, SOLUTION.md and submission CSV have no diff. Nothing was pushed.

All 17 staged inputs are checksum-verified on Rivanna and in the private Mac
backup. All seven IDSSE conversions are retained. Videos, homography and
broadcast files were not needed and were not downloaded.

Artifacts retained: 3 tree fold models, 18 CNN fold checkpoints, 49 original
OOF probability files, and 105 stress-case archives with 840 member/ensemble
probability matrices. These represent 15,039,660 frame evaluations but only
SEVEN independent matches; noise seeds are repetitions.

Frozen CSV regenerated twice on Linux and both outputs match the submitted
SHA-256 exactly. Artifact backup verification passed; all probabilities and
labels are aligned and finite.

GPU/CPU spot checks: maximum probability difference 2.086e-6, minimum argmax
agreement 1.0. CPU replay matched all saved scores exactly, and Linux/Mac replay
rows are identical. Linux replay took 11 seconds; Mac replay took 14.55 seconds
with neither Torch nor LightGBM installed in its isolated environment.

Period audit reconstructed all paired baseline masks/scores exactly. Inserting
a half separator changes at most 0.00255 spellF1 in an individual case; changes
to case means are below 0.0004. Legacy means below remain the preregistered
comparison; period-safe means are saved alongside them.

| Missing-ball case | Ensemble spellF1 | smooth50 spellF1 |
|---|---:|---:|
| visible | 0.6942 | 0.2707 |
| target_like | 0.6306 | 0.1876 |
| iid40 | 0.6908 | 0.2451 |
| burst2s40 | 0.6524 | 0.2028 |
| burst10s40 | 0.6331 | 0.1895 |

At about 40% removal, IID missingness has little effect (.6942 -> .6908),
while 2-second and 10-second burst gaps reduce the mean to .6524 and .6331.
The target-like pattern yields .6306. These are synthetic, selected-CV results,
not a private BAR-ATH score or an untouched test. The target-like mask comes
from licensed observations; exact template-dependent reproduction needs the
permitted inputs and the mask itself must not be redistributed.

Allocated cluster usage, including failures and cancellations: 3.1533 GPU-hours / 24 allowed; 19.4394 CPU core-hours / 32 allowed. Measured storage across cluster and local project: 15.69 GiB / 20 allowed.

Final housekeeping removed 17 checksum-proven intermediate duplicates
(629,264,810 bytes) from cluster home. A historical status file and an unmatched
old rsync partial file were deliberately retained privately; they are not used
by the pipeline. Root .gitignore now excludes all private research backups.

Cheap remaining submission work: confirm registered team name, create renamed
copies of CSV/doc, have the team review document wording, and designate the
single Google Form submitter. No registration identity was guessed and no form
was submitted. Further decoder/ensemble/uncertainty analyses can reuse cached
probabilities without GPU training.

## Continued analysis and submission preparation

Official organizer submission email dated November 13 confirms the deadline is
November 17, 2026, 23:59 CET = **17:59 America/New_York**. Found the official
submission form. The registration confirmation did not contain the registered
team name; requested that exact name from the user. No email or form was sent.

Manual code review identified a precision confound in research/replay.py:
penalty-only overrides unnecessarily recomputed the frozen ensemble from
float32 member caches. Fixed the research replay helper to preserve the exact
cached ensemble whenever mixture weight/member set is unchanged. Added validated
configuration handling and optional period-safe scoring. Seven research tests
pass, including exact invariance under unchanged mixture overrides. Frozen
submission code and CSV remain unchanged.

Queued CPU-only sensitivity job 23046173: 26 predeclared configurations x
105 cases = 2,730 score evaluations. Variants change one penalty at a time,
mixture weight, or member composition; they are descriptive sensitivity checks,
not an automatic selection of a replacement model. Protocol is saved in
sensitivity-protocol.json. The job requests one CPU, 4 GiB RAM and 30 minutes,
with no GPU, within the remaining original CPU budget.

Staged exact source CSV and unchanged document in private
artifacts/submission-packages/review-staging/. The packaged CSV passes the
official validator and preserves the submitted SHA-256. package_submission.py
can create named copies once the exact registered team name is supplied.
The manifest includes the verified deadline and form URL and explicitly marks
team/document review outstanding. Form submission is not performed by this tool.

Prepared solution-review.patch and a private proposed Markdown preview. The
patch qualifies selected-CV claims, records separate reconstructed results,
corrects the external dataset citation and mask-rotation wording, explains
GroupNorm's global context, and records the frozen source/CSV hashes. It applies
cleanly but has NOT been applied to SOLUTION.md. The team should review and
rewrite the final wording as requested by the teammate.

### Fixed normalization/context probe

Context protocol committed before evaluation: four CNN sequence contexts on
all seven external matches, fixed held-out fold models, unchanged play feature
vectors, unchanged tree probabilities and decoder. Compare compact full match,
independent halves, neutral standardized ungraded padding and observed automatic
BAR-ATH context outside its published halves. No BAR-ATH labels are loaded; no
context feature vectors are exported. Interior comparisons exclude 60 seconds
from each half boundary to separate direct convolution-edge effects from the
network's longer-range context dependence.

Budget: seven one-GPU/one-CPU tasks, at most 15 minutes each, plus a five-minute
CPU summary. Maximum new reservation 1.75 GPU-hours and 1.84 CPU core-hours;
new storage estimated below 0.8 GiB across cluster and local backup. This remains
within the cumulative 24 GPU-hour / 32 core-hour / 20 GiB caps.

Pilot 23046726; remaining fold probe 23046727; summary 23046728.

Context provenance clarification from code inspection: the inserted rows are
selected at BAR-ATH ungraded timestamps, but the frozen feature extractor's
rolling and gap features can reference automatic observations across half
boundaries. Thus this is observed *feature context at ungraded times*, not
features computed exclusively from ungraded raw observations. All observations
are permitted automatic inputs and no BAR-ATH labels are read. No algorithm,
case or parameter changed; the protocol wording has been corrected accordingly.

The sensitivity study completed all 2,730 evaluations in 125.8 seconds. Under
the same fixed decoder, target-like spellF1 was .63091 for the frozen six-CNN
mixture, .62264 for the three-seed subset and .61257 for the base-network subset.
Removing trees (all-CNN mixture) gave .62680; trees alone with the frozen mixed
decoder gave .55508. These are conditional ablations, not comparisons after
separately optimizing each model's decoder. Halving/doubling the team-switch
penalty changed target-like spellF1 by about -.00354/-.00277. No replacement
configuration was selected; the frozen submission is preserved.

### Matched-capacity normalization ablation

The context probe completed 28 evaluations. Observed ungraded feature context
reduced period-safe external-match spellF1 from .63856 to .61745 (paired
match-bootstrap delta -.02111, 95% interval [-.02838,-.01276]). Independent
halves changed the mean by -.00035; neutral context by +.00190. Observed context
changed 1.38% of interior argmax labels, even >60 seconds from half boundaries.
The maximum compact CPU/GPU check discrepancy was 3.16e-6.

This motivates one controlled research ablation, fixed in frame-norm-protocol.json
before training: replace sequence-wide GroupNorm with the same channel grouping
and affine parameters applied independently at each frame. Keep width 96, seed
0, four noise realizations, 30 epochs, optimizer/sampling and decoder unchanged.
Three original held-out folds; compare against the same-capacity stock base CNN
on four context cases and all seven external matches. Only one training seed is
used, so training variance remains a limitation. No BAR-ATH labels, no new
submission prediction, and no change to the frozen model.

Maximum additional reservation: 2.5 GPU-hours and 2.59 CPU core-hours, below
the remaining cumulative caps; output storage estimated below 0.4 GiB. Unit
check requires per-frame norm to be invariant to appended unrelated context;
whole-CNN interior prediction differences must stay below .002. All trained
checkpoint parameter shapes must match the stock base model.

Training pilot 23047267; other folds 23047268; evaluations 23047269; summary 23047270.

### Retained full-data tree and separate inference-scope candidate

Queued CPU job 23047439, eight CPUs / 15 minutes / 32 GiB RAM (maximum two
additional core-hours, no GPU). It fits and retains the original full-seven-match
LightGBM model, checks save/reload predictions, then requires the committed
stock CNN weights plus saved tree to reproduce the frozen CSV hash exactly.
Only after that gate, it creates a separately named half-isolated CNN inference
research candidate, keeping all features, tree probabilities, weights and decoder
unchanged. Neither source submission file nor document is overwritten. No
BAR-ATH label CSV is read and no target accuracy score is computed. The package
tool remains pinned to the original submission hash.

This saves the final tree for later CPU inference without retraining and makes
the context-isolation proposal concrete, while preserving the official frozen
submission and team review process.

### Completed follow-up findings and submission staging

All normalization jobs completed successfully. Per-frame normalization had zero
interior probability change under every inserted-context condition. With the
fixed tree mixture/decoder, compact spellF1 was .61675 versus .61420 for the
same-capacity stock base CNN; the paired difference .00255 had interval
[-.00805,.01273]. Under observed ungraded feature context the corresponding
scores were .61541 versus .60300, a difference .01241 with interval
[.00391,.01993]. This supports the context-invariance mechanism, but one seed
on seven previously selected games does not establish an overall accuracy gain.
No variant was selected and no extra ensemble training is justified by this run.

Full-data tree job 23047439 completed in 5m19s. Its saved/reloaded probabilities
were identical, and its original six-CNN inference regenerated the frozen CSV
exactly. The half-isolated research candidate passed format validation and
differed on 2,386 active frames; this is a prediction disagreement count, not
an accuracy comparison. No BAR-ATH label score was computed. A separate
CPU repeat job 23048161 reuses the saved tree and requires byte-identical
original/candidate CSV hashes across two inference runs. Maximum reservation
is two additional CPU core-hours, zero GPU-hours.

The user confirmed the registered team name `encara_messi`. The original CSV
and frozen technical-document draft have been staged as encara_messi.csv and
encara_messi.md under the private submission package directory. The CSV hash
is checked before packaging and the validator passed (199,319 frames). A
separate PROPOSED_encara_messi_FOR_TEAM_REVIEW.md is generated from the clean
solution-review.patch; it is not applied to the frozen source or substituted
for the team-reviewed document. The team still needs to review its wording and
designate a single form submitter before November 17, 2026 at 23:59 CET /
17:59 America/New_York. No form was sent and no branch was pushed.

### Final verification checkpoint

Repeat-inference job 23048161 completed successfully in 2m19s. Original and
research candidate CSVs were byte-identical between runs. Retained tree SHA-256:
51ce8b796907ff4b5f703bac7aa2e7c95f0edc306da345c3a50e03ed09744b85.
The half-isolated research candidate remains private and unsubmitted, hash
8c3449d054bbd5b16a1935175faa8308c04baaf4d864fee1d443b33782ac19fa.

After final rsync collection, verify_backup.py passed all 17 source-data
checksums, 105 stress archives / 840 member matrices, 56 additional context
archives, 18 stock and three per-frame fold checkpoints, three fold trees,
and the retained final tree / repeated candidate CSV hashes. The original CPU
replay scores matched exactly. All 75 source/model manifest hashes matched;
eight focused tests passed. The proposed solution patch passes apply --check.
Both the main user checkout and the frozen submission paths are unchanged.

Final allocated use including failed/cancelled jobs: 20.80694 CPU core-hours
and 3.4625 GPU-hours. Conservative gross storage across cluster scratch,
cluster checkout and this full local research checkout: 16.72332 GiB (original
checkout files included). All are within the 32 / 24 / 20 caps. No experiment
jobs remain unfinished. Further analysis of retained probabilities needs only
the lightweight CPU environment; all required tracking data and trained models
are privately backed up. Detailed machine-readable records are
resource-accounting.json, storage-accounting.json, followup-results.json and
backup-inventory.json. The remaining submission steps are team document review
and one designated form response; no push or form submission has occurred.

### Submission-document review and transfer bundle

Reviewed the staged technical document against the frozen classifier, CNN,
decoder, run.sh and submission rules. The revised draft explicitly labels
unretained historical scores as team-reported, preserves independently
reconstructed scores separately, explains offline Viterbi behavior and the
limits of inferred possession rules, and warns that the commented optional
ensemble command can overwrite the frozen decoder. The IDSSE paper title,
author order and CC-BY attribution were checked against the publisher record
and PubMed (https://pubmed.ncbi.nlm.nih.gov/39893199/).

The named private encara_messi.md now contains this edited draft, pending
team wording review. ORIGINAL_FROZEN_SOLUTION.md preserves the previous copy.
No repository SOLUTION.md or inference file was changed. The complete review
is retained as research/solution-review.patch; it applies cleanly to frozen
SOLUTION.md but has not been applied there.

package_submission.py now accepts an explicit Markdown source and optionally
builds a deterministic ZIP containing only the named CSV and Markdown. Existing
different documents are preserved with an error. Revised bundles receive a
content-hash filename so the prior bundle is preserved. Two consecutive
packaging runs produced identical bundles. ZIP entry names, hashes, CRCs and
the exact CSV validator passed. The ZIP is a convenience transfer artifact;
the Google Form takes the individual CSV and Markdown. No licensed tracking
assets, research probabilities, or research candidate are included.

This continuation consumed no additional cluster compute. Remaining external
actions: team document review and one designated form submission. No response
has been sent, and no branch has been pushed.

### Submission release cleanup, November 17

Created a clean submission branch from the frozen model source. The branch
contains the final 1,015-word Downloads document, the unchanged CSV, source and
input hash manifests, a standard-library verification/packaging command, an
isolated reproduction wrapper, eight release tests, and pinned GitHub Actions.
The organizer README and root SOLUTION.md remain available as historical
material. Research files remain on this branch. Submission is the GitHub
default branch and the main local workspace now tracks it. No source, weights,
decoder, or original CSV on possession-model changed.

The release CI passed at 6c45bc7. Both release package files matched Downloads
exactly, and the official validator passed. The initial end-to-end guarded
wrapper job 23076854 timed out at 15m09s after converting all external matches.
The job used eight CPUs, so it remains included in allocated accounting. A
single retry, 23077630, has a 30-minute limit (maximum four additional core-
hours). The retry is not recorded as passed until its exact CSV check succeeds.
No new model variants or BAR-ATH label evaluations are part of these checks.

The guarded fresh-cache retry 23077630 completed in 7m46s. It regenerated the
exact frozen CSV hash and the final Downloads document hash. The wrapper source
hash matched the code published on submission. Both the timeout and successful
retry are recorded publicly in submission/verification.json; a copy is retained
as research/release-verification.json. The final release is tagged
submission-2026-11-17. Model source, six checkpoints, decoder, and original CSV
remain byte-identical to 602b613. New CI verifies source/artifact hashes and safe
packaging without model training.

All required raw inputs were verified again using the release input manifest.
The release package matches Downloads exactly, and the official validator passed.
Latest allocated use including the timeout: 23.8625 CPU core-hours and 3.4625 GPU-
hours, within the 32/24 caps. No more model jobs are required for this cleanup.
