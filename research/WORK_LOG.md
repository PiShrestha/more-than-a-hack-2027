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
