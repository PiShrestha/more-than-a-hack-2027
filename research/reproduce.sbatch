#!/bin/bash
#SBATCH --account=nssac_students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export MTAH_DATA_DIR=/scratch/jpj8rf/possession-research-20261116/data
export RESEARCH_RESULTS=/scratch/jpj8rf/possession-research-20261116/results
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
module purge
module load miniforge/26.9.4
hostname
date -u
source /scratch/jpj8rf/possession-research-20261116/venv/bin/activate
expected=557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c
for repetition in 1 2; do
  python -W ignore -m possession.model predict
  python evaluation/validate_submission.py submission/submission.csv
  got=$(sha256sum submission/submission.csv | cut -d ' ' -f 1)
  cp submission/submission.csv "$RESEARCH_RESULTS/reproduction-${repetition}.csv"
  printf '%s %s\n' "$repetition" "$got" | tee -a "$RESEARCH_RESULTS/reproduction-hashes.txt"
  if [[ "$got" != "$expected" ]]; then
    echo 'HASH MISMATCH: preserve outputs for investigation; do not change the frozen submission.'
    exit 2
  fi
done
