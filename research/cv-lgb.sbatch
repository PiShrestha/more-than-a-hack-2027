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
export POSS_TAG=_audit
python -W ignore -m possession.model cv
# Research CV chooses/writes only decoder_audit.json, never the frozen decoder.json.
for f in "$MTAH_DATA_DIR"/external/idsse/cv_audit_*.npz; do
  cp "$f" "${f/cv_audit_/cv_}"
done
