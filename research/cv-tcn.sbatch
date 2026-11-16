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
case "$SLURM_ARRAY_TASK_ID" in
  0) export TCN_TAG='' ;;
  1) export TCN_TAG=_r8 TCN_REAL=8 ;;
  2) export TCN_TAG=_e60 TCN_EPOCHS=60 ;;
  3) export TCN_TAG=_w128 TCN_WIDTH=128 ;;
  4) export TCN_TAG=_s1 TCN_SEED=1 ;;
  5) export TCN_TAG=_s2 TCN_SEED=2 ;;
  *) exit 2 ;;
esac
nvidia-smi
python -W ignore research/cv_tcn.py
