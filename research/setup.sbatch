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
mkdir -p "$RESEARCH_RESULTS"
python -m venv /scratch/jpj8rf/possession-research-20261116/venv
source /scratch/jpj8rf/possession-research-20261116/venv/bin/activate
python -m pip install --no-cache-dir -r requirements.txt -r requirements-solution.txt
python -m pip freeze > "$RESEARCH_RESULTS/environment-lock.txt"
python research/prepare_data.py
python -m possession.idsse
python -m possession.data
python - <<'CHECK'
import torch,json,os
from possession import idsse
print('torch',torch.__version__,'built CUDA',torch.version.cuda)
summary=[]
for m in idsse.MATCHES:
 d=idsse.load(m)
 assert len(d['y'])==len(d['period']) and len(d['y'])>100000
 assert set(d['y']).issubset({-1,0,1,2})
 summary.append(dict(match=m,frames=len(d['y']),class_counts={str(k):int((d['y']==k).sum()) for k in (-1,0,1,2)}))
open(os.environ['RESEARCH_RESULTS']+'/dataset-summary.json','w').write(json.dumps(summary,indent=2))
CHECK
