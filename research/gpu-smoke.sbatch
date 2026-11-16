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
nvidia-smi
python - <<'CHECK'
import torch,json,os
assert torch.cuda.is_available(), 'CUDA unavailable'
x=torch.randn(8,111,4096,device='cuda')
net=torch.nn.Conv1d(111,96,3,padding=1).cuda()
net(x).sum().backward()
d=dict(torch=torch.__version__,built_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability())
print(d)
open(os.environ['RESEARCH_RESULTS']+'/cuda-ready.json','w').write(json.dumps(d,indent=2))
CHECK
