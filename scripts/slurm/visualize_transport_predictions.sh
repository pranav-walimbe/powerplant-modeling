#!/bin/bash
#SBATCH --job-name=transport-predictions
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src:/global/home/users/pranavwalimbe/no2-modeling/src/delta-model"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/global/home/users/pranavwalimbe/.cache/matplotlib"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

output_dir="/global/home/users/pranavwalimbe/vis/transport-predictions-${SLURM_JOB_ID}"
output_path="${output_dir}/transport_predictions.png"
mkdir -p "${output_dir}" "${MPLCONFIGDIR}"

srun python -u -m modeling.visualize_transport \
    --split test \
    --samples 10 \
    --seed 42 \
    --transition 3 \
    --output "${output_path}"

bash scripts/slurm/email_transport_predictions.sh \
    "${output_path}" \
    "${SLURM_JOB_ID}" \
    "pranav.walimbe@berkeley.edu"
