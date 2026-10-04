#!/bin/bash
#SBATCH --job-name=train-no2-classifier
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src:/global/home/users/pranavwalimbe/no2-modeling/src/delta-model"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/global/home/users/pranavwalimbe/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

shared_dataset_dir="/global/scratch/projects/fc_nitrates/ddp/nox/dataset"
node_work_dir="/tmp/no2-model-training-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
mkdir -p "${node_work_dir}"
if [[ ! -w "${node_work_dir}" ]]; then
    echo "Job-local storage is not writable: ${node_work_dir}" >&2
    exit 1
fi
node_dataframe_dir="${node_work_dir}/dataframes"
echo "Staging split metadata into ${node_dataframe_dir}"
cp -a "${shared_dataset_dir}/dataframes" "${node_dataframe_dir}"
export NO2_DATASET_DF="${node_dataframe_dir}"
df -h "${node_work_dir}"

mkdir -p "/global/home/users/pranavwalimbe/.cache"
training_output=$(mktemp "/global/home/users/pranavwalimbe/.cache/train-no2-${SLURM_JOB_ID}.XXXXXX.log")
trap 'rm -f -- "${training_output}"' EXIT
recipient="pranav.walimbe@berkeley.edu"
raster_loss_weight="${RASTER_LOSS_WEIGHT:-1.0}"
contribution_splat_sigma="${CONTRIBUTION_SPLAT_SIGMA:-1.0}"
sinkhorn_blur_cells="${SINKHORN_BLUR_CELLS:-1.0}"
sinkhorn_reach_cells="${SINKHORN_REACH_CELLS:-6.0}"
sinkhorn_iterations="${SINKHORN_ITERATIONS:-30}"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m modeling.train \
    --device cuda \
    --completed-raster-dir "${node_work_dir}/filled-rasters" \
    --batch-size 128 \
    --epochs 100 \
    --seasonal-epochs 75 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --prefetch-factor 2 \
    --seed 42 \
    --model-width 64 \
    --head-dim 48 \
    --dropout 0.20 \
    --learning-rate 6.0378112892784365e-5 \
    --weight-decay 1e-4 \
    --gradient-clip-norm 5.0 \
    --scheduler-patience 10 \
    --scheduler-factor 0.50 \
    --early-stop-patience 10 \
    --raster-loss-weight "${raster_loss_weight}" \
    --contribution-splat-sigma "${contribution_splat_sigma}" \
    --sinkhorn-blur-cells "${sinkhorn_blur_cells}" \
    --sinkhorn-reach-cells "${sinkhorn_reach_cells}" \
    --sinkhorn-iterations "${sinkhorn_iterations}" \
    | tee "${training_output}"
echo "Training command completed; locating result artifacts"

run_dir="$(sed -n 's/^Training .*; outputs: //p' "${training_output}" | tail -n 1)"
if [[ -z "${run_dir}" ]]; then
    echo "Could not determine the model run directory from the training output." >&2
    exit 1
fi

echo "Emailing seasonal and vision-model results"
bash scripts/slurm/email_model_results.sh \
    "${run_dir}" \
    "${SLURM_JOB_ID}" \
    "${recipient}"
