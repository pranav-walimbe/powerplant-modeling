#!/bin/bash
#SBATCH --job-name=optimize-no2-model
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<EMAIL>
#SBATCH --output=/global/home/users/<USER>/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/<USER>/no2-modeling/logs/%x-%j.err

set -euo pipefail

cd "/global/home/users/<USER>/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/<USER>/no2-modeling/src:/global/home/users/<USER>/no2-modeling/src/delta-model"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="/global/home/users/<USER>/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

shared_dataset_dir="/global/scratch/projects/<ACCOUNT>/ddp/nox/dataset"
node_work_dir="/tmp/no2-model-hpo-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
mkdir -p "${node_work_dir}"
if [[ ! -w "${node_work_dir}" ]]; then
    echo "Job-local storage is not writable: ${node_work_dir}" >&2
    exit 1
fi
node_dataframe_dir="${node_work_dir}/dataframes"
echo "Staging split metadata into ${node_dataframe_dir}"
staging_succeeded=false
for attempt in 1 2 3; do
    staging_dir="${node_work_dir}/dataframes-stage-${attempt}"
    echo "Dataset staging attempt ${attempt}/3 on ${HOSTNAME}"
    if cp -a "${shared_dataset_dir}/dataframes" "${staging_dir}"; then
        mv "${staging_dir}" "${node_dataframe_dir}"
        staging_succeeded=true
        break
    fi
    sleep 10
done
if [[ "${staging_succeeded}" != true ]]; then
    echo "Could not stage split metadata from shared storage on ${HOSTNAME}" >&2
    exit 1
fi
export NO2_DATASET_DF="${node_dataframe_dir}"
df -h "${node_work_dir}"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m modeling.optimize \
    --device cuda \
    --completed-raster-dir "${node_work_dir}/filled-rasters" \
    --trials "${HPO_TRIALS:-24}" \
    --batch-size 128 \
    --epochs 100 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --prefetch-factor 2 \
    --seed 42
