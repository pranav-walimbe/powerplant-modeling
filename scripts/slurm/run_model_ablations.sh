#!/bin/bash
#SBATCH --job-name=ablate-no2-model
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_gpu
#SBATCH --qos=a5k_gpu4_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --time=08:00:00
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

shared_dataset_dir="/global/scratch/projects/fc_nitrates/ddp/nox/dataset"
node_work_dir="/tmp/no2-model-ablation-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
node_dataframe_dir="${node_work_dir}/dataframes"
mkdir -p "${node_work_dir}"
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

mkdir -p "/global/home/users/pranavwalimbe/.cache"
ablation_output=$(mktemp "/global/home/users/pranavwalimbe/.cache/ablate-no2-${SLURM_JOB_ID}.XXXXXX.log")
trap 'rm -f -- "${ablation_output}"' EXIT
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m modeling.ablation \
    --device cuda \
    --completed-raster-dir "${node_work_dir}/filled-rasters" \
    | tee "${ablation_output}"

run_dir="$(sed -n 's/^Running ablation study .*; outputs: //p' "${ablation_output}" | tail -n 1)"
summary_path="${run_dir}/ablation_summary.txt"
if [[ ! -s "${summary_path}" ]]; then
    echo "Ablation summary is missing or empty: ${summary_path}" >&2
    exit 1
fi

echo "Emailing ablation summary to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'NO2 input-ablation study completed successfully.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${SLURM_JOB_ID}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 ablation results (${SLURM_JOB_ID})' -a '${summary_path}' '${recipient}'"

delivery_confirmed=false
for _ in {1..30}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        tail -c "+$((mail_log_offset + 1))" "${mail_log}" \
        | grep -F "to=<${recipient}>" \
        | grep -Fq 'status=sent'; then
        delivery_confirmed=true
        break
    fi
    sleep 1
done
if [[ "${delivery_confirmed}" != true ]]; then
    echo "Could not confirm successful SMTP delivery in ${mail_log}" >&2
    exit 1
fi
echo "Confirmed SMTP delivery of ablation results to ${recipient}"
