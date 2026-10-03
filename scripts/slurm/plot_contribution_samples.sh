#!/bin/bash
#SBATCH --job-name=plot-contributions
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
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
export MPLBACKEND=Agg
export MPLCONFIGDIR="/global/home/users/pranavwalimbe/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

shared_dataframe="/global/scratch/projects/fc_nitrates/ddp/nox/dataset/dataframes/val_df.csv"
node_work_dir="/tmp/no2-contribution-plot-${SLURM_JOB_ID:?SLURM_JOB_ID is not set}"
node_dataframe_dir="${node_work_dir}/dataframes"
mkdir -p "${node_dataframe_dir}"

for attempt in 1 2 3; do
    if cp "${shared_dataframe}" "${node_dataframe_dir}/val_df.csv"; then
        break
    fi
    if [[ "${attempt}" -eq 3 ]]; then
        echo "Could not stage validation metadata from shared storage on ${HOSTNAME}" >&2
        exit 1
    fi
    rm -f "${node_dataframe_dir}/val_df.csv"
    sleep 10
done

output="/global/home/users/pranavwalimbe/vis/contribution-samples-${SLURM_JOB_ID}.png"
recipient="pranav.walimbe@berkeley.edu"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m modeling.plot_contribution_samples \
    --split val \
    --dataframe-dir "${node_dataframe_dir}" \
    --samples-per-class 100 \
    --columns 20 \
    --sigma 1.0 \
    --output "${output}"

if [[ ! -s "${output}" ]]; then
    echo "Expected contribution contact sheet is missing or empty: ${output}" >&2
    exit 1
fi

mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'Attached are 100 validation decrease and 100 validation increase spatial contribution targets. No NO2 rasters are included.\n\nArtifact: %s\nJob: %s\n' \
    "${output}" \
    "${SLURM_JOB_ID}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'Spatial contribution raster samples (${SLURM_JOB_ID})' -a '${output}' '${recipient}'"

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
echo "Confirmed SMTP delivery of ${output} to ${recipient}"
