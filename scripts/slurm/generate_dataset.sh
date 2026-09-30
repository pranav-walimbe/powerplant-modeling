#!/bin/bash
#SBATCH --job-name=generate-dataset
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%A_%a.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%A_%a.err

set -euo pipefail

cd "/global/home/users/pranavwalimbe/no2-modeling"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="/global/home/users/pranavwalimbe/no2-modeling/src:/global/home/users/pranavwalimbe/no2-modeling/src/delta-model"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m preprocessing.generate_dataset "$@"

if [[ "${DATASET_GENERATION_STAGE:-generate}" == "finalize" ]]; then
    recipient="pranav.walimbe@berkeley.edu"
    mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
    mail_log="/var/log/maillog"
    result_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
    diagnostic="/global/home/users/pranavwalimbe/vis/dataset-window-behavior-${result_job_id}.png"

    if [[ ! -s "${diagnostic}" ]]; then
        echo "Expected post-filter diagnostic was not created: ${diagnostic}" >&2
        exit 1
    fi

    remote_artifact_visible=false
    for _ in {1..60}; do
        if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" "test -s '${diagnostic}'"; then
            remote_artifact_visible=true
            break
        fi
        sleep 2
    done
    if [[ "${remote_artifact_visible}" != true ]]; then
        echo "Post-filter diagnostic did not become visible from ${mail_host}" >&2
        exit 1
    fi

    mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
    printf '%s\n' \
        'Five-timestep dataset generation completed successfully.' \
        'The attached plot shows median and interquartile emissions trajectories after raster filtering and class balancing.' \
        "Diagnostic: ${diagnostic}" \
        | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
            "mailx -s 'NO2 post-filter dataset diagnostic (${result_job_id})' -a '${diagnostic}' '${recipient}'"

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
    echo "Confirmed SMTP delivery of post-filter diagnostic to ${recipient}"
fi
