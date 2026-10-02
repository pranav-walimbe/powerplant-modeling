#!/bin/bash
#SBATCH --job-name=generate-ops-dataset
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
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
export POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
summary="/global/home/users/pranavwalimbe/vis/ops-dataset-summary-${SLURM_JOB_ID}.txt"

srun python -u -m preprocessing.generate_ops_dataset "$@" | tee "${summary}"

if [[ ! -s "${summary}" ]]; then
    echo "Expected operations dataset summary was not created: ${summary}" >&2
    exit 1
fi

remote_artifact_visible=false
for _ in {1..60}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" "test -s '${summary}'"; then
        remote_artifact_visible=true
        break
    fi
    sleep 2
done
if [[ "${remote_artifact_visible}" != true ]]; then
    echo "Operations dataset summary did not become visible from ${mail_host}" >&2
    exit 1
fi

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Operations evaluation dataset generation completed successfully.' \
    'The attached text file summarizes AOI and class counts.' \
    "Summary: ${summary}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 operations dataset summary (${SLURM_JOB_ID})' -a '${summary}' '${recipient}'"

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
echo "Confirmed SMTP delivery of operations dataset summary to ${recipient}"
