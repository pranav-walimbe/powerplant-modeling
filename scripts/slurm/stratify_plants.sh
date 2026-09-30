#!/bin/bash
#SBATCH --job-name=stratify-plants
#SBATCH --account=fc_nitrates
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --constraint=savio4_m512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pranav.walimbe@berkeley.edu
#SBATCH --output=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.log
#SBATCH --error=/global/home/users/pranavwalimbe/no2-modeling/logs/%x-%j.err

set -euo pipefail

repo_dir="${NO2_REPOSITORY_DIR:-/global/home/users/pranavwalimbe/no2-modeling}"
strat_dir="/global/scratch/projects/fc_nitrates/ddp/nox/nox_powerplant_data"
recipient="pranav.walimbe@berkeley.edu"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
diagnostic="/global/home/users/pranavwalimbe/vis/stratification-window-behavior-${SLURM_JOB_ID}.png"
monthly_prevalence="/global/home/users/pranavwalimbe/vis/stratification-monthly-prevalence-${SLURM_JOB_ID}.png"
solar_hour_prevalence="/global/home/users/pranavwalimbe/vis/stratification-solar-hour-prevalence-${SLURM_JOB_ID}.png"
aoi_characteristics_plot="/global/home/users/pranavwalimbe/vis/stratification-aoi-characteristics-${SLURM_JOB_ID}.png"
aoi_characteristics_table="/global/home/users/pranavwalimbe/vis/stratification-aoi-characteristics-${SLURM_JOB_ID}.csv"
aoi_selection_table="/global/home/users/pranavwalimbe/vis/stratification-aoi-selection-${SLURM_JOB_ID}.csv"

cd "${repo_dir}"
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
export PYTHONPATH="${repo_dir}/src:${repo_dir}/src/delta-model"

# Slurm 22.05 and later stopped propagating --cpus-per-task into srun
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

srun python -u -m preprocessing.stratify_plants \
    --diagnostic-output "${diagnostic}" \
    --monthly-prevalence-output "${monthly_prevalence}" \
    --solar-hour-prevalence-output "${solar_hour_prevalence}" \
    --aoi-characteristics-plot "${aoi_characteristics_plot}" \
    --aoi-characteristics-output "${aoi_characteristics_table}" \
    --aoi-selection-output "${aoi_selection_table}" \
    "$@"

for artifact in \
    "${diagnostic}" \
    "${monthly_prevalence}" \
    "${solar_hour_prevalence}" \
    "${aoi_characteristics_plot}" \
    "${aoi_characteristics_table}" \
    "${aoi_selection_table}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected stratification artifact was not created: ${artifact}" >&2
        exit 1
    fi
done
train_records=$(($(wc -l < "${strat_dir}/train_records.csv") - 1))
val_records=$(($(wc -l < "${strat_dir}/val_records.csv") - 1))
test_records=$(($(wc -l < "${strat_dir}/test_records.csv") - 1))
total_records=$((train_records + val_records + test_records))

remote_artifacts_visible=false
for _ in {1..60}; do
    if ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "test -s '${diagnostic}' && test -s '${monthly_prevalence}' && test -s '${solar_hour_prevalence}' && test -s '${aoi_characteristics_plot}' && test -s '${aoi_characteristics_table}' && test -s '${aoi_selection_table}'"; then
        remote_artifacts_visible=true
        break
    fi
    sleep 2
done
if [[ "${remote_artifacts_visible}" != true ]]; then
    echo "Stratification artifacts did not become visible from ${mail_host}" >&2
    exit 1
fi

mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" stat -c %s "${mail_log}")
printf '%s\n' \
    'Five-timestep window stratification completed successfully.' \
    'Selected the half of AOIs with the highest activity-conditioned median NOx.' \
    'Change threshold: 250 lb/hr absolute.' \
    'Five causal rasters retained; labels compare time-weighted t0-t2 and t2-t4 averages.' \
    'Each timestep NOx value is linearly interpolated between its surrounding CAMPD hours.' \
    'Geographic AOI clusters were assigned by AOI and candidate-record counts to approximately 70/15/15 splits.' \
    'Training classes are downsampled to the smallest training class with a fixed seed.' \
    'Validation and test retain every change record, then uniformly sample steady to match the larger change class.' \
    "Train: ${train_records} records" \
    "Validation: ${val_records} records" \
    "Test: ${test_records} records" \
    "Total: ${total_records} records" \
    "Diagnostic: ${diagnostic}" \
    "Monthly prevalence: ${monthly_prevalence}" \
    "Solar-hour prevalence: ${solar_hour_prevalence}" \
    "AOI characteristics: ${aoi_characteristics_plot}" \
    "AOI characteristics table: ${aoi_characteristics_table}" \
    "AOI selection audit: ${aoi_selection_table}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 stratification diagnostics (${SLURM_JOB_ID})' -a '${diagnostic}' -a '${monthly_prevalence}' -a '${solar_hour_prevalence}' -a '${aoi_characteristics_plot}' -a '${aoi_characteristics_table}' -a '${aoi_selection_table}' '${recipient}'"

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
echo "Confirmed SMTP delivery to ${recipient}"
