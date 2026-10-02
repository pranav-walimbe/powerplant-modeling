#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 RUN_DIR JOB_ID RECIPIENT" >&2
    exit 2
fi

run_dir="$1"
job_id="$2"
recipient="$3"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"
class_f1_plot="${run_dir}/split_class_f1.png"
loss_plot="${run_dir}/training_curves.png"
characteristic_plot="${run_dir}/accuracy_by_characteristic.png"
joint_samples_plot="${run_dir}/joint_prediction_samples.png"
ops_evaluation_plot="${run_dir}/plant_ops_evaluation.png"

for artifact in \
    "${class_f1_plot}" \
    "${loss_plot}" \
    "${characteristic_plot}" \
    "${joint_samples_plot}" \
    "${ops_evaluation_plot}"; do
    if [[ ! -s "${artifact}" ]]; then
        echo "Expected result artifact is missing or empty: ${artifact}" >&2
        exit 1
    fi
done

echo "Emailing seasonal and vision-model results to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'NO2 classification training completed successfully.\n\nThe attachments compare seasonal and context-aware vision models, including their per-class F1 scores on the pooled plant-operations cohort over validation and test AOIs. The joint-sample image shows 20 test examples with NO2 inputs, transport residuals, true targets, and predictions.\n\nRun: %s\nJob: %s\n' \
    "${run_dir}" \
    "${job_id}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 classification results (${job_id})' \
            -a '${class_f1_plot}' -a '${loss_plot}' -a '${characteristic_plot}' \
            -a '${joint_samples_plot}' -a '${ops_evaluation_plot}' '${recipient}'"

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
echo "Confirmed SMTP delivery of model results to ${recipient}"
