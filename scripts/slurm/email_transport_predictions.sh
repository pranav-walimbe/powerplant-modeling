#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 PNG_PATH JOB_ID RECIPIENT" >&2
    exit 2
fi

png_path="$1"
job_id="$2"
recipient="$3"
mail_host="${SLURM_SUBMIT_HOST:-ln002.brc}"
mail_log="/var/log/maillog"

if [[ ! -s "${png_path}" ]]; then
    echo "Expected transport visualization is missing or empty: ${png_path}" >&2
    exit 1
fi

echo "Emailing transport predictions to ${recipient} via ${mail_host}"
mail_log_offset=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    "${mail_host}" stat -c %s "${mail_log}")
printf 'The attached figure compares horizontal-advection predictions with observed TEMPO NO2 for ten class-balanced test records. Missing NO2 values were filled by spatial interpolation. No masked autoencoder, classifier, decay, or elevated-wind adjustment was used.\n\nJob: %s\nArtifact: %s\n' \
    "${job_id}" \
    "${png_path}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=15 "${mail_host}" \
        "mailx -s 'NO2 transport predictions (${job_id})' -a '${png_path}' '${recipient}'"

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
echo "Confirmed SMTP delivery of transport predictions to ${recipient}"
