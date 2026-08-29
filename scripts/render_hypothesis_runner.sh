#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: RENDER_SERVICE_ID=... RENDER_APPROVAL_ID=... RENDER_APPROVED_MAX_MINUTES=N $0 HYPOTHESIS_ID BASE_SHA CANDIDATE_SHA [health|smoke]" >&2
}

[[ $# -ge 3 && $# -le 4 ]] || { usage; exit 64; }

hypothesis_id="$1"
base_sha="$2"
candidate_sha="$3"
tier="${4:-smoke}"
service_id="${RENDER_SERVICE_ID:-}"
approval_id="${RENDER_APPROVAL_ID:-}"
approved_minutes="${RENDER_APPROVED_MAX_MINUTES:-}"
plan_id="${RENDER_PLAN_ID:-plan-srv-013}"
required_isa="${FLOCK_REQUIRED_ISA:-}"

[[ "${hypothesis_id}" =~ ^H-[A-Z0-9][A-Z0-9_-]{2,63}$ ]] || { echo "invalid hypothesis id" >&2; exit 64; }
[[ "${base_sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "base must be a full lowercase Git SHA" >&2; exit 64; }
[[ "${candidate_sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "candidate must be a full lowercase Git SHA" >&2; exit 64; }
[[ "${tier}" == health || "${tier}" == smoke ]] || { echo "unsupported tier" >&2; exit 64; }
[[ "${service_id}" =~ ^srv-[a-z0-9]+$ ]] || { echo "RENDER_SERVICE_ID is required" >&2; exit 64; }
[[ "${approval_id}" =~ ^APPROVAL-[A-Za-z0-9_-]{4,64}$ ]] || { echo "a per-run RENDER_APPROVAL_ID is required" >&2; exit 64; }
[[ "${approved_minutes}" =~ ^[0-9]+$ ]] || { echo "RENDER_APPROVED_MAX_MINUTES is required" >&2; exit 64; }
(( approved_minutes >= 5 && approved_minutes <= 60 )) || { echo "approved minutes outside 5..60" >&2; exit 64; }
[[ "${plan_id}" =~ ^plan-srv-[0-9]{3}$ ]] || { echo "invalid Render plan id" >&2; exit 64; }
[[ -z "${required_isa}" || "${required_isa}" =~ ^[a-z0-9_,-]+$ ]] || { echo "invalid required ISA list" >&2; exit 64; }

receipt_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/local/flock-council/receipts/render"
mkdir -p "${receipt_dir}"

printf -v remote_command '%q ' \
  /usr/local/bin/flock-render-job \
  --hypothesis "${hypothesis_id}" \
  --base "${base_sha}" \
  --candidate "${candidate_sha}" \
  --tier "${tier}" \
  --required-isa "${required_isa}"

dispatch_json="$(render jobs create "${service_id}" --start-command "${remote_command}" --plan-id "${plan_id}" --confirm --output json)"
job_id="$(jq -r '.id // empty' <<< "${dispatch_json}")"
[[ "${job_id}" =~ ^job-[a-z0-9]+$ ]] || { echo "Render returned no job id" >&2; exit 65; }

jq -n \
  --arg approval_id "${approval_id}" \
  --arg hypothesis_id "${hypothesis_id}" \
  --arg base_sha "${base_sha}" \
  --arg candidate_sha "${candidate_sha}" \
  --arg tier "${tier}" \
  --arg service_id "${service_id}" \
  --arg job_id "${job_id}" \
  --arg plan_id "${plan_id}" \
  --argjson approved_minutes "${approved_minutes}" \
  '{approval_id:$approval_id,hypothesis_id:$hypothesis_id,base_sha:$base_sha,candidate_sha:$candidate_sha,tier:$tier,service_id:$service_id,job_id:$job_id,plan_id:$plan_id,approved_minutes:$approved_minutes}' \
  > "${receipt_dir}/${job_id}.dispatch.json"

echo "Dispatched ${job_id} for ${hypothesis_id}; maximum approved runtime ${approved_minutes} minutes."

deadline=$(( $(date +%s) + approved_minutes * 60 ))
status=""
while (( $(date +%s) < deadline )); do
  jobs_json="$(render jobs list "${service_id}" --output json)"
  status="$(jq -r --arg job_id "${job_id}" 'map(select(.id == $job_id))[0].status // "unknown"' <<< "${jobs_json}")"
  case "${status}" in
    succeeded|failed|cancelled) break ;;
    pending|running|unknown) sleep 10 ;;
    *) echo "unexpected Render status: ${status}" >&2; exit 65 ;;
  esac
done

[[ "${status}" == succeeded || "${status}" == failed || "${status}" == cancelled ]] || {
  echo "job did not reach a terminal state inside the approved window" >&2
  exit 66
}

log_file="${receipt_dir}/${job_id}.log"
render logs --resources "${service_id}" --task-id "${job_id}" --limit 5000 > "${log_file}"

receipt_file="${receipt_dir}/${job_id}.receipt.json"
encoded_receipt="$(rg -o 'FLOCK_RECEIPT_JSON=[A-Za-z0-9+/=]+' "${log_file}" | tail -n 1 | cut -d= -f2-)"
[[ -n "${encoded_receipt}" ]] || { echo "job logs contain no durable receipt" >&2; exit 67; }
printf '%s' "${encoded_receipt}" | base64 -d > "${receipt_file}"
jq -e --arg job_id "${job_id}" --arg status "${status}" 'select(.kind == "flock_render_job" and .status == $status and .resolved_sha != "") | . + {job_id:$job_id}' "${receipt_file}" > "${receipt_file}.validated"
mv "${receipt_file}.validated" "${receipt_file}"

if [[ "${status}" != succeeded ]]; then
  echo "Render job ${job_id} ended ${status}; receipt saved at ${receipt_file}" >&2
  exit 1
fi

echo "Render job ${job_id} succeeded with a validated directional receipt at ${receipt_file}."
