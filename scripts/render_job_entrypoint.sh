#!/usr/bin/env bash
set -euo pipefail

umask 077

usage() {
  echo "usage: flock-render-job --hypothesis ID --base SHA --candidate SHA --tier health|smoke [--required-isa csv]" >&2
}

hypothesis=""
base_sha=""
candidate_sha=""
tier=""
required_isa=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hypothesis) hypothesis="${2:-}"; shift 2 ;;
    --base) base_sha="${2:-}"; shift 2 ;;
    --candidate) candidate_sha="${2:-}"; shift 2 ;;
    --tier) tier="${2:-}"; shift 2 ;;
    --required-isa) required_isa="${2:-}"; shift 2 ;;
    *) usage; exit 64 ;;
  esac
done

[[ "${hypothesis}" =~ ^H-[A-Z0-9][A-Z0-9_-]{2,63}$ ]] || { echo "invalid hypothesis id" >&2; exit 64; }
[[ "${base_sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "base must be a full lowercase Git SHA" >&2; exit 64; }
[[ "${candidate_sha}" =~ ^[0-9a-f]{40}$ ]] || { echo "candidate must be a full lowercase Git SHA" >&2; exit 64; }
[[ "${tier}" == health || "${tier}" == smoke ]] || { echo "tier must be health or smoke" >&2; exit 64; }
[[ -z "${required_isa}" || "${required_isa}" =~ ^[a-z0-9_,-]+$ ]] || { echo "invalid required ISA list" >&2; exit 64; }

repo_url="${FLOCK_JOB_REPO_URL:-https://github.com/Amal-David/flock-challenge-multi.git}"
[[ "${repo_url}" == "https://github.com/Amal-David/flock-challenge-multi.git" ]] || { echo "unapproved repository URL" >&2; exit 64; }

max_seconds="${FLOCK_JOB_MAX_SECONDS:-900}"
[[ "${max_seconds}" =~ ^[0-9]+$ ]] || { echo "FLOCK_JOB_MAX_SECONDS must be an integer" >&2; exit 64; }
(( max_seconds >= 60 && max_seconds <= 3600 )) || { echo "FLOCK_JOB_MAX_SECONDS outside 60..3600" >&2; exit 64; }

workspace="$(mktemp -d /tmp/flock-render-job.XXXXXX)"
repo_dir="${workspace}/repo"
log_file="${workspace}/job.log"
phase="bootstrap"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
toolchain="$(rustc --version 2>/dev/null || true)"
cpu_model="$(lscpu 2>/dev/null | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
cpu_flags_sha="$(sha256sum /proc/cpuinfo 2>/dev/null | awk '{print $1}')"
nproc_value="$(nproc)"
cgroup_cpu_max="$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)"
cgroup_memory_max="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo unknown)"

emit_receipt() {
  local exit_code="$1"
  local status="failed"
  [[ "${exit_code}" -eq 0 ]] && status="succeeded"
  local resolved_sha=""
  local verifier_sha=""
  local score_json="null"
  if [[ -d "${repo_dir}/.git" ]]; then
    resolved_sha="$(git -C "${repo_dir}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -f "${repo_dir}/benchmark-tools/trusted/flock_benchmark_verifier" ]]; then
    verifier_sha="$(sha256sum "${repo_dir}/benchmark-tools/trusted/flock_benchmark_verifier" | awk '{print $1}')"
  fi
  if [[ -s "${repo_dir}/score.json" ]]; then
    score_json="$(jq -c . "${repo_dir}/score.json")"
  fi
  local receipt_json
  receipt_json="$(jq -cn \
    --arg authority "RENDER_DIRECTIONAL" \
    --arg kind "flock_render_job" \
    --arg hypothesis_id "${hypothesis}" \
    --arg base_sha "${base_sha}" \
    --arg candidate_sha "${candidate_sha}" \
    --arg resolved_sha "${resolved_sha}" \
    --arg tier "${tier}" \
    --arg required_isa "${required_isa}" \
    --arg status "${status}" \
    --arg phase "${phase}" \
    --arg started_at "${started_at}" \
    --arg finished_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg toolchain "${toolchain}" \
    --arg cpu_model "${cpu_model}" \
    --arg cpu_flags_sha256 "${cpu_flags_sha}" \
    --arg nproc "${nproc_value}" \
    --arg cgroup_cpu_max "${cgroup_cpu_max}" \
    --arg cgroup_memory_max "${cgroup_memory_max}" \
    --arg verifier_sha256 "${verifier_sha}" \
    --arg repo_url "${repo_url}" \
    --argjson exit_code "${exit_code}" \
    --argjson score "${score_json}" \
    '{authority:$authority,kind:$kind,hypothesis_id:$hypothesis_id,base_sha:$base_sha,candidate_sha:$candidate_sha,resolved_sha:$resolved_sha,tier:$tier,required_isa:$required_isa,status:$status,phase:$phase,started_at:$started_at,finished_at:$finished_at,toolchain:$toolchain,cpu_model:$cpu_model,cpu_flags_sha256:$cpu_flags_sha256,nproc:($nproc|tonumber),cgroup_cpu_max:$cgroup_cpu_max,cgroup_memory_max:$cgroup_memory_max,verifier_sha256:$verifier_sha256,repo_url:$repo_url,exit_code:$exit_code,score:$score}')"
  echo "FLOCK_RECEIPT_JSON=$(printf '%s' "${receipt_json}" | base64 -w0)"
  rm -rf "${workspace}"
}

on_exit() {
  local exit_code="$?"
  trap - EXIT
  emit_receipt "${exit_code}"
  exit "${exit_code}"
}
trap on_exit EXIT

[[ "${toolchain}" == rustc\ 1.97.0* ]] || { echo "wrong Rust toolchain: ${toolchain}" >&2; exit 65; }

phase="clone"
git clone --filter=blob:none --no-checkout "${repo_url}" "${repo_dir}" 2>&1 | tee -a "${log_file}"
git -C "${repo_dir}" fetch --no-tags origin "${base_sha}" "${candidate_sha}" 2>&1 | tee -a "${log_file}"
git -C "${repo_dir}" checkout --detach "${candidate_sha}" 2>&1 | tee -a "${log_file}"
[[ "$(git -C "${repo_dir}" rev-parse HEAD)" == "${candidate_sha}" ]] || { echo "resolved SHA mismatch" >&2; exit 65; }
[[ -z "$(git -C "${repo_dir}" status --porcelain)" ]] || { echo "candidate checkout is dirty" >&2; exit 65; }

phase="scope"
git -C "${repo_dir}" cat-file -e "${base_sha}^{commit}"
git -C "${repo_dir}" diff --check "${base_sha}..${candidate_sha}"
while IFS= read -r changed_path; do
  [[ -z "${changed_path}" ]] && continue
  case "${changed_path}" in
    crates/flock-core/src/*|crates/flock-prover/src/*) ;;
    *) echo "changed path outside editable surface: ${changed_path}" >&2; exit 65 ;;
  esac
done < <(git -C "${repo_dir}" diff --name-only "${base_sha}..${candidate_sha}")

phase="isa"
if [[ -n "${required_isa}" ]]; then
  IFS=',' read -r -a isa_items <<< "${required_isa}"
  cpu_flags=" $(awk -F: '/^flags/ {print $2; exit}' /proc/cpuinfo) "
  for feature in "${isa_items[@]}"; do
    [[ "${cpu_flags}" == *" ${feature} "* ]] || { echo "required ISA is absent: ${feature}" >&2; exit 66; }
  done
fi

phase="verifier"
(
  cd "${repo_dir}/benchmark-tools/trusted"
  sha256sum -c SHA256SUMS
) 2>&1 | tee -a "${log_file}"

phase="setup"
timeout "${max_seconds}" bash -lc "cd '${repo_dir}' && ./setup.sh" 2>&1 | tee -a "${log_file}"

phase="verified_smoke"
timeout "${max_seconds}" bash -lc "cd '${repo_dir}' && FLOCK_REQUIRE_SANDBOX=1 BLAKE3_LOG2=8 BLAKE3_THREADS=1 BLAKE3_WARMUP_RUNS=1 BLAKE3_RUNS=3 BENCHMARK_OUTPUT_DIR=benchmark-results/render-${hypothesis} ./benchmark.sh" 2>&1 | tee -a "${log_file}"
jq -e '.metrics.verified == true and .metrics.measured_runs == 3 and .metrics.warmup_runs == 1' "${repo_dir}/score.json" >/dev/null

phase="complete"
