#!/usr/bin/env bash
set -euo pipefail

label="${1:?label}"
mode="${2:?mode: off|gap|all}"
seed="${3:-1311768467463790320}"
worker=/home/ubuntu/stfmp-control-worker
out="/home/ubuntu/cff-once-${label}"

[[ ! -e "${out}" ]] || {
  echo "output already exists: ${out}" >&2
  exit 2
}
mkdir -p "${out}/scratch"
mkfifo "${out}/seed.fifo"
exec 3<>"${out}/seed.fifo"

profile_env=()
case "${mode}" in
  off) ;;
  gap)
    profile_env+=(FLOCK_GAP_TIMING=1)
    ;;
  all)
    profile_env+=(
      FLOCK_GAP_TIMING=1
      FLOCK_COMMIT_TIMING=1
      FLOCK_ZC_TIMING=1
      FLOCK_OPEN_TIMING=1
      LIG_PROVE_TRACE=1
      LINCHECK_TRACE=1
      PCS_TRACE=1
      PERM_TRACE=1
      MERKLE_TRACE=1
    )
    ;;
  *)
    echo "unknown mode: ${mode}" >&2
    exit 2
    ;;
esac

env RAYON_NUM_THREADS=16 TMPDIR="${out}/scratch" "${profile_env[@]}" \
  "${worker}" 18 "${out}/run.ready" "${out}/run.proof" \
  <&3 >"${out}/worker.stdout" 2>"${out}/worker.stderr" &
worker_pid=$!

for _ in $(seq 1 6000); do
  [[ -f "${out}/run.ready" ]] && break
  kill -0 "${worker_pid}" 2>/dev/null || {
    wait "${worker_pid}"
    exit 1
  }
  sleep 0.01
done
[[ -f "${out}/run.ready" ]]

start_ns=$(date +%s%N)
printf '%s\n' "${seed}" >&3
exec 3>&-
set +e
wait "${worker_pid}"
worker_status=$?
set -e
end_ns=$(date +%s%N)

test -s "${out}/run.proof"
sha256sum "${worker}" "${out}/run.proof" >"${out}/sha256.txt"
printf 'label=%s\nmode=%s\nseed=%s\nelapsed_ns=%s\nworker_status=%s\n' \
  "${label}" "${mode}" "${seed}" "$((end_ns - start_ns))" "${worker_status}" \
  >"${out}/elapsed.txt"
cat "${out}/elapsed.txt"
cat "${out}/sha256.txt"
