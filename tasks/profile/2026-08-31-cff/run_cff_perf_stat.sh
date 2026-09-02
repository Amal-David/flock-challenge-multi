#!/usr/bin/env bash
set -euo pipefail

label="${1:?label}"
events="${2:?comma-separated perf events}"
seed="${3:-1311768467463790320}"
worker=/home/ubuntu/stfmp-control-worker
out="/home/ubuntu/cff-perf-${label}"

[[ ! -e "${out}" ]] || {
  echo "output already exists: ${out}" >&2
  exit 2
}
mkdir -p "${out}/scratch"
mkfifo "${out}/seed.fifo"
exec 3<>"${out}/seed.fifo"

env RAYON_NUM_THREADS=16 TMPDIR="${out}/scratch" \
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

tids=$(find "/proc/${worker_pid}/task" -mindepth 1 -maxdepth 1 -printf '%f\n' | paste -sd, -)
sudo perf stat -x, -o "${out}/perf.csv" -t "${tids}" -e "${events}" &
perf_pid=$!
sleep 0.2
start_ns=$(date +%s%N)
printf '%s\n' "${seed}" >&3
exec 3>&-
set +e
wait "${worker_pid}"
worker_status=$?
set -e
end_ns=$(date +%s%N)
wait "${perf_pid}" 2>/dev/null || true

test -s "${out}/run.proof"
sha256sum "${worker}" "${out}/run.proof" >"${out}/sha256.txt"
printf 'label=%s\nevents=%s\nseed=%s\nelapsed_ns=%s\nworker_status=%s\n' \
  "${label}" "${events}" "${seed}" "$((end_ns - start_ns))" "${worker_status}" \
  >"${out}/elapsed.txt"
cat "${out}/elapsed.txt"
cat "${out}/perf.csv"
cat "${out}/sha256.txt"
