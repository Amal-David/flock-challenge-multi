#!/usr/bin/env bash
set -euo pipefail

label="${1:?label}"
seed="${2:-1311768467463790320}"
worker=/home/ubuntu/frontier796-worker
out="/home/ubuntu/frontier796-record-timed-${label}"

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

# Attach only to the already-created worker/Rayon threads. perf exits when the
# observed threads exit, so capture cannot drift into post-proof idle time.
record_tids=()
while IFS= read -r tid; do
  record_tids+=(-t "${tid}")
done < <(find "/proc/${worker_pid}/task" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort -n)
[[ "${#record_tids[@]}" -gt 1 ]]

sudo perf record -F 997 -g --call-graph dwarf,8192 \
  -o "${out}/perf.data" "${record_tids[@]}" &
perf_pid=$!
sleep 0.2
start_ns=$(date +%s%N)
printf '%s\n' "${seed}" >&3
exec 3>&-
set +e
wait "${worker_pid}"
worker_status=$?
wait "${perf_pid}"
perf_status=$?
set -e
end_ns=$(date +%s%N)

[[ "${worker_status}" == 0 ]]
[[ "${perf_status}" == 0 ]]
test -s "${out}/run.proof"
test -s "${out}/perf.data"
sudo perf report --stdio --no-children --call-graph none --percent-limit 0.05 \
  --sort comm,dso,symbol -i "${out}/perf.data" >"${out}/report.txt"
sudo sha256sum "${worker}" "${out}/run.proof" "${out}/perf.data" \
  >"${out}/sha256.txt"
printf 'label=%s\nseed=%s\ncapture_mode=worker_threads\nelapsed_ns=%s\nworker_status=%s\nperf_status=%s\n' \
  "${label}" "${seed}" "$((end_ns - start_ns))" "${worker_status}" "${perf_status}" \
  >"${out}/elapsed.txt"
cat "${out}/elapsed.txt"
sed -n '1,260p' "${out}/report.txt"
cat "${out}/sha256.txt"
