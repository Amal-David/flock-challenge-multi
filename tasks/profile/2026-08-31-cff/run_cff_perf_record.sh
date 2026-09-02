#!/usr/bin/env bash
set -euo pipefail

label="${1:?label}"
seed="${2:-1311768467463790320}"
worker=/home/ubuntu/stfmp-control-worker
out="/home/ubuntu/cff-record-${label}"

[[ ! -e "${out}" ]] || {
  echo "output already exists: ${out}" >&2
  exit 2
}
mkdir -p "${out}/scratch"
mkfifo "${out}/seed.fifo"
exec 3<>"${out}/seed.fifo"

sudo perf record -F 997 -g --call-graph dwarf,8192 \
  -o "${out}/perf.data" -- \
  env RAYON_NUM_THREADS=16 TMPDIR="${out}/scratch" \
  "${worker}" 18 "${out}/run.ready" "${out}/run.proof" \
  <&3 >"${out}/worker.stdout" 2>"${out}/worker.stderr" &
perf_pid=$!

for _ in $(seq 1 6000); do
  [[ -f "${out}/run.ready" ]] && break
  kill -0 "${perf_pid}" 2>/dev/null || {
    wait "${perf_pid}"
    exit 1
  }
  sleep 0.01
done
[[ -f "${out}/run.ready" ]]

start_ns=$(date +%s%N)
printf '%s\n' "${seed}" >&3
exec 3>&-
set +e
wait "${perf_pid}"
perf_status=$?
set -e
end_ns=$(date +%s%N)

test -s "${out}/run.proof"
sudo perf report --stdio --no-children --percent-limit 0.05 \
  --sort comm,dso,symbol -i "${out}/perf.data" >"${out}/report.txt"
sudo sha256sum "${worker}" "${out}/run.proof" "${out}/perf.data" >"${out}/sha256.txt"
printf 'label=%s\nseed=%s\nelapsed_ns=%s\nperf_status=%s\n' \
  "${label}" "${seed}" "$((end_ns - start_ns))" "${perf_status}" \
  >"${out}/elapsed.txt"
cat "${out}/elapsed.txt"
sed -n '1,240p' "${out}/report.txt"
cat "${out}/sha256.txt"
