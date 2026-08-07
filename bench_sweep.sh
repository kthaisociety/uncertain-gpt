#!/usr/bin/env bash
# The batch-size amortization sweep. One `bench:` line per (arm, B) into
# runs/bench/, which paper/tables/make_tables.py reads.
#
#     ./bench_sweep.sh [tag]        tag defaults to "pre" (before fusion)
#
# Sweeps B with T held fixed. Step counts fall as B rises so every point
# takes a comparable wall-clock time; the median is over the retained steps
# either way, and the count is printed on the line so it is checkable.
#
# The card has a known cooling fault. This script idles between points until
# the GPU is back under COOL_TO, and the binaries themselves abort over
# BENCH_TEMP_ABORT (89 C) rather than report throttled numbers.
set -u

TAG="${1:-pre}"
OUT="runs/bench/$TAG"
mkdir -p "$OUT"

COOL_TO=72          # C — start each point from a comparable thermal state
COOL_MAX=180        # s — give up waiting rather than hang the sweep

# B -> "steps warmup". Below 8 the step is a few ms, so more of them.
sched() {
  case "$1" in
    1|2|4|8)   echo "300 30" ;;
    16|32)     echo "200 20" ;;
    64)        echo "120 15" ;;
    128)       echo "80 10"  ;;
    *)         echo "50 10"  ;;
  esac
}

gputemp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null; }

cool() {
  local t waited=0
  while :; do
    t=$(gputemp); [ -z "$t" ] && return 0
    [ "$t" -le "$COOL_TO" ] && return 0
    [ "$waited" -ge "$COOL_MAX" ] && { echo "  (still ${t}C after ${waited}s, continuing)"; return 0; }
    sleep 10; waited=$((waited + 10))
  done
}

ARMS="bbb:train_bnn_gpu vadam:train_bnn_vadam lowr:train_bnn_lowrank ens:train_ensemble"
BS_LIST="${BS_LIST:-1 2 4 8 16 32 64 128 256}"

for pair in $ARMS; do
  arm="${pair%%:*}"; bin="${pair##*:}"
  for B in $BS_LIST; do
    read -r steps warm <<<"$(sched "$B")"
    cool
    log="$OUT/${arm}_B${B}.log"
    printf '%-6s B=%-4s steps=%-4s ' "$arm" "$B" "$steps"
    if ! ./"$bin" bench "$B" "$steps" "$warm" > "$log" 2>"$log.err"; then
      rc=$?
      if grep -q "THERMAL ABORT" "$log"; then
        echo "THERMAL ABORT — stopping the sweep"
        grep "THERMAL ABORT" "$log"
        exit 2
      fi
      # Running out of VRAM headroom is the expected way a sweep ends. Keep
      # the log when the binary said why (it holds the OUT OF RANGE line, and
      # "where we stopped and why" is part of the result); drop an empty one.
      if grep -q "OUT OF RANGE" "$log"; then
        echo "out of range — $(grep -o 'only [0-9]* MB.*free' "$log" | head -1)"
      else
        echo "stopped (rc=$rc) — see $log.err"
        [ -s "$log" ] || rm -f "$log"
      fi
      break
    fi
    grep -oE 'total [0-9.]+ ms \| posterior [0-9.]+ ms.*share [0-9.]+' "$log" | head -1
  done
done

cat "$OUT"/*.log > "runs/bench_${TAG}.log" 2>/dev/null
echo
echo "wrote $OUT/ and runs/bench_${TAG}.log"
grep -c '^bench: ' "runs/bench_${TAG}.log" | sed 's/^/points: /'
