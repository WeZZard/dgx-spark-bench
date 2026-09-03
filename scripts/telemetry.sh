#!/usr/bin/env bash
# Node telemetry sampler: MemAvailable, GPU utilisation, GPU power, container
# state, once a second, to /tmp/sparkbench-telemetry.log.
#
#   bash scripts/telemetry.sh start [container]
#   bash scripts/telemetry.sh stop
#   bash scripts/telemetry.sh tail [n]
#
# MemAvailable rather than free/used because it is the only counter that tracks
# GB10 driver memory. On 2026-09-03 the head showed 43 MB of MemAvailable while
# the OOM killer's own dump accounted for barely 17 GB of the 121 GiB installed
# -- the other 104 GB was held by the driver as unified memory and appeared in
# no counter the killer scores.
#
# GPU utilisation because container liveness is not workload liveness. A vLLM
# rank once failed compiling a kernel and stayed "running" for 8.5 hours
# holding 90 GB with the GPU flat at 0%. An idle GPU on a node that should be
# loading is the earliest true failure signal there is.
#
# power.draw is GPU-domain power only. Module and memory power read N/A on
# GB10, so any tokens-per-joule figure derived from this is tokens per GPU
# joule and must be labelled that way -- it excludes the 20 CPU cores and the
# memory system. See docs/baselines.md.
#
# Stop is by pidfile, never `pkill -f`: a pattern broad enough to catch the
# sampler also matches the ssh command line that carries it, which kills the
# session instead (observed, exit 255).
set -uo pipefail

PIDFILE=/tmp/sparkbench-telemetry.pid
LOGFILE=/tmp/sparkbench-telemetry.log

sample_loop() {
  local container="${1:-}"
  while true; do
    local ts ma gpu cs
    ts=$(date +%H:%M:%S)
    ma=$(awk '/^MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    if [ -n "$container" ]; then
      cs=$(timeout 10 docker inspect -f '{{.State.Status}}' "$container" 2>/dev/null || echo absent)
    else
      cs=-
    fi
    echo "$ts memavail_gib=$ma gpu_util%,power_w,sm_mhz=$gpu container=$cs"
    sleep 1
  done
}

case "${1:-}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already sampling (pid $(cat "$PIDFILE"))"; exit 0
    fi
    : > "$LOGFILE"
    setsid nohup bash "$0" _loop "${2:-}" > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 1
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null \
      && echo "telemetry started on $(hostname), pid $(cat "$PIDFILE")" \
      || { echo "telemetry failed to start" >&2; exit 1; }
    ;;
  stop)
    [ -f "$PIDFILE" ] || { echo "not running"; exit 0; }
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "telemetry stopped on $(hostname)"
    ;;
  tail) tail -n "${2:-20}" "$LOGFILE" ;;
  _loop) sample_loop "${2:-}" ;;
  *) echo "usage: $0 {start [container]|stop|tail [n]}" >&2; exit 2 ;;
esac
