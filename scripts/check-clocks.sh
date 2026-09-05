#!/usr/bin/env bash
# Refuse to measure on a GPU that will not clock up.
#
#   bash scripts/check-clocks.sh probe [worker-host]
#
# Why this exists. On 2026-09-05 the head node was found running every
# workload at 682-695 MHz against its peer's 2288-2340 MHz -- 29% of the
# clock, 17 W against 91 W, on identical hardware, identical driver
# (580.173.02), both in performance state P0 with no throttle reason active
# and no locked or applications clocks set. nvidia-smi reported nothing wrong.
# Telemetry showed the node had been like that for its entire 34-hour uptime,
# which is to say for every measurement this repo had taken. In a TP2
# lock-step job the slow rank sets the pace, so all of it was measured on
# half a machine. A reboot of that node restored 2379-2405 MHz at 94 W.
#
# Nothing in the engine, the container or the launcher can see this. The only
# way to catch it is to put a known load on each GPU and watch the clock, and
# the only safe moment is BEFORE bring-up, when the GPUs are idle and a
# 4096x4096 matmul cannot collide with a model.
#
# This is fatal on purpose, unlike check-rdma.sh. RoCE being down makes the
# numbers 1.74x pessimistic and still real; a rank at 29% clock makes them
# fiction, and the cost of finding out afterwards is every campaign since the
# last reboot.
set -uo pipefail

MIN_MHZ="${CLOCK_MIN_MHZ:-2000}"
SECS="${CLOCK_PROBE_SECS:-20}"
IMAGE="${CLOCK_PROBE_IMAGE:-vllm-dsv4:ray}"
WORKER="${2:-${WORKER_HOST:-spark-right-fast}}"

# One matmul loop, one clock sample stream, on one host. Prints
# "<median_mhz> <median_watt> <max_util>".
probe_one() {
  local host="$1" run
  run='docker run --rm --gpus all --entrypoint python3 '"$IMAGE"' -c "
import torch, time
a=torch.randn(4096,4096,device=\"cuda\",dtype=torch.bfloat16)
b=torch.randn(4096,4096,device=\"cuda\",dtype=torch.bfloat16)
t=time.time()
while time.time()-t < '"$SECS"':
    for _ in range(50): c=a@b
    torch.cuda.synchronize()
" >/dev/null 2>&1 &
sleep 8
for i in $(seq 1 6); do nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits; sleep 1; done
wait'
  if [ -z "$host" ]; then bash -c "$run"; else ssh -o ConnectTimeout=30 "$host" "$run"; fi
}

# stdin: "mhz, watt, util" lines -> "<median_mhz> <mean_watt> <max_util>".
#
# python3, not awk: median needs a sort, gawk's asort() is not in mawk, and a
# `awk || python3` fallback inside a pipeline is a trap -- awk consumes stdin
# before it fails, so the fallback reads nothing and reports a healthy zero.
summarise() {
  python3 -c "
import sys
rows=[l.split(',') for l in sys.stdin if l.strip()]
if not rows:
    print('0 0 0'); raise SystemExit
mhz=sorted(int(r[0]) for r in rows)
w=[float(r[1]) for r in rows]; u=[int(r[2]) for r in rows]
print('%d %.1f %d' % (mhz[len(mhz)//2], sum(w)/len(w), max(u)))"
}

case "${1:-probe}" in
  probe)
    rc=0
    for host in "" "$WORKER"; do
      label="${host:-$(hostname)}"
      out="$(probe_one "$host" | grep -E '^[0-9]+,' | summarise)"
      mhz="${out%% *}"; rest="${out#* }"; watt="${rest%% *}"; util="${rest##* }"
      printf "   %-22s median %5s MHz, %6s W, peak util %s%%\n" "$label" "$mhz" "$watt" "$util"
      if [ "${util:-0}" -lt 50 ]; then
        echo "!! $label: the probe never loaded the GPU (peak util ${util}%). Cannot judge the clock." >&2
        rc=1
      elif [ "${mhz:-0}" -lt "$MIN_MHZ" ]; then
        echo "!! $label: ${mhz} MHz under load, against a ${MIN_MHZ} MHz floor." >&2
        echo "   This node is in the GB10 low-clock state. Every number measured" >&2
        echo "   in TP2 with it is paced by it and is not a real result." >&2
        echo "   Reboot this node, then re-run this probe." >&2
        rc=1
      fi
    done
    [ "$rc" = 0 ] && echo "   both nodes clock up: OK"
    exit $rc
    ;;
  *) echo "usage: check-clocks.sh probe [worker-host]" >&2; exit 2 ;;
esac
