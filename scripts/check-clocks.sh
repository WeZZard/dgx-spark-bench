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
# Power is the counter that does not lie; see the verdict below.
MIN_WATT="${CLOCK_MIN_WATT:-50}"
SECS="${CLOCK_PROBE_SECS:-30}"
export SECS
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
sleep 4
for i in $(seq 1 20); do nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader,nounits; sleep 1; done
wait'
  if [ -z "$host" ]; then bash -c "$run"; else ssh -o ConnectTimeout=30 "$host" "$run"; fi
}

# stdin: "mhz, watt, util" lines -> "<median_mhz_under_load> <mean_watt> <loaded_samples>".
#
# ONLY SAMPLES WHERE THE GPU IS ACTUALLY LOADED COUNT.
#
# The first version medianed every sample in the window and reported it as the
# clock "under load". The probe starts a container and then samples; on the
# worker the container started late, so most samples caught an idle GPU at
# 208 MHz while peak utilisation still saw the tail of the load. The check
# then refused a perfectly healthy node -- 2307 MHz, 96%, 90 W when asked
# again a moment later -- and aborted a legitimate two-hour run.
#
# A guard that measures the sampling window and calls it the load is the same
# defect this repo has now found five times. The quantity wanted is the median
# clock over loaded samples, so that is what this computes, and it says how
# many loaded samples it had rather than averaging an unknown mixture.
#
# python3, not awk: median needs a sort, gawk's asort() is not in mawk, and a
# `awk || python3` fallback inside a pipeline is a trap -- awk consumes stdin
# before it fails, so the fallback reads nothing and reports a healthy zero.
summarise() {
  python3 -c "
import sys
rows=[l.split(',') for l in sys.stdin if l.strip()]
loaded=[r for r in rows if int(r[2]) >= 50]
if not loaded:
    print('0 0 0'); raise SystemExit
mhz=sorted(int(r[0]) for r in loaded)
w=[float(r[1]) for r in loaded]
print('%d %.1f %d' % (mhz[len(mhz)//2], sum(w)/len(w), len(loaded)))"
}

case "${1:-probe}" in
  probe)
    rc=0
    for host in "" "$WORKER"; do
      label="${host:-$(hostname)}"
      # Retry once, longer, before condemning a node. The failure mode being
      # guarded against here is my own: a probe whose container starts late
      # collects mostly idle samples, and aborting a two-hour campaign on that
      # is worse than spending another 60 seconds to be sure.
      out="$(probe_one "$host" | grep -E '^[0-9]+,' | summarise)"
      if [ "${out##* }" -lt 5 ]; then
        echo "   $label: only ${out##* } loaded samples, retrying with a longer probe"
        out="$(SECS=$((SECS * 2)) probe_one "$host" | grep -E '^[0-9]+,' | summarise)"
      fi
      mhz="${out%% *}"; rest="${out#* }"; watt="${rest%% *}"; n="${rest##* }"
      printf "   %-22s median %5s MHz, %6s W, over %s loaded samples\n" "$label" "$mhz" "$watt" "$n"
      if [ "${n:-0}" -lt 5 ]; then
        echo "!! $label: only ${n} loaded samples; the probe did not run long enough" >&2
        echo "   to judge the clock. Raise CLOCK_PROBE_SECS and try again." >&2
        rc=1
      elif [ "${mhz:-0}" -lt "$MIN_MHZ" ]; then
        # TWO COUNTERS, AND ONLY ONE OF THEM IS RELIABLE.
        #
        # clocks.sm on GB10 intermittently returns the idle value under full
        # load: this guard read "208 MHz over 18 loaded samples" on a node
        # that was simultaneously drawing 91.5 W, and which read 2288-2307 MHz
        # at 90 W when asked again a minute later. 208 MHz and 91 W cannot
        # both be true. Power can: the genuine low-clock state drew 17 W
        # against a healthy 91 W for the same matmul, and that is what made
        # the original diagnosis stick.
        #
        # So a low clock alone is not enough to condemn a node. Both counters
        # have to agree.
        if awk "BEGIN{exit !($watt < $MIN_WATT)}"; then
          echo "!! $label: ${mhz} MHz and ${watt} W under load, against a" >&2
          echo "   ${MIN_MHZ} MHz / ${MIN_WATT} W floor. This node is in the GB10" >&2
          echo "   low-clock state. Every number measured in TP2 with it is paced" >&2
          echo "   by it and is not a real result. Reboot it, then re-run this probe." >&2
          rc=1
        else
          echo "   $label: clocks.sm read ${mhz} MHz but power is ${watt} W, which"
          echo "   cannot both be true. Trusting power; clocks.sm misreports here."
        fi
      fi
    done
    [ "$rc" = 0 ] && echo "   both nodes clock up: OK"
    exit $rc
    ;;
  *) echo "usage: check-clocks.sh probe [worker-host]" >&2; exit 2 ;;
esac
