#!/usr/bin/env bash
# One configuration, end to end: bring up, wait, measure every cell, tear down.
#
#   CONFIG_NOTE="rdma" NCCL_DEBUG=INFO bash scripts/sweep-qwen.sh
#   CONFIG_NOTE="rdma+mtp" QWEN_SPEC=NEXTN bash scripts/sweep-qwen.sh
#
# Knobs are environment variables and reach both ranks through bring_up, which
# forwards by prefix. CONFIG_NOTE is written into every run's notes so the
# database can separate one configuration's cells from another's -- otherwise
# two sweeps become one undifferentiated pile of numbers and the comparison
# that motivated them cannot be made.
#
# Change ONE thing per sweep. The whole point of running the control first was
# to be able to attribute a difference; a sweep that turns on RDMA and MTP
# together measures neither.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

source scripts/lib/wait-ready.sh

NOTE="${CONFIG_NOTE:?set CONFIG_NOTE to name this configuration}"
BASE="${BASE:-http://127.0.0.1:30000}"
NAME="${QWEN_NAME:-sglang_qwen38fn}"
WORKER="${WORKER_HOST:-spark-right-fast}"

log=/tmp/qwen-sweep-$NOTE.log
echo "== sweep '$NOTE' starting $(date -Is), log $log"

# Tear down both ranks and wait for the memory back, rather than sleeping. A
# rank left holding 90 GB is what turned a failed launch into a wedged node.
for host in "" "$WORKER"; do
  if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
done

( setsid nohup bash scripts/bring-up.sh scripts/launch-qwen-sglang.sh > "$log" 2>&1 < /dev/null & )

wait_ready "$NAME" "$BASE" "$log" || exit 1

# BRACKET THE CAMPAIGN, DO NOT SAMPLE INSIDE IT.
#
# check-rdma.sh's primary test reads the HCA's rx_write_requests counter, and
# needs RDMA verbs in flight to see any. Calling it here, before a single
# request has been served, could only ever report "could not tell" -- which is
# what every sweep did for weeks, so every campaign's RoCE evidence was
# gathered by hand afterwards.
#
# Sampling inside the campaign instead was the obvious fix and was also wrong:
# a 20-second window landed in an idle gap between warmup requests and
# reported "delta 0 in 20s" for a campaign that had in fact issued 11,040,888
# RDMA writes. The counter is monotonic, so bracketing is strictly better --
# no timing to get right, and the window is exactly as long as the thing being
# measured.
rdma_before="$(bash scripts/check-rdma.sh --snapshot 2>/dev/null || echo "")"

CAMPAIGN_NOTE="$NOTE" bash scripts/campaign-qwen.sh "$BASE"


# The RoCE verdict, over the whole campaign. Not fatal: the cells are recorded
# either way, and a sweep that threw away real measurements over a transport
# check would be worse. But a campaign that ran on TCP is worth about 1.74x
# less than it looks (docs/interconnect.md), so the answer belongs beside the
# numbers rather than in someone's memory.
echo "== RoCE check, bracketing the campaign"
if [ -n "$rdma_before" ]; then
  bash scripts/check-rdma.sh --delta "$rdma_before" "$NAME" || true
else
  echo "   no counter snapshot was taken before the campaign; cannot tell."
fi

# TEAR DOWN BOTH RANKS, ALWAYS, INCLUDING AFTER A FAILED CAMPAIGN.
#
# The teardown at the top of this script runs before the NEXT sweep, which is
# no help at all if there is not going to be one. On 2026-09-04 a GLM run hit
# a scheduler watchdog timeout, rank 0 died, and rank 1 stayed up holding
# 120 of 121 GiB for as long as nobody looked -- the exact "a rank left
# holding 90 GB" state the header at the top warns about, reached from the
# other end.
#
# A rank that survived its peer is not serving anything; it is only holding
# memory. Remove it here so the machine is left usable whatever happened
# above.
echo "== tearing down both ranks"
for host in "" "$WORKER"; do
  if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
done

echo "== sweep '$NOTE' done $(date -Is)"
