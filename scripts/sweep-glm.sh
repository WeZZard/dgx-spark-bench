#!/usr/bin/env bash
# One configuration, end to end: bring up, wait, measure every cell, tear down.
#
#   CONFIG_NOTE="rdma" NCCL_DEBUG=INFO bash scripts/sweep-glm.sh
#   CONFIG_NOTE="rdma+mtp" QWEN_SPEC=NEXTN bash scripts/sweep-glm.sh
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
NAME="${GLM_NAME:-sglang_glm53}"
WORKER="${WORKER_HOST:-spark-right-fast}"

log=/tmp/glm-sweep-$NOTE.log
echo "== sweep '$NOTE' starting $(date -Is), log $log"

# Tear down both ranks and wait for the memory back, rather than sleeping. A
# rank left holding 90 GB is what turned a failed launch into a wedged node.
for host in "" "$WORKER"; do
  if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
done

( setsid nohup bash scripts/bring-up.sh scripts/launch-glm-sglang.sh > "$log" 2>&1 < /dev/null & )

wait_ready "$NAME" "$BASE" "$log" || exit 1

# RUN THIS DURING THE CAMPAIGN, NOT BEFORE IT.
#
# check-rdma.sh's primary test reads the HCA's rx_write_requests counter and
# needs RDMA verbs in flight to see any. Called here, before a single request
# has been served, it can only ever report "could not tell" -- which is what
# every sweep did, so every campaign's RoCE evidence had to be gathered by
# hand afterwards. The check said so in its own header ("it answers during a
# campaign and not before one") and the sweep called it before one anyway.
#
# Detached, it samples while the campaign generates traffic. Its verdict is
# collected after.
rdma_out=/tmp/rdma-$NOTE.txt
( sleep "${RDMA_DELAY_S:-90}"; bash scripts/check-rdma.sh "$NAME" > "$rdma_out" 2>&1 ) &
rdma_pid=$!

CAMPAIGN_NOTE="$NOTE" bash scripts/campaign-glm.sh "$BASE"


# The RoCE verdict, sampled while the campaign was running. Not fatal: the
# cells are recorded either way, and a sweep that threw away real
# measurements over a transport check would be worse. But a campaign that ran
# on TCP is worth about 1.74x less than it looks (docs/interconnect.md), so
# the answer belongs beside the numbers.
wait "$rdma_pid" 2>/dev/null || true
echo "== RoCE check, sampled during the campaign"
sed 's/^/   /' "$rdma_out" 2>/dev/null || echo "   (no result)"

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
