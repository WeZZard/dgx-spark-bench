#!/usr/bin/env bash
# One configuration, end to end: bring up, wait, measure every cell, tear down.
#
#   CONFIG_NOTE="rdma" NCCL_DEBUG=INFO bash scripts/sweep-deepseek.sh
#   CONFIG_NOTE="rdma+mtp" QWEN_SPEC=NEXTN bash scripts/sweep-deepseek.sh
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
BASE="${BASE:-http://127.0.0.1:8888}"
NAME="${DS_NAME:-vllm_dsv4}"
WORKER="${WORKER_HOST:-spark-right-fast}"

log=/tmp/ds-sweep-$NOTE.log
echo "== sweep '$NOTE' starting $(date -Is), log $log"

# TEAR DOWN BOTH RANKS, ALWAYS, ON EVERY EXIT PATH.
#
# A rank left holding 90 GB is what turned a failed launch into a wedged node,
# and this script reaches that state from three directions: the next sweep
# never happening, one rank outliving its peer mid-campaign, and -- the one
# that was missed until 2026-09-04 -- an early `exit` between bring-up and the
# teardown at the bottom. wait_ready failing, or a preflight refusing to
# measure, jumped straight over it and left both containers up. A rank that is
# not serving anything is only holding memory.
#
# So the teardown is a trap, not a block of code at the end that an exit can
# step around.
teardown() {
  echo "== tearing down both ranks"
  for host in "" "$WORKER"; do
    if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
    else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
  done
}

# Once before, to reclaim whatever a previous run left behind...
teardown
# ...and once on every way out of this script, however it ends.
trap teardown EXIT

( setsid nohup bash scripts/bring-up.sh scripts/launch-deepseek-vllm.sh > "$log" 2>&1 < /dev/null & )

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

# Before: the loader checks, which are all that can be asked of a cold server.
bash scripts/check-patch4.sh "$NAME" "$BASE" || {
  echo "== refusing to measure: patch-4 preflight failed. Decode would run at" >&2
  echo "   about half speed with output quality intact, which is exactly the" >&2
  echo "   number this repo must not record." >&2
  exit 1
}

CAMPAIGN_NOTE="$NOTE" bash scripts/campaign-deepseek.sh "$BASE"

# After: the acceptance rate, which needs traffic and is the only one of the
# three that observes the draft actually working. Not fatal here -- the cells
# are already recorded -- but a failure means they must not be believed.
echo "== patch-4 re-check, now that DSpark has had traffic"
REQUIRE_ACCEPTANCE=1 bash scripts/check-patch4.sh "$NAME" "$BASE" \
  || echo "!! DSpark acceptance is in unpatched territory: the numbers above are suspect" >&2


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

# The teardown runs from the EXIT trap installed at the top, so it happens
# after this line whether the campaign finished, refused, or died.
echo "== sweep '$NOTE' done $(date -Is)"
