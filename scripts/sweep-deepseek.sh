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

# Tear down both ranks and wait for the memory back, rather than sleeping. A
# rank left holding 90 GB is what turned a failed launch into a wedged node.
for host in "" "$WORKER"; do
  if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
done

( setsid nohup bash scripts/bring-up.sh scripts/launch-deepseek-vllm.sh > "$log" 2>&1 < /dev/null & )

wait_ready "$NAME" "$BASE" "$log" || exit 1

bash scripts/check-rdma.sh "$NAME" || echo "(rdma check inconclusive; see above)"

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

echo "== sweep '$NOTE' done $(date -Is)"
