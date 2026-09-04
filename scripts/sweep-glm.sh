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

bash scripts/check-rdma.sh "$NAME" || echo "(rdma check inconclusive; see above)"

CAMPAIGN_NOTE="$NOTE" bash scripts/campaign-glm.sh "$BASE"

echo "== sweep '$NOTE' done $(date -Is)"
