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

NOTE="${CONFIG_NOTE:?set CONFIG_NOTE to name this configuration}"
BASE="${BASE:-http://127.0.0.1:30000}"
NAME="${QWEN_NAME:-sglang_qwen38fn}"
WORKER="${WORKER_HOST:-spark-right-fast}"
READY_DEADLINE_S="${READY_DEADLINE_S:-1800}"

log=/tmp/qwen-sweep-$NOTE.log
echo "== sweep '$NOTE' starting $(date -Is), log $log"

# Tear down both ranks and wait for the memory back, rather than sleeping. A
# rank left holding 90 GB is what turned a failed launch into a wedged node.
for host in "" "$WORKER"; do
  if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
done

( setsid nohup bash scripts/bring-up.sh scripts/launch-qwen-sglang.sh > "$log" 2>&1 < /dev/null & )

echo "== waiting up to ${READY_DEADLINE_S}s for /health"
started=$(date +%s)
deadline=$(( started + READY_DEADLINE_S ))
# The container does not exist for the first few seconds: bring_up has to reach
# the worker over ssh, drop the page cache and wait on MemAvailable before it
# runs `docker run`. Treating "absent" as fatal from the first iteration is a
# race, and it lost -- the wrapper exited while the detached bring-up carried on
# loading a 126 GiB model that nothing was then going to measure.
STARTUP_GRACE_S="${STARTUP_GRACE_S:-180}"
while :; do
  if curl -sf --max-time 5 "$BASE/health" >/dev/null 2>&1; then
    echo "== ready after $(( $(date +%s) - started ))s"; break
  fi
  st=$(timeout 15 docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo absent)
  if [ "$st" != "running" ] && [ $(( $(date +%s) - started )) -gt "$STARTUP_GRACE_S" ]; then
    echo "== container is '$st' after ${STARTUP_GRACE_S}s grace; last log lines:" >&2
    tr '\r' '\n' < "$log" | grep -vE '^\s*$|Multi-thread' | tail -20 >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "== DEADLINE: not ready in ${READY_DEADLINE_S}s. Still running is not progress." >&2
    exit 1
  fi
  sleep 10
done

bash scripts/check-rdma.sh "$NAME" || echo "(rdma check inconclusive; see above)"

CAMPAIGN_NOTE="$NOTE" bash scripts/campaign-qwen.sh "$BASE"

echo "== sweep '$NOTE' done $(date -Is)"
