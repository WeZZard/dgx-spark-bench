#!/usr/bin/env bash
# Start a two-node server: rank 1 on the worker, rank 0 here, same knobs.
#
#   bash scripts/bring-up.sh scripts/launch-qwen-sglang.sh
#
# Run this ON THE HEAD. It reaches the worker over the 200 GbE link by its
# /etc/hosts name (`spark-right-fast`); the LAN name is not resolvable from
# the nodes, and the fast link is the right path for the ~63 GiB of weights
# each rank pulls over NFS anyway.
#
# The whole reason this is a script rather than two ssh calls is in
# scripts/lib/cluster.sh: ssh carries no environment, so a tuning knob set in
# front of a hand-written pair of calls reaches rank 0 and silently stops at
# the ssh boundary. bring_up derives the forwarded set from the knob prefixes
# and prints it for both ranks.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$here/scripts/lib/cluster.sh"

launcher="${1:?usage: bring-up.sh <launcher path relative to repo root>}"
worker="${2:-${WORKER_HOST:-spark-right-fast}}"

cd "$here"

# Rule 3, checked rather than remembered. A launcher whose image has never
# recorded a clean run is unproven, and an unproven launcher is run with
# someone watching. The database is what knows; ATTENDED=1 is the human saying
# they are in fact watching.
if [ "${ATTENDED:-0}" != "1" ]; then
  img="$(grep -oE '^[A-Z_]*IMAGE="\$\{[A-Z_]*IMAGE:-[^}]+\}"' "$launcher" \
         | head -1 | sed -E 's/.*:-([^}]+)\}"/\1/')"
  if [ -n "$img" ]; then
    if ! "$here/scripts/sparkbench" preflight --engine-image "$img"; then
      echo "refusing to bring up an unproven launcher unattended." >&2
      exit 3
    fi
  else
    echo "could not determine the image from $launcher; treating as unproven." >&2
    echo "set ATTENDED=1 if you are watching." >&2
    exit 3
  fi
fi
echo "== bring-up $(date -Is): $launcher, worker $worker"
bring_up "$worker" "$here" "$launcher"
