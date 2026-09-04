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
# THE WORKER RUNS ITS OWN COPY OF THE LAUNCHER, SO SEND IT THE CURRENT ONE.
#
# bring_up ssh's into the worker and runs $launcher from the worker's own
# checkout. Nothing kept that checkout current, so every edit to a launcher
# had to be copied to two machines by hand, and the day one was not, the two
# ranks ran different code: rank 0 with --mem-fraction-static 0.85, rank 1
# with 0.92, from the same command. A tensor-parallel pair whose ranks
# disagree about how much memory to reserve is not a configuration anyone
# meant to measure, and nothing in the log says the ranks differ -- each
# prints its own flags and both look right.
#
# Only scripts/ and src/ go: the code that has to agree. Model stores,
# results and anything worker-local are untouched. --delete is deliberately
# NOT used, so a stray file on the worker is left alone rather than removed
# by a bring-up.
if [ "${BRINGUP_SYNC:-1}" = "1" ]; then
  echo "== syncing scripts/ and src/ to $worker"
  rsync -az --exclude '__pycache__' \
    "$here/scripts/" "$worker:$here/scripts/" || {
      echo "refusing to bring up: could not sync scripts/ to $worker." >&2
      echo "The ranks would run different code. BRINGUP_SYNC=0 skips this." >&2
      exit 4
    }
  rsync -az --exclude '__pycache__' \
    "$here/src/" "$worker:$here/src/" || {
      echo "refusing to bring up: could not sync src/ to $worker." >&2
      exit 4
    }
  # Prove it rather than trust it: the launcher is the file whose divergence
  # caused this, so compare the two copies before either rank starts.
  mine="$(md5sum "$here/$launcher" | cut -d" " -f1)"
  theirs="$(ssh -o ConnectTimeout=30 "$worker" "md5sum '$here/$launcher' 2>/dev/null | cut -d' ' -f1")"
  if [ "$mine" != "$theirs" ]; then
    echo "refusing to bring up: $launcher differs between ranks after sync" >&2
    echo "  head:   $mine" >&2
    echo "  worker: ${theirs:-<missing>}" >&2
    exit 4
  fi
  echo "   launcher matches on both ranks ($mine)"
fi

echo "== bring-up $(date -Is): $launcher, worker $worker"
bring_up "$worker" "$here" "$launcher"
