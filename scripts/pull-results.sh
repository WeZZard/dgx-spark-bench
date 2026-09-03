#!/usr/bin/env bash
# Copy the canonical results database off the head, safely.
#
#   bash scripts/pull-results.sh [head-ssh-host]
#
# Two rules, both from failures in docs/baselines.md.
#
# The head's copy is canonical and this pull is ONE WAY. The previous attempt
# let every entry point open `results/runs.sqlite` by a relative path, so the
# Mac and the head each accumulated runs the other did not have -- 6 runs only
# on the head, 150 only on the Mac -- and neither was a superset. Copying
# either over the other would have destroyed real measurements, and the report
# would have rendered a clean table with whole phases missing. Here the Mac
# never writes, so there is nothing to merge and nothing to lose.
#
# The snapshot is taken with sqlite's own `.backup`, never `scp` on the live
# file. A benchmark can be mid-write, and this database has already come back
# `database disk image is malformed` once, in a build of sqlite with no
# `.recover` to fall back on.
set -euo pipefail

HEAD="${1:-spark-left}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_db="${REMOTE_DB:-/home/station/dgx-spark-bench/results/runs.sqlite}"
local_db="$here/results/runs.sqlite"

mkdir -p "$here/results"
echo "== snapshotting $HEAD:$remote_db"
ssh -o ConnectTimeout=30 "$HEAD" \
  "test -f '$remote_db' && sqlite3 '$remote_db' \".backup '/tmp/runs-snapshot.sqlite'\" && \
   sqlite3 /tmp/runs-snapshot.sqlite 'PRAGMA integrity_check' | head -1"

scp -o ConnectTimeout=30 "$HEAD:/tmp/runs-snapshot.sqlite" "$local_db"
echo "== local copy at $local_db"
sqlite3 "$local_db" "SELECT COUNT(*) || ' runs, ' || (SELECT COUNT(*) FROM metrics) || ' metrics' FROM runs"
