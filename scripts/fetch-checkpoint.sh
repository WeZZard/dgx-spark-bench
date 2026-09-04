#!/usr/bin/env bash
# Fetch a checkpoint by PINNED REVISION through the China acceleration path.
#
# Three things this script exists to enforce:
#
# 1. A revision, never a branch. Rule 1 says a result names the checkpoint
#    *and its revision*; "main" is not a revision, it is a moving target, and
#    a run recorded against it cannot be reproduced later.
#
# 2. hf-mirror.com, never huggingface.co. Plain huggingface.co is not
#    reachable from the nodes without the VPN, and its hf.co CDN hosts were
#    silently eating VPN traffic before HF_ENDPOINT was set
#    (docs/baselines.md, "Download path").
#
# 3. It must not run while anything is being measured. A 156 GiB download
#    evicts the page cache the engine is loading weights through and competes
#    for the same disk, so a sweep running beside it is measuring the
#    download. The script refuses to start if a serving container is up.
#    Override with FETCH_ALLOW_BUSY=1 only if you are not recording numbers.
#
# Usage: scripts/fetch-checkpoint.sh <repo-id> <revision> <dest-dir-name>

set -euo pipefail

REPO="${1:?repo id, e.g. deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
REV="${2:?full 40-char commit sha — a branch name is refused}"
NAME="${3:?destination directory name under the model store}"

STORE="${MODEL_STORE:-/home/station/models/hf}"
DEST="$STORE/$NAME"

# ~/.local/bin holds `hf` but is not on a non-interactive ssh PATH, so a
# script that assumes an interactive shell finds no CLI at all.
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# The Xet backend fails on these nodes -- "CAS Client Error: Request
# middleware error" from `hf download`, huggingface_hub 1.28.0, aarch64.
# Auth and plain HTTPS are fine; with Xet off, downloads run at the WAN
# limit. See docs/baselines.md, "Gotchas found during provisioning".
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! "$REV" =~ ^[0-9a-f]{40}$ ]]; then
  echo "refusing: '$REV' is not a 40-char commit sha." >&2
  echo "Resolve the branch to a sha first and record that sha." >&2
  exit 2
fi

if [ "${FETCH_ALLOW_BUSY:-0}" != "1" ]; then
  running="$(docker ps --format '{{.Names}}' | grep -E 'sglang_|vllm_' || true)"
  if [ -n "$running" ]; then
    echo "refusing: serving container(s) up — a download would evict the page" >&2
    echo "cache under them and contend for the same disk:" >&2
    echo "$running" | sed 's/^/  /' >&2
    echo "Set FETCH_ALLOW_BUSY=1 only if nothing is being recorded." >&2
    exit 3
  fi
fi

avail_gib="$(df -BG --output=avail "$STORE" | tail -1 | tr -dc '0-9')"
echo "== endpoint     $HF_ENDPOINT"
echo "== repo         $REPO"
echo "== revision     $REV"
echo "== destination  $DEST"
echo "== free space   ${avail_gib} GiB"

mkdir -p "$DEST"

# ONE FETCH PER DESTINATION, ENFORCED.
#
# Two fetches writing the same directory do not share the work, they destroy
# it: each `hf download` treats the other's .incomplete files as its own,
# and the loser's partials are discarded. On 2026-09-04 this happened by
# accident and cost hours. The sequence is worth knowing because it looks
# reasonable at every step:
#
#   1. A fetch was running and looked slow.
#   2. Its `hf` child was killed by PID, to restart with more workers.
#   3. The SCRIPT survived -- the retry loop below catches a failed child and
#      spawns another -- so killing the child restarted it rather than
#      stopping it.
#   4. A second fetch was then started, and from that moment two scripts wrote
#      the same destination for two and a half hours.
#
# The visible symptoms all pointed elsewhere: "Temporary failure in name
# resolution" from the contention, a destination that shrank from 12 GB to
# 5 GB, and a throughput comparison between "4 workers" and "16 workers" that
# was really 4 against 20-in-two-processes and is therefore worthless.
#
# Killing the child is not stopping the fetch. Kill the script.
LOCK="$DEST/.fetch.lock"
if [ -e "$LOCK" ]; then
  other="$(cat "$LOCK" 2>/dev/null || echo "")"
  if [ -n "$other" ] && kill -0 "$other" 2>/dev/null; then
    echo "refusing: a fetch for $DEST is already running as PID $other." >&2
    echo "  To stop it:      kill $other        # the SCRIPT, not its hf child" >&2
    echo "  To watch it:     tail -f the log it was started with" >&2
    echo "  A second fetch would discard this one's partial files." >&2
    exit 5
  fi
  echo "== stale lock from PID ${other:-unknown} (not running); taking over"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT


# `huggingface-cli` is deprecated in huggingface_hub 1.x and *no longer
# works* -- it prints a deprecation warning and exits without downloading.
# `hf` is the current entry point. Check rather than assume.
HF_BIN="${HF_BIN:-hf}"
command -v "$HF_BIN" >/dev/null || {
  echo "refusing: '$HF_BIN' not found. huggingface-cli is deprecated and" >&2
  echo "no longer downloads; install/point HF_BIN at the 'hf' CLI." >&2
  exit 4
}

# Resume is the default; retries cover the mirror dropping a connection
# mid-file, which it does.
attempt=0
# WORKER COUNT: the honest state of the evidence is that it has not been
# measured cleanly, and the two attempts recorded here were both wrong.
#
# What was measured, and why none of it is a controlled comparison:
#
#    4 workers:    240 MB in 60s  =  4.0 MB/s   one fetch running
#   16 workers:  2,640 MB in 60s  = 44.0 MB/s   TWO fetches running
#   16 workers:  3,960 MB in 176s = 22.5 MB/s   TWO fetches running
#   16 workers:  5,920 MB in 240s = 24.0 MB/s   TWO fetches running
#
# Every 16-worker figure was taken while a second fetch script was also
# downloading the same checkpoint, unnoticed, so they measure roughly twenty
# workers split across two processes that were destroying each other's
# partial files. The "11x from one flag" this script claimed, and the "5.6x"
# it was corrected to, are both artefacts. The only clean number here is the
# 4-worker one.
#
# 16 is kept as the default because more connections plausibly helps on a
# per-connection-limited mirror, not because it was demonstrated. Measure it
# properly -- one fetch, from scratch, on a quiet link -- before quoting a
# figure for it.
FETCH_WORKERS="${FETCH_WORKERS:-16}"
echo "== workers       $FETCH_WORKERS"
until "$HF_BIN" download "$REPO" \
        --revision "$REV" \
        --local-dir "$DEST" \
        --max-workers "$FETCH_WORKERS"; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 20 ]; then
    echo "!! gave up after $attempt attempts" >&2
    exit 1
  fi
  echo "-- attempt $attempt failed, retrying in 30s (partial files are kept)"
  sleep 30
done

# Record what was actually fetched, so the revision survives beside the files
# rather than only in this terminal. checkpoint.py reads this.
cat > "$DEST/.provenance.json" <<PROV
{
  "repo_id": "$REPO",
  "revision": "$REV",
  "endpoint": "$HF_ENDPOINT",
  "fetched_at": "$(date -Iseconds)",
  "fetched_by": "scripts/fetch-checkpoint.sh"
}
PROV

echo "== fetched. Audit before any run:"
echo "   scripts/audit-checkpoint.py $DEST --verify"
