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
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

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
until "$HF_BIN" download "$REPO" \
        --revision "$REV" \
        --local-dir "$DEST" \
        --max-workers 4; do
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
