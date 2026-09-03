#!/usr/bin/env bash
# DeepSeek-V4-Flash cells, same shape and same load generator as the others.
#
# The checkpoint defaults to the text 0731 model, not Vision-Exp, and that is a
# considered choice rather than a substitution. BASELINE.md's DeepSeek row is
# labelled Vision-Exp, but re-reading the sources shows its headline decode
# figures -- 84.3 peak, 197 aggregate at c6 -- are quoted for the text model;
# what the same repository reports for Vision-Exp is an image latency. 0731 is
# already on disk. See the source-check section in BASELINE.md.
#
# DS_MODEL selects between the two checkpoints on disk, and the difference is
# the most interesting comparison this model offers: the official one carries
# MXFP4 experts (E8M0 scales, block 32), the nvidia one NVFP4 (E4M3, block 16),
# and they hold the SAME 138.00 GiB of packed 4-bit weights. Any speed
# difference between them is therefore a kernel-path difference, not a weight
# bandwidth one -- which is exactly the claim B12X makes.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:-http://127.0.0.1:8888}"
IMAGE="${DS_IMAGE:-vllm-dsv4:ray}"
MODEL_DIR="${DS_MODEL_DIR:-/home/station/models/hf/dsv4-flash-fp8}"
MODEL_ID="${DS_MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
CONTAINER="${DS_NAME:-vllm_dsv4}"
MAXTOK="${MAXTOK:-512}"

FLAGS="${CAMPAIGN_FLAGS:-$(docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null)}"
[ -n "$FLAGS" ] || { echo "could not read launch flags from $CONTAINER" >&2; exit 2; }
echo "== flags, read from the running container:"; echo "   $FLAGS"

warmup() {
  echo "== warmup (discarded)"
  for i in 1 2 3; do
    curl -s --max-time 600 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Write one sentence about caching."}],"max_tokens":64,"temperature":0}' \
      -o /dev/null -w "   warmup $i: http=%{http_code} %{time_total}s\n" || true
  done
}

cell() {
  local workload="$1" users="$2" note="$3"
  echo; echo "############ $workload @ ${users} users ############"
  "$here/scripts/sparkbench" measure \
    --base "$BASE" --served-model deepseek-v4-flash \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --engine-image "$IMAGE" --container "$CONTAINER" \
    --workload "$workload" --users "$users" --max-tokens "$MAXTOK" \
    --flags "$FLAGS" --notes "${CAMPAIGN_NOTE:+[$CAMPAIGN_NOTE] }$note" \
    || echo "!! cell failed: $workload @ $users (continuing)"
}

warmup
cell short 1 "short-prompt cell, matches published conditions"
cell short 6 "published concurrency c6"
cell code  1 "code: the content DSpark accepts best, ~78% per the source"
cell code  6 "code at published concurrency"
cell prose 1 "prose: ~34% DSpark acceptance, the honest agentic floor"
cell agentic-4k  1 "agentic context, short"
cell agentic-4k  6 "agentic context, 6 users"
cell agentic-33k 1 "agentic context, 33k"
cell agentic-33k 6 "agentic context, 33k, 6 users"
