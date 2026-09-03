#!/usr/bin/env bash
# GLM-5.3-Flash reproduction cells. Same shape as the Qwen campaign so the two
# models are measured by the same code and are therefore comparable.
#
# The concurrency ladder is the published one for this model -- c1, c2, c4, c8
# -- because BASELINE.md gives figures at each of those points and a ladder
# that does not line up with the source cannot be checked against it.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:-http://127.0.0.1:30000}"
IMAGE="${GLM_IMAGE:-lmsysorg/sglang:glm-5.3-flash}"
MODEL_DIR="${GLM_MODEL_DIR:-/home/station/models/hf/glm53-flash-nvfp4-ref}"
CONTAINER="${GLM_NAME:-sglang_glm53}"
MAXTOK="${MAXTOK:-512}"

FLAGS="${CAMPAIGN_FLAGS:-$(docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null)}"
[ -n "$FLAGS" ] || { echo "could not read launch flags from $CONTAINER" >&2; exit 2; }
echo "== flags, read from the running container:"; echo "   $FLAGS"

warmup() {
  echo "== warmup (discarded)"
  for i in 1 2 3; do
    curl -s --max-time 300 "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
      -d '{"model":"glm53-flash","messages":[{"role":"user","content":"Write one sentence about caching."}],"max_tokens":64,"temperature":0}' \
      -o /dev/null -w "   warmup $i: http=%{http_code} %{time_total}s\n" || true
  done
}

cell() {
  local workload="$1" users="$2" note="$3"
  echo; echo "############ $workload @ ${users} users ############"
  "$here/scripts/sparkbench" measure \
    --base "$BASE" --served-model glm53-flash \
    --model-dir "$MODEL_DIR" --model-id LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --engine-image "$IMAGE" --container "$CONTAINER" \
    --workload "$workload" --users "$users" --max-tokens "$MAXTOK" \
    --flags "$FLAGS" --notes "${CAMPAIGN_NOTE:+[$CAMPAIGN_NOTE] }$note" \
    || echo "!! cell failed: $workload @ $users (continuing)"
}

warmup
cell short 1 "short-prompt cell, matches published conditions"
cell code  1 "code, the content the published c1 figure uses"
cell code  2 "published ladder c2"
cell code  4 "published ladder c4"
cell code  8 "published ladder c8, --max-running-requests 8"
cell agentic-4k  1 "agentic context, short"
cell agentic-4k  8 "agentic context at published concurrency"
cell agentic-33k 1 "agentic context, 33k"
cell agentic-33k 8 "agentic context, 33k, 8 users"
