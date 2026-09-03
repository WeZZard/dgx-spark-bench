#!/usr/bin/env bash
# The Qwen3.8-Flash-Next reproduction, as a fixed sequence of measured cells.
#
#   bash scripts/campaign-qwen.sh [base-url]
#
# Order is deliberate. The short cell runs first because it is the only one
# that can be compared with the published figure at all: every number in
# BASELINE.md is a decode rate taken at a ~22-token prompt, and the previous
# attempt compared those against its own 2,457-token runs. Measuring the
# matching cell is the fix Rule 2 demands -- not switching formula.
#
# `--max-tokens 512` is held constant across every cell on purpose. Served rate
# divides output tokens by the whole wall clock, so a cell that generates fewer
# tokens pays a larger share of prefill and reads slower. Varying the output
# length between cells would make the ladder measure the output length rather
# than the configuration.
#
# Every cell writes a complete six-field tuple or writes nothing: the harness
# reads the engine, its version, the KV cache dtype and the KV pool off the
# running server, and reads the revision and weight quantization off the
# checkpoint. Nothing about the tuple is typed here except the image, which the
# server cannot report.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:-http://127.0.0.1:30000}"
IMAGE="${QWEN_IMAGE:-sglang-qwen38fn:sm121-ours}"
MODEL_DIR="${QWEN_MODEL_DIR:-/home/station/models/hf/qwen38-flash-next-nvfp4}"
CONTAINER="${QWEN_NAME:-sglang_qwen38fn}"
MAXTOK="${MAXTOK:-512}"
FLAGS="${CAMPAIGN_FLAGS:---max-total-tokens 600000 --mem-fraction-static 0.80 --cuda-graph-max-bs 8 --disable-cuda-graph-padding --ple-offload-embedding --max-mamba-cache-size 97 --sampling-backend pytorch --disable-radix-cache --context-length 120000}"

cell() {
  local workload="$1" users="$2" note="$3"
  echo
  echo "############ $workload @ ${users} users ############"
  "$here/scripts/sparkbench" measure \
    --base "$BASE" \
    --served-model qwen38-flash-next \
    --model-dir "$MODEL_DIR" \
    --model-id RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --engine-image "$IMAGE" \
    --container "$CONTAINER" \
    --workload "$workload" \
    --users "$users" \
    --max-tokens "$MAXTOK" \
    --flags "$FLAGS" \
    --notes "$note" \
    || echo "!! cell failed: $workload @ $users (continuing)"
}

# The like-for-like cell first: published conditions are a ~22-token prompt.
cell short 1 "short-prompt cell, matches the conditions every published figure was taken under"
cell short 6 "published concurrency: 6 agents"

# The content the published peak was measured on.
cell code   1 "code generation, the content DSpark-style drafters accept best"
cell code   6 "code at published concurrency"
cell prose  1 "prose, the honest floor for agentic content"

# What this project is actually for.
cell agentic-4k   1 "agentic context, short"
cell agentic-4k   6 "agentic context, short, 6 users"
cell agentic-33k  1 "agentic context, 33k"
cell agentic-33k  6 "agentic context, 33k, 6 users"

echo
echo "############ recorded tuples ############"
"$here/scripts/sparkbench" report
