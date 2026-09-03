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

# Read the flags off the RUNNING container rather than restating them here. A
# campaign that describes the configuration it meant to launch, while the server
# runs a different one, is how 2026-09-03 produced a result about expert
# parallelism from a rank that never had it. `docker inspect` reports the argv
# the process was actually started with.
FLAGS="${CAMPAIGN_FLAGS:-$(docker inspect --format '{{join .Args " "}}' "${QWEN_NAME:-sglang_qwen38fn}" 2>/dev/null)}"
if [ -z "$FLAGS" ]; then
  echo "could not read the launch flags from container ${QWEN_NAME:-sglang_qwen38fn}." >&2
  echo "The flags are the record of what was measured; set CAMPAIGN_FLAGS to proceed." >&2
  exit 2
fi
echo "== flags, read from the running container:"
echo "   $FLAGS"

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

# Warm up first, and discard it. SGLang compiles Triton kernels lazily, after
# serving has started -- the log says so plainly:
#
#   Triton kernel 'chunk_gated_delta_rule_fwd_kkt_solve_kernel' took 1.71 s to
#   compile after serving started. Serving-time compilation can stall the ...
#
# Served rate divides output tokens by the whole wall clock, so a few seconds of
# one-time compilation lands entirely on whichever cell runs first and makes it
# look like a slow configuration. Warming is not massaging the number: the
# compilation happens once per server, not once per request, so charging it to
# one cell would misreport every cell.
warmup() {
  echo "== warmup (discarded): compiling kernels at serving time"
  for i in 1 2 3; do
    curl -s --max-time 300 "$BASE/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"Write one sentence about caching."}],"max_tokens":64,"temperature":0}' \
      -o /dev/null -w "   warmup $i: http=%{http_code} %{time_total}s\n" || true
  done
  # a long prompt too, so the prefill path is compiled before it is measured
  python3 - "$BASE" <<'PY' || true
import json, sys, urllib.request
base = sys.argv[1]
body = {"model": "qwen38-flash-next",
        "messages": [{"role": "user", "content": "ok " * 6000 + "\nReply with one word."}],
        "max_tokens": 16, "temperature": 0}
req = urllib.request.Request(base + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=600).read()
    print("   warmup long-prompt: ok")
except Exception as e:
    print(f"   warmup long-prompt: {e}")
PY
}
warmup

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
