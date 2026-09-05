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
source "$here/scripts/lib/warmup.sh"
BASE="${1:-http://127.0.0.1:30000}"
# Read the image off the RUNNING container rather than repeating the
# launcher's default here. Those two defaults drifted the moment the GLM
# launcher moved to the patched sglang-glm53:gb10-tilelang image: the server
# ran the patched image and the database recorded the base one, which is
# precisely the confusion the separate tag was built to prevent. `docker
# inspect` cannot drift.
IMAGE="$(docker inspect --format '{{.Config.Image}}' "${QWEN_NAME:-sglang_qwen38fn}" 2>/dev/null)"
IMAGE="${IMAGE:-${QWEN_IMAGE:-unknown-image}}"
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
  # --ignore-eos below: every cell must emit exactly MAXTOK tokens.
  #
  # Without it a cell measures whatever length the model felt like. Served rate
  # is output tokens over the whole wall clock, so a short completion amortises
  # the fixed prefill and scheduling cost over fewer tokens and reads low. Same
  # DeepSeek server, same prompt, two runs minutes apart:
  #
  #   prose @ 1 user   512 tokens / 11.1 s  ->  46.0 tok/s
  #   prose @ 1 user   144 tokens /  7.9 s  ->  18.1 tok/s
  #
  # A 2.5x swing that is entirely how long the answer was. Across models it is
  # a systematic bias, not noise: GLM ran nearly every cell to the full 512
  # while DeepSeek often stopped early, so the comparison would read verbosity
  # as speed.
  #
  # This comment lives ABOVE the command on purpose. Put it between two
  # backslash-continued lines and the continuation joins the comment onto the
  # command line, the "#" swallows the rest of it, and every later argument
  # becomes a separate command. That is not hypothetical -- it is what the
  # first version of this change did, silently dropping --ignore-eos, --flags
  # and --notes from every cell of a whole campaign.
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
    --ignore-eos \
    --flags "$FLAGS" \
    --notes "${CAMPAIGN_NOTE:+[$CAMPAIGN_NOTE] }$note" \
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
warmup() { warmup_engine "$BASE" "qwen38-flash-next"; }
# Shape coverage by construction: warm every (workload, users) pair the
# campaign is about to measure, using the same generator and the same seed,
# recording nothing.
#
# The hand-written warmup could not do this. It sent a burst of 10-token
# prompts, and campaigns then compiled kernels INSIDE measured cells --
# charging tens of seconds to their time-to-first-token and making those cells
# swing by up to 2.2x between samples of one unchanged server. The engines say
# so themselves: vLLM's jit_monitor prints "TileLang JIT compilation during
# inference ... consider extending warmup to cover this shape/config", and
# SGLang prints "Triton kernel ... took 1.91 s to compile after serving
# started. Serving-time compilation can stall the engine".
#
# max_tokens is the SAME as the measured cells, not a token or two.
#
# The first version used 8, on the theory that decode-graph capture keys on
# batch size and the prefill cost is paid in full either way. That fixed the
# cells whose cost was prefill compilation -- code @ 6 went from 1.64x spread
# to 1.07x -- and left the ones whose cost is not: `short @ 6` still read
# 90.0, 138.8, 139.0 across three samples of one server with DSpark
# acceptance flat at 75/70/80%, so nothing about drafting explains it. An
# 8-token warmup never reaches the sequence lengths, KV block counts or
# attention buckets a 512-token cell reaches, so those are still compiled or
# tuned inside the first measured sample.
#
# Warming at full length costs one extra campaign per sweep. That is cheap
# against publishing a number that is 35% low because it went first.
warm_cell() {
  local workload="$1" users="$2"
  printf '   warm %s @ %s ... ' "$workload" "$users"
  if "$here/scripts/sparkbench" measure \
    --base "$BASE" --served-model qwen38-flash-next \
    --model-dir "$MODEL_DIR" --model-id RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --engine-image "$IMAGE" --container "$CONTAINER" \
       --workload "$workload" --users "$users" --max-tokens "$MAXTOK" \
       --ignore-eos --no-record >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAILED (continuing; this cell may compile while measured)"
  fi
}

warmup

# The like-for-like cell first: published conditions are a ~22-token prompt.
echo "== warming every shape the campaign will measure (discarded)"
warm_cell short 1
warm_cell short 6
warm_cell code 1
warm_cell code 6
warm_cell prose 1
warm_cell agentic-4k 1
warm_cell agentic-4k 6
warm_cell agentic-33k 1
warm_cell agentic-33k 6
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
