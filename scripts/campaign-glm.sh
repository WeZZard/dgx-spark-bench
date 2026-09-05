#!/usr/bin/env bash
# GLM-5.3-Flash reproduction cells. Same shape as the Qwen campaign so the two
# models are measured by the same code and are therefore comparable.
#
# The concurrency ladder is the published one for this model -- c1, c2, c4, c8
# -- because BASELINE.md gives figures at each of those points and a ladder
# that does not line up with the source cannot be checked against it.
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
IMAGE="$(docker inspect --format '{{.Config.Image}}' "${GLM_NAME:-sglang_glm53}" 2>/dev/null)"
IMAGE="${IMAGE:-${GLM_IMAGE:-unknown-image}}"
MODEL_DIR="${GLM_MODEL_DIR:-/home/station/models/hf/glm53-flash-nvfp4-ref}"
CONTAINER="${GLM_NAME:-sglang_glm53}"
MAXTOK="${MAXTOK:-512}"

FLAGS="${CAMPAIGN_FLAGS:-$(docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null)}"
[ -n "$FLAGS" ] || { echo "could not read launch flags from $CONTAINER" >&2; exit 2; }
echo "== flags, read from the running container:"; echo "   $FLAGS"

warmup() { warmup_engine "$BASE" "glm53-flash"; }

cell() {
  local workload="$1" users="$2" note="$3"
  echo; echo "############ $workload @ ${users} users ############"
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
    --base "$BASE" --served-model glm53-flash \
    --model-dir "$MODEL_DIR" --model-id LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --engine-image "$IMAGE" --container "$CONTAINER" \
    --workload "$workload" --users "$users" --max-tokens "$MAXTOK" \
    --ignore-eos \
    --flags "$FLAGS" --notes "${CAMPAIGN_NOTE:+[$CAMPAIGN_NOTE] }$note" \
    || echo "!! cell failed: $workload @ $users (continuing)"
}

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
    --base "$BASE" --served-model glm53-flash \
    --model-dir "$MODEL_DIR" --model-id LibertAIDAI/GLM-5.3-Flash-NVFP4 \
    --engine-image "$IMAGE" --container "$CONTAINER" \
       --workload "$workload" --users "$users" --max-tokens "$MAXTOK" \
       --ignore-eos --no-record >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAILED (continuing; this cell may compile while measured)"
  fi
}

warmup
echo "== warming every shape the campaign will measure (discarded)"
warm_cell short 1
warm_cell code 1
warm_cell code 2
warm_cell code 4
warm_cell code 8
warm_cell agentic-4k 1
warm_cell agentic-4k 8
warm_cell agentic-33k 1
warm_cell agentic-33k 8
cell short 1 "short-prompt cell, matches published conditions"
cell code  1 "code, the content the published c1 figure uses"
cell code  2 "published ladder c2"
cell code  4 "published ladder c4"
cell code  8 "published ladder c8, --max-running-requests 8"
cell agentic-4k  1 "agentic context, short"
cell agentic-4k  8 "agentic context at published concurrency"
cell agentic-33k 1 "agentic context, 33k"
cell agentic-33k 8 "agentic context, 33k, 8 users"
