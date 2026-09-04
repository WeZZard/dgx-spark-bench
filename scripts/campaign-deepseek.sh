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
source "$here/scripts/lib/warmup.sh"
BASE="${1:-http://127.0.0.1:8888}"
# Read the image off the RUNNING container rather than repeating the
# launcher's default here. Those two defaults drifted the moment the GLM
# launcher moved to the patched sglang-glm53:gb10-tilelang image: the server
# ran the patched image and the database recorded the base one, which is
# precisely the confusion the separate tag was built to prevent. `docker
# inspect` cannot drift.
IMAGE="$(docker inspect --format '{{.Config.Image}}' "${DS_NAME:-vllm_dsv4}" 2>/dev/null)"
IMAGE="${IMAGE:-${DS_IMAGE:-unknown-image}}"
MODEL_DIR="${DS_MODEL_DIR:-/home/station/models/hf/dsv4-flash-fp8}"
MODEL_ID="${DS_MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
CONTAINER="${DS_NAME:-vllm_dsv4}"
MAXTOK="${MAXTOK:-512}"

FLAGS="${CAMPAIGN_FLAGS:-$(docker inspect --format '{{join .Args " "}}' "$CONTAINER" 2>/dev/null)}"
[ -n "$FLAGS" ] || { echo "could not read launch flags from $CONTAINER" >&2; exit 2; }
echo "== flags, read from the running container:"; echo "   $FLAGS"

warmup() { warmup_engine "$BASE" "deepseek-v4-flash"; }

# DSpark acceptance, PER CELL, by bracketing the counters.
#
# The campaign-wide aggregate cannot answer the question it is asked. The
# source puts patched acceptance at ~78% on structured code, ~56% on mixed
# agent traffic and ~34% on prose (BASELINE.md), against ~25% for the
# unpatched loader. A five-workload campaign averages straight through that
# spread: this campaign's 0731 run read 52.8% and its Vision-Exp run 33.5%,
# and neither number separates "prose-heavy and healthy" from "broken", which
# is what check-patch4.sh's 40% line assumed it could do.
#
# Bracketing the `code` cell does separate them: ~78% patched against ~25%
# unpatched is a 3x gap with room for a threshold in the middle.
#
# The counters are monotonic totals, so a difference over a cell is that
# cell's acceptance and nothing else. Empty output means speculation is off.
CODE_CELL_ACCEPTANCE=""

acc_snapshot() {
  curl -sf --max-time 5 "$BASE/metrics" 2>/dev/null | awk '
    /^vllm:spec_decode_num_accepted_tokens_total/ {a=$2}
    /^vllm:spec_decode_num_draft_tokens_total/    {d=$2}
    END { if (d > 0) printf "%.0f %.0f", a, d }'
}

cell() {
  local workload="$1" users="$2" note="$3"
  echo; echo "############ $workload @ ${users} users ############"
  local before after
  before="$(acc_snapshot)"
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
    --base "$BASE" --served-model deepseek-v4-flash \
    --model-dir "$MODEL_DIR" --model-id "$MODEL_ID" \
    --engine-image "$IMAGE" --container "$CONTAINER" \
    --workload "$workload" --users "$users" --max-tokens "$MAXTOK" \
    --ignore-eos \
    --flags "$FLAGS" --notes "${CAMPAIGN_NOTE:+[$CAMPAIGN_NOTE] }$note" \
    || echo "!! cell failed: $workload @ $users (continuing)"

  after="$(acc_snapshot)"
  if [ -n "$before" ] && [ -n "$after" ]; then
    local acc
    acc=$(awk -v b="$before" -v a="$after" 'BEGIN{
      split(b, x, " "); split(a, y, " ");
      da = y[1] - x[1]; dd = y[2] - x[2];
      if (dd > 0) printf "%.1f", 100*da/dd
    }')
    if [ -n "$acc" ]; then
      echo "dspark acceptance, this cell only: ${acc}%"
      # An `a && b && c` chain here would make cell() return non-zero for
      # every workload that is not `code`, which is a misleading exit status
      # for a function whose caller may one day check it.
      if [ "$workload" = code ] && [ "$users" = 1 ]; then
        CODE_CELL_ACCEPTANCE="$acc"
      fi
    fi
  fi
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

# THE DISCRIMINATING CHECK, on the one cell where patched and unpatched are
# far apart. ~78% published patched, ~25% unpatched; 50% sits between them
# with margin on both sides, which is more than can be said for a threshold
# applied to a campaign-wide average.
#
# Not fatal: the cells above are recorded either way and a campaign that threw
# away real measurements over a diagnostic would be worse. But a run that
# fails this should not have its DeepSeek decode numbers believed.
echo
if [ -z "$CODE_CELL_ACCEPTANCE" ]; then
  echo "== DSpark acceptance on the code cell: not measured (speculation off,"
  echo "   or /metrics had no spec_decode counters)."
elif awk "BEGIN{exit !($CODE_CELL_ACCEPTANCE < 50)}"; then
  echo "!! DSpark acceptance on the code cell is ${CODE_CELL_ACCEPTANCE}%, against ~78%" >&2
  echo "   published for this content and ~25% for the unpatched draft loader." >&2
  echo "   Treat the decode behaviour of this configuration as unexplained." >&2
else
  echo "== DSpark acceptance on the code cell: ${CODE_CELL_ACCEPTANCE}% (published ~78%,"
  echo "   unpatched ~25%). The draft is working."
fi
