#!/usr/bin/env bash
# Does an fp8 KV cache break Qwen at ONE user, or only under concurrency?
#
#   bash scripts/probe-qwen-fp8kv.sh
#
# The `[mtp+fp8kv]` campaign died at `agentic-4k @ 6 users` with
#
#   AssertionError: Unsupported rhs dtype fp8e4nv
#
# out of Triton's tl.dot, raised from the QSA chunked-prefill kernel
# sparse_gqa_fwd_interface_triton_ck. It is tempting to read that as "fp8 KV
# does not survive concurrency", and that reading would be wrong in a way
# that matters: it would leave fp8 KV on the table as a single-user option,
# doubling the pool from 166,016 to 332,480 tokens.
#
# srt/layers/attention/qwen_sparse_attn_backend.py says otherwise.
# forward_extend branches on `if not any(prefix_lens)`: with no prefix it
# passes the freshly projected bfloat16 k/v to the plain kernel, and with any
# prefix it hands the _ck kernel K/V read straight out of the paged pool --
# which is fp8 exactly when --kv-cache-dtype made it so. Concurrency is
# incidental. A non-zero prefix is what chunked prefill produces from its
# second chunk onward, and chunked_prefill_size is 8192.
#
# So the prediction is that ONE request with a 30,301-token prompt crashes it,
# with nothing else in flight. This script runs the control that must pass and
# then the cell that must fail, and prints the scheduler traceback either way.
#
# Rule 3 is satisfied: this exact configuration served for seven cells on
# 2026-09-03 before it died, so the launcher is not unproven.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
source scripts/lib/wait-ready.sh

BASE="${BASE:-http://127.0.0.1:30000}"
NAME="${QWEN_NAME:-sglang_qwen38fn}"
WORKER="${WORKER_HOST:-spark-right-fast}"
log=/tmp/qwen-fp8kv-probe.log

for host in "" "$WORKER"; do
  if [ -z "$host" ]; then docker rm -f "$NAME" >/dev/null 2>&1 || true
  else ssh -o ConnectTimeout=30 "$host" "docker rm -f '$NAME' >/dev/null 2>&1 || true"; fi
done

echo "== bringing up with --kv-cache-dtype fp8_e4m3, log $log"
( setsid nohup env QWEN_KV_DTYPE=fp8_e4m3 QWEN_SPEC=NEXTN \
    bash scripts/bring-up.sh scripts/launch-qwen-sglang.sh > "$log" 2>&1 < /dev/null & )

wait_ready "$NAME" "$BASE" "$log" || exit 1

echo "== KV pool and dtype, read from the running server"
curl -sf "$BASE/metrics" | grep -iE "kv_cache_size_tokens|cache_dtype" | head -5

# One cell, one user, through the campaign so the tuple is complete.
probe() {
  local workload="$1" expect="$2"
  echo
  echo "############ $workload @ 1 user -- expected: $expect ############"
  local image flags
  image="$(docker inspect --format '{{.Config.Image}}' "$NAME" 2>/dev/null)"
  flags="$(docker inspect --format '{{join .Args " "}}' "$NAME" 2>/dev/null)"
  # --ignore-eos and a fixed --max-tokens for the same reason every cell in
  # campaign-qwen.sh carries them: served rate divides by output tokens.
  # This comment is ABOVE the command, never inside the continuation.
  scripts/sparkbench measure \
    --base "$BASE" \
    --served-model qwen38-flash-next \
    --model-dir "${QWEN_MODEL_DIR:-/home/station/models/hf/qwen38-flash-next-nvfp4}" \
    --model-id RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --engine-image "$image" \
    --container "$NAME" \
    --workload "$workload" \
    --users 1 \
    --max-tokens 64 \
    --ignore-eos \
    --flags "$flags" \
    --notes "[qwen-fp8kv-probe] fp8 KV at one user, $workload: expected $expect" \
    || echo "!! cell failed"
  if docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null | grep -q running; then
    echo "-- server still running"
  else
    echo "-- SERVER IS GONE"
  fi
  tr '\r' '\n' < "$log" | grep -A4 "Scheduler hit an exception" | tail -20
  tr '\r' '\n' < "$log" | grep -E "AssertionError|kill_process_tree" | tail -5
}

# Control: 4,318 tokens, one chunk under chunked_prefill_size 8192, so
# prefix_lens is all zero and the plain kernel runs. This cell read 21.8 tok/s
# with fp8 KV on 2026-09-03; if it fails now the probe proves nothing.
probe agentic-4k "PASS (fits one 8192-token prefill chunk)"

# The prediction: 30,301 tokens cannot be one chunk, so chunk 2 carries a
# prefix and the _ck kernel gets an fp8 right-hand side.
probe agentic-33k "CRASH (chunked, so prefix_lens is non-zero)"

echo
echo "== probe done $(date -Is)"
