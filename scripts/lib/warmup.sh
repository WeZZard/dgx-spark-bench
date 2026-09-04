#!/usr/bin/env bash
# warmup_engine <base-url> <served-model-name>
#
# Discarded traffic, sent before any cell is measured, for one reason: these
# engines compile kernels at serving time, and the compile lands in whichever
# cell touches a shape first. That is not a property of the configuration, so
# it must not be inside a recorded number.
#
# It has to include a LONG prompt, not just short ones. Measured on GLM,
# 2026-09-04, with short-only warmup:
#
#   agentic-4k  @ 1 user   3.5 tok/s served   (512 tokens / 148.2 s wall)
#   agentic-33k @ 1 user   5.9 tok/s served   (512 tokens /  87.1 s wall)
#
# The 33k prompt finished in 60% of the wall clock of the 4k one. A longer
# prompt is not faster to prefill; the 4k cell was simply the first to touch
# the prefill path and paid to compile it. Both cells were discarded and
# re-measured. Short warmups alone would not have caught this, because the
# short path was already compiled by then.
set -uo pipefail

warmup_engine() {
  local base="$1" model="$2"
  echo "== warmup (discarded): compiling kernels at serving time"
  local i
  for i in 1 2 3; do
    curl -s --max-time 300 "$base/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Write one sentence about caching.\"}],\"max_tokens\":64,\"temperature\":0}" \
      -o /dev/null -w "   warmup $i: http=%{http_code} %{time_total}s\n" || true
  done

  # Concurrency, not just prompt length. Repeating the GLM campaign three
  # times against one unchanged server showed the code ladder was still
  # settling after a single-stream warmup:
  #
  #   users   run 1    run 2    run 3      (tok/s served)
  #      1      9.9     10.0     10.6
  #      2     16.4     17.3     16.9
  #      4     23.1     28.0     29.3
  #      8     31.5     44.2     44.9
  #
  # Runs 2 and 3 agree within a few percent; run 1 does not, and the gap grows
  # with concurrency -- 4% at one user, 42% at eight. Those prompts are 47
  # tokens, so prefill and the radix cache explain none of it (the engine
  # reported #cached-token: 0 on 73 of 75 prefill batches). Whatever is
  # one-time here scales with batch, so the warmup has to reach the widest
  # batch a cell will use, or the first campaign measures it.
  local users="${WARMUP_USERS:-8}"
  echo "   warmup burst: $users concurrent"
  local pids=() i
  for ((i = 0; i < users; i++)); do
    curl -s --max-time 600 "$base/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Count to twenty, one number per line.\"}],\"max_tokens\":128,\"temperature\":0}" \
      -o /dev/null &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null || true
  echo "   warmup burst: done"

  # Long prompt, so the prefill path is compiled before it is measured.
  python3 - "$base" "$model" <<'PY' || true
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
for label, words in (("long", 6000), ("longer", 24000)):
    body = {"model": model,
            "messages": [{"role": "user", "content": "ok " * words + "\nReply with one word."}],
            "max_tokens": 16, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        urllib.request.urlopen(req, timeout=900).read()
        print(f"   warmup {label}-prompt ({words} words): ok in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"   warmup {label}-prompt ({words} words): {e}")
PY
}
