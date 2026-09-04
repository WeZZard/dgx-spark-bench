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
