#!/usr/bin/env bash
# Did the DSpark draft's shared experts actually load, or is decode running at
# half speed while looking perfectly healthy?
#
#   bash scripts/check-patch4.sh <container> [base-url]
#
# Patch 4 is the DSpark draft shared-expert loader fix. Without it the draft
# layer's shared-expert tensors load uninitialized, DSpark acceptance collapses
# from 60-80% to about 25%, and decode runs at roughly half speed. Nothing is
# logged at INFO. Output quality is unaffected, so the run looks fine and the
# number is wrong -- which is precisely the shape of failure this repo exists
# to refuse. BASELINE.md records the source's own verification as "6
# shared_experts matches expected; grep for 'Skipping' should return empty".
#
# Three checks, weakest to strongest:
#
#   1. "Skipping" in the loader output. The source's grep. Cheap, and a hit is
#      conclusive -- but silence proves nothing, which is the whole problem.
#   2. The expected shared-expert count, read from the checkpoint's own index
#      rather than hard-coded. On dsv4-flash-fp8 every layer carries exactly 6
#      (w1/w2/w3 x weight+scale) and num_nextn_predict_layers is 1, so the
#      draft layer is the last one and 6 is where the source's number comes
#      from. Read it per checkpoint; do not assume it.
#   3. The DSpark acceptance rate from /metrics. This is the only check that
#      observes the draft actually working. It needs traffic to have been
#      served first, so it is skipped on a cold server and reported as skipped.
#
# A note on the count. The four-node report says *twelve* tensors load
# uninitialized; this checkpoint's draft layer holds six. The two are not
# reconciled here and this script does not pretend otherwise -- it checks the
# six it can see and says so.
set -uo pipefail

C="${1:?usage: check-patch4.sh <container name> [base-url]}"
BASE="${2:-http://127.0.0.1:8888}"
rc=0

echo "== 1. loader 'Skipping' lines (source's own grep)"
skips=$(docker logs "$C" 2>&1 | grep -i "skipping" | grep -i "expert\|weight\|tensor" | head -10)
if [ -n "$skips" ]; then
  echo "!! FAIL: the loader reported skipped tensors:" >&2
  printf '   %s\n' "$skips" >&2
  rc=1
else
  echo "   none. (Necessary, not sufficient: the unpatched loader is silent.)"
fi

echo "== 2. shared-expert tensors expected in the draft layer"
model=$(docker inspect --format '{{join .Args " "}}' "$C" 2>/dev/null \
        | tr ' ' '\n' | grep -A1 -x -- '--model' | tail -1)
if [ -z "$model" ]; then
  echo "   could not read --model off the container; skipping."
else
  docker exec "$C" python3 - "$model" <<'PY'
import json, os, re, sys, collections
path = sys.argv[1]
try:
    cfg = json.load(open(os.path.join(path, "config.json")))
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
except OSError as e:
    print(f"   could not read checkpoint metadata: {e}"); sys.exit(0)

n_layers = cfg.get("num_hidden_layers")
n_draft  = cfg.get("num_nextn_predict_layers", 0)
if not n_layers or not n_draft:
    print(f"   no draft layer declared (num_nextn_predict_layers={n_draft}); nothing to check.")
    sys.exit(0)

names = [n for n in idx["weight_map"] if "shared_expert" in n]
per_layer = collections.Counter()
for n in names:
    m = re.search(r"layers\.(\d+)\.", n)
    if m:
        per_layer[int(m.group(1))] += 1

draft_layers = range(n_layers - n_draft, n_layers)
print(f"   {n_layers} layers, last {n_draft} is the DSpark draft")
for L in draft_layers:
    got = per_layer.get(L, 0)
    body = [c for l, c in per_layer.items() if l not in draft_layers]
    expect = max(set(body), key=body.count) if body else got
    verdict = "matches the body layers" if got == expect else f"DIFFERS from body ({expect})"
    print(f"   layer {L}: {got} shared-expert tensors -- {verdict}")
PY
fi

echo "== 3. DSpark acceptance rate (needs traffic already served)"
metrics=$(curl -sf --max-time 5 "$BASE/metrics" 2>/dev/null)
if [ -z "$metrics" ]; then
  echo "   /metrics unreachable at $BASE; SKIPPED."
  echo "   This is the only check that observes the draft working. Re-run it"
  echo "   after a campaign cell rather than treating the run as verified."
else
  spec=$(printf '%s' "$metrics" | grep -E "^vllm:spec_decode.*(accepted|draft|emitted).*[0-9]" | head -8)
  if [ -z "$spec" ]; then
    echo "   no vllm:spec_decode_* series present. Either speculation is off or"
    echo "   this build names them differently; SKIPPED rather than guessed."
  else
    printf '   %s\n' "$spec"
    acc=$(printf '%s' "$metrics" | awk '
      /^vllm:spec_decode_num_accepted_tokens_total/ {a=$2}
      /^vllm:spec_decode_num_draft_tokens_total/    {d=$2}
      END { if (d > 0) printf "%.1f", 100*a/d }')
    if [ -n "$acc" ]; then
      echo "   acceptance: ${acc}%"
      # 25% unpatched against 60-80% patched, per the four-node report.
      if awk "BEGIN{exit !($acc < 40)}"; then
        echo "!! FAIL: ${acc}% is in unpatched territory (~25%), not the 60-80%" >&2
        echo "   a working DSpark draft reaches. Do not believe decode numbers" >&2
        echo "   from this server until this is resolved." >&2
        rc=1
      fi
    fi
  fi
fi

[ "$rc" = 0 ] && echo "== patch-4 checks passed (see the caveats above)" \
             || echo "== patch-4 CHECK FAILED" >&2
exit "$rc"
