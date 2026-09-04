#!/usr/bin/env bash
# Did the DSpark draft's shared experts actually load, or is decode running at
# half speed while looking perfectly healthy?
#
#   bash scripts/check-patch4.sh <container> [base-url]
#   REQUIRE_ACCEPTANCE=1 bash scripts/check-patch4.sh <container> [base-url]
#
# Patch 4 is the DSpark draft shared-expert loader fix. Without it the draft
# layer's shared-expert tensors load uninitialized, DSpark acceptance collapses
# from 60-80% to about 25%, and decode runs at roughly half speed. Nothing is
# logged at INFO and output quality is unaffected, so the run looks fine and
# the number is wrong. BASELINE.md records the source's verification as "6
# shared_experts matches expected; grep for 'Skipping' should return empty".
#
# Three checks, weakest first. The script reports what it actually verified:
# an earlier version printed "patch-4 checks passed" while checks 2 and 3 had
# both silently done nothing -- check 2 because `docker exec` without -i
# discards stdin so the heredoc never reached python, and check 3 because it
# ran before any traffic existed. A false pass on the guard against a silent
# half-speed failure is worse than no guard at all.
#
#   1. "Skipping" in the loader output. The source's grep. A hit is
#      conclusive; silence is not, because the unpatched loader is silent.
#   2. The expected shared-expert count, read from the checkpoint's own index
#      rather than hard-coded.
#   3. The DSpark acceptance rate from /metrics. The only check that observes
#      the draft working, and it needs traffic. Set REQUIRE_ACCEPTANCE=1 when
#      calling it after a campaign, where "no counters" means speculation is
#      not running rather than "too early to tell".
#
# The four-node report says twelve tensors load uninitialized; this
# checkpoint's draft layer holds six. The two are not reconciled here, and
# this script checks the six it can see rather than pretending otherwise.
set -uo pipefail

C="${1:?usage: check-patch4.sh <container name> [base-url]}"
BASE="${2:-http://127.0.0.1:8888}"
REQUIRE_ACCEPTANCE="${REQUIRE_ACCEPTANCE:-0}"
rc=0
tensors_checked=0
acceptance_checked=0

echo "== 1. loader 'Skipping' lines (source's own grep)"
skips=$(docker logs "$C" 2>&1 | grep -i "skipping" | grep -i "expert\|weight\|tensor" | head -10)
if [ -n "$skips" ]; then
  echo "!! FAIL: the loader reported skipped tensors:" >&2
  printf '   %s\n' "$skips" >&2
  rc=1
else
  echo "   none. (Necessary, not sufficient: the unpatched loader is silent.)"
fi

echo "== 2. shared-expert tensors in the draft layer, read from the checkpoint"
model=$(docker inspect --format '{{join .Args " "}}' "$C" 2>/dev/null \
        | tr ' ' '\n' | grep -A1 -x -- '--model' | tail -1)
if [ -z "$model" ]; then
  echo "!! could not read --model off the container" >&2
  rc=1
else
  # -i is required: without it docker exec discards stdin and the heredoc
  # below never reaches python, which then prints nothing and exits 0.
  out=$(docker exec -i "$C" python3 - "$model" <<'INNER'
import json, os, re, sys, collections
path = sys.argv[1]
try:
    cfg = json.load(open(os.path.join(path, "config.json")))
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
except OSError as e:
    print(f"   could not read checkpoint metadata: {e}")
    sys.exit(2)

n_layers = cfg.get("num_hidden_layers")
n_draft = cfg.get("num_nextn_predict_layers", 0)
if not n_layers or not n_draft:
    print(f"   no draft layer declared (num_nextn_predict_layers={n_draft})")
    sys.exit(2)

per_layer = collections.Counter()
for n in idx["weight_map"]:
    if "shared_expert" in n:
        m = re.search(r"layers\.(\d+)\.", n)
        if m:
            per_layer[int(m.group(1))] += 1

draft_layers = list(range(n_layers - n_draft, n_layers))
body = [c for l, c in per_layer.items() if l not in draft_layers]
expect = max(set(body), key=body.count) if body else None
print(f"   {n_layers} layers, last {n_draft} is the DSpark draft; "
      f"body layers carry {expect}")
bad = False
for L in draft_layers:
    got = per_layer.get(L, 0)
    ok = (got == expect)
    print(f"   layer {L}: {got} shared-expert tensors -- "
          f"{'matches the body layers' if ok else f'DIFFERS from body ({expect})'}")
    bad = bad or not ok
sys.exit(3 if bad else 0)
INNER
)
  st=$?
  [ -n "$out" ] && printf '%s\n' "$out"
  case "$st" in
    0) tensors_checked=1 ;;
    3) echo "!! FAIL: the draft layer is missing shared-expert tensors" >&2; rc=1 ;;
    2) echo "   (checkpoint metadata unavailable; this check could not run)" ;;
    *) echo "!! the tensor check produced no result -- it verified nothing" >&2; rc=1 ;;
  esac
fi

echo "== 3. DSpark acceptance rate (needs traffic already served)"
metrics=$(curl -sf --max-time 5 "$BASE/metrics" 2>/dev/null)
acc=""
if [ -z "$metrics" ]; then
  echo "   /metrics unreachable at $BASE"
else
  # *_created are Prometheus creation timestamps, not counts -- reading one as
  # a count gives ~1.79e9, a Unix time, and a nonsense acceptance rate.
  acc=$(printf '%s' "$metrics" | awk '
    /^vllm:spec_decode_num_accepted_tokens_total/ {a=$2}
    /^vllm:spec_decode_num_draft_tokens_total/    {d=$2}
    END { if (d > 0) printf "%.1f", 100*a/d }')
  drafted=$(printf '%s' "$metrics" | awk '/^vllm:spec_decode_num_draft_tokens_total/ {print $2}')
  echo "   draft tokens so far: ${drafted:-none}"
fi

if [ -n "$acc" ]; then
  acceptance_checked=1
  echo "   acceptance: ${acc}%"
  # ~25% unpatched against 60-80% patched, per the four-node report.
  if awk "BEGIN{exit !($acc < 40)}"; then
    echo "!! FAIL: ${acc}% is unpatched territory (~25%), not the 60-80% a" >&2
    echo "   working DSpark draft reaches. Do not believe decode numbers." >&2
    rc=1
  fi
elif [ "$REQUIRE_ACCEPTANCE" = 1 ]; then
  echo "!! FAIL: traffic has been served and there is still no DSpark draft" >&2
  echo "   token count. Speculation is not running, so the configuration is" >&2
  echo "   not the one being claimed." >&2
  rc=1
else
  echo "   no draft tokens yet; deferred to the post-campaign re-check."
fi

if [ "$rc" != 0 ]; then
  echo "== patch-4 CHECK FAILED" >&2
  exit 1
fi
echo "== patch-4: verified tensors=$tensors_checked acceptance=$acceptance_checked"
if [ "$tensors_checked" = 0 ] && [ "$acceptance_checked" = 0 ]; then
  echo "== INCONCLUSIVE: nothing was actually verified. Not a pass." >&2
  exit 2
fi
echo "== patch-4 checks passed (see the caveats in the header)"
