# Two-node bring-up, and the one mistake it exists to prevent.
#
# A two-node server is ONE server. If the ranks disagree about how the model is
# sharded, the launch is not a worse measurement -- it is not a measurement.
# On 2026-09-03 rank 1 was started with
#
#     ssh "$WORKER" "cd $REPO && bash $launcher 1"
#
# and rank 0 with a plain `bash "$launcher" 0`. Every tuning knob in this
# project travels as an environment variable, and ssh carries no environment,
# so a knob set in front of the call reached rank 0 and stopped at the ssh
# boundary. Rank 0 came up with expert parallelism, loaded further than before,
# and looked like the hypothesis being confirmed; rank 1 was still on plain
# tensor parallelism and died. Rank 0 then held 109 GiB waiting for a rank that
# was gone. Nothing was learned, and the failure looked exactly like the idea
# being wrong.
#
# So the forwarded set is derived from the variable PREFIXES the launchers use,
# never from a list of names -- a knob added later travels without anyone
# having to remember this function -- and the chosen environment is printed on
# both sides so the two can be compared in the logs.

: "${KNOB_PREFIXES:=SG_ QWEN_ GLM_ DS_ SPARK_ SGLANG_ VLLM_}"

# Print `A=1 B=2` for every exported variable whose name starts with a prefix.
collect_knobs() {
  local out=() name prefix
  while IFS='=' read -r name _; do
    for prefix in $KNOB_PREFIXES; do
      case "$name" in
        "$prefix"*) out+=("$name=$(printf '%q' "${!name}")"); break ;;
      esac
    done
  done < <(env)
  printf '%s\n' "${out[*]}"
}

# bring_up <worker-ssh-host> <repo-path> <launcher-relative-path>
# Starts rank 1 on the worker with the same knobs as rank 0, then rank 0 here.
bring_up() {
  local worker="$1" repo="$2" launcher="$3"
  local knobs; knobs="$(collect_knobs)"

  echo "######## rank 1 environment: ${knobs:-<none>}"
  echo "######## rank 0 environment: ${knobs:-<none>}"

  # Rank 1 first: it waits for rank 0, not the other way round, and a worker
  # that refuses to launch (memguard exit 4) should be known before rank 0
  # allocates 70-100 GB here.
  ssh -o ConnectTimeout=30 "$worker" \
    "cd '$repo' && env $knobs bash '$launcher' 1" &
  local worker_pid=$!

  sleep 5
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    wait "$worker_pid" || {
      echo "rank 1 refused to launch; not starting rank 0" >&2
      return 4
    }
  fi

  env $knobs bash "$launcher" 0
  local rc=$?
  wait "$worker_pid" 2>/dev/null || true
  return $rc
}

# Read a scalar out of inventory.local.yml without a YAML parser.
inv() {
  local path="${INVENTORY:-$(dirname "${BASH_SOURCE[0]}")/../../inventory.local.yml}"
  local section="$1" key="$2"
  awk -v s="$section" -v k="$key" '
    /^[a-z_]+:/ { in_s = ($0 ~ "^" s ":") }
    in_s && $1 == k":" { print $2; exit }
  ' "$path"
}
