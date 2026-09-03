#!/usr/bin/env bash
# wait_ready <container> <base-url> <log-file>
#
# Waits for a serving container to answer /health, and gives up on two
# different grounds, because they are two different failures:
#
#   STALLED  -- nothing has moved for STALL_S. This is the 2026-08-31 zombie:
#               a vLLM rank died compiling a kernel, the container stayed
#               *running* holding 90 GB with the GPU at 0%, and every
#               "is it running?" check said fine for 8.5 hours. Caught here in
#               STALL_S rather than at the wall-clock deadline.
#
#   DEADLINE -- still making progress, but has taken longer than any load
#               should. A backstop, not the primary signal.
#
# The distinction matters because a flat deadline cannot tell a slow load from
# a stuck one, and the two models here differ by 60 GiB of weights: Qwen
# (127 GiB) reached /health in ~1050-1115 s, so a 1800 s flat deadline that is
# generous for Qwen is tight for GLM (189 GiB) and says nothing useful about
# either. Progress is the signal; time is only the backstop.
#
# Progress means either of:
#   - the launcher log grew, or
#   - MemAvailable moved by more than NOISE_MIB (weight loading and KV
#     allocation both consume it steadily; a stuck process consumes nothing).
#
# Neither is gameable by a container that is merely *running*.

wait_ready() {
  local name="$1" base="$2" log="$3"
  local deadline_s="${READY_DEADLINE_S:-3600}"
  local stall_s="${STALL_S:-600}"
  local grace_s="${STARTUP_GRACE_S:-180}"
  local noise_mib="${NOISE_MIB:-64}"

  local started last_change last_size last_mem now size mem st
  started=$(date +%s); last_change=$started
  last_size=0; last_mem=0

  echo "== waiting for /health: deadline ${deadline_s}s, stall ${stall_s}s"
  while :; do
    if curl -sf --max-time 5 "$base/health" >/dev/null 2>&1; then
      echo "== ready after $(( $(date +%s) - started ))s"; return 0
    fi

    now=$(date +%s)

    # The container does not exist for the first few seconds: bring_up has to
    # reach the worker over ssh, drop the page cache and wait on MemAvailable
    # before it runs `docker run`. Treating "absent" as fatal from the first
    # iteration is a race, and it lost once -- the wrapper exited while the
    # detached bring-up carried on loading a 126 GiB model that nothing was
    # then going to measure.
    st=$(timeout 15 docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo absent)
    if [ "$st" != "running" ] && [ $(( now - started )) -gt "$grace_s" ]; then
      echo "== container is '$st' after ${grace_s}s grace; last log lines:" >&2
      tr '\r' '\n' < "$log" | grep -vE '^\s*$|Multi-thread' | tail -20 >&2
      return 1
    fi

    size=$(wc -c < "$log" 2>/dev/null || echo 0)
    mem=$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)
    if [ "$size" -ne "$last_size" ] || \
       [ "$(( mem > last_mem ? mem - last_mem : last_mem - mem ))" -gt "$noise_mib" ]; then
      last_change=$now; last_size=$size; last_mem=$mem
    fi

    if [ $(( now - last_change )) -ge "$stall_s" ] && [ $(( now - started )) -gt "$grace_s" ]; then
      echo "== STALLED: no log growth and no memory movement for ${stall_s}s." >&2
      echo "   'Still running' is not progress -- see docs/baselines.md," >&2
      echo "   'Incident 2026-08-31: the nine-hour zombie container'." >&2
      tr '\r' '\n' < "$log" | grep -vE '^\s*$|Multi-thread' | tail -20 >&2
      return 1
    fi

    if [ "$now" -ge $(( started + deadline_s )) ]; then
      echo "== DEADLINE: not ready in ${deadline_s}s, though it was still moving." >&2
      echo "   The container is left up: it may simply be a slower load than" >&2
      echo "   this deadline allows. Poll $base/health and run the campaign" >&2
      echo "   directly, or re-run with a larger READY_DEADLINE_S." >&2
      return 1
    fi
    sleep 10
  done
}
