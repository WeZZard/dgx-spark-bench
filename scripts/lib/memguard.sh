# Shared memory guard for every container launch on a Spark.
#
# What the head's kernel ring buffer recorded on 2026-09-03 at 06:19:29, the
# moment the global OOM killer finally fired, about fifty minutes into the
# lockup:
#
#     Node 0 Normal   free 43,840 kB of 126,901,560 kB managed
#     active_anon 5,613,836 kB     inactive_anon 6,506,252 kB
#     active_file 0 kB             inactive_file 1,496 kB
#     shmem 2,671,356 kB           slab_unreclaimable 583,227 pages
#     SwapFree unchanged; swapents 0 for both server processes
#
# Add those up and roughly 17 GB is accounted for. The other 104 GB is held by
# the NVIDIA driver as unified memory, and it appears in no counter the OOM
# killer scores. Four things follow, and all four are measured:
#
#   * The page cache was already at zero (active_file 0 kB). That is why sshd
#     could not fault its own text back in and no banner ever arrived, while
#     ICMP -- handled in the kernel, touching no page cache -- still answered
#     in 0.2 ms.
#   * The kill was "constraint=CONSTRAINT_NONE ... global_oom", never a memcg
#     OOM, even though the container ran under --memory 116g. A cgroup cap does
#     not bound what the driver allocates on GB10.
#   * The killer's first victim was (sd-pam), a one-megabyte process, because
#     the largest RSS in the entire table was about 5 GB. It needed two more
#     passes and 37 more seconds to reach the actual server.
#   * The node never swapped. An earlier note in docs/baselines.md blamed swap;
#     that was wrong.
#
# MemAvailable does track the loss -- 43 MB free -- so wait_for_free_memory
# below is the check that works. The computed cap is kept because it still
# bounds the container's host-side memory, but it cannot stop this failure and
# must not be trusted to.

HOST_RESERVE_GIB="${HOST_RESERVE_GIB:-12}"

mem_total_gib() { awk '/^MemTotal/ {print int($2/1048576)}' /proc/meminfo; }
mem_avail_gib() { awk '/^MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }

# Print the number to pass to both --memory and --memory-swap.
container_cap_gib() {
  local total cap
  total=$(mem_total_gib)
  cap=$(( total - HOST_RESERVE_GIB ))
  if [ "$cap" -lt 32 ]; then
    echo "host reserve of ${HOST_RESERVE_GIB}GiB leaves only ${cap}GiB" >&2
    return 1
  fi
  echo "$cap"
}

# A container whose process is blocked inside the driver cannot be killed.
# `docker rm -f` returned at 05:29:59 on 2026-09-03 and the previous server's
# scheduler thread was still in state D at 06:19 holding its memory:
# "INFO: task sglang::schedul:301733 blocked for more than 1228 seconds",
# tgid 301520. SIGKILL cannot reap a task blocked in the kernel. Say so,
# because only a reboot frees that memory and a silent wait looks like progress.
report_stuck_servers() {
  local stuck
  stuck=$(ps -eo pid,stat,comm --no-headers 2>/dev/null |
          awk '$2 ~ /D/ && $3 ~ /sglang|llama|python/ {print "  pid "$1" "$3}')
  [ -n "$stuck" ] || return 0
  echo "these processes are blocked in the kernel and cannot be killed:" >&2
  echo "$stuck" >&2
  echo "their memory does not come back until this node reboots." >&2
  return 1
}

# Wait a bounded time for a previous container's memory to come back, then
# refuse. Starting a 70-100 GB server on a node that still owes its memory is
# what locked the head up; refusing is the far cheaper failure.
wait_for_free_memory() {
  local need="$1" free=0 i
  for i in $(seq 1 20); do
    free=$(mem_avail_gib)
    if [ "${free:-0}" -ge "$need" ]; then
      echo "memory ok: ${free}GiB available, need ${need}GiB"
      return 0
    fi
    echo "waiting for memory: ${free}GiB available, need ${need}GiB"
    sleep 15
  done
  echo "refusing to launch: only ${free}GiB available, need ${need}GiB." >&2
  report_stuck_servers ||
    echo "no stuck process found; something else holds this node's memory." >&2
  return 4
}
