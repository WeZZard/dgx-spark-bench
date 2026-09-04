#!/usr/bin/env bash
# Did NCCL actually use RoCE, or did it fall back to sockets?
#
#   bash scripts/check-rdma.sh <container>
#
# Ask, because the failure is silent. On 2026-09-04 a two-node server ran its
# entire tensor-parallel traffic over TCP: the container had no /dev/infiniband
# but did have /sys/class/infiniband, so NCCL listed four RoCE HCAs it could not
# open and said nothing at default log level. Nothing in the server log, no
# warning, no degraded-mode message -- just decode about twice slower than it
# should have been.
#
# Run this after every bring-up. NCCL_DEBUG=INFO makes it print the transport
# it selected per channel; the launchers default to WARN, so pass
# NCCL_DEBUG=INFO to the bring-up when you want this check to have something
# to read.
set -uo pipefail

C="${1:?usage: check-rdma.sh <container name>}"

echo "== devices visible inside $C"
docker exec "$C" bash -c 'ls /dev/infiniband 2>/dev/null | tr "\n" " " || echo "NONE"' \
  | sed 's/^/   verbs devices: /'
docker exec "$C" bash -c 'ls /sys/class/infiniband 2>/dev/null | tr "\n" " "' \
  | sed 's/^/   sysfs HCAs:    /'

# Check 1: the HCA's own counters. Log-independent, and therefore the one
# that works on a normal sweep -- the log check below needs NCCL_DEBUG=INFO,
# the launchers default to WARN, and every sweep so far has come back
# "inconclusive" because of it.
#
# rx_write_requests under hw_counters/ is an RDMA verb: TCP cannot increment
# it, whatever else shares the physical port. Do NOT substitute
# ports/1/counters/port_xmit_data -- on mlx5 those are physical-port counters
# that can include ordinary Ethernet, so a growing byte count there does not
# separate RoCE from a TCP fallback, which is the entire question.
#
# It needs traffic in flight, so it answers during a campaign and not before
# one.
rdma_counter_check() {
  local hca="${RDMA_HCA:-rocep1s0f0}"
  local f="/sys/class/infiniband/$hca/ports/1/hw_counters/rx_write_requests"
  [ -r "$f" ] || { echo "   no hw_counters for $hca"; return 2; }
  local a b
  a=$(cat "$f"); sleep "${RDMA_SAMPLE_S:-20}"; b=$(cat "$f")
  echo "   rx_write_requests: $a -> $b (delta $((b - a)) in ${RDMA_SAMPLE_S:-20}s)"
  [ "$((b - a))" -gt 0 ]
}

echo "== RDMA verbs actually issued (needs traffic in flight)"
if rdma_counter_check; then
  echo "== RoCE confirmed by the HCA's own counters"
  exit 0
fi

# Check 2: what NCCL said it chose. Only useful at NCCL_DEBUG=INFO.
echo "== transport NCCL chose"
log=$(docker logs "$C" 2>&1 | grep -iE "NCCL INFO NET/|Using network|NET/IB|NET/Socket|NCCL WARN" | tail -20)
if [ -n "$log" ]; then
  printf '   %s\n' "$log"
  if printf '%s' "$log" | grep -qi "NET/IB"; then
    echo "== RoCE in use"; exit 0
  fi
  if printf '%s' "$log" | grep -qi "NET/Socket"; then
    echo "== SOCKETS in use -- the 200 GbE link is carrying TCP, not RDMA." >&2
    echo "   Decode on cross-node TP is latency-bound; this is worth roughly 2x." >&2
    exit 1
  fi
else
  echo "   nothing logged; NCCL_DEBUG is probably WARN."
fi

echo "== could not tell: no RDMA verbs seen and no transport in the log."
echo "   If this ran before any traffic, the counter check cannot answer;"
echo "   re-run it during a campaign. Do not assume RoCE from the presence"
echo "   of devices -- that assumption cost 1.74x once already."
exit 2
