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

echo "== transport NCCL chose"
log=$(docker logs "$C" 2>&1 | grep -iE "NCCL INFO NET/|Using network|NET/IB|NET/Socket|NCCL WARN" | tail -20)
if [ -z "$log" ]; then
  echo "   nothing logged. NCCL_DEBUG is probably WARN; relaunch with NCCL_DEBUG=INFO"
  echo "   to make this answerable. Do not assume RoCE from the presence of devices."
  exit 2
fi
printf '   %s\n' "$log"

if printf '%s' "$log" | grep -qi "NET/IB"; then
  echo "== RoCE in use"
  exit 0
fi
if printf '%s' "$log" | grep -qi "NET/Socket"; then
  echo "== SOCKETS in use -- the 200 GbE link is carrying TCP, not RDMA." >&2
  echo "   Decode on cross-node TP is latency-bound; this is worth roughly 2x." >&2
  exit 1
fi
echo "== could not tell from the log"
exit 2
