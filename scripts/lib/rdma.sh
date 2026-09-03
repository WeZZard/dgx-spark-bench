# RDMA passthrough for inference containers, and why decode was slow without it.
#
# These nodes are joined back-to-back at 200 GbE and carry four RoCE devices
# (`rocep1s0f0` and friends). docs/baselines.md measures 109 Gb/s with
# `ib_write_bw` on the host. None of that reaches a container started with
# `--gpus all --network host` alone.
#
# Measured on 2026-09-04, inside a running two-node SGLang server:
#
#     host:      /dev/infiniband/{uverbs0..3, rdma_cm}   present
#     container: /dev/infiniband                          ABSENT
#     container: /sys/class/infiniband                    present (sysfs is mounted)
#
# The sysfs entries being visible is the trap. NCCL can enumerate the HCAs and
# see them listed, but it cannot open the verbs character devices, so it falls
# back to TCP sockets over the same wire and says nothing about it at default
# log level. The link is fast either way; what changes is latency, and decode on
# a cross-node tensor-parallel server is latency-bound -- every layer needs an
# all-reduce, so a fixed per-operation cost is multiplied by the layer count on
# every single token.
#
# Two things are needed besides the devices. IPC_LOCK, because RDMA registers
# pinned memory regions and registration fails without the capability. And
# NCCL_IB_HCA, so NCCL binds to the port pair that is actually cross-connected
# rather than probing all four -- the published DeepSeek launcher sets
# `NCCL_IB_HCA=rocep1s0f0` for the same reason.
#
# Set NCCL_DEBUG=INFO to see which transport was chosen. At default level NCCL
# reports neither the fallback nor the choice, which is why this went unnoticed.

RDMA_HCA="${RDMA_HCA:-rocep1s0f0}"

# Emit the docker arguments needed for RoCE, or nothing if the host has no
# RDMA devices. Printing nothing is deliberate: a node without RDMA should run
# over TCP rather than fail to launch, and the tuple records which it was.
rdma_docker_args() {
  [ -d /dev/infiniband ] || return 0
  local args=()
  local d
  for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
    [ -e "$d" ] && args+=( --device "$d" )
  done
  [ ${#args[@]} -eq 0 ] && return 0
  args+=( --cap-add IPC_LOCK --ulimit memlock=-1:-1 )
  printf '%s\n' "${args[*]}"
}

# Say plainly whether this launch will have RDMA, so the log records it.
rdma_report() {
  if [ -d /dev/infiniband ] && ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
    echo "RDMA: passing through $(ls /dev/infiniband/uverbs* | wc -l) verbs devices, HCA=$RDMA_HCA"
  else
    echo "RDMA: no verbs devices on this host -- NCCL will use TCP sockets"
  fi
}
