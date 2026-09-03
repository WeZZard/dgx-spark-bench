#!/usr/bin/env bash
# Qwen3.8-Flash-Next, NVFP4, SGLang, TP2 across both Sparks.
#
#   bash scripts/launch-qwen-sglang.sh <rank>     # 0 = head, 1 = worker
#
# Reproduces the published configuration in BASELINE.md. Every knob is an
# environment variable with the recipe's value as its default, so a sweep
# changes one name and `bring_up` forwards it to both ranks unchanged.
#
# Three things here are not tuning and must not be "cleaned up".
#
# 1. `sync; echo 3 > /proc/sys/vm/drop_caches` before launch. The source recipe
#    calls it mandatory: on unified memory the page cache and the model share
#    one pool, and a warm cache starves the load. It is also the guard's
#    precondition -- MemAvailable is only meaningful once the cache is dropped.
#
# 2. `SGLANG_HOST_IP` set per rank. Without it multi-node shm_broadcast hangs
#    at startup; this is one of the four GB10 fixes the GLM recipe reports and
#    it costs nothing to apply to every SGLang launch.
#
# 3. `--ple-offload-embedding`. Not optional. The checkpoint carries 47.68 GiB
#    of per-layer n-gram embedding tables (`scripts/audit-checkpoint.py`, and
#    `docs/checkpoints.md`), 38% of its bytes. Leaving them resident is what
#    makes a 600,000-token pool impossible.
#
# The QSA guard is enforced below rather than documented: this image ships the
# SM121 QSA kernel and NOT the pr36845 packed-varlen successor, and the source
# reports silent output corruption past about 120K context. Silent is the
# problem -- nothing in the log says so. See docs/qwen-qsa.md.
set -euo pipefail

RANK="${1:?usage: launch-qwen-sglang.sh <rank 0|1>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$here/scripts/lib/memguard.sh"
source "$here/scripts/lib/rdma.sh"

QWEN_IMAGE="${QWEN_IMAGE:-sglang-qwen38fn:sm121-ours}"
QWEN_NAME="${QWEN_NAME:-sglang_qwen38fn}"
QWEN_PORT="${QWEN_PORT:-30000}"
QWEN_DIST_ADDR="${QWEN_DIST_ADDR:-10.10.10.1:20000}"
QWEN_HOST_IP="${QWEN_HOST_IP:-$( [ "$RANK" = 0 ] && echo 10.10.10.1 || echo 10.10.10.2 )}"
QWEN_MODEL="${QWEN_MODEL:-$( [ "$RANK" = 0 ] \
    && echo /home/station/models/hf/qwen38-flash-next-nvfp4 \
    || echo /mnt/models/hf/qwen38-flash-next-nvfp4 )}"

# --- the published recipe, as defaults -------------------------------------
QWEN_TOTAL_TOKENS="${QWEN_TOTAL_TOKENS:-600000}"     # pinned pool
QWEN_MEM_FRACTION="${QWEN_MEM_FRACTION:-0.80}"
QWEN_CUDA_GRAPH_BS="${QWEN_CUDA_GRAPH_BS:-8}"
QWEN_MAMBA_CACHE="${QWEN_MAMBA_CACHE:-97}"
# The published recipe passes --context-length 262144. The SAME source warns
# that the SM121 QSA kernel corrupts output past about 120K. Those two
# statements cannot both be safe, and this image has QSA and not the pr36845
# successor, so the default here is the largest length that stays inside the
# stated safe zone. Reproducing the published 262144 means measuring corruption,
# not throughput; QWEN_ALLOW_LONG_QSA=1 exists for that experiment.
QWEN_CTX="${QWEN_CTX:-120000}"
QWEN_SAMPLING="${QWEN_SAMPLING:-pytorch}"
QWEN_RADIX="${QWEN_RADIX:-off}"      # recipe disables it; re-decided here, see below
QWEN_KV_DTYPE="${QWEN_KV_DTYPE:-}"   # empty => engine default, recorded as such
# The published recipe does not mention this flag, and on this build the run
# dies without it, at the first forward pass rather than at load:
#
#   NotImplementedError: Unsupported moe_runner_backend for NVFP4 MoE:
#   MoeRunnerBackend.FLASHINFER_TRTLLM. Use --moe-runner-backend
#   flashinfer_cutlass instead.
#
# The auto-selection picks a TRTLLM MoE runner that has no NVFP4 path. The
# warning is there twenty minutes earlier, at load time, and says nothing about
# being fatal: "FlashInfer TRTLLM MoE deferred finalize is disabled
# (moe_runner_backend=auto, quant_method=ModelOptNvFp4FusedMoEMethod)".
QWEN_MOE_BACKEND="${QWEN_MOE_BACKEND:-flashinfer_cutlass}"
# MTP. The published recipe reports 69.7 tok/s peak with it and "~20 without",
# so this is the single largest lever on this model and not a detail. The
# checkpoint carries the draft head already -- `mtp.fc_embedding.weight`,
# `mtp.pre_fc_norm_embedding.weight` and friends are in the safetensors -- and
# qwen4_exp.py takes an `is_nextn` path and loads tensors whose names contain
# "mtp". SGLang reaches it through --speculative-algorithm NEXTN, with no
# separate draft model to point at.
#
# Default off for the first pass on purpose: the no-MTP number is the control,
# and BASELINE.md quotes a "~20 without MTP" figure to check it against. Turn it
# on with QWEN_SPEC=NEXTN once the control is recorded.
QWEN_SPEC="${QWEN_SPEC:-off}"
QWEN_SPEC_STEPS="${QWEN_SPEC_STEPS:-1}"
QWEN_SPEC_TOPK="${QWEN_SPEC_TOPK:-1}"
QWEN_SPEC_DRAFT_TOKENS="${QWEN_SPEC_DRAFT_TOKENS:-2}"
QWEN_EXTRA="${QWEN_EXTRA:-}"

# --- QSA long-context guard -------------------------------------------------
# 120K is the source's stated onset of silent corruption. Refusing is cheap;
# a corrupt long-context measurement that reads as a valid one is not.
QSA_SAFE_CTX="${QSA_SAFE_CTX:-120000}"
if [ "$QWEN_CTX" -gt "$QSA_SAFE_CTX" ] && [ "${QWEN_ALLOW_LONG_QSA:-0}" != "1" ]; then
  cat >&2 <<MSG
refusing to launch: --context-length $QWEN_CTX exceeds the QSA safe limit of
$QSA_SAFE_CTX. This image ships the SM121 QSA kernel and not the pr36845
packed-varlen successor, and QSA is reported to corrupt output past roughly
120K context WITHOUT logging anything.

If you are deliberately measuring where that corruption starts, set
QWEN_ALLOW_LONG_QSA=1 and use an agentic workload -- its needle check is what
turns the warning into a measurement.
MSG
  exit 5
fi

echo "== rank $RANK: tearing down any previous $QWEN_NAME"
docker rm -f "$QWEN_NAME" >/dev/null 2>&1 || true

echo "== rank $RANK: sync + drop_caches (mandatory for this model)"
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

CAP_GIB="$(container_cap_gib)"
NEED_GIB="${QWEN_NEED_GIB:-90}"
echo "== rank $RANK: container cap ${CAP_GIB}GiB, need ${NEED_GIB}GiB free"
wait_for_free_memory "$NEED_GIB"     # exits 4 rather than wedging the node

args=(
  --model-path "$QWEN_MODEL"
  --served-model-name qwen38-flash-next
  --tp-size 2 --nnodes 2 --node-rank "$RANK"
  --dist-init-addr "$QWEN_DIST_ADDR"
  --host 0.0.0.0 --port "$QWEN_PORT"
  --context-length "$QWEN_CTX"
  --max-total-tokens "$QWEN_TOTAL_TOKENS"
  --mem-fraction-static "$QWEN_MEM_FRACTION"
  --cuda-graph-max-bs "$QWEN_CUDA_GRAPH_BS"
  --disable-cuda-graph-padding
  --ple-offload-embedding
  --max-mamba-cache-size "$QWEN_MAMBA_CACHE"
  --sampling-backend "$QWEN_SAMPLING"
  --moe-runner-backend "$QWEN_MOE_BACKEND"
)
# BASELINE.md flags this one for re-decision rather than copying: the published
# recipe passes --disable-radix-cache because their workload is not agentic,
# and a prefix cache is worth roughly 3x on multi-turn agent traffic. So it is
# a knob here, defaulting to the recipe so the reproduction is faithful.
[ "$QWEN_RADIX" = "off" ] && args+=( --disable-radix-cache )
if [ "$QWEN_SPEC" != "off" ]; then
  args+=(
    --speculative-algorithm "$QWEN_SPEC"
    --speculative-num-steps "$QWEN_SPEC_STEPS"
    --speculative-eagle-topk "$QWEN_SPEC_TOPK"
    --speculative-num-draft-tokens "$QWEN_SPEC_DRAFT_TOKENS"
  )
fi
[ -n "$QWEN_KV_DTYPE" ] && args+=( --kv-cache-dtype "$QWEN_KV_DTYPE" )
# shellcheck disable=SC2206
[ -n "$QWEN_EXTRA" ] && args+=( $QWEN_EXTRA )

# The head exports /home/station/models; the worker mounts it at /mnt/models
# under x-systemd.automount, so the path only materialises once something looks
# at it. Bind-mounting an unvisited automount point hands the container an
# empty directory and the load fails on a missing shard, which reads as a bad
# checkpoint rather than a missing mount. Touch it first, and only mount what
# this node actually has.
mounts=()
[ -d /home/station/models ] && mounts+=( -v /home/station/models:/home/station/models:ro )
if ls /mnt/models >/dev/null 2>&1; then
  mounts+=( -v /mnt/models:/mnt/models:ro )
fi
case "$QWEN_MODEL" in
  /mnt/models/*)
    [ -d "$QWEN_MODEL" ] || { echo "rank $RANK: $QWEN_MODEL not readable; is the NFS automount up?" >&2; exit 6; } ;;
  *)
    [ -d "$QWEN_MODEL" ] || { echo "rank $RANK: $QWEN_MODEL does not exist" >&2; exit 6; } ;;
esac

echo "== rank $RANK: launching $QWEN_IMAGE"
printf '   %s\n' "${args[*]}"

rdma_report
read -r -a RDMA_ARGS <<< "$(rdma_docker_args)"

exec docker run --rm --name "$QWEN_NAME" \
  --gpus all --ipc=host --network host \
  --memory "${CAP_GIB}g" --memory-swap "${CAP_GIB}g" \
  "${RDMA_ARGS[@]}" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e SGLANG_HOST_IP="$QWEN_HOST_IP" \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  "${mounts[@]}" \
  "$QWEN_IMAGE" \
  python3 -m sglang.launch_server "${args[@]}"
