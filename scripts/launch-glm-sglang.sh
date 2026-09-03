#!/usr/bin/env bash
# GLM-5.3-Flash, LibertAIDAI NVFP4, SGLang, TP2 across both Sparks.
#
#   bash scripts/launch-glm-sglang.sh <rank>     # 0 = head, 1 = worker
#
# Three things here come straight from failures already recorded in
# docs/baselines.md, and none of them is optional.
#
# 1. `--ep-size 2` (GLM_EP=on, the default). With plain tensor parallelism the
#    NVFP4 packed shards fail to load: "The size of tensor a (512) must match
#    the size of tensor b (1024) at non-singleton dimension 1". Expert
#    parallelism loads them cleanly. That was established on 2026-09-03 -- and
#    established twice, because the first attempt set the knob in front of a
#    bring_up that did not forward it, so only rank 0 got it.
#
# 2. The DSA backend is a knob, not a constant, because tilelang does not fit
#    this hardware. Its sparse kernel asks the driver for 169,984 bytes of
#    dynamic shared memory per block; GB10 allows 101,376
#    (MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, measured). The request is 1.68x the
#    budget, so the driver refuses it, and no flag makes a kernel fit a budget
#    it overruns by two thirds. Of the nine values this build accepts, six are
#    ruled out by their packages being absent from the image (flash_mla, aiter,
#    tensorrt_llm). The two survivors are flashinfer_sparse_mla -- first, it
#    ships Blackwell kernels -- and fa3, which is Hopper-first, as the fallback.
#
# 3. `SGLANG_HOST_IP` per rank, or multi-node shm_broadcast hangs at startup.
#
# The drafter is where the speed is. BASELINE.md puts DFlash2 at 2.15x against
# MTP-4 (46.9 vs 21.8 tok/s single-stream at 74.1% acceptance), and the previous
# attempt ran GLM with no drafter at all. GLM_SPEC=off exists only to measure
# what the drafter is worth on this hardware, never as a default.
set -euo pipefail

RANK="${1:?usage: launch-glm-sglang.sh <rank 0|1>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$here/scripts/lib/memguard.sh"
source "$here/scripts/lib/rdma.sh"

GLM_IMAGE="${GLM_IMAGE:-lmsysorg/sglang:glm-5.3-flash}"
GLM_NAME="${GLM_NAME:-sglang_glm53}"
GLM_PORT="${GLM_PORT:-30000}"
GLM_DIST_ADDR="${GLM_DIST_ADDR:-10.10.10.1:20000}"
GLM_HOST_IP="${GLM_HOST_IP:-$( [ "$RANK" = 0 ] && echo 10.10.10.1 || echo 10.10.10.2 )}"

store="$( [ "$RANK" = 0 ] && echo /home/station/models/hf || echo /mnt/models/hf )"
GLM_MODEL="${GLM_MODEL:-$store/glm53-flash-nvfp4-ref}"
GLM_DRAFT="${GLM_DRAFT:-$store/glm53-dflash2-draft}"

GLM_EP="${GLM_EP:-on}"                      # expert parallelism: see note 1
GLM_DSA="${GLM_DSA:-flashinfer_sparse_mla}" # see note 2; fallback fa3
GLM_KV_DTYPE="${GLM_KV_DTYPE:-fp8_e4m3}"    # PR #36904's fp8 KV; bf16 before it
GLM_MEM="${GLM_MEM:-0.92}"
GLM_MAMBA="${GLM_MAMBA:-40}"
GLM_MAMBA_DTYPE="${GLM_MAMBA_DTYPE:-bfloat16}"
GLM_RUNNING="${GLM_RUNNING:-8}"
GLM_CTX="${GLM_CTX:-131072}"
GLM_SPEC="${GLM_SPEC:-on}"
GLM_SPEC_TOKENS="${GLM_SPEC_TOKENS:-8}"
GLM_EXTRA="${GLM_EXTRA:-}"

echo "== rank $RANK: tearing down any previous $GLM_NAME"
docker rm -f "$GLM_NAME" >/dev/null 2>&1 || true

echo "== rank $RANK: sync + drop_caches"
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

CAP_GIB="$(container_cap_gib)"
NEED_GIB="${GLM_NEED_GIB:-95}"
echo "== rank $RANK: container cap ${CAP_GIB}GiB, need ${NEED_GIB}GiB free"
wait_for_free_memory "$NEED_GIB"

[ -d "$GLM_MODEL" ] || { echo "rank $RANK: $GLM_MODEL not readable" >&2; exit 6; }

args=(
  --model-path "$GLM_MODEL"
  --served-model-name glm53-flash
  --tp-size 2 --nnodes 2 --node-rank "$RANK"
  --dist-init-addr "$GLM_DIST_ADDR"
  --host 0.0.0.0 --port "$GLM_PORT"
  --context-length "$GLM_CTX"
  --kv-cache-dtype "$GLM_KV_DTYPE"
  --mem-fraction-static "$GLM_MEM"
  --max-mamba-cache-size "$GLM_MAMBA"
  --mamba-ssm-dtype "$GLM_MAMBA_DTYPE"
  --max-running-requests "$GLM_RUNNING"
  --dsa-prefill-backend "$GLM_DSA"
  --dsa-decode-backend "$GLM_DSA"
)
[ "$GLM_EP" = "on" ] && args+=( --ep-size 2 )
if [ "$GLM_SPEC" = "on" ]; then
  [ -d "$GLM_DRAFT" ] || { echo "rank $RANK: drafter $GLM_DRAFT not readable" >&2; exit 6; }
  args+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$GLM_DRAFT"
    --speculative-num-draft-tokens "$GLM_SPEC_TOKENS"
  )
fi
# shellcheck disable=SC2206
[ -n "$GLM_EXTRA" ] && args+=( $GLM_EXTRA )

mounts=()
[ -d /home/station/models ] && mounts+=( -v /home/station/models:/home/station/models:ro )
ls /mnt/models >/dev/null 2>&1 && mounts+=( -v /mnt/models:/mnt/models:ro )

echo "== rank $RANK: launching $GLM_IMAGE"
printf '   %s\n' "${args[*]}"

rdma_report
read -r -a RDMA_ARGS <<< "$(rdma_docker_args)"

exec docker run --rm --name "$GLM_NAME" \
  --gpus all --ipc=host --network host \
  --memory "${CAP_GIB}g" --memory-swap "${CAP_GIB}g" \
  "${RDMA_ARGS[@]}" \
  -e NCCL_IB_HCA="$RDMA_HCA" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e SGLANG_HOST_IP="$GLM_HOST_IP" \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  "${mounts[@]}" \
  "$GLM_IMAGE" \
  python3 -m sglang.launch_server "${args[@]}"
