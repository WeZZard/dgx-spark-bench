#!/usr/bin/env bash
# DeepSeek-V4-Flash on vLLM, TP2 across both Sparks, with DSpark speculation.
#
#   bash scripts/launch-deepseek-vllm.sh <rank>     # 0 = head, 1 = worker
#
# This is NOT the published configuration and does not pretend to be. The
# published one wants `--kv-cache-dtype nvfp4_ds_mla` and a B12X MoE backend,
# and neither exists in any vLLM image on these nodes (docs/engines.md). What
# this launcher reaches is the nearest configuration that is real here:
# `fp8_ds_mla`, which this build does implement with a documented packed layout
# and a Blackwell sparse-MLA backend behind it. The result is a different tuple
# with a smaller KV pool, recorded as such -- not a failed reproduction.
#
# Flags follow the sources rather than BASELINE.md's summary of them, which was
# checked on 2026-09-04 and differed in five places; see the source-check
# section there. In particular max-num-seqs is 12, gpu-memory-utilization 0.85
# and max-num-batched-tokens 8192.
#
# Patch 4 is the failure to watch for. The DSpark draft's shared-expert loader
# silently drops twelve tensors and halves decode while leaving output quality
# intact, and nothing is logged at INFO. This launcher cannot apply the patch
# -- it lives in the source repo -- so scripts/check-patch4.sh runs the source's
# own verification against the running container and must be run before any
# DeepSeek number is believed.
#
# Multi-node here is NOT the same shape as SGLang's. SGLang takes --nnodes and
# --node-rank and every rank runs the server. vLLM has neither: it forms a Ray
# cluster first, and then ONE `vllm serve` runs on the head and places workers
# across the cluster. So rank 1 starts a Ray worker and blocks, rank 0 starts
# the Ray head and then serves. Writing this the SGLang way would start two
# independent single-node servers, each trying to fit a 155 GiB checkpoint into
# 121 GiB -- which is exactly the "two ranks launched with different
# configurations" failure in a new costume.
set -euo pipefail

RANK="${1:?usage: launch-deepseek-vllm.sh <rank 0|1>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$here/scripts/lib/memguard.sh"
source "$here/scripts/lib/rdma.sh"

# Built by scripts/build-vllm-ray.sh: the stock image has no ray, and vLLM
# cannot span two nodes without it (or without an external launcher).
DS_IMAGE="${DS_IMAGE:-vllm-dsv4:ray}"
DS_NAME="${DS_NAME:-vllm_dsv4}"
DS_PORT="${DS_PORT:-8888}"
DS_MASTER="${DS_MASTER:-10.10.10.1}"
DS_MASTER_PORT="${DS_MASTER_PORT:-25440}"
DS_HOST_IP="${DS_HOST_IP:-$( [ "$RANK" = 0 ] && echo 10.10.10.1 || echo 10.10.10.2 )}"

# Which checkpoint. The published decode figures -- 84.3 peak, 197 aggregate at
# c6 -- are the TEXT 0731 model's, not Vision-Exp's, so 0731 is the default and
# it is already on disk. The official checkpoint carries MXFP4 experts; the
# nvidia NVFP4 one carries the same 138.00 GiB of 4-bit weights re-blocked to
# 16 (docs/checkpoints.md). Switching between them is the cleanest available
# test of whether the MoE kernel path, not weight bandwidth, is the difference.
DS_MODEL="${DS_MODEL:-$( [ "$RANK" = 0 ] \
    && echo /home/station/models/hf/dsv4-flash-fp8 \
    || echo /mnt/models/hf/dsv4-flash-fp8 )}"

DS_KV_DTYPE="${DS_KV_DTYPE:-fp8_ds_mla}"
DS_MAX_LEN="${DS_MAX_LEN:-262144}"
DS_MAX_SEQS="${DS_MAX_SEQS:-12}"
DS_BATCHED_TOKENS="${DS_BATCHED_TOKENS:-8192}"
DS_GPU_UTIL="${DS_GPU_UTIL:-0.85}"
DS_SPEC_TOKENS="${DS_SPEC_TOKENS:-5}"
DS_SPEC="${DS_SPEC:-on}"
DS_EXTRA="${DS_EXTRA:-}"

echo "== rank $RANK: tearing down any previous $DS_NAME"
docker rm -f "$DS_NAME" >/dev/null 2>&1 || true

echo "== rank $RANK: sync + drop_caches"
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

CAP_GIB="$(container_cap_gib)"
NEED_GIB="${DS_NEED_GIB:-95}"
echo "== rank $RANK: container cap ${CAP_GIB}GiB, need ${NEED_GIB}GiB free"
wait_for_free_memory "$NEED_GIB"

[ -d "$DS_MODEL" ] || { echo "rank $RANK: $DS_MODEL not readable" >&2; exit 6; }

args=(
  --model "$DS_MODEL"
  --served-model-name deepseek-v4-flash
  --tensor-parallel-size 2
  --host 0.0.0.0 --port "$DS_PORT"
  --kv-cache-dtype "$DS_KV_DTYPE"
  --max-model-len "$DS_MAX_LEN"
  --max-num-seqs "$DS_MAX_SEQS"
  --max-num-batched-tokens "$DS_BATCHED_TOKENS"
  --gpu-memory-utilization "$DS_GPU_UTIL"
  --trust-remote-code
  # Not optional, and not inferred. vLLM picks its executor backend in
  # config/parallel.py in strict elif order: nnodes > 1 forces "mp"; failing
  # that, `device_count() < world_size` RAISES. Each Spark presents one GPU
  # and world_size is 2, so with --nnodes left at its default of 1 that branch
  # fires and the server dies at config validation with "World size (2) is
  # larger than the number of available GPUs (1) in this node". The later
  # branch that would have chosen ray because a cluster is up is never
  # reached, so building the Ray cluster is not by itself enough.
  #
  # And it has to be this flag rather than --nnodes 2: a few lines further on,
  # nnodes > 1 is only allowed with mp, uni or external_launcher, so --nnodes 2
  # would reject "ray" outright. Ray is what this launcher actually builds.
  --distributed-executor-backend ray
)
if [ "$DS_SPEC" = "on" ]; then
  args+=( --speculative-config \
    "{\"method\":\"dspark\",\"num_speculative_tokens\":${DS_SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}" )
fi
# shellcheck disable=SC2206
[ -n "$DS_EXTRA" ] && args+=( $DS_EXTRA )

mounts=()
[ -d /home/station/models ] && mounts+=( -v /home/station/models:/home/station/models:ro )
ls /mnt/models >/dev/null 2>&1 && mounts+=( -v /mnt/models:/mnt/models:ro )

rdma_report
read -r -a RDMA_ARGS <<< "$(rdma_docker_args)"

common=(
  --rm --name "$DS_NAME"
  # The base image ENTRYPOINT is ["vllm","serve"], so without this every
  # command below would be appended to it as arguments.
  --entrypoint bash
  --gpus all --ipc=host --network host
  --memory "${CAP_GIB}g" --memory-swap "${CAP_GIB}g"
  "${RDMA_ARGS[@]}"
  -e NCCL_IB_HCA="$RDMA_HCA"
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  -e VLLM_HOST_IP="$DS_HOST_IP"
  -e VLLM_TRITON_MLA_SPARSE=1
  -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0
)

if [ "$RANK" = 1 ]; then
  # Join the head's Ray cluster and stay up. No serving happens here; the head
  # places a TP worker onto this node's GPU.
  echo "== rank 1: joining ray at ${DS_MASTER}:${DS_MASTER_PORT}"
  exec docker run "${common[@]}" "${mounts[@]}" "$DS_IMAGE" \
    -c "ray start --address=${DS_MASTER}:${DS_MASTER_PORT} \
                       --num-gpus=1 --block"
fi

echo "== rank 0: starting ray head and serving $DS_IMAGE"
printf '   %s\n' "${args[*]}"
exec docker run "${common[@]}" "${mounts[@]}" "$DS_IMAGE" \
  -c "ray start --head --port=${DS_MASTER_PORT} --num-gpus=1 && \
           echo '== waiting for 2 ray nodes ==' && \
           for i in \$(seq 1 60); do \
             n=\$(python3 -c \"import ray;ray.init(address='auto');print(len([x for x in ray.nodes() if x['Alive']]))\" 2>/dev/null || echo 0); \
             [ \"\$n\" -ge 2 ] && break; sleep 5; \
           done; \
           echo \"== ray sees \$n alive node(s) ==\" && \
           [ \"\$n\" -ge 2 ] || { echo 'refusing to serve: worker never joined ray' >&2; exit 7; } && \
           exec vllm serve $(printf '%q ' "${args[@]}")"
