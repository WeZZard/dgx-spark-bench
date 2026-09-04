#!/usr/bin/env bash
# Build the text-only DeepseekV4 loader image on both nodes.
#
#   bash scripts/build-vllm-dsv4-textonly.sh
#
# Built on each node rather than pushed, for the same reason as
# build-vllm-ray.sh: there is no registry here and a 20 GB image over NFS is
# slower than re-running a one-file patch.
#
# What this image is for, and what it is not: DeepSeek-V4-Flash-Vision-Exp
# carries 316 tensors that a text-only DeepseekV4ForCausalLM cannot place, and
# AutoWeightsLoader raises rather than skipping them, so the stock image
# cannot load the checkpoint at all. This one skips them. A run made with it
# has no vision path, and its tuple must say so.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:-vllm-dsv4:textonly}"
BASE="${BASE:-vllm-dsv4:ray}"
WORKER="${WORKER_HOST:-spark-right-fast}"

# The worker builds from ITS OWN docker/ directory, so send the current one.
# Nothing kept it in sync, and the day a new Dockerfile was added the worker
# build failed while the head's succeeded -- the same divergence bring-up.sh
# now guards against for scripts/ and src/, one directory over.
sync_worker() {
  # scripts/ as well as docker/: the remote step re-invokes THIS script from
  # the worker's own checkout, so a build script that syncs only its inputs
  # fails with "No such file or directory" the first time it is added.
  echo "== syncing scripts/ and docker/ to $WORKER"
  for d in scripts docker; do
    rsync -az --exclude '__pycache__' "$here/$d/" "$WORKER:$here/$d/" || {
      echo "refusing: could not sync $d/ to $WORKER; the two images would" >&2
      echo "be built from different sources." >&2
      exit 4
    }
  done
}

build_here() {
  echo "== building $TAG on $(hostname)"
  docker build --build-arg BASE="$BASE" -t "$TAG" \
    -f "$here/docker/Dockerfile.vllm-dsv4-textonly" "$here/docker"
  # Verify the patch is in the image rather than trusting the build. The
  # patch script refuses if its anchor is missing, but a cached layer from an
  # older base would not re-run it.
  docker run --rm --entrypoint bash "$TAG" -c '
    F=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py
    head -1 "$F" | grep -q PATCHED-VLLM-DSV4-TEXTONLY || { echo "marker missing" >&2; exit 1; }
    grep -q "skip_substrs=\[\"mtp.\", \"vision.\", \"aligner.\", \"image_\", \".bias_vl\"\]" "$F" \
      || { echo "skip list missing" >&2; exit 1; }
    echo "   patch verified in image"'
}

if [ "${1:-}" = "--local-only" ]; then build_here; exit 0; fi
build_here
sync_worker
echo "== building on $WORKER"
ssh -o ConnectTimeout=30 "$WORKER" "cd '$here' && TAG='$TAG' BASE='$BASE' bash scripts/build-vllm-dsv4-textonly.sh --local-only"
