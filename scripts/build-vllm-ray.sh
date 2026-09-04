#!/usr/bin/env bash
# Build the Ray-enabled vLLM image on both nodes.
#
#   bash scripts/build-vllm-ray.sh
#
# Built on each node rather than built once and pushed, because there is no
# registry here and a 20 GB image over the 161 MB/s NFS write path is slower
# than a pip install on each side.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:-vllm-dsv4:ray}"
BASE="${BASE:-vllm-glm53-flash:sm121-v11-dflash2}"
WORKER="${WORKER_HOST:-spark-right-fast}"

# The worker builds from ITS OWN checkout, so send the current docker/ first.
# build-sglang-glm53-gb10.sh scp's the two files it needs and is therefore
# immune; this one re-invokes itself over ssh and is not. The day a new
# Dockerfile was added, the head built and the worker did not.
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
  docker build --build-arg BASE="$BASE" -t "$TAG" -f "$here/docker/Dockerfile.vllm-ray" "$here/docker"
  docker run --rm --entrypoint python3 "$TAG" -c "import ray;print('ray ok', ray.__version__)"
}

if [ "${1:-}" = "--local-only" ]; then build_here; exit 0; fi
build_here
sync_worker
echo "== building on $WORKER"
ssh -o ConnectTimeout=30 "$WORKER" "cd '$here' && TAG='$TAG' BASE='$BASE' bash scripts/build-vllm-ray.sh --local-only"
