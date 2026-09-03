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

build_here() {
  echo "== building $TAG on $(hostname)"
  docker build --build-arg BASE="$BASE" -t "$TAG" -f "$here/docker/Dockerfile.vllm-ray" "$here/docker"
  docker run --rm --entrypoint python3 "$TAG" -c "import ray;print('ray ok', ray.__version__)"
}

if [ "${1:-}" = "--local-only" ]; then build_here; exit 0; fi
build_here
echo "== building on $WORKER"
ssh -o ConnectTimeout=30 "$WORKER" "cd '$here' && TAG='$TAG' BASE='$BASE' bash scripts/build-vllm-ray.sh --local-only"
