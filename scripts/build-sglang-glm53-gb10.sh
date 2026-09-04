#!/usr/bin/env bash
# Build the GB10-tuned SGLang GLM image on both nodes.
#
#   bash scripts/build-sglang-glm53-gb10.sh
#
# One layer on top of lmsysorg/sglang:glm-5.3-flash. Built on each node
# rather than pushed, because there is no registry between them and the
# base image is already present on both.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${TAG:-sglang-glm53:gb10-tilelang}"
WORKER="${WORKER_HOST:-spark-right-fast}"

for host in "" "$WORKER"; do
  where="${host:-local}"
  echo "== building $TAG on $where"
  if [ -z "$host" ]; then
    docker build -t "$TAG" -f "$here/docker/Dockerfile.sglang-glm53-gb10" "$here/docker"
  else
    ssh -o ConnectTimeout=30 "$host" "mkdir -p ~/dgx-spark-bench/docker"
    scp -q "$here/docker/Dockerfile.sglang-glm53-gb10" \
           "$here/docker/patch-tilelang-gb10.py" "$host:~/dgx-spark-bench/docker/"
    ssh -o ConnectTimeout=600 "$host" \
      "docker build -t '$TAG' -f ~/dgx-spark-bench/docker/Dockerfile.sglang-glm53-gb10 ~/dgx-spark-bench/docker"
  fi
done

echo "== verifying the patch is present in both images"
for host in "" "$WORKER"; do
  where="${host:-local}"
  cmd="docker run --rm --entrypoint grep '$TAG' -c 'dgx-spark-bench GB10 patch' /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"
  if [ -z "$host" ]; then n=$(eval "$cmd"); else n=$(ssh -o ConnectTimeout=60 "$host" "$cmd"); fi
  echo "   $where: $n patch marker(s)"
done
