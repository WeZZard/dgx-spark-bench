#!/usr/bin/env python3
"""Audit a Hugging Face checkpoint without downloading its weights.

    scripts/audit-remote-checkpoint.py deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
        --revision 86f746b36186f0e567729a5c06a8c918caba82a9 --verify

A safetensors file begins with an 8-byte little-endian header length followed
by that many bytes of JSON describing every tensor's dtype, shape and byte
range. That is all `sparkbench.checkpoint.audit_headers` needs, so a 156 GiB
checkpoint can be measured over HTTP range requests for a few megabytes.

This exists because `BASELINE.md` has a question that blocks a comparison --
is DeepSeek-V4-Flash-Vision-Exp built like its `0731` sibling, whose "FP8"
label covers 4-bit experts? -- and answering it by downloading the checkpoint
costs 156 GiB and about three hours. Reading 48 headers costs eight megabytes.

The tally is the same function the on-disk auditor calls. Do not reimplement
it here: the point of one formula per quantity is that the remote answer and
the local answer are the same measurement, so that when the weights do land,
`scripts/audit-checkpoint.py --verify` must agree with this to the byte.

Downloads go through HF_ENDPOINT, which defaults to the hf-mirror endpoint
this project uses for everything else.
"""
import argparse
import json
import os
import struct
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sparkbench.checkpoint import (  # noqa: E402
    GIB, audit_headers, derive_formats, weight_field,
)

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


# The mirror answers 403 to urllib's default `Python-urllib/3.x` agent and
# 200 to anything else. Found by sending the same request twice with only this
# header changed, so it is the header and not the range, the revision or the
# redirect.
UA = os.environ.get("HF_USER_AGENT", "sparkbench-audit/1 (+curl-compatible)")


def _get(url: str, headers: dict | None = None) -> bytes:
    h = {"User-Agent": UA}
    h.update(headers or {})
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def shard_header(base: str, shard: str) -> dict:
    url = f"{base}/{shard}"
    n = struct.unpack("<Q", _get(url, {"Range": "bytes=0-7"}))[0]
    return json.loads(_get(url, {"Range": f"bytes=8-{8 + n - 1}"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo")
    ap.add_argument("--revision", required=True,
                    help="a 40-character commit sha; a branch name is not a revision")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if len(a.revision) != 40:
        sys.exit("refusing: --revision must be a 40-character commit sha, "
                 "so the audit names a checkpoint that cannot move under it")

    base = f"{ENDPOINT}/{a.repo}/resolve/{a.revision}"
    index = json.loads(_get(f"{base}/model.safetensors.index.json"))
    shards = sorted(set(index["weight_map"].values()))
    print(f"### {a.repo}@{a.revision[:12]}  shards={len(shards)}  via {ENDPOINT}")

    hdrs = []
    for i, s in enumerate(shards, 1):
        print(f"\r    reading header {i}/{len(shards)}", end="", file=sys.stderr)
        hdrs.append(shard_header(base, s))
    print("", file=sys.stderr)

    r = audit_headers(a.repo, hdrs)
    total = r.total_bytes
    print(f"    tensors={r.ntensors} total={total / GIB:.2f} GiB")
    # The index's own total is an independent check on the header arithmetic.
    declared = index.get("metadata", {}).get("total_size")
    if declared:
        print(f"    index declares {declared / GIB:.2f} GiB "
              f"({'agrees' if abs(declared - total) < 2**20 else 'DISAGREES'})")
    print(f"    weight field: {weight_field(r)}")
    print("-- by dtype --")
    for k, v in r.by_dtype.most_common():
        print(f"   {k:>10} {v / GIB:9.2f} GiB  {100 * v / total:5.1f}%")
    print("-- by category --")
    for k, v in r.by_cat.most_common():
        print(f"   {k:>20} {v / GIB:9.2f} GiB  {100 * v / total:5.1f}%")
    if not a.verify:
        return
    print("-- format, derived from the weight:scale byte ratio --")
    for c, fmt, blk, ok in derive_formats(r):
        w = r.pairs[c]["weight"]
        s = sum(v for k, v in r.pairs[c].items() if k != "weight")
        print(f"   {c:>20}  {w / GIB:7.2f} GiB packed 4-bit + {s / GIB:6.2f} GiB "
              f"scales -> block {blk:.2f}  = {fmt}  [{'OK' if ok else 'UNEXPECTED'}]")


if __name__ == "__main__":
    main()
