#!/usr/bin/env python3
"""Read what a checkpoint actually is, from the safetensors headers alone.

Thin CLI over `sparkbench.checkpoint`, which holds the logic so that the
harness fills the tuple's `weight` and `model_revision` fields with the same
measurement this prints. See `docs/checkpoints.md` for what it found on this
project's own store, and why size is not evidence of bit width.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sparkbench.checkpoint import GIB, audit, derive_formats, revision, weight_field  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--verify", action="store_true",
                    help="derive the quantization block size and name the format")
    a = ap.parse_args()

    r = audit(a.path)
    total = r.total_bytes
    print(f"### {os.path.basename(a.path.rstrip('/'))}  shards={r.shards} "
          f"tensors={r.ntensors} total={total / GIB:.2f} GiB")
    rev = revision(a.path)
    print(f"    revision: {rev or 'unknown (no hf download metadata)'}")
    print(f"    weight field: {weight_field(r)}")
    print("-- by dtype --")
    for k, v in r.by_dtype.most_common():
        print(f"   {k:>10} {v / GIB:9.2f} GiB  {100 * v / total:5.1f}%")
    print("-- by category --")
    for k, v in r.by_cat.most_common():
        print(f"   {k:>20} {v / GIB:9.2f} GiB  {100 * v / total:5.1f}%")
    print("-- category x dtype --")
    for (c, dt), v in sorted(r.cat_dtype.items(), key=lambda x: -x[1]):
        if v / GIB < 0.005:
            continue
        print(f"   {c:>20} {dt:>10} {v / GIB:9.2f} GiB")

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
