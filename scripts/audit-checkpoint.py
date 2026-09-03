#!/usr/bin/env python3
"""Read what a checkpoint actually is, from the safetensors headers alone.

Rule 1 in CLAUDE.md needs a `weight` field that names the real quantization.
Directory names and model cards do not supply it: on this project's own store,
the checkpoint called `dsv4-flash-fp8` carries 4-bit experts, and the one
called `nvfp4` is the *larger* of the two. Both facts are visible only here.

The header of a safetensors shard is a JSON dictionary of
{name: {dtype, shape, data_offsets}}, stored after an 8-byte little-endian
length. Reading it touches kilobytes, not gigabytes, so a 189 GiB checkpoint
audits in seconds and never has to be loaded onto a GPU.

How to read the output. A 4-bit checkpoint stores two values per byte in an
integer container (U8 or I8), alongside one scale byte per block of values.
The scale dtype and the block size are what separate the formats:

    NVFP4   E4M3 scales, block 16   -> scale bytes = weight bytes * 2 / 16
    MXFP4   E8M0 scales, block 32   -> scale bytes = weight bytes * 2 / 32
    FP8     E4M3 weights, no packing

so NVFP4 costs exactly twice the scale bytes of MXFP4 over identical weights.
`--verify` checks that identity against the tensors actually present and
prints the implied block size, which is the evidence for the format claim.
"""
import argparse, collections, json, os, re, struct, sys

GIB = 1024 ** 3
PACKED = {"U8", "I8", "UINT8", "INT8"}
SCALE = {"F8_E4M3": 16, "F8_E8M0": 32}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def category(name):
    if ".experts." in name or re.search(r"experts\.\d", name):
        return "routed_experts"
    if "shared_expert" in name:
        return "shared_experts"
    if "ple." in name or "ple_embedding" in name:
        return "per_layer_embedding"
    if ".attn." in name or "self_attn" in name or "attention" in name:
        return "attention"
    if "embed" in name or name.startswith("lm_head") or ".head" in name:
        return "embed/head"
    return "other"


def audit(path):
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no safetensors shards in {path}")
    by_dtype = collections.Counter()
    by_cat = collections.Counter()
    cat_dtype = collections.Counter()
    # per (category, role) bytes, where role is weight / scale, for --verify
    pairs = collections.defaultdict(lambda: collections.Counter())
    ntensors = 0
    for s in shards:
        for name, meta in read_header(os.path.join(path, s)).items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            nb, dt, c = b - a, meta["dtype"], category(name)
            ntensors += 1
            by_dtype[dt] += nb
            by_cat[c] += nb
            cat_dtype[(c, dt)] += nb
            if dt in PACKED:
                pairs[c]["weight"] += nb
            elif dt in SCALE and ("scale" in name.lower()):
                pairs[c][dt] += nb
    return dict(shards=len(shards), ntensors=ntensors, by_dtype=by_dtype,
                by_cat=by_cat, cat_dtype=cat_dtype, pairs=pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--verify", action="store_true",
                    help="derive the quantization block size and name the format")
    a = ap.parse_args()
    r = audit(a.path)
    total = sum(r["by_dtype"].values())
    print(f"### {os.path.basename(a.path.rstrip('/'))}  "
          f"shards={r['shards']} tensors={r['ntensors']} total={total/GIB:.2f} GiB")
    print("-- by dtype --")
    for k, v in r["by_dtype"].most_common():
        print(f"   {k:>10} {v/GIB:9.2f} GiB  {100*v/total:5.1f}%")
    print("-- by category --")
    for k, v in r["by_cat"].most_common():
        print(f"   {k:>20} {v/GIB:9.2f} GiB  {100*v/total:5.1f}%")
    print("-- category x dtype --")
    for (c, dt), v in sorted(r["cat_dtype"].items(), key=lambda x: -x[1]):
        if v / GIB < 0.005:
            continue
        print(f"   {c:>20} {dt:>10} {v/GIB:9.2f} GiB")

    if not a.verify:
        return
    print("-- format, derived from the weight:scale byte ratio --")
    for c, p in sorted(r["pairs"].items(), key=lambda x: -sum(x[1].values())):
        w = p.get("weight", 0)
        if not w:
            continue
        for dt, blk in SCALE.items():
            s = p.get(dt, 0)
            if not s:
                continue
            # two 4-bit values per weight byte; one scale byte per block
            implied = (w * 2) / s
            fmt = {"F8_E4M3": "NVFP4", "F8_E8M0": "MXFP4"}[dt]
            ok = "OK" if abs(implied - blk) < 0.01 else f"UNEXPECTED (want {blk})"
            print(f"   {c:>20}  {w/GIB:7.2f} GiB packed 4-bit + {s/GIB:6.2f} GiB "
                  f"{dt} scales -> block {implied:.2f}  = {fmt}  [{ok}]")
        if not any(p.get(dt) for dt in SCALE):
            print(f"   {c:>20}  {w/GIB:7.2f} GiB packed, no scale tensor found")


if __name__ == "__main__":
    main()
