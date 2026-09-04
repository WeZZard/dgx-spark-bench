#!/usr/bin/env python3
"""Are two checkpoints the same weights, or only the same shape?

    scripts/compare-checkpoints.py /path/to/local \
        deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
        --revision 86f746b36186f0e567729a5c06a8c918caba82a9

Prints the tensor-name difference between a local checkpoint and a remote one,
then hashes a sample of the tensors they share and says whether the bytes
match. The remote side is read with HTTP range requests, so a shared tensor
costs its own size and nothing more.

This exists because of a trap that is easy to walk into. Vision-Exp contains
every one of `0731`'s 72,317 tensor names, at identical dtypes and identical
byte counts, plus 316 more. Two checkpoints that agree that exactly look like
one checkpoint with a vision tower bolted on, and if that were true then this
project's `0731` measurements would be Vision-Exp measurements for text work
and the largest reproduction gap in `BASELINE.md` would close for free.

They are not. Sampled expert tensors from layers 3, 20 and 40 hash
differently, so the text backbones are separately trained weights that merely
share an architecture. That is Rule 1's "a vision variant is not the text
model", demonstrated rather than asserted -- and it is worth demonstrating,
because the structural evidence points the other way.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

UA = os.environ.get("HF_USER_AGENT", "sparkbench-audit/1 (+curl-compatible)")
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

# Spread across depth and across expert index, because a checkpoint that
# differs only in its last layers would pass a sample taken from layer 0.
DEFAULT_SAMPLE = [
    "layers.3.ffn.experts.0.w1.weight",
    "layers.3.ffn.experts.0.w1.scale",
    "layers.20.ffn.experts.7.w2.weight",
    "layers.40.ffn.experts.255.w3.weight",
]


def _get(url, rng=None):
    h = {"User-Agent": UA}
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1]}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=600).read()


class Local:
    def __init__(self, path):
        self.path = path
        self.wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self._h = {}

    def header(self, shard):
        if shard not in self._h:
            with open(os.path.join(self.path, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._h[shard] = (json.loads(f.read(n)), 8 + n)
        return self._h[shard]

    def tensor(self, name):
        hdr, base = self.header(self.wm[name])
        lo, hi = hdr[name]["data_offsets"]
        with open(os.path.join(self.path, self.wm[name]), "rb") as f:
            f.seek(base + lo)
            return hdr[name]["dtype"], f.read(hi - lo)


class Remote:
    def __init__(self, repo, revision):
        self.base = f"{ENDPOINT}/{repo}/resolve/{revision}"
        self.wm = json.loads(_get(f"{self.base}/model.safetensors.index.json"))["weight_map"]
        self._h = {}

    def header(self, shard):
        if shard not in self._h:
            url = f"{self.base}/{shard}"
            n = struct.unpack("<Q", _get(url, (0, 7)))[0]
            self._h[shard] = (json.loads(_get(url, (8, 8 + n - 1))), 8 + n)
        return self._h[shard]

    def tensor(self, name):
        shard = self.wm[name]
        hdr, base = self.header(shard)
        lo, hi = hdr[name]["data_offsets"]
        b = _get(f"{self.base}/{shard}", (base + lo, base + hi - 1))
        if len(b) != hi - lo:
            raise IOError(f"{name}: asked for {hi - lo} bytes, got {len(b)}")
        return hdr[name]["dtype"], b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("local")
    ap.add_argument("repo")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--sample", action="append", default=None,
                    help="tensor name to hash; repeatable, defaults to four expert tensors")
    a = ap.parse_args()
    if len(a.revision) != 40:
        sys.exit("refusing: --revision must be a 40-character commit sha")

    L, R = Local(a.local), Remote(a.repo, a.revision)
    ln, rn = set(L.wm), set(R.wm)
    print(f"### {os.path.basename(a.local.rstrip('/'))}  vs  {a.repo}@{a.revision[:12]}")
    print(f"    shared names {len(ln & rn)}   local-only {len(ln - rn)}   remote-only {len(rn - ln)}")
    if rn - ln:
        import collections
        pre = collections.Counter(n.split(".")[0] for n in rn - ln)
        print(f"    remote-only prefixes: {dict(pre)}")

    print("-- sampled tensors --")
    same = diff = 0
    for name in (a.sample or DEFAULT_SAMPLE):
        if name not in ln or name not in rn:
            print(f"   {name}  not in both, skipped")
            continue
        ldt, lb = L.tensor(name)
        rdt, rb = R.tensor(name)
        lh, rh = hashlib.sha256(lb).hexdigest(), hashlib.sha256(rb).hexdigest()
        ok = lh == rh and ldt == rdt and len(lb) == len(rb)
        same, diff = same + ok, diff + (not ok)
        print(f"   {name}\n      local  {ldt:>8} {len(lb):>12,} B  {lh[:32]}"
              f"\n      remote {rdt:>8} {len(rb):>12,} B  {rh[:32]}   {'SAME' if ok else 'DIFFERENT'}")

    print(f"-- {same} identical, {diff} different --")
    if diff:
        print("   These are different weights. Shared names and shared byte counts mean a")
        print("   shared architecture and a shared quantization, not a shared checkpoint;")
        print("   measurements of one are not measurements of the other (Rule 1).")
    elif same:
        print("   Every sampled tensor matched. That is evidence, not proof: only the")
        print("   sampled tensors were read.")


if __name__ == "__main__":
    main()
