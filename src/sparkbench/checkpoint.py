"""What a checkpoint is, read from its own bytes.

Two of Rule 1's six fields describe the checkpoint -- `model` (the exact
revision) and `weight` (the quantization). Both are read here rather than typed
into a launcher, for the reason `docs/checkpoints.md` records: the directory
called `dsv4-flash-fp8` holds 4-bit MXFP4 experts, and the one called `nvfp4` is
the larger of the two. A hand-written weight field would have been wrong on
both.

Header reading only: a safetensors shard begins with an 8-byte little-endian
length followed by that many bytes of JSON describing every tensor. A 189 GiB
checkpoint therefore audits in seconds, with no GPU and no load.
"""
from __future__ import annotations

import collections
import json
import os
import re
import struct
from dataclasses import dataclass, field

GIB = 1024 ** 3
PACKED = {"U8", "I8", "UINT8", "INT8"}
# scale dtype -> the quantization block size that dtype implies, and its name
SCALE_FORMATS = {"F8_E4M3": (16, "NVFP4"), "F8_E8M0": (32, "MXFP4")}


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def category(name: str) -> str:
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


@dataclass
class Audit:
    path: str
    shards: int = 0
    ntensors: int = 0
    by_dtype: collections.Counter = field(default_factory=collections.Counter)
    by_cat: collections.Counter = field(default_factory=collections.Counter)
    cat_dtype: collections.Counter = field(default_factory=collections.Counter)
    pairs: dict = field(default_factory=lambda: collections.defaultdict(collections.Counter))

    @property
    def total_bytes(self) -> int:
        return sum(self.by_dtype.values())


def audit_headers(path: str, headers) -> Audit:
    """Tally an already-read sequence of safetensors headers.

    Split out of `audit` so a checkpoint that is not on this disk can be
    measured by the SAME arithmetic. The headers sit in the first few hundred
    kilobytes of each shard, so a remote checkpoint can be audited over HTTP
    range requests for a few megabytes instead of downloading 156 GiB -- but
    only if the two paths share this function. Two audit implementations would
    be two formulas for one quantity, which is the thing CLAUDE.md forbids.
    """
    headers = list(headers)
    a = Audit(path=path, shards=len(headers))
    for hdr in headers:
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            nb, dt, c = hi - lo, meta["dtype"], category(name)
            a.ntensors += 1
            a.by_dtype[dt] += nb
            a.by_cat[c] += nb
            a.cat_dtype[(c, dt)] += nb
            if dt in PACKED:
                a.pairs[c]["weight"] += nb
            elif dt in SCALE_FORMATS and "scale" in name.lower():
                a.pairs[c][dt] += nb
    return a


def audit(path: str) -> Audit:
    shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {path}")
    return audit_headers(path, (read_header(os.path.join(path, s)) for s in shards))


def derive_formats(a: Audit) -> list[tuple[str, str, float, bool]]:
    """[(category, format, derived_block_size, matches_expectation)].

    A 4-bit checkpoint packs two values per byte and carries one scale byte per
    block of values, so `weight_bytes * 2 / scale_bytes` *is* the block size.
    That identity is what separates NVFP4 (E4M3 scales, block 16) from MXFP4
    (E8M0 scales, block 32) at identical weight bytes -- and it is the whole
    reason the NVFP4 DeepSeek checkpoint is 8.63 GiB larger than the MXFP4 one.
    """
    out = []
    for c, p in sorted(a.pairs.items(), key=lambda x: -sum(x[1].values())):
        w = p.get("weight", 0)
        if not w:
            continue
        for dt, (blk, fmt) in SCALE_FORMATS.items():
            s = p.get(dt, 0)
            if not s:
                continue
            implied = (w * 2) / s
            out.append((c, fmt, implied, abs(implied - blk) < 0.01))
    return out


def weight_field(a: Audit) -> str:
    """The `weight` field of the tuple, as a sentence, from the measurement.

    Names the format of the experts -- which carry the overwhelming majority of
    the bytes in every model here -- with the block size that was derived, not
    the one the directory name claims.
    """
    fmts = derive_formats(a)
    expert = [f for f in fmts if f[0] == "routed_experts"]
    total = a.total_bytes
    if expert:
        _, fmt, blk, ok = expert[0]
        share = 100 * a.by_cat.get("routed_experts", 0) / total
        note = "" if ok else " [BLOCK SIZE UNEXPECTED]"
        return f"{fmt} experts (block {blk:.0f}, {share:.0f}% of bytes){note}"
    # No packed experts: report the dominant dtype instead of inventing a name.
    dt, nb = a.by_dtype.most_common(1)[0]
    pretty = {"F8_E4M3": "FP8 E4M3", "BF16": "BF16", "F16": "FP16"}.get(dt, dt)
    return f"{pretty} ({100 * nb / total:.0f}% of bytes)"


def revision(path: str) -> str | None:
    """The commit the files were fetched at.

    Two sources, in order:

    1. `.provenance.json`, written by scripts/fetch-checkpoint.sh. Preferred
       because it survives `.cache` being pruned to reclaim disk, which is a
       thing that happens on a node holding 700 GiB of checkpoints.
    2. `hf download`'s own `.cache/huggingface/download/<file>.metadata`, whose
       first line is the commit hash. Any one of them will do; they all come
       from the same snapshot. This is the only source for the checkpoints that
       were already on disk before the fetch script existed.

    Returns None when neither is present — and None is not "main". A run whose
    checkpoint cannot state its revision fails the tuple rule.
    """
    sidecar = os.path.join(path, ".provenance.json")
    if os.path.isfile(sidecar):
        try:
            with open(sidecar) as fh:
                rev = json.load(fh).get("revision", "")
            if re.fullmatch(r"[0-9a-f]{40}", rev or ""):
                return rev
        except (OSError, ValueError):
            pass

    root = os.path.join(path, ".cache", "huggingface", "download")
    if not os.path.isdir(root):
        return None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".metadata"):
                try:
                    with open(os.path.join(dirpath, f)) as fh:
                        first = fh.readline().strip()
                    if re.fullmatch(r"[0-9a-f]{40}", first):
                        return first
                except OSError:
                    continue
    return None


def describe(path: str) -> dict:
    """Everything the tuple needs about a checkpoint on disk."""
    a = audit(path)
    return {
        "path": path,
        "name": os.path.basename(path.rstrip("/")),
        "revision": revision(path),
        "weight": weight_field(a),
        "total_gib": a.total_bytes / GIB,
        "audit": a,
    }
