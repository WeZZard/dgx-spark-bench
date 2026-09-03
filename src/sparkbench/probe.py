"""Read the tuple's engine and KV-pool fields off the running server.

Rule 1 asks for "the measured pool in tokens, read from the running server" and
for an engine version "because the fixes live in specific builds". Both are
facts about the process that is about to be measured, so both are read from it
rather than copied out of the launcher that was *meant* to configure it. The
launcher is a hypothesis; the server is the evidence. On 2026-09-03 a two-node
run was launched with the two ranks holding different configurations and the
head's log looked healthy throughout -- reading back from the server is the
habit that catches that class of mistake.

Every probe records *how* it learned the number, and that string is stored as
`kv_pool_source`. A pool the harness could not read is not defaulted to
anything: `probe_kv_pool` raises, and `RunTuple.validate()` would refuse the
run anyway.
"""
from __future__ import annotations

import json
import re
import subprocess

import requests


class ProbeFailed(RuntimeError):
    pass


def _get(base: str, path: str, timeout: float = 20.0):
    r = requests.get(base.rstrip("/") + path, timeout=timeout)
    r.raise_for_status()
    return r


def server_info(base: str) -> dict:
    """Whatever the server will tell us about itself. Never raises."""
    for path in ("/get_server_info", "/version", "/v1/models"):
        try:
            return {"path": path, "body": _get(base, path).json()}
        except Exception:
            continue
    return {}


def probe_engine(base: str) -> tuple[str, str]:
    """(engine, version) as the build reports it, not as we believe it to be."""
    # SGLang answers /get_server_info with its full ServerArgs plus version.
    try:
        b = _get(base, "/get_server_info").json()
        v = b.get("version") or b.get("sglang_version") or ""
        if not v:
            # Older builds nest it under the server args.
            v = (b.get("server_args") or {}).get("version", "")
        return "sglang", str(v) or "unknown"
    except Exception:
        pass
    try:
        b = _get(base, "/version").json()
        return "vllm", str(b.get("version", "unknown"))
    except Exception:
        pass
    raise ProbeFailed(
        f"{base} answered neither /get_server_info nor /version; "
        "the engine version is a required tuple field and must not be guessed"
    )


_POOL_PATTERNS = [
    # vLLM v1, e.g. "GPU KV cache size: 3,068,415 tokens"
    (re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"), "log:vllm-gpu-kv-cache-size"),
    # SGLang, e.g. "max_total_num_tokens=600000" / "KV Cache is allocated. #tokens: 600000"
    (re.compile(r"max_total_num_tokens\s*[=:]\s*([\d,]+)"), "log:sglang-max-total-num-tokens"),
    (re.compile(r"#tokens:\s*([\d,]+)"), "log:sglang-kv-cache-tokens"),
]


def probe_kv_pool(base: str, container: str | None = None,
                  ssh_host: str | None = None) -> tuple[int, str]:
    """(tokens, source). Raises rather than returning a number it invented."""
    # 1. SGLang states the pool directly in its server info.
    try:
        b = _get(base, "/get_server_info").json()
        for key in ("max_total_num_tokens", "max_total_tokens"):
            v = b.get(key) or (b.get("server_args") or {}).get(key)
            if v:
                return int(v), f"http:/get_server_info:{key}"
    except Exception:
        pass

    # 2. vLLM publishes the block count on /metrics; tokens = blocks * block_size.
    try:
        text = _get(base, "/metrics").text
        blocks = re.search(r'num_gpu_blocks="?([\d.]+)"?', text)
        bsz = re.search(r'block_size="?([\d.]+)"?', text)
        if blocks and bsz:
            n = int(float(blocks.group(1)) * float(bsz.group(1)))
            if n > 0:
                return n, "http:/metrics:num_gpu_blocks*block_size"
    except Exception:
        pass

    # 3. The engine printed it once at startup. Read its own log back.
    if container:
        cmd = ["docker", "logs", "--tail", "4000", container]
        if ssh_host:
            cmd = ["ssh", "-o", "ConnectTimeout=20", ssh_host, " ".join(cmd)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            blob = (out.stdout or "") + (out.stderr or "")
            for pat, src in _POOL_PATTERNS:
                hits = pat.findall(blob)
                if hits:
                    return int(hits[-1].replace(",", "")), src
        except Exception:
            pass

    raise ProbeFailed(
        f"could not read the KV pool from {base}"
        + (f" or from container {container}" if container else "")
        + ".\nThe pool in tokens is a required tuple field (Rule 1) and it is the "
          "field this project got wrong last time. Find it before measuring: it "
          "is printed once at startup by both engines."
    )


# "KV Cache is allocated. dtype: torch.bfloat16" -- SGLang says what it actually
# allocated, which is not the same thing as what the flag asked for.
_KV_DTYPE_ALLOCATED = re.compile(r"KV Cache is allocated\.\s*dtype:\s*(\S+)")


def probe_kv_cache_dtype(base: str, container: str | None = None,
                         ssh_host: str | None = None) -> tuple[str, str]:
    """(dtype, source).

    CLAUDE.md: "If no flag is passed, record `default` and go and find out what
    the default actually was." Going and finding out is this function's job, so
    it prefers the dtype the engine says it *allocated* over the one the
    server args say was *requested*.

    That distinction is not academic. Launching Qwen with no --kv-cache-dtype
    leaves `kv_cache_dtype: "auto"` in the server args while the engine logs
    "KV Cache is allocated. dtype: torch.bfloat16". Recording "auto" would put
    a non-answer in the field this project got wrong last time; recording
    "bfloat16 (engine default)" records what the pool is actually made of, and
    makes the cost of the default visible -- bf16 is twice fp8 per token, and
    the pool is what caps the users.
    """
    if container:
        cmd = ["docker", "logs", "--tail", "4000", container]
        if ssh_host:
            cmd = ["ssh", "-o", "ConnectTimeout=20", ssh_host, " ".join(cmd)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            blob = (out.stdout or "") + (out.stderr or "")
            hits = _KV_DTYPE_ALLOCATED.findall(blob)
            if hits:
                dtype = hits[-1].replace("torch.", "")
                return dtype, "log:allocated (engine's own report of the pool it built)"
        except Exception:
            pass
    try:
        b = _get(base, "/get_server_info").json()
        args = b.get("server_args") or b
        for key in ("kv_cache_dtype", "cache_dtype"):
            v = args.get(key)
            if v and str(v) != "auto":
                return str(v), f"http:/get_server_info:{key} (requested, not confirmed allocated)"
        if args.get("kv_cache_dtype") == "auto":
            return ("auto -- UNRESOLVED",
                    "server args say auto and the allocation was not read; "
                    "find the allocated dtype before publishing this run")
    except Exception:
        pass
    return "default", "unread: engine did not report it -- resolve before publishing"
