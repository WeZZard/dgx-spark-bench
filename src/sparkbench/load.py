"""The load generator: one request path, one set of timings, for every engine.

Both SGLang and vLLM serve the OpenAI-compatible `/v1/chat/completions`, so
there is exactly one client here and no per-engine variant. That is on purpose:
if DeepSeek on vLLM and GLM on SGLang were measured by different code, the
comparison between them would be a comparison of load generators.

Timing. `started_at` is taken immediately before the request is dispatched, so
connection setup and the whole prefill are inside the measured wall clock --
that is what served rate means. Each streamed chunk that carries content stamps
`token_times`. The token *count* comes from the engine's own
`usage.completion_tokens` rather than from counting chunks, because a chunk is
not guaranteed to be one token; `stream_options.include_usage` asks for it.
When an engine declines to send usage, the chunk count is used and the run is
flagged, so the substitution is visible rather than silent.

Concurrency is real concurrency: `users` worker threads share a queue and the
run's wall clock spans from the first dispatch to the last completion. Threads
rather than asyncio because `requests` is the only HTTP client present on the
nodes, and because at the rates these machines reach -- a few hundred chunks a
second -- the GIL is not the bottleneck.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass

import requests

from .rates import RequestTrace
from .workload import Prompt


@dataclass
class LoadResult:
    traces: list[RequestTrace]
    texts: list[str]
    needle_hits: int
    needle_total: int
    usage_reported: bool
    prompt_tokens_seen: list[int]


def _one_request(base: str, model: str, p: Prompt, max_tokens: int,
                 temperature: float, timeout: float, ignore_eos: bool) -> tuple[RequestTrace, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": p.text}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True

    token_times: list[float] = []
    chunks = 0
    out_tokens = 0
    in_tokens = 0
    text_parts: list[str] = []
    err = None
    started = time.perf_counter()
    try:
        with requests.post(base.rstrip("/") + "/v1/chat/completions", json=body,
                           stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = ev.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        token_times.append(time.perf_counter())
                        chunks += 1
                        text_parts.append(piece)
                if ev.get("usage"):
                    u = ev["usage"]
                    out_tokens = int(u.get("completion_tokens") or 0)
                    in_tokens = int(u.get("prompt_tokens") or 0)
    except Exception as e:  # a failure is a measurement, not an exception to hide
        err = f"{type(e).__name__}: {e}"
    finished = time.perf_counter()

    counted_from_usage = out_tokens > 0
    if not counted_from_usage:
        out_tokens = chunks
    return (
        RequestTrace(
            started_at=started,
            finished_at=finished,
            output_tokens=out_tokens,
            input_tokens=in_tokens,
            token_times=token_times,
            ok=err is None and out_tokens > 0,
            error=err,
        ),
        "".join(text_parts),
    ), counted_from_usage


def run_load(base: str, model: str, prompts: list[Prompt], users: int,
             max_tokens: int = 256, temperature: float = 0.0,
             timeout: float = 1800.0, ignore_eos: bool = False) -> LoadResult:
    """Drive `len(prompts)` requests through `users` concurrent workers."""
    if users < 1:
        raise ValueError("decoding users must be at least 1")
    q: queue.Queue[tuple[int, Prompt]] = queue.Queue()
    for i, p in enumerate(prompts):
        q.put((i, p))

    traces: list[RequestTrace | None] = [None] * len(prompts)
    texts: list[str] = [""] * len(prompts)
    usage_flags: list[bool] = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                i, p = q.get_nowait()
            except queue.Empty:
                return
            (tr, txt), from_usage = _one_request(
                base, model, p, max_tokens, temperature, timeout, ignore_eos)
            with lock:
                traces[i] = tr
                texts[i] = txt
                usage_flags.append(from_usage)
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    done = [t for t in traces if t is not None]
    hits = total = 0
    for p, txt in zip(prompts, texts):
        if p.needle:
            total += 1
            hits += int(p.needle in txt)
    return LoadResult(
        traces=done,
        texts=texts,
        needle_hits=hits,
        needle_total=total,
        usage_reported=all(usage_flags) if usage_flags else False,
        prompt_tokens_seen=[t.input_tokens for t in done if t.input_tokens],
    )
