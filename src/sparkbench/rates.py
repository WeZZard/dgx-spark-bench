"""The one speed formula, and the one place it is computed.

Rule 2 in CLAUDE.md says this project reports **served rate** and that decode
rate may appear only where the question is how generation holds up as the
prompt grows, labelled as decode rate every time. The previous attempt broke
that rule by computing both ad hoc and quoting whichever suited the sentence;
the same DeepSeek run appeared as 22.6 and 40.3 tok/s in one document.

Prose cannot enforce that. This module makes it structural:

* both rates are computed here and nowhere else;
* a rate is never a bare float. `Rate` carries the name of the formula that
  produced it, and `Rate.label()` prints the name alongside the number, so a
  figure cannot reach a table having forgotten where it came from;
* `served()` and `decode()` are the only constructors, and neither can produce
  the other's `kind`.

If a third rate is ever needed, add it here with its own name. Do not compute
one at a call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean


@dataclass(frozen=True)
class RequestTrace:
    """One request's timings, in seconds on a monotonic clock.

    `token_times` holds the arrival time of every streamed output chunk,
    including the first, so `first_token_at` is `token_times[0]` when the
    request produced anything at all.
    """

    started_at: float
    finished_at: float
    output_tokens: int
    input_tokens: int
    token_times: list[float] = field(default_factory=list)
    ok: bool = True
    error: str | None = None

    @property
    def first_token_at(self) -> float | None:
        return self.token_times[0] if self.token_times else None

    @property
    def ttft_ms(self) -> float | None:
        t = self.first_token_at
        return None if t is None else (t - self.started_at) * 1000.0


@dataclass(frozen=True)
class Rate:
    """A throughput figure that knows which formula made it."""

    kind: str  # "served" or "decode"
    value: float
    n_requests: int
    output_tokens: int
    wall_s: float

    def label(self) -> str:
        return f"{self.value:.1f} tok/s {self.kind} rate"

    def __str__(self) -> str:  # so an accidental f-string still carries the name
        return self.label()


def served(traces: list[RequestTrace]) -> Rate:
    """Served rate: all output tokens divided by the whole wall clock.

    The wall clock runs from the first request being dispatched to the last
    response completing, so the wait for the first token counts against the
    number. At one decoding user this is the single-stream served rate; at N it
    is the aggregate. It is the same formula either way, which is why this
    project can rank a c1 cell against a c12 cell without switching measure.

    Failed requests contribute their wall-clock time but not their tokens.
    That is deliberate: a configuration that errors under load is slower, and
    hiding that by dropping the request would flatter it.
    """
    if not traces:
        raise ValueError("served rate of no requests is not a number")
    out = sum(t.output_tokens for t in traces if t.ok)
    wall = max(t.finished_at for t in traces) - min(t.started_at for t in traces)
    if wall <= 0:
        raise ValueError(f"non-positive wall clock ({wall}s); the clock is wrong")
    return Rate("served", out / wall, len(traces), out, wall)


def decode(traces: list[RequestTrace]) -> Rate:
    """Decode rate: 1000 divided by the mean gap between output tokens.

    The prompt read time is removed, so this answers "how fast does generation
    itself run", and only that. Every published figure in `BASELINE.md` is a
    decode rate. Use it to show how generation holds up as the prompt grows;
    never as a headline, and never in a table beside a served rate.

    Computed per request, then averaged across requests, so one long request
    cannot dominate.

    Implemented as (tokens - 1) / (last token time - first token time) rather
    than as the mean of chunk-to-chunk gaps. The two are identical when the
    engine streams one token per chunk, which is the usual case, but they
    diverge the moment a chunk carries several tokens -- and then the gap form
    silently undercounts. The span form uses the authoritative token count from
    the engine's own `usage`, so it stays correct either way.
    """
    if not traces:
        raise ValueError("decode rate of no requests is not a number")
    per_request = []
    for t in traces:
        if not t.ok or len(t.token_times) < 2 or t.output_tokens < 2:
            continue
        span = t.token_times[-1] - t.token_times[0]
        if span > 0:
            per_request.append((t.output_tokens - 1) / span)
    if not per_request:
        raise ValueError("no request produced two timed tokens; no decode rate exists")
    out = sum(t.output_tokens for t in traces if t.ok)
    wall = max(t.finished_at for t in traces) - min(t.started_at for t in traces)
    return Rate("decode", mean(per_request), len(traces), out, wall)


def ttft_p50_ms(traces: list[RequestTrace]) -> float | None:
    vals = sorted(t.ttft_ms for t in traces if t.ok and t.ttft_ms is not None)
    if not vals:
        return None
    return vals[len(vals) // 2]
