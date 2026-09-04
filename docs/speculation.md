# MTP is worth 1.5x, and it costs three quarters of the KV pool

Measured 2026-09-04 on Qwen3.8-Flash-Next, one variable against the RoCE
configuration. Both sets are in `results/runs.sqlite`, tagged `rdma` and
`rdma+mtp`.

## What it buys

The checkpoint already carries the draft head — `mtp.fc_embedding.weight`,
`mtp.pre_fc_norm_embedding.weight` and friends are in the safetensors, and
`qwen4_exp.py` has an `is_nextn` path — so this costs nothing but a flag:
`--speculative-algorithm NEXTN`. SGLang resolves it to EAGLE internally with
the draft model path set to the target model.

Served rate:

| workload | users | RoCE | RoCE + MTP | ratio |
|---|---:|---:|---:|---:|
| short | 1 | 16.8 | **25.9** | 1.54x |
| short | 6 | 74.8 | **92.9** | 1.24x |
| code | 1 | 17.4 | **22.1** | 1.27x |
| code | 6 | 72.8 | **77.9** | 1.07x |
| prose | 1 | 17.0 | **23.2** | 1.36x |
| agentic-4k | 1 | 15.7 | **22.2** | 1.41x |
| agentic-4k | 6 | 50.9 | **59.0** | 1.16x |
| agentic-33k | 1 | 10.9 | **12.5** | 1.15x |
| agentic-33k | 6 | 20.4 | **19.2** | **0.94x** |

Acceptance is excellent: SGLang reports `accept len: 1.88` to `1.91` against a
maximum of 2.0 at `--speculative-num-draft-tokens 2`, an acceptance rate of
0.88 to 0.91. The draft is nearly always right. What limits the gain is the
budget, not the draft.

## What it costs, and why the tuple caught it

The last row is a regression, and the reason is in a column that a throughput
table would not have:

```
rdma       agentic-33k  6 users   KV pool 600,000   19.2 -> 20.4 tok/s
rdma+mtp   agentic-33k  6 users   KV pool 145,024   ...
```

**Turning MTP on shrank the KV pool from 600,000 tokens to 145,024**, a factor
of 4.1, with `--max-total-tokens 600000` passed unchanged in both runs. The
draft model needs its own KV cache and the speculative path needs its own mamba
state, and SGLang takes both out of the same budget before sizing the pool.

That is fatal exactly where this project cares most. Six concurrent requests at
30,000 tokens of context need 180,000 tokens of pool. The pool is 145,024. So
the requests queue, and the configuration that is 1.5x faster on a short prompt
is 6% *slower* on six long agentic ones.

This is the clearest argument for Rule 1 the project has produced. The six-field
tuple is not bookkeeping: **the KV pool is a field precisely because it silently
changes when something else is tuned**, and a report of "MTP is worth 1.5x"
would have been true, useful-sounding and wrong for the workload the machines
were bought for.

## What follows

Three things to measure, in this order.

1. **fp8 KV cache.** The pool is bf16 by engine default — two bytes a token
   where fp8 is one — because the published recipe never sets
   `--kv-cache-dtype` and neither did the first runs here. Halving the bytes a
   token is the obvious way to buy back what speculation took.
   `QWEN_KV_DTYPE=fp8_e4m3`.

   **Tried. It buys the pool back and then crashes under load.**

   The pool goes from 145,024 tokens to **332,480**, a factor of 2.29, which is
   what was wanted: six 30,000-token contexts need 180,000 and now fit. Speed on
   the cells that completed is a wash — 23.7 against 25.9 on `short` at one
   user, 84.0 against 77.9 on `code` at six.

   Then `agentic-4k` at six users killed the server, with 6 of 6 requests
   failing and no output at all:

   ```
   [TP1] Scheduler hit an exception: ...
     File ".../triton/language/semantic.py", line 1500, in dot
       assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32, ...
   AssertionError: Unsupported rhs dtype fp8e4nv
   ```

   A Triton `dot` in the QSA attention path cannot take an fp8 right-hand side
   in this build. Both nodes came back clean — 117 GiB available, nothing in
   state D — and the harness recorded the cell as `0.0 tok/s, 6 requests,
   6 failed` rather than dropping it, which is the behaviour that keeps a
   broken configuration from looking merely absent.

   **The "concurrent" in this paragraph used to say the trigger was load. It
   is not, and the correction matters.** Reading
   `qwen_sparse_attn_backend.py`, `forward_extend` branches on `if not
   any(prefix_lens)`: no prefix and it passes freshly projected bfloat16 k/v
   to the plain kernel, any prefix and the `_ck` kernel reads K/V straight out
   of the paged pool — which is fp8 exactly when `--kv-cache-dtype` made it
   so. A non-zero prefix is what chunked prefill produces, and
   `chunked_prefill_size` is 8192. `scripts/probe-qwen-fp8kv.sh` tested the
   prediction that follows: **one** request with a 30,301-token prompt, no
   concurrency, same assertion, server dead. The 4,318-token control passed at
   the same one user.

   So the real rule is that fp8 KV fails on any prompt longer than
   `chunked_prefill_size`, which is worse than "fails under load" and is
   precisely the long-context case. Full working in `docs/qwen-qsa.md`.

   **So fp8 KV is not available on this image**, and the pool that speculation
   takes cannot be bought back this way. What remains: `nvfp4` and
   `fp4_mx_block16` are also offered as KV dtypes and may hit the same Triton
   limit; reducing `--max-mamba-cache-size` frees budget the mamba state is
   holding; and accepting the smaller pool with concurrency capped at long
   context is a legitimate answer rather than a failure.
2. **A longer draft chain.** Acceptance of 0.88-0.91 at a 2-token budget says
   the chain is the constraint. `--speculative-num-steps 3` with
   `--speculative-num-draft-tokens 4`.

   **Tried, and it does not launch at this memory setting.** After loading, both
   ranks refuse:

   ```
   ValueError: Loaded weights leave no GPU memory for the KV cache under
   --mem-fraction-static=0.8. ... If using speculative decoding, draft weights
   are now counted.
   ```

   That last clause is the whole story, and it is the same accounting that took
   the pool from 600,000 to 145,024: a longer chain needs more draft state, and
   at a 2-token budget the pool was already down to a quarter. So the ordering
   in this list was wrong when it was written — a longer chain cannot be
   measured until the pool has been bought back, because there is no room to put
   it in. Do (1) first, then return here.

   The failure was clean: both nodes came back to 117 GiB available with no
   container and no process stuck in state D. The guard did its job and nothing
   needed a power cycle.

   **This error is not Qwen-specific and it caught GLM too, on 2026-09-04.**
   Adding the DFlash2 drafter to GLM and then *lowering*
   `--mem-fraction-static` to 0.85 to make room produced the same refusal, for
   the same reason: the fraction is the static budget, weights are charged to
   it, and draft weights are now counted. Lowering it does not buy headroom —
   it takes the headroom from the KV cache. The engine names the floor it
   computed (`minimum viable = 1 - available/pre`), so read it out of the log
   rather than guessing. `scripts/launch-glm-sglang.sh` records GLM's window.
3. **The ReplaySSM verify path.** `enable_linear_replayssm_spec` defaults to
   False and was False for the run above. This model is 48 layers of mostly
   linear attention, and SGLang has a fold-every-commit verify valid only for a
   linear draft chain — which NEXTN is — with a slower recurrent fallback
   otherwise. At a 2-token budget the verify is cheap either way; at a 4-token
   chain it should matter, which is why it is worth pairing with (2) rather
   than measuring alone.

## The honest summary

At one user, MTP is a clear win everywhere: 1.15x to 1.54x. At six users it
narrows to 1.07x-1.24x, and at six users on long agentic context it is a small
loss. Whether to run it is therefore a workload decision, not a tuning default,
until the pool is bought back.

---

# GLM's DFlash2 drafter is worth 2.75x at one user and nothing at eight

Measured 2026-09-04, `[glm-dflash2-mem90]` in `REPORT.md`, against
`[glm-eos-warm]` — the same server configuration with `GLM_SPEC=off`. Served
rate, `--ignore-eos`, zero failures, needle 100% in every agentic cell.

| workload | users | no drafter | DFlash2 | ratio |
|---|---:|---:|---:|---:|
| short | 1 | 10.3 | **28.3** | **2.75x** |
| agentic-4k | 1 | 9.7 | **15.7** | 1.62x |
| code | 1 | 10.0 | **15.4** | 1.54x |
| agentic-33k | 1 | 7.3 | **10.8** | 1.48x |
| code | 2 | 17.4 | **24.1** | 1.39x |
| agentic-4k | 8 | 24.4 | **33.8** | 1.39x |
| code | 4 | 29.7 | **38.5** | 1.30x |
| agentic-33k | 8 | 13.5 | 14.5 | 1.07x |
| code | 8 | 45.7 | 45.3 | 0.99x |

**The ratio falls monotonically with concurrency, to nothing.** That is what
speculative decoding is: it spends compute to buy latency, and at eight
in-flight requests there is no spare compute to spend. The 8-user code cell is
a dead heat, and one repeat of a GLM campaign moves cells by up to 1.3x
anyway, so 0.99x means "no effect" rather than "slightly worse".

`short` at 2.75x against `code` at 1.54x, at the same one user, is
content: a short predictable continuation is drafted well and code is not.
Judge an agentic workload against the 1.48-1.62x rows, never against 2.75x.

## What it costs

**The KV pool falls from 802,560 tokens to 138,304 — a factor of 5.8**, and
it is worth splitting that in two, because both halves are measured:

| configuration | mem-fraction | KV pool |
|---|---:|---:|
| `glm-eos-warm`, no drafter | 0.92 | 802,560 |
| `glm-dflash2-mem92` | 0.92 | 244,480 |
| `glm-dflash2-mem90` | 0.90 | 138,304 |

The drafter itself costs **3.3x** of the pool at an unchanged memory fraction:
draft weights and draft KV are charged to the same static budget. Dropping the
fraction to 0.90 to buy back the dynamic headroom that `mem92` hung for costs a
further **1.77x**. `mem92` is in `REPORT.md` with its four cells and no more,
because it hit a scheduler watchdog on the fifth; see
`scripts/launch-glm-sglang.sh` for the window and why 0.85 does not exist.

That is not free at long context. Eight concurrent 33k prompts need 264,192
tokens of pool and there are 138,304, so the 8-user 33k cell cannot have held
all eight contexts resident — some of it is queueing, which is part of why
that row is the one with no gain. It answered all eight needles verbatim, so
this is a throughput cost and not a correctness one.

## Acceptance, measured rather than quoted

41 logged decode batches across the campaign:

```
mean accept len = 3.78 of 8 drafted     mean accept rate = 0.396
```

`BASELINE.md` quotes **74.1%** for DFlash2. This is 39.6%, roughly half, on
this project's workloads — which are code, prose and agentic context rather
than the short predictable continuations published figures are usually taken
on. The `short` cell's 2.75x is what a high-acceptance workload looks like
here; everything else is what 40% buys.

## Decode rate tells a different and also true story

Labelled as decode rate throughout, and used here for the one thing it is for
— how generation holds up as the prompt grows:

| workload | users | no drafter | DFlash2 | ratio |
|---|---:|---:|---:|---:|
| short | 1 | 10.3 | 29.1 | 2.83x |
| agentic-33k | 1 | 10.7 | 27.2 | 2.54x |
| code | 1 | 10.1 | 15.9 | 1.57x |
| code | 8 | 5.8 | 8.0 | 1.38x |

`agentic-33k @ 1` is the pair worth staring at. Decode is 2.54x faster and
served rate is only 1.48x, because time-to-first-token on a 26,535-token
prompt is 28.6 seconds and the drafter does nothing for prefill. A drafter
speeds up generation; it does not speed up reading. Served rate is this
project's formula precisely so that the 28.6 seconds is not quietly dropped.

## The published warning did not hold

`BASELINE.md` carries a warning that the tilelang DSA kernel "overflows shared
memory on GB10 at 8-token verify", which is why GLM was measured without a
drafter for so long. With the `block_I=32, num_stages=1, threads=128` tuning
in `sglang-glm53:gb10-tilelang` it does not: the draft-verify CUDA graph
captures at `num_tokens_per_req=8` for `bs=[1..8]` and the campaign runs
clean. The overflow the warning describes is the *stock* tuning's, and it was
already fixed for the no-speculation rows — nobody had checked whether the fix
also covered verify. It does.


## DSpark acceptance is a property of the content, not only of the loader

`check-patch4.sh` failed a healthy Vision-Exp campaign on 2026-09-04 with
"33.5% is unpatched territory". It was not. Its 40% line came from the
four-node report's "~25% unpatched against 60-80% patched", and that pair of
numbers is not a range a campaign-wide average can be compared against.

The Vision-Exp source (quoted in `BASELINE.md`) gives patched acceptance by
content:

| content | published DSpark acceptance, patched |
|---|---:|
| structured code | ~78% |
| mixed agent traffic | ~56% |
| prose | ~34% |
| — unpatched, any content | ~25% |

`campaign-deepseek.sh` runs `short`, `code`, `prose`, `agentic-4k` and
`agentic-33k`, so its aggregate is an average over that whole spread. Two
measured campaigns on this hardware, both with the patched image:

| configuration | campaign-wide acceptance |
|---|---:|
| `0731`, DSpark k=5 probabilistic | 52.8% |
| Vision-Exp text-only, DSpark k=6 probabilistic | 33.5% |

The 40% line ran between them and separated neither from a broken loader.
33.5% sits exactly where the published prose figure says a working draft
lands.

The fix is to measure the quantity that discriminates. `code` is ~78% patched
against ~25% unpatched — a 3x gap — so `campaign-deepseek.sh` now brackets the
spec-decode counters around **each cell** and asserts on the `code @ 1` one at
50%, midway between the two with margin on both sides. The campaign-wide check
survives, narrowed to the unpatched floor (28%) and explicitly reporting the
28–40% band as inconclusive rather than as a failure.

The counters (`vllm:spec_decode_num_accepted_tokens_total`,
`vllm:spec_decode_num_draft_tokens_total`) are monotonic totals, so a
difference across a cell is that cell's acceptance and nothing else — the same
bracketing argument `check-rdma.sh` uses for `rx_write_requests`.

Acceptance also falls with the draft length `k`, and the two DeepSeek
configurations here do not share one: `0731` declares
`num_nextn_predict_layers=1` and runs k=5, Vision-Exp declares 3 and runs k=6
(k must be a multiple of it). So 52.8% against 33.5% is not a like-for-like
comparison of two checkpoints either, and is not written down as one.
