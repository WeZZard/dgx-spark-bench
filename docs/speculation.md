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
