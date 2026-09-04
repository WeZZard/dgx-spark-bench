# dgx-spark-bench

Reproducible serving benchmarks for three open-weight MoE models on a pair of
NVIDIA DGX Spark machines (GB10, sm_121, 121 GiB unified memory each, joined
back-to-back at 200 GbE).

- **DeepSeek-V4-Flash** — the text `0731` checkpoint; Vision-Exp is not yet on disk
- **Qwen3.8-Flash-Next**
- **GLM-5.3-Flash**

## What this measures

Speed is reported as **served rate**: all output tokens divided by the whole
wall clock, so the wait for the first token counts against it. That is the
rate a person waiting on an answer actually experiences. Where a second
measure appears it is named **decode rate** and it is used only to show how
generation holds up as the prompt grows.

Every result is recorded as a six-field tuple — model, weight, KV cache,
engine, decoding users, KV pool — because a throughput number without those
fields cannot be compared with anything.

Every cell here is generated with `--ignore-eos`, so each request emits
exactly 512 tokens. Without that, served rate moves with how long the answer
happened to be: four samples of one DeepSeek cell produced 179, 266, 279 and
512 tokens and read 5.8, 8.1, 8.3 and 13.9 tok/s, while time-to-first-token
stayed flat within 2%. Same work, different denominator. `REPORT.md` prints
the output-token count beside every row so incomparable rows are visible.

## Status

All three models serve on the pair. Zero failed requests and 100% verbatim
needle retrieval in every agentic cell, including 26,500-token prompts at
full concurrency. NCCL over RoCE, confirmed during each campaign from the
HCA's own `rx_write_requests` counter rather than from a log.

**One decoding user** — the only cells where the three are directly
comparable, since GLM's ladder runs to 8 users and DeepSeek's and Qwen's to
6, each matching its own published concurrency, and users is a tuple field:

| workload | DeepSeek-V4-Flash-0731 | Qwen3.8-Flash-Next | GLM-5.3-Flash |
|---|---:|---:|---:|
| short prompt | **44.1** | 26.5 | 10.3 |
| code | **32.8** | 23.9 | 10.0 |
| agentic, 4k context | **26.2** | 22.7 | 9.7 |
| agentic, 33k context | **13.9** | 11.0 | 7.3 |

At each model's own published concurrency:

| workload | DeepSeek @ 6 | Qwen @ 6 | GLM @ 8 |
|---|---:|---:|---:|
| short prompt | 89.6 | **96.6** | – |
| code | 40.2 | **87.3** | 45.7 |
| agentic, 4k context | **54.6** | 43.4 | 24.4 |
| agentic, 33k context | **17.7** | 11.2 | 13.5 |

Served rate throughout. Full six-field tuples, with the KV pool each was
measured against, are in `REPORT.md` — generated, never hand-edited.

The configurations behind those numbers, and why each differs from the
published one:

- **DeepSeek**: fp8_ds_mla KV, 505,407-token pool, DSpark k=5, measured
  acceptance 54.3% over 68,115 draft tokens. The published `nvfp4_ds_mla` KV
  dtype does not exist in any vLLM build on these nodes, and these are the
  **text** model's numbers, not Vision-Exp's. A different tuple, recorded as
  one.
- **Qwen**: NVFP4 with NEXTN speculation over RoCE, bf16 KV. Enabling MTP
  cut the measured KV pool from 600,000 tokens to 145,024 with the same
  `--max-total-tokens`. `docs/speculation.md`, `docs/interconnect.md`.
- **GLM**: bfloat16 KV, 800,448-token pool, **no speculation**. The published
  fp8 KV path is unreachable for this checkpoint's architecture on GB10, and
  the DSA kernel needed a shared-memory fix to run at all. `docs/glm-dsa.md`.

Findings, each written up with the evidence that supports it:

- **The checkpoint labelled FP8 has 4-bit experts.** Both DeepSeek checkpoints
  carry exactly 138.00 GiB of packed 4-bit expert weights; the "NVFP4" one is
  larger only because it stores twice the scale bytes. `docs/checkpoints.md`.
- **The container could not see the RDMA devices**, so a 200 GbE link
  cross-connected between the two nodes was carrying NCCL over TCP. Fixing it
  was worth 1.74x across every cell measured. `docs/interconnect.md`.
- **GLM-5.3-Flash's sparse-attention kernel does not fit on GB10.** The stock
  tilelang kernel asks for 169,984 bytes of dynamic shared memory against a
  101,376-byte limit. The working tuning was found by compiling the kernel
  standalone in about a minute, after three seventeen-minute weight loads had
  each ended in a different failure. `docs/glm-dsa.md`.
- **The published DeepSeek row does not say what it appears to say.** Its
  "NVFP4" is the KV cache, its headline throughput belongs to the text model
  rather than the vision one, and its engine version matches neither source.
  `BASELINE.md`, "What the sources actually say".

## Layout

    BASELINE.md          the published tuples to reproduce, with sources
    CLAUDE.md            the rules that govern any work in this repo
    REPORT.md            every recorded tuple, generated from the database
    docs/baselines.md    measured hardware facts and the incidents behind them
    docs/checkpoints.md  what is actually on disk, read from the headers
    docs/engines.md      what each engine image can actually do
    docs/glm-dsa.md      which GLM sparse-attention kernel can run on GB10
    docs/interconnect.md the RoCE finding, with the A/B that supports it
    docs/qwen-qsa.md     the long-context corruption question, and how to settle it
    docs/speculation.md  what MTP costs in KV pool, and what it buys
    src/sparkbench/      the harness: one rate formula, one tuple, one database
    scripts/             launchers, sweeps, the GB10 memory guard and the checks
    docker/              the two derived images, and why each exists
    tests/               harness checks that need no GPU

## Running it

    scripts/sparkbench selftest                    # the formulas and the tuple rule
    scripts/audit-checkpoint.py <path> --verify    # what a checkpoint really is
    bash scripts/build-sglang-glm53-gb10.sh        # GLM needs a patched kernel
    bash scripts/build-vllm-ray.sh                 # DeepSeek needs ray in the image
    CONFIG_NOTE=<name> bash scripts/sweep-glm.sh   # one configuration, every cell
    scripts/sparkbench report --out REPORT.md

`bring-up.sh` refuses to start a launcher that has no clean run behind it
unless `ATTENDED=1`, because on GB10 a container memory cap does not bound what
the driver allocates and an unproven launcher once needed a power cycle.

Two things worth knowing before reading any number here:

- **The first campaign after a load reads low, and the gap grows with
  concurrency.** Repeating one GLM campaign three times against an unchanged
  server moved 8-user code from 31.5 to 44.2 to 44.9 tok/s while 1-user moved
  9.9 to 10.0 to 10.6. The warmup now includes a concurrent burst as well as
  long prompts, which reduces but does not eliminate it.
- **Some concurrent cells are genuinely noisy.** DeepSeek's 6-user short-prompt
  cell read 41.2, 101.8 and 89.6 tok/s across three samples with identical
  token counts and no GPU throttling (clocks steady at ~690 MHz, 51 °C).

## Licence

MIT.
