# dgx-spark-bench

Reproducible serving benchmarks for three open-weight MoE models on a pair of
NVIDIA DGX Spark machines (GB10, sm_121, 121 GiB unified memory each, joined
back-to-back at 200 GbE).

- **DeepSeek-V4-Flash-Vision-Exp**
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

## Status

Rebuilt from scratch on 2026-09-04. `BASELINE.md` holds the published
configurations to reproduce, with sources.

Measured so far, all on Qwen3.8-Flash-Next
(`RadixArk/Qwen3.8-Flash-Next-NVFP4` rev `7b71922`, NVFP4 experts at block 16,
KV bf16 by engine default, 600,000-token pool, SGLang `0.0.0.dev1+gd91c3682b`,
two nodes):

| | 1 user | 6 users |
|---|---:|---:|
| short prompt (74 tok), NCCL over TCP | 10.4 | 42.3 |
| short prompt (74 tok), NCCL over RoCE | **16.8** | **74.8** |
| agentic 33k context, NCCL over RoCE | **10.9** | **20.4** |

Served rate throughout. Full tuples in `docs/interconnect.md`.

Three things found on the way, each written up with its evidence:

- **The checkpoint labelled FP8 has 4-bit experts.** Both DeepSeek checkpoints
  carry exactly 138.00 GiB of packed 4-bit expert weights; the "NVFP4" one is
  larger only because it stores twice the scale bytes. `docs/checkpoints.md`.
- **The container could not see the RDMA devices**, so a 200 GbE link
  cross-connected between the two nodes was carrying NCCL over TCP. Fixing it
  is worth 1.74x across every cell measured. `docs/interconnect.md`.
- **The published DeepSeek row does not say what it appears to say.** Its
  "NVFP4" is the KV cache, its headline throughput belongs to the text model
  rather than the vision one, and its engine version does not match either
  source. `BASELINE.md`, "What the sources actually say".

## Layout

    BASELINE.md          the published tuples to reproduce, with sources
    CLAUDE.md            the rules that govern any work in this repo
    docs/baselines.md    measured hardware facts and the incidents behind them
    docs/checkpoints.md  what is actually on disk, read from the headers
    docs/engines.md      what each engine image can actually do
    docs/interconnect.md the RoCE finding, with the A/B that supports it
    docs/qwen-qsa.md     the long-context corruption question, and how to settle it
    src/sparkbench/      the harness: one rate formula, one tuple, one database
    scripts/             launchers, sweeps and the GB10 memory guard
    tests/               harness checks that need no GPU

## Running it

    scripts/sparkbench selftest                    # the formulas and the tuple rule
    scripts/audit-checkpoint.py <path> --verify    # what a checkpoint really is
    bash scripts/bring-up.sh scripts/launch-qwen-sglang.sh
    CONFIG_NOTE=<name> bash scripts/sweep-qwen.sh  # one configuration, every cell
    scripts/sparkbench report --out REPORT.md

`bring-up.sh` refuses to start a launcher that has no clean run behind it
unless `ATTENDED=1`, because on GB10 a container memory cap does not bound what
the driver allocates and an unproven launcher once needed a power cycle.

## Licence

MIT.
