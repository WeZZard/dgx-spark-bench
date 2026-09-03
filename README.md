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
configurations to reproduce, with sources. No measurement of our own has been
taken yet under these rules.

## Layout

    BASELINE.md        the published tuples to reproduce, with sources
    CLAUDE.md          the rules that govern any work in this repo
    docs/baselines.md  measured hardware facts and the incidents behind them
    scripts/lib/       shared shell library, including the GB10 memory guard

## Licence

MIT.
