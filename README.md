# dgx-spark-bench

Reproducible serving benchmarks for three open-weight MoE models on a pair of
NVIDIA DGX Spark machines (GB10, sm_121, 121 GiB unified memory each, joined
back-to-back at 200 GbE).

- **DeepSeek-V4-Flash** — served here as the text `0731` checkpoint, and also
  as `Vision-Exp`, the checkpoint the published row actually names, loaded
  text-only through a patched weight loader
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

Every cell in the tables below is generated with `--ignore-eos`, so each
request emits exactly 512 tokens. Without that, served rate moves with how
long the answer happened to be: four samples of one DeepSeek cell produced
179, 266, 279 and 512 tokens and read 5.8, 8.1, 8.3 and 13.9 tok/s, while
time-to-first-token stayed flat within 2%. Same work, different denominator.

`REPORT.md` holds earlier configurations that predate that flag, and two
correctness probes run at 64 tokens, so it prints the output-token count
beside every row. Rows with different counts in that column are not
comparable on served rate and are not meant to be.

## Status

All three models serve on the pair. In the three shipped configurations
below: zero failed requests, and 100% verbatim needle retrieval in every
agentic cell, including 26,500-token prompts at full concurrency. NCCL over
RoCE, confirmed during each campaign from the HCA's own `rx_write_requests`
counter rather than from a log.

Configurations that did *not* work are in `REPORT.md` too, with their
failures — an fp8 KV cache on Qwen serves short prompts and then kills the
server on the first chunked one (`docs/qwen-qsa.md`), and three GLM sparse-
attention backends never reached a forward pass (`docs/glm-dsa.md`).

**One decoding user** — the only cells where the three are directly
comparable, since GLM's ladder runs to 8 users and DeepSeek's and Qwen's to
6, each matching its own published concurrency, and users is a tuple field:

| workload | DeepSeek-V4-Flash-0731 | Qwen3.8-Flash-Next | GLM-5.3-Flash | lead |
|---|---:|---:|---:|---:|
| short prompt | **44.1** | 26.5 | 28.3 | 1.56x |
| code | **32.8** | 23.9 | 15.4 | 1.37x |
| agentic, 4k context | 26.2 | 22.7 | 15.7 | 1.15x |
| agentic, 33k context | 13.9 | 11.0 | 10.8 | 1.26x |

At each model's own published concurrency:

| workload | DeepSeek @ 6 | Qwen @ 6 | GLM @ 8 | lead |
|---|---:|---:|---:|---:|
| short prompt | 89.6 | 96.6 | – | 1.08x |
| code | 40.2 | **87.3** | 45.3 | 1.93x |
| agentic, 4k context | 54.6 | 43.4 | 33.8 | 1.26x |
| agentic, 33k context | 17.7 | 11.2 | 14.5 | 1.22x |

Served rate throughout, one sample per cell. **Lead** is the top cell over
the runner-up. **The bold marks are currently unsupported and are being
re-derived**: they were set against a single 1.3x threshold, and repeating one
DeepSeek campaign three times against an unchanged server has since measured
per-cell spreads from 1.01x to 2.24x, with the two worst on concurrency cells
(see the end of this file). Error bars for GLM and Qwen are being measured the
same way; the marks will be redone once all three models have them, from
medians rather than from single samples.

Adding GLM's drafter moved one row in the direction people usually do not
report: at 33k context and full concurrency GLM went 13.5 → 14.5, narrowing
DeepSeek's lead from 1.31x to 1.22x. Under the old single 1.3x threshold that
put the row below the line; under the per-cell spreads now measured it may be
the opposite, because `agentic-33k @ 6` is one of the *stable* cells — three
DeepSeek samples reproduce it to within 1.03x. Whether 1.22x clears GLM's own
spread on that cell is what GLM's repeats are being run to answer.

Full six-field tuples, with the KV pool each was measured against, are in
`REPORT.md` — generated, never hand-edited.

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
- **GLM**: bfloat16 KV, **DFlash2 speculation**, 138,304-token pool,
  `--mem-fraction-static 0.90`. The published fp8 KV path is unreachable for
  this checkpoint's architecture on GB10 and the DSA kernel needed a
  shared-memory fix to run at all (`docs/glm-dsa.md`). The drafter is worth
  **2.75x at one user and nothing at eight**, and costs 5.8x of the KV pool —
  the no-speculation rows are still in `REPORT.md` as `glm-eos-warm` and are
  the better configuration above four users. `docs/speculation.md`.

Findings, each written up with the evidence that supports it:

- **The checkpoint labelled FP8 has 4-bit experts, and so does the one the
  published row actually names.** All three DeepSeek checkpoints in play —
  `0731`, nvidia's NVFP4 build, and `Vision-Exp` — carry exactly **138.00 GiB**
  of packed 4-bit expert weights. The "NVFP4" one is larger only because
  block-16 scaling stores twice the scale bytes of block-32. So the published
  49.6–53.8 against 80.1 tok/s is not 8-bit against 4-bit and cannot be a
  weight-bandwidth result; it is a kernel-path result wearing a precision
  label. Vision-Exp was settled without downloading it, by reading its 48
  safetensors headers over HTTP — eight megabytes against 156 GiB.
  `docs/checkpoints.md`, `scripts/audit-remote-checkpoint.py`.
- **Vision-Exp holds every one of `0731`'s 72,317 tensor names, and none of
  its weights.** Identical dtypes, identical byte counts, 316 extra tensors
  for the vision tower — and every sampled expert tensor hashes differently,
  at layers 3, 20 and 40. Same architecture, same quantization, separately
  trained. So the `0731` numbers here are not Vision-Exp numbers, however
  much the structure invites that shortcut. `scripts/compare-checkpoints.py`.
- **The container could not see the RDMA devices**, so a 200 GbE link
  cross-connected between the two nodes was carrying NCCL over TCP. Fixing it
  was worth 1.74x across every cell measured. `docs/interconnect.md`.
- **GLM-5.3-Flash's sparse-attention kernel does not fit on GB10.** The stock
  tilelang kernel asks for 169,984 bytes of dynamic shared memory against a
  101,376-byte limit. The working tuning was found by compiling the kernel
  standalone in about a minute, after three seventeen-minute weight loads had
  each ended in a different failure. `docs/glm-dsa.md`.
- **GLM's drafter is worth 2.75x at one user and nothing at eight.** DFlash2
  raises short-prompt served rate from 10.3 to 28.3 tok/s, and the ratio falls
  monotonically with concurrency until the 8-user code cell is a dead heat.
  Measured acceptance is 39.6%, not the published 74.1%, on workloads that are
  code and agentic context rather than short continuations. It costs 5.8x of
  the KV pool. `docs/speculation.md`.
- **The published DeepSeek row does not say what it appears to say.** Its
  "NVFP4" is the KV cache, its headline throughput belongs to the text model
  rather than the vision one, and its engine version matches neither source.
  `BASELINE.md`, "What the sources actually say".
- **Two concurrency cells swing 2.2x on an unchanged server, and it is all
  time-to-first-token.** Three campaigns run back to back against one DeepSeek
  process, nothing restarted between them:

  | cell | s1 | s2 | s3 | spread | TTFT p50, ms |
  |---|---:|---:|---:|---:|---|
  | short @ 6 | 46.9 | 105.2 | 94.4 | **2.24x** | 35,364 / 1,278 / 1,076 |
  | agentic, 4k @ 6 | 24.7 | 50.4 | 54.8 | **2.22x** | 50,155 / 8,996 / 12,639 |
  | prose @ 1 | 54.8 | 36.2 | 50.1 | 1.51x | ~450 |
  | code @ 1 | 48.0 | 33.7 | 46.1 | 1.42x | ~480 |
  | code @ 6 | 73.2 | 72.4 | 73.1 | 1.01x | ~1,270 |
  | agentic, 33k @ 6 | 19.7 | 19.9 | 20.4 | 1.03x | ~64,600 |

  Decode rate barely moves across those samples; the slow ones are requests
  that waited tens of seconds to start. So the noise floor is **per cell, not
  one number**, and two of the four concurrency cells cannot support a
  comparison finer than about 2.2x. `SWEEP_REPEATS=N` in the sweep scripts is
  what produces this.
- **A 2.5x speculation result was measured here and then withdrawn, by the
  above.** Changing DSpark's `draft_sample_method` from `probabilistic` to
  `greedy` appeared to be worth 1.5x-2.5x on short and code cells. It was one
  sample against one sample, and the two cells carrying the largest claimed
  gains are the two that swing 2.2x by themselves. Against a properly sampled
  baseline — probabilistic n=4, greedy n=2, medians:

  | workload | probabilistic | greedy | ratio |
  |---|---:|---:|---:|
  | code @ 1 | 39.9 | 51.4 | 1.29x |
  | code @ 6 | 72.7 | 66.5 | 0.91x |
  | short @ 6 | 92.0 | 45.9 | 0.50x — both greedy samples in the slow mode |
  | short @ 1 | 47.9 | 49.0 | 1.02x |
  | agentic, 4k @ 1 | 26.5 | 27.7 | 1.04x |
  | agentic, 33k @ 1 | 13.7 | 13.6 | 0.99x |
  | agentic, 33k @ 6 | 19.8 | 20.7 | 1.05x |

  What survives is a possible 1.29x on `code @ 1` and nothing else; the
  agentic rows this project cares about are flat. The Vision-Exp
  greedy-against-probabilistic pair in `REPORT.md` has the same defect and is
  not read as a result either. `greedy` remains the value BASELINE.md's
  published configuration uses, which is why it was worth measuring.
- **Acceptance had to be measured per cell before any of that could be seen.**
  A campaign-wide average runs through a spread the source itself documents —
  ~78% on code, ~56% on mixed agent traffic, ~34% on prose — so `0731` reading
  52.8% and Vision-Exp reading 33.5% told nothing apart, and a preflight
  threshold set at 40% between them refused a healthy server. The counters are
  monotonic totals, so bracketing each cell gives that cell's acceptance and
  nothing else. `docs/speculation.md`.
- **Vision-Exp serves, and it takes two loader patches rather than one.** The
  stock image cannot place 313 of the checkpoint's tensors and raises on the
  first. Skipping them in `DeepseekV4ForCausalLM` is not enough: the DSpark
  draft model reads the same files afterwards through its own loop, with its
  own skip logic and a bare `params_dict[name]`, so a clean target load is no
  evidence at all about the draft. Both skips also have to be written in the
  name the loader sees *after* the weights mapper runs, not the name in the
  file. `docs/checkpoints.md`.

## Layout

    BASELINE.md          the published tuples to reproduce, with sources
    CLAUDE.md            the rules that govern any work in this repo
    REPORT.md            every recorded tuple, generated from the database
    docs/baselines.md    measured hardware facts and the incidents behind them
    docs/checkpoints.md  what is actually on disk, read from the headers
    docs/engines.md      what each engine image can actually do
    docs/glm-dsa.md      which GLM sparse-attention kernel can run on GB10
    docs/interconnect.md the RoCE finding, with the A/B that supports it
    docs/qwen-qsa.md     the long-context corruption question, and the fp8 KV crash
    docs/speculation.md  what MTP costs in KV pool, and what it buys
    src/sparkbench/      the harness: one rate formula, one tuple, one database
    scripts/             launchers, sweeps, the GB10 memory guard and the checks
    docker/              the two derived images, and why each exists
    tests/               harness checks that need no GPU

## Running it

    scripts/sparkbench selftest                    # the formulas and the tuple rule
    scripts/audit-checkpoint.py <path> --verify    # what a checkpoint really is
    scripts/audit-remote-checkpoint.py <repo> --revision <sha> --verify
    scripts/compare-checkpoints.py <path> <repo> --revision <sha>
    bash scripts/build-sglang-glm53-gb10.sh        # GLM needs a patched kernel
    bash scripts/build-vllm-ray.sh                 # DeepSeek needs ray in the image
    CONFIG_NOTE=<name> bash scripts/sweep-glm.sh   # one configuration, every cell
    scripts/sparkbench report --out REPORT.md

The two remote tools read safetensors headers over HTTP range requests, so
they answer "what is this checkpoint" and "are these the same weights" for
megabytes rather than for the checkpoint's size. Both take a 40-character sha
and refuse a branch name.

`bring-up.sh` refuses to start a launcher that has no clean run behind it
unless `ATTENDED=1`, because on GB10 a container memory cap does not bound what
the driver allocates and an unproven launcher once needed a power cycle.

Two things worth knowing before reading any number here:

- **Repeating a campaign against an unchanged server moves some cells by
  2.2x, and not in a consistent direction.** This bullet said "up to 1.3x"
  until 2026-09-05, when three back-to-back campaigns against one DeepSeek
  process measured `short @ 6` at 46.9, 105.2 and 94.4 tok/s and
  `agentic-4k @ 6` at 24.7, 50.4 and 54.8 — 2.24x and 2.22x, with identical
  token counts and nothing restarted. The 1.3x came from paired passes that
  happened not to include either cell's slow mode.

  The spread is **per cell**, not one number for the repo: `code @ 6` and
  `agentic-33k @ 6` reproduce to within 1.03x across the same three samples,
  while `code @ 1` and `prose @ 1` move by 1.42x and 1.51x. Any comparison
  finer than the relevant cell's spread is not supported by this data.

  Earlier observations, kept because they are the same phenomenon seen with
  two samples instead of three: two passes over one GLM server read higher
  the second time in 8 of 9 cells (8-user code 36.3 → 45.7) while every 1-user
  cell moved by 0.7 or less; the same treatment of one Qwen server went the
  *other* way in 7 of 9 cells (6-user short 96.6 → 65.2, decode 16.8 → 11.2).
- **In the DeepSeek cells, the swing is time-to-first-token, not decoding.**
  Across the three samples above, decode rate for `short @ 6` moved 16.4 →
  19.5 tok/s while TTFT p50 moved 35,364 → 1,076 ms. A slow sample is six
  requests that waited tens of seconds to start, not a server generating
  slowly. Served rate is this project's formula precisely because that wait is
  real to a caller, so these are not artefacts to be excluded — but a single
  sample of such a cell says nothing, and every headline number here is
  currently a single sample. The cause of the bimodality is not established.
  Qwen's case above is different again: there decode rate moved with served
  rate and TTFT improved, so it is not the same mechanism.

The three-sample repeats predate `--ignore-eos` and cannot be used to size
this, because output length varied between samples and served rate divides by
it — one DeepSeek prose cell emitted 512, 144 and 512 tokens for 46.0, 18.1
and 36.2 tok/s. Only the paired `--ignore-eos` passes above hold the
denominator fixed.

## Licence

MIT.
