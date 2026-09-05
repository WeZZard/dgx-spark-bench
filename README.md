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
below — twelve campaigns, three per configuration against one unchanged
server — zero failed requests, and 100% verbatim needle retrieval in every
agentic cell, including 26,500-token prompts at full concurrency. NCCL over
RoCE, confirmed during each campaign from the HCA's own `rx_write_requests`
counter rather than from a log.

**Everything here was re-measured on 2026-09-05.** One of the two GPUs had
been stuck at 29% of its clock since before the first campaign, and Qwen had
been the only model denied a prefix cache. Numbers taken before that date are
in `REPORT.md` for the record and should not be compared against these.

Configurations that did *not* work are in `REPORT.md` too, with their
failures — an fp8 KV cache on Qwen serves short prompts and then kills the
server on the first chunked one (`docs/qwen-qsa.md`), and three GLM sparse-
attention backends never reached a forward pass (`docs/glm-dsa.md`).

Every number below is the **median of three campaigns** against one
unchanged server, and every cell carries the spread between those three
samples. A lead is only called when it exceeds the worst spread among the
cells being compared — otherwise the table says the models are not separated
by this data, which for four of the eight rows is the honest answer.

**One decoding user** — the only cells where the three are directly
comparable, since GLM's ladder runs to 8 users and DeepSeek's and Qwen's to
6, each matching its own published concurrency, and users is a tuple field:

| workload | DeepSeek-V4-Flash-0731 | Qwen3.8-Flash-Next | GLM-5.3-Flash | lead | worst spread | |
|---|---:|---:|---:|---:|---:|---|
| short prompt | **72.4** (1.29x) | 42.6 (1.02x) | 41.1 (1.08x) | 1.70x | 1.29x | separated |
| code | 59.2 (**2.02x**) | 39.0 (1.03x) | 22.5 (1.37x) | 1.52x | 2.02x | not separated |
| agentic, 4k context | **41.8** (1.04x) | 32.1 (1.06x) | 24.9 (1.12x) | 1.30x | 1.12x | separated |
| agentic, 33k context | 22.5 (1.04x) | 21.0 (1.01x) | 17.7 (1.32x) | 1.07x | 1.32x | not separated |

At each model's own published concurrency:

| workload | DeepSeek @ 6 | Qwen @ 6 | GLM @ 8 | lead | worst spread | |
|---|---:|---:|---:|---:|---:|---|
| short prompt | 124.5 (1.20x) | 131.1 (1.11x) | – | 1.05x | 1.20x | not separated |
| code | 99.5 (1.04x) | **116.6** (1.07x) | 77.3 (1.08x) | 1.17x | 1.08x | separated |
| agentic, 4k context | 74.8 (1.06x) | **99.3** (1.16x) | 51.8 (1.03x) | 1.33x | 1.16x | separated |
| agentic, 33k context | 29.2 (1.06x) | 28.6 (1.10x) | 23.7 (1.02x) | 1.02x | 1.10x | not separated |

Served rate throughout, `--ignore-eos`, 512 tokens per request. DeepSeek leads
at one user; **Qwen leads at concurrency**, on `code` and on 4k agentic
context. Every earlier version of this table said otherwise, and every earlier
version was measured on a pair where one GPU ran at 29% clock and where Qwen
was the only model denied a prefix cache. Both are fixed; see the findings.

Full six-field tuples, with the KV pool each was measured against, are in
`REPORT.md` — generated, never hand-edited.

The configurations behind those numbers, and why each differs from the
published one:

All three run with the **prefix cache off**, which is new: until 2026-09-05
only Qwen disabled it and the other two silently took the engine default of
on. See the findings.

- **DeepSeek**: fp8_ds_mla KV, **517,484**-token pool, DSpark k=5
  probabilistic. The published `nvfp4_ds_mla` KV dtype does not exist in any
  vLLM build on these nodes, and these are the **text** model's numbers, not
  Vision-Exp's. A different tuple, recorded as one. A greedy-drafting arm was
  measured alongside and is within noise on eight of nine cells; it is in
  `REPORT.md` as `ds-greedy-v3`.
- **Qwen**: NVFP4 with NEXTN speculation over RoCE, bf16 KV, **161,728**-token
  pool. Enabling MTP cut the measured pool from 600,000 tokens with the same
  `--max-total-tokens`. `docs/speculation.md`, `docs/interconnect.md`.
- **GLM**: bfloat16 KV, **DFlash2 speculation**, **162,688**-token pool,
  `--mem-fraction-static 0.90`. The published fp8 KV path is unreachable for
  this checkpoint's architecture on GB10 and the DSA kernel needed a
  shared-memory fix to run at all (`docs/glm-dsa.md`). `docs/speculation.md`.

Findings, each written up with the evidence that supports it:

- **One of the two GPUs had been running at 29% of its clock, for every
  measurement this project ever took.** Under an identical matmul the head
  node read **690 MHz and 17 W** against its peer's **2340 MHz and 91 W** — on
  identical hardware, identical driver 580.173.02, both in performance state
  P0, with no throttle reason active and no clock lock set. `nvidia-smi`
  reported nothing wrong, and telemetry showed the node had been like that for
  its whole 34-hour uptime. In TP2 lock-step the slow rank sets the pace, so
  every campaign in this repo before 2026-09-05 was measured on half a machine.
  A reboot restored it; the same configurations then read **1.1x to 2.2x**
  faster, the largest gains on the agentic cells. `scripts/check-clocks.sh` now
  refuses to start a sweep on a node that will not clock up, and it condemns a
  node only when clock **and** power agree — `clocks.sm` on GB10
  intermittently reports the idle value under full load, and believing it
  alone made that guard refuse a healthy node twice.
- **Qwen was the only model measured without a prefix cache.** It passed
  `--disable-radix-cache`, copied from its published recipe; DeepSeek and GLM
  passed nothing and took the engine default, which is on in both vLLM and
  SGLang. None of Rule 1's six fields records this. Correcting it is most of
  why Qwen now leads at concurrency. It also corrupted the measurement: every
  sample replays the same seeded prompt, so a sample that hits the cache times
  a lookup rather than a prefill — DeepSeek's `agentic-33k @ 1` read 45.5 tok/s
  at a **508 ms** time-to-first-token on a 27,000-token prompt against 22.6 at
  12,263 ms for the samples that ran one, and the cell swung 2.0x on which
  sample got the hit. All three now disable it, and that cell reproduces to
  1.04x.
- **Engines compile kernels inside the first measured cell, and both say so.**
  vLLM prints *"TileLang JIT compilation during inference … consider extending
  warmup to cover this shape/config"*; SGLang prints *"Serving-time
  compilation can stall the engine"*. The warmup here sent a burst of 10-token
  prompts, so `short @ 6` and `agentic-4k @ 6` compiled while being measured
  and carried 35 s and 50 s inside their time-to-first-token. Across two nodes
  it is worse than one: each rank has its own container and its own cache, so
  the same kernel compiles twice and the ranks serialise — one compiles while
  the other spins in NCCL, which is what a cool head node beside a hot worker
  looks like. Warmup now drives every `(workload, users)` pair at full
  generation length through the same generator and seed, and the JIT caches are
  mounted per node instead of being deleted with the container. Concurrency
  spreads went from 1.5x to **1.04x**, and bring-up from 1,274 s to **502 s**.
- **Draft sampling is worth nothing here, and an earlier 2.5x claim was
  withdrawn.** Changing DSpark's `draft_sample_method` from `probabilistic` to
  `greedy` first appeared to be worth 1.5x–2.5x. That was one sample against
  one, on the two cells that turn out to be the noisiest in the set. Properly
  sampled, greedy is within 1.05x on eight of nine cells; only `code @ 1`
  differs, and that cell is unreliable in both configurations.
- **One cell is still not understood.** DeepSeek's `code @ 1` swings 2.0x, and
  it is an acceptance collapse, not a queueing artefact: across three samples
  of one unchanged server the DSpark acceptance rate reads 82.8%, **28.0%**,
  76.6%, with time-to-first-token flat at ~340 ms and decode rate halving with
  it. It happens under **greedy** drafting too — 94.8%, **37.4%**, 98.8% — so a
  stochastic draft does not explain it. Same prompt, same seed, temperature 0.
  The cause is not established, so that row is published with its spread and
  is not used to separate the models.

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
- **GLM's drafter is worth 2.75x at one user and nothing at eight** — measured
  before the clock fix, and not yet re-measured. DFlash2 raised short-prompt
  served rate from 10.3 to 28.3 tok/s, with the ratio falling monotonically
  with concurrency until the 8-user code cell was a dead heat. Both halves of
  that comparison were taken on the crippled pair, so the ratio may survive
  and the absolute numbers do not; the no-drafter arm needs re-running.
  Measured acceptance is 39.6%, not the published 74.1%, on workloads that are
  code and agentic context rather than short continuations. It costs 5.8x of
  the KV pool. `docs/speculation.md`.
- **The published DeepSeek row does not say what it appears to say.** Its
  "NVFP4" is the KV cache, its headline throughput belongs to the text model
  rather than the vision one, and its engine version matches neither source.
  `BASELINE.md`, "What the sources actually say".
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

- **Run-to-run spread is per cell, and it is now small — but it was not.**
  Every headline number is the median of three campaigns against one unchanged
  server, and the tables print each cell's spread. In the current
  configuration Qwen's worst cell is 1.16x, GLM's 1.37x, and DeepSeek's are
  all under 1.30x except `code @ 1`. Any comparison finer than the relevant
  cell's spread is not supported by this data, which is why four of the eight
  headline rows are marked *not separated*.

  Getting there took removing three separate causes, each of which had been
  read as noise: a GPU stuck at 29% clock, kernels compiling inside the first
  measured cell, and a prefix cache that turned a 27,000-token prefill into a
  508 ms lookup on whichever sample won the eviction race. Before those fixes
  the same repeats measured `short @ 6` at 46.9, 105.2 and 94.4 tok/s and
  `agentic-4k @ 6` at 24.7, 50.4 and 54.8 — 2.24x and 2.22x. This section
  claimed "up to 1.3x" for weeks, on the strength of paired passes that
  happened to miss both slow modes.

- **`code @ 1` on DeepSeek still moves 2.0x, and it is not queueing.** Across
  three samples of one unchanged server, DSpark acceptance reads 82.8%, 28.0%,
  76.6% with time-to-first-token flat at ~340 ms and decode rate halving in
  step. It happens under greedy drafting too (94.8%, 37.4%, 98.8%), so a
  stochastic draft does not explain it, and the prompt, seed and temperature
  are identical across samples. The cause is not established. That row is
  published with its spread and is not used to separate the models.

Served rate is output tokens over the whole wall clock, so a wait to start
counts against it. That is deliberate — it is what a caller experiences — and
it is why these cells are sensitive to anything that delays a first token,
which is how all three causes above were found.

## Licence

MIT.
