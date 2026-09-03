# The configurations to reproduce

This file is the target. Every row is a published two-node DGX Spark result
with the source that reports it. Nothing here has been measured by this
project yet; the point of the project is to reproduce these numbers first and
only then try to beat them.

Read `CLAUDE.md` before running anything. Two rules govern this repo: a run is
recorded as a complete six-field tuple or not at all, and there is one speed
formula for the whole project.

## The tuple

Every measurement names all six fields. A number without its tuple is not a
result.

| field | why it is in the tuple |
|---|---|
| **model** | the exact checkpoint and revision, not the family name |
| **weight** | the weight quantization: FP8, NVFP4, EXL3 4bpw, GGUF quant |
| **KV cache** | the KV cache dtype. This is the field this project got wrong: no dtype was ever set, and the KV pool was a fraction of what every good recipe reports |
| **engine** | engine, version, and image, because the fixes live in specific builds |
| **decoding users** | how many requests were being served at once |
| **KV pool** | the measured pool in tokens, which is what caps the users |

## The chosen baseline, one configuration per model

These are the configurations with the best published measured numbers on two
DGX Sparks. They are the target to reproduce. Whether a format is "native" to
the model is not a criterion here; the measured number is.

| model | weight | KV cache | engine | users | KV pool | best published |
|---|---|---|---|---:|---:|---|
| DeepSeek-V4-Flash-Vision-Exp | NVFP4 | `nvfp4_ds_mla`, block 256 | vLLM 0.25.2.dev0 + DSpark k=5-6 | 6 | 2.79-3.07M | **84.3 tok/s** peak single-stream, **197.3** aggregate at 6 |
| Qwen3.8-Flash-Next | NVFP4 | `--max-mamba-cache-size 97` | SGLang `radixark/sglang-qwen38flashnext:sm121-qsa-mrope1` | 6 | 600,000 pinned | **69.7 tok/s** peak single-stream |
| GLM-5.3-Flash | LibertAIDAI NVFP4 | **fp8** (SGLang PR #36904) | SGLang PR #36507 + #36904 + DFlash2 drafter | 8 | mamba-governed | **37.3 tok/s** single-stream, **83.5** aggregate at 12 |

For GLM there is a second path with a much larger pool but slower single
streams — EXL3 4bpw weights with a 288-byte-a-token NVFP4 KV cache reaching a
2,196,850-token pool and 67.6 tok/s aggregate 12-way. Measure it only after
the SGLang path above is reproduced.

Every published figure in this file is a **decode rate**. This project reports
**served rate**. Do not put them in one table until a matching short-prompt
cell has been measured here; see rule 2 in `CLAUDE.md`.

The full detail for each configuration follows.

## DeepSeek-V4-Flash-Vision-Exp

| field | value |
|---|---|
| model | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` rev `86f746b36186f0e567729a5c06a8c918caba82a9` |
| weight | published as "FP8 (official)" — **unverified for this checkpoint; the label is wrong on its `0731` sibling**, see below |
| KV cache | `nvfp4_ds_mla`, block size 256 |
| engine | vLLM `0.25.2.dev0+g752a3a504.d20260714` (anemll 0.1.1), DSpark k=6 greedy |
| decoding users | `max_num_seqs 6` |
| KV pool | **3,068,415 tokens** at TP2; 7,092,846 at TP4 |
| decode rate | 49.6 → 53.8 tok/s at depth (nvfp4 KV against fp8 KV) |
| quality | 4/4 needle retrieval verbatim at 33K / 131K / 262K / 524K |

Also: `gpu_memory_utilization 0.835`, `max_num_batched_tokens 4096`,
`tensor_parallel_size 2`, 1,048,576-token context.

A sibling build on **NVFP4 weights** reports single-stream 80.1 tok/s on
"count to 300", 51.8 on code and **33.2 on prose**, with aggregate C1 61.0 /
C2 91.7 / C4 151.1 / C6 197.3. The spread is DSpark acceptance varying with
content — ~78% on structured code, ~34% on prose, ~56% on mixed agent traffic
— not tuning. Judge any agentic result against the prose and mixed figures,
never against "count to 300".

**The "FP8" label is wrong on the sibling checkpoint, and that probably
changes what this comparison means.** Vision-Exp is not on disk here, so it has
not been audited. Its text sibling has: `scripts/audit-checkpoint.py` finds
**138.00 GiB of packed 4-bit experts in both** DeepSeek checkpoints we hold.
`deepseek-ai/DeepSeek-V4-Flash-0731` — the one the same sources call FP8 —
carries MXFP4 experts (E8M0 scales, block 32), and
`nvidia/DeepSeek-V4-Flash-nvfp4-DSpark` carries NVFP4 experts (E4M3 scales,
block 16). `0731`'s own config declares `expert_dtype: fp4`.

If Vision-Exp is built the same way, then 49.6–53.8 against 80.1 is **not**
8-bit against 4-bit and cannot be explained by weight bandwidth, because both
builds would move the same expert bytes. The difference would have to be in
the kernel path, which makes `VLLM_USE_B12X_MOE` and DSpark acceptance the two
things to separate by measurement. **Audit Vision-Exp with
`scripts/audit-checkpoint.py --verify` the moment it is downloaded, before any
run**, and settle this rather than assuming either way. Evidence for the
sibling in `docs/checkpoints.md`.

Two failures in this stack are silent:

- **Patch 4**, the DSpark draft shared-expert loader fix. Without it decode
  runs at about half speed; twelve tensors load uninitialized and nothing is
  logged at INFO. A four-node report puts the same fix at 23-26 → ~90 tok/s
  and DSpark acceptance at 25% → 60-80%. There is a `check-patch4.sh`
  preflight in the source repo.
- **`VLLM_USE_B12X_MOE=1`**, described as "the entire speed difference"; unset,
  it falls back to a slower kernel without saying so.

Sources:
- https://forums.developer.nvidia.com/t/deepseek-v4-flash-vision-exp-is-released-as-open-weights/381911
- https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
- https://forums.developer.nvidia.com/t/4-node-dgx-spark-cluster-with-deepseek-v4-flash-0731-dspark-benchmark-prefill-2-500-t-s-decode-90-t-s/378878

### What the sources actually say, checked 2026-09-04

The two sources above were re-read against the row at the top of this file.
Five things do not match, and three of them change the tuple.

**1. "NVFP4" in this row is the KV cache, not the weights.** The repository is
named `...-1M-NVFP4-KV-...`, its launcher passes `--kv-cache-dtype
nvfp4_ds_mla`, and it passes **no `--quantization` flag at all** -- so the
weights are whatever the checkpoint carries, which for `0731` this project has
measured as MXFP4 experts plus FP8 attention (`docs/checkpoints.md`). The
`weight` column at the top of this file should not read NVFP4 on the strength
of that repository name.

**2. The 84.3 / 197 figures are the text model, not Vision-Exp.** The
repository reports Vision-Exp as *1.03 s end-to-end for a 336x336 image and a
26-token answer*. The decode figures -- 78.4 tok/s peak at 98.9% acceptance,
84.3 structured, 67.6 mean, 197 aggregate at c6 -- are quoted for the text
`0731` checkpoint. The row therefore pairs a Vision-Exp model field with text
throughput.

**3. The engine version does not match.** Both sources say
`0.21.1rc1.dev339+g1967a5627bc3`. This file says
`0.25.2.dev0+g752a3a504.d20260714`. Rule 1 puts the version in the tuple
"because the fixes live in specific builds", so this is not a cosmetic
difference: reproducing against the wrong version is reproducing nothing.

**4. B12X is a flag as well as an environment variable.** The four-node thread
uses `--moe-backend b12x`. Setting only `VLLM_USE_B12X_MOE=1` may not be
enough, and this project's own vLLM image defines **neither** -- see
`docs/engines.md`.

**5. The published aggregates lean on prefix-cache hits.** The four-node thread
gives 208.74 tok/s aggregate at c6 **with cache hits**, and says plainly that
cold 150K unique prompts performed significantly worse. This project's agentic
workload sends distinct prompts by construction. A lower number here is
therefore a different measurement condition, not a tuning gap, and the
like-for-like comparison has to be built rather than assumed.

Other flag differences, smaller but worth carrying into the launcher:
`--max-num-seqs 12` (not 6), `--gpu-memory-utilization 0.85` (not 0.835),
`--max-num-batched-tokens 8192` (not 4096), and
`--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'`.
Environment: `VLLM_TRITON_MLA_SPARSE=1`, `VLLM_USE_FLASHINFER_SAMPLER=1`,
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`, `NCCL_SOCKET_IFNAME=enp1s0f0np0`.
Patch 4 verifies as "6 shared_experts matches expected; grep for 'Skipping'
should return empty".

None of this is a reason to discard the row. It is a reason to reproduce the
configuration the sources describe rather than the one this file recorded.

## Qwen3.8-Flash-Next

| field | value |
|---|---|
| model | Qwen3.8-Flash-Next (125B-A3B hybrid MoE), `qwen4_exp` |
| weight | NVFP4 |
| KV cache | not stated in the source; `--max-mamba-cache-size 97` governs the recurrent state |
| engine | **SGLang**, image `radixark/sglang-qwen38flashnext:sm121-qsa-mrope1`, TP2 |
| decoding users | 6 concurrent agents with tool-carrying load |
| KV pool | **600,000 tokens** pinned for safety; 1,049,344 reachable unpinned |
| decode rate | 69.7 peak (code), ~50 typical, ~20 without MTP |

Flags: `--max-total-tokens 600000 --mem-fraction-static 0.80
--cuda-graph-max-bs 8 --disable-cuda-graph-padding --ple-offload-embedding
--disable-radix-cache --max-mamba-cache-size 97 --sampling-backend pytorch
--context-length 262144`.

Mandatory before launch: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`.
Unified memory page-cache starvation otherwise.

**Warning carried from the source:** the SM121 QSA kernel guard causes silent
corruption past about 120K context. The successor is the Triton packed-varlen
kernel, `pr36845.diff`. Do not run the QSA guard at long context.

Note `--disable-radix-cache` in that recipe. Their workload is not agentic. A
prefix cache is worth roughly 3x on multi-turn agent traffic, so this flag
must be re-decided here rather than copied.

Sources:
- https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark
- https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks
- https://github.com/x00byte/Qwen3.8-Flash-Dual-Spark-Recipe

## GLM-5.3-Flash

Two paths. The SGLang one is the closer fit; the EXL3 one has the larger pool.

### SGLang path

| field | value |
|---|---|
| model | GLM-5.3-Flash (320B total / 18B active MoE) |
| weight | LibertAIDAI NVFP4 |
| KV cache | **fp8** (SGLang PR #36904); bf16 before that |
| engine | SGLang PR #36507 + #36904, TP2 |
| drafter | **incoai DFlash2** block-diffusion |
| decoding users | `--max-running-requests 8` |
| KV pool | governed by mamba state; about 5x per-request amplification |
| decode rate | 29.3 tok/s code with DFlash2; **37.3 with fp8-KV tuning** |

Concurrency, aggregate, DFlash2 with fp8 KV: c1 37.3 · c2 49.5 · c4 51.7
(12.9 a stream) · c8 78 · c12 83.5. Without speculation: c1 14.5 · c2 27.1 ·
c4 36.8 · c8 55.1 · c12 55.0.

Flags: `--max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16
--mem-fraction-static 0.92 --enable-multimodal`.

Their DSA tilelang setting at 8-token verify is **`block_I=32,
num_stages=1, threads=128`**.

Four GB10 fixes they report: `SGLANG_HOST_IP` per rank or multi-node
shm_broadcast hangs; mamba pool needs `per_req*(1+D)` plus ~5x amplification;
DSA tilelang shared-memory overflow at 8-token verify; and the upstream
`residual=None` crash in the DFLASH adapter (sglang #36755).

A separate report puts DFlash2 at 46.9 tok/s single-stream against 21.8 for
MTP-4 — 2.15x — at 74.1% draft acceptance.

### EXL3 path

| field | value |
|---|---|
| model | `neko-legends/GLM-5.3-Flash-Uncensored-EXL3` |
| weight | EXL3 4 bits per weight |
| KV cache | NVFP4, **288 bytes a token**, per-16 e4m3 block scales |
| engine | MiaAI-Lab EXL3 kit (vLLM fork), custom FlashInfer sparse-MLA kernel |
| decoding users | 16 slots |
| KV pool | **2,196,850 tokens** — eight full 262K sessions |
| decode rate | 20-24 tok/s single-stream, peak 27-29; 67.6 aggregate at 12-way |
| quality | 59/60 on GSM8K |

Prefill 680-705 tok/s. 900K context a request. Host-side fixes for unified
memory livelock, a memory watchdog that sends SIGTERM below 0.75 GiB free,
and `vm.swappiness` tuning.

Sources:
- https://forums.developer.nvidia.com/t/glm-5-3-flash-on-2x-dgx-spark-nvfp4-kv-cache-288-b-token-2-2m-token-pool-16-way-serving-900k-context-full-recipe/382120
- https://forums.developer.nvidia.com/t/glm-5-3-flash-dflash2-on-sglang-2x-dgx-spark-first-sglang-path-recipe-4-gb10-fixes-honest-numbers-concurrency-curve/381703
- https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark
- https://huggingface.co/randomllama/GLM-5.3-Flash-DFlash2-SGLang-2x-DGX-Spark

## What the previous attempt got wrong

Recorded so the same ground is not lost twice. The measurements themselves
were taken with one load generator throughout; what failed was the reporting
and the configuration.

1. **No KV cache dtype was ever set, on any model.** Every recipe above sets
   one. The pools that resulted were a fraction of the published ones:
   510,464 against 3,068,415 on DeepSeek; 100,672-121,600 against a
   2,196,850-token pool on GLM. The pool caps the users, and the users cap
   aggregate throughput, so this one omission bounded everything.
2. **GLM ran with no drafter at all.** DFlash2 is measured at 2.15x.
3. **Two speed formulas were used and neither was named.** Served rate (output
   tokens over the whole wall clock) and decode rate (1000 over the mean gap
   between tokens) differ by more than twofold on a long prompt. The same run
   was reported as 22.6 and 40.3 tok/s in one document.
4. **Published figures were compared against ours without matching
   conditions.** Theirs used a 22-token prompt; ours sent 2,457.
5. **A launcher that had never produced a successful run was scheduled
   unattended**, and hung both machines for ninety minutes.
   `scripts/lib/memguard.sh` is carried over from that incident and its
   header explains why a cgroup cap does not help on GB10.
