# What is actually on disk

Rule 1 needs a `weight` field that names the real quantization of the
checkpoint being served. Directory names do not supply it and neither do model
cards. Every claim below was measured with `scripts/audit-checkpoint.py`, which
reads the safetensors headers only — kilobytes per shard, no GPU, no load.

Audited 2026-09-04 on the head node against `/home/station/models/hf`.

## The store

| directory | repo | revision | on disk | weight, as measured |
|---|---|---|---|---|
| `dsv4-flash-nvfp4-dspark` | `nvidia/DeepSeek-V4-Flash-nvfp4-DSpark` | `7ac53121c19350518c374b67e3883ceb189fb26e` | 164.0 GiB | **NVFP4 experts** (block 16), FP8 E4M3 attention |
| `dsv4-flash-fp8` | `deepseek-ai/DeepSeek-V4-Flash-0731` | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` | 155.4 GiB | **MXFP4 experts** (block 32), FP8 E4M3 attention |
| `qwen38-flash-next-nvfp4` | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | 125.9 GiB | NVFP4 experts (block 16), BF16 attention, FP8 PLE |
| `glm53-flash-nvfp4-ref` | `LibertAIDAI/GLM-5.3-Flash-NVFP4` | `aa28e1f54130286c95fee10d0705c74ce8743734` | 181.3 GiB | NVFP4 experts (block 16), BF16 attention |
| `glm53-flash-fp8` | `zai-org/GLM-5.3-Flash` | `04c4e9e95c5da8862dced7e5056455116f83a7e0` | 129.8 GiB | FP8 E4M3 experts — genuinely 8-bit |
| `glm53-dflash2-draft` | `incoai/GLM-5.3-Flash-DFlash2` | `dc77ff1c99eeb2df044ee3d4f0094eb033fee410` | 2.2 GiB | BF16, `DFlash2DraftModel`, 5 layers |

Revisions come from the `.cache/huggingface/download/*.metadata` sidecars that
`hf download` leaves in each directory; the first line of each is the commit
the file was fetched at. They agree with `models/hub/models--*/refs/main`.

## The 165 GiB "NVFP4" checkpoint is genuine 4-bit

It was flagged for being *larger* than the 156 GiB checkpoint labelled FP8,
which a 4-bit checkpoint should not be. It is genuine, and the reason it is
larger is exact:

| | `dsv4-flash-nvfp4-dspark` | `dsv4-flash-fp8` |
|---|---|---|
| packed expert weights | **138.00 GiB** `U8` | **138.00 GiB** `I8` |
| expert scales | 17.25 GiB `F8_E4M3` | 8.62 GiB `F8_E8M0` |
| derived block size | **16.00** | **32.00** |
| format | NVFP4 | MXFP4 |

The weight payload is byte-identical in size — 138.00 GiB either way, because
both store two 4-bit values per byte. The entire 8.63 GiB difference is scale
bytes, and it is a factor of exactly 2.00 because NVFP4 scales blocks of 16
values where MXFP4 scales blocks of 32. Reading one expert tensor pair shows
the same thing directly:

```
nvfp4:  layers.3.ffn.experts.0.w1.weight        U8       [2048, 2048]  4,194,304 B
        layers.3.ffn.experts.0.w1.weight_scale  F8_E4M3  [2048,  256]    524,288 B   -> 4096/256 = block 16
        layers.3.ffn.experts.0.w1.weight_scale_2 F32     []                    4 B   -> NVFP4's second-level global scale
fp8:    layers.3.ffn.experts.0.w1.weight        I8       [2048, 2048]  4,194,304 B
        layers.3.ffn.experts.0.w1.scale         F8_E8M0  [2048,  128]    262,144 B   -> 4096/128 = block 32
```

The checkpoint's own `cast_mxfp4_to_nvfp4.log` records the conversion, tensor
by tensor, as `lossless=100.0000%` with `mse(v3.5)=0.0000e+00`. Nothing was
upcast; the experts were re-blocked from 32 to 16 and re-scaled.

**Verdict: trust it.** Size is not evidence of bit width when the scale
granularity differs.

## The checkpoint labelled FP8 does not have FP8 experts

The same audit settles a second thing. `deepseek-ai/DeepSeek-V4-Flash-0731` —
the official text checkpoint, and the one the DSpark sources on which
`BASELINE.md` draws describe as FP8 — stores **94.3% of its weight bytes as
4-bit MXFP4 experts**. Only attention (4.75 GiB) and the shared experts
(1.08 GiB) are FP8 E4M3. The checkpoint says so itself:

```
$ python3 -c 'import json; print(json.load(open("config.json"))["expert_dtype"])'
fp4
```

`expert_dtype: fp4` is present in **both** DeepSeek checkpoints.

This probably changes what the published DeepSeek comparison means, and the
qualifier is deliberate. `BASELINE.md`'s DeepSeek row is for
**Vision-Exp**, which is not on disk here and has therefore not been audited;
what is audited is its text sibling. The baseline reports 49.6 → 53.8 tok/s for
the "FP8" build against 80.1 tok/s for the "NVFP4" one, and the natural reading
is that 4-bit weights beat 8-bit weights on memory bandwidth. For the two
checkpoints measured here that reading is simply false: **both run 4-bit
experts and move the same 138.00 GiB of expert weights.** Whatever separates
them is not weight traffic.

Audit Vision-Exp the same way as soon as it is downloaded, before it is served.
If it too is `expert_dtype: fp4`, the published comparison is a kernel-path
comparison wearing a precision label.

The remaining candidates are all kernel-path, and they are testable:

- an NVFP4 MoE kernel that requires block-16 E4M3 scales and cannot consume
  MXFP4's block-32 E8M0 scales, so the MXFP4 checkpoint silently falls to a
  dequantize-then-BF16 path. This is the shape of the `VLLM_USE_B12X_MOE=1`
  trap recorded in `BASELINE.md` — "the entire speed difference".
- DSpark draft acceptance differing between the two builds, which `BASELINE.md`
  already attributes to content (78% on code, 34% on prose) rather than tuning.

Both are worth separating by measurement, and neither is a weight-precision
effect.

## Qwen3.8-Flash-Next is 38% embedding table

The NVFP4 Qwen checkpoint is 125.9 GiB, which does not fit one 121 GiB node.
Where the bytes are matters for the memory budget:

| category | GiB | dtype |
|---|---:|---|
| per-layer embedding (`ple.ple_embedding.ngram*`) | **47.68** | F8_E4M3 |
| routed experts | 63.28 | NVFP4 block 16 (56.25 packed + 7.03 scales) |
| everything else (attention, shared experts, lm_head, vision tower, MTP) | 14.9 | BF16 |

The PLE tensors are `[2500012, 160]` per layer — a 2.5-million-entry n-gram
table. This is what `--ple-offload-embedding` in the published recipe exists
for: moving 47.68 GiB to host memory is what leaves room for a 600,000-token
KV pool on two nodes. It is not an optional tuning flag.

Two other structural facts fall out of the same audit, both relevant later:
the checkpoint carries `mtp.*` tensors (the MTP draft head — the baseline's
"~20 tok/s without MTP" is measured against these) and a `model.visual.*`
tower, with `architectures: ["Qwen4ExpForConditionalGeneration"]`.

## Reproducing this

```bash
scripts/audit-checkpoint.py /path/to/checkpoint --verify
```

`--verify` derives the block size from the ratio of packed-weight bytes to
scale bytes and names the format. It prints `OK` only when the derived block
size matches the one that dtype implies, so a checkpoint that has been
repacked, truncated or mixed shows up as `UNEXPECTED` rather than being taken
on trust.
