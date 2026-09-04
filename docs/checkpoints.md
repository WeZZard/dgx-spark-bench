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

This changes what the published DeepSeek comparison means. The baseline
reports 49.6 → 53.8 tok/s for the "FP8" build against 80.1 tok/s for the
"NVFP4" one, and the natural reading is that 4-bit weights beat 8-bit weights
on memory bandwidth. For the two checkpoints measured here that reading is
simply false: **both run 4-bit experts and move the same 138.00 GiB of expert
weights.** Whatever separates them is not weight traffic.

`BASELINE.md`'s DeepSeek row is for **Vision-Exp**, which is not on disk here,
so for a while this was said of the text sibling and left hanging for the
model the row actually names. It no longer is — see the next section.

## Vision-Exp is the same build, and it did not need downloading

That qualifier can now be dropped. `scripts/audit-remote-checkpoint.py` reads
the 48 safetensors headers over HTTP range requests — about eight megabytes
against the checkpoint's 156 GiB — and runs them through the same
`audit_headers()` the on-disk auditor uses:

```
### deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186  shards=48
    tensors=72633 total=156.29 GiB
    index declares 156.29 GiB (agrees)
    weight field: MXFP4 experts (block 32, 94% of bytes)
           I8    138.00 GiB   88.3%
      F8_E8M0      8.63 GiB    5.5%
      F8_E4M3      5.87 GiB    3.8%
   routed_experts  138.00 GiB packed 4-bit + 8.62 GiB scales -> block 32.00 = MXFP4  [OK]
```

Its `config.json` agrees in its own words: `"expert_dtype": "fp4"` beside a
`quantization_config` of `{"quant_method": "fp8", "fmt": "e4m3",
"scale_fmt": "ue8m0", "weight_block_size": [128, 128]}`. The fp8 block applies
to attention. The experts are fp4.

Set beside the two checkpoints on disk, all three lines are the same length:

| | Vision-Exp | `0731` (text) | nvidia NVFP4 |
|---|---|---|---|
| packed expert weights | **138.00 GiB** `I8` | **138.00 GiB** `I8` | **138.00 GiB** `U8` |
| expert scales | 8.62 GiB `F8_E8M0` | 8.62 GiB `F8_E8M0` | 17.25 GiB `F8_E4M3` |
| derived block size | **32.00** | **32.00** | **16.00** |
| format | MXFP4 | MXFP4 | NVFP4 |

**So the published DeepSeek comparison is a kernel-path comparison wearing a
precision label.** `BASELINE.md` reports 49.6 → 53.8 tok/s for the build it
calls FP8 against 80.1 for the one it calls NVFP4, and the natural reading —
8-bit weights against 4-bit weights — is false for every checkpoint involved.
All three move exactly 138.00 GiB of expert weight. Nothing about weight
bandwidth can produce a 1.6x difference between them.

What is left is the kernel path and the drafter, and both are testable rather
than arguable: an NVFP4 MoE kernel that requires block-16 E4M3 scales and
cannot consume block-32 E8M0 ones would run a different code path at identical
byte traffic, and DSpark acceptance varies from ~34% on prose to ~78% on
structured code on the published figures alone. `VLLM_USE_B12X_MOE` and the
measured acceptance rate are the two things to separate.

### It is the same architecture, and it is not the same weights

The audit above invites a much stronger conclusion than it supports, and the
temptation is worth naming because acting on it would be a Rule 1 violation
with a large payoff attached. Vision-Exp contains **every one of `0731`'s
72,317 tensor names**, at identical dtypes and identical byte counts, and adds
316 more:

```
### dsv4-flash-fp8  vs  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186
    shared names 72317   local-only 0   remote-only 316
    remote-only prefixes: {'vision': 259, 'layers': 46, 'aligner': 4, 'mtp': 3,
                           'image_newline': 1, 'image_start': 1,
                           'image_pad': 1, 'image_end': 1}
```

That is exactly the shape of "the text model with a vision tower bolted on",
and if it were true then this project's `0731` measurements would *be*
Vision-Exp measurements for text work, and the largest reproduction gap in
`BASELINE.md` would close without measuring anything.

It is not true. `scripts/compare-checkpoints.py` hashes the shared tensors —
each costs its own size over an HTTP range request, twelve megabytes for four
of them — and every one differs:

```
   layers.3.ffn.experts.0.w1.weight
      local        I8    4,194,304 B  a40fb64af3da643c01af3abb1774286d
      remote       I8    4,194,304 B  ac5ebb0a9e3e574fe1c194b6a491717d   DIFFERENT
   layers.20.ffn.experts.7.w2.weight    ... DIFFERENT
   layers.40.ffn.experts.255.w3.weight  ... DIFFERENT
   layers.3.ffn.experts.0.w1.scale      ... DIFFERENT
-- 0 identical, 4 different --
```

Sampled at layers 3, 20 and 40 and at expert indices 0, 7 and 255, so a
checkpoint differing only in its last layers or its rarely-routed experts
could not have passed. Shared names and shared byte counts mean a shared
architecture and a shared quantization; they do not mean a shared checkpoint.
`0731` is not Vision-Exp, which is what Rule 1 says, now shown rather than
asserted.

One serving fact came out of the same eight megabytes. Vision-Exp's top-level
`config.json` declares `"architectures": ["DeepseekV4ForCausalLM"]`,
`model_type: deepseek_v4` — a text causal LM. The vision tower lives in the
repo's own `inference/vision.py` and `inference/config.json`, a bespoke
reference implementation. An engine that reads `config.json` serves this
checkpoint as a text model, which is consistent with `BASELINE.md`'s finding
that the 84.3 / 197.3 figures are text throughput and not a vision result.

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
for: moving 47.68 GiB out of the GPU's budget is what makes room for a
600,000-token KV pool on two nodes.

An earlier version of this file called that flag "not optional". That was an
overstatement, and the arithmetic does not support it. At TP2 the rest of the
checkpoint is ~39 GiB a node, and the bf16 600,000-token pool is ~7 GB, so a
node carrying the table too needs ~94 GiB if it is replicated and ~70 GiB if it
is TP-sharded — both inside the 109 GiB container cap. Whether offloading helps
or costs is therefore a measurement, and `QWEN_PLE_OFFLOAD=off` exists to take
it. It matters because 48 per-layer gathers from a 2.5-million-row table in
host memory is the leading explanation for the ~1.2 ms a layer that remains
unaccounted for after the interconnect was fixed.

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

## The engine here cannot load Vision-Exp, and that was cheap to establish

Downloading 156 GiB to find out whether it serves would be the expensive way
round. The image already on the nodes answers it.

`vllm-dsv4:ray` registers `DeepseekV4ForCausalLM` to
`vllm.models.deepseek_v4`, and that class loads weights through

```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
    loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

`AutoWeightsLoader` raises on any name it cannot place —
`vllm/model_executor/models/utils.py:389`, *"There is no module or parameter
named {prefix!r}"* — unless the name matches `skip_substrs`, which here is
`mtp.` and nothing else. The inner `DeepseekV4Model.load_weights` ends its
dispatch chain on a bare `param = params_dict[name]`, so an unplaceable name
is a `KeyError` there rather than a skip.

Vision-Exp's 316 extra tensors are:

| group | n | placeable in `DeepseekV4ForCausalLM`? |
|---|---|---|
| `vision.*` (44-block ViT) | 259 | no |
| `layers.N.ffn.gate.bias_vl` | 43 | no |
| `aligner.w{1,2}.{weight,bias}` | 4 | no |
| `image_{start,end,pad,newline}` | 4 | no |
| `mtp.N.ffn.gate.bias_vl` | 3 | yes — `mtp.` is skipped |
| `layers.{0,1}.ffn.gate.bias` | 3 | no |

So 313 of the 316 are fatal. **This build cannot load the checkpoint.**

`bias_vl` is the one worth pausing on, because it is not a bolt-on. It is a
*second expert-routing bias, on all 43 layers*, alongside the `bias` that
`0731` also has — a separate MoE route for image tokens. Vision-Exp is not the
text model plus a tower; the vision path reaches into routing in every layer.
That is consistent with the tensor hashes differing, and it is a second reason
not to read `0731`'s numbers as Vision-Exp's.

It also says what a text-only load would have to do: skip `vision.`,
`aligner.`, `image_` and `.bias_vl`, leaving the `bias` route that text tokens
use and that is present in both checkpoints. That is a `skip_substrs` change —
a derived image, in the pattern of `sglang-glm53:gb10-tilelang` and
`vllm-dsv4:ray` — and any tuple it produces must record in its flags that the
vision tower was not loaded. Which is the honest form of the comparison
anyway: `BASELINE.md` establishes that the published 84.3 / 197.3 figures are
text throughput, not a vision result.
