# GLM-5.3-Flash: which sparse-attention kernel can actually run on GB10

Short answer: **bf16 KV with the tilelang DSA backend**. Not the fp8 KV the
published row names. This is a different tuple, recorded as one — not a
failed reproduction. Two seventeen-minute weight loads bought this.

## The published row is not reachable for this checkpoint

`BASELINE.md` gives the GLM SGLang path as **fp8** KV, crediting SGLang
PR #36904, "bf16 before that". Attempting it produced, after the model was
fully loaded:

```
ValueError: flashinfer_sparse_mla supports only GLM DSA with FP8 KV cache
on NVIDIA SM120/SM121; got model_arch='Glm5NextForConditionalGeneration',
sm_major=12, kv_cache_dtype=torch.float8_e4m3fn
```

Both conditions the message states were satisfied: GB10 is SM121, and the
KV cache was fp8_e4m3. The condition it does not state is the architecture.
The allow-list in `kernels/ops/attention/flash_mla_sm120.py` is

```python
_GLM_DSA_MODEL_ARCHS = ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN")
```

and both GLM checkpoints on this node — `glm53-flash-nvfp4-ref` (189 G,
NVFP4) and `glm53-flash-fp8` (146 G) — declare
`Glm5NextForConditionalGeneration`, `model_type: glm5_next`.

These are genuinely two different models in the same image, not two names
for one:

| architecture | implementation | family |
|---|---|---|
| `GlmMoeDsaForCausalLM` | `srt/models/glm4_moe.py` | GLM-4 MoE DSA |
| `Glm5NextForConditionalGeneration` | `srt/models/glm5_next.py` | GLM-5 Next |

Both are in `_DEEPSEEK_FAMILY_ARCHS` and both satisfy `is_deepseek_dsa()`,
so both take the DSA path — but the SM120/121 fp8 fast kernel was written
for the GLM-4 one. No flag reaches it from a glm5_next checkpoint.

## What auto-detect picks instead, and why that also fails

Omitting the flags is the right instinct — there is no `auto` choice, and
SGLang auto-detects only when the flag is **absent**. What it detects is
decided in `srt/arg_groups/overrides.py` and was predicted before the run:
`is_glm_sm12_fp8` requires the arch to be exactly `GlmMoeDsaForCausalLM`,
so it is False, and the fp8 branch gives `"trtllm" if major >= 10`. GB10 is
major 12. The log confirmed it exactly:

```
Set DSA backends for fp8_e4m3 KV Cache: prefill=trtllm, decode=trtllm.
```

That prediction was right and the backend still does not work. After a
second full load:

```
tvm.error.InternalError: Error in function 'TllmGenFmhaRunner' at
flashinfer/trtllm/fmha/fmhaRunner.cuh:37: Unsupported architecture
[AutoTuner]: Tuning trtllm_batch_decode_mla: 0%| | 0/21
```

TensorRT-LLM's generated FMHA kernels do not cover sm_121.

## The backends that remain

| backend | status on GB10 + glm5_next |
|---|---|
| `flashinfer_sparse_mla` | arch-gated to the GLM-4 DSA family |
| `trtllm` | `TllmGenFmhaRunner`: unsupported architecture (measured) |
| `flashmla_sparse_q8` | explicit `device_sm_major != 9` guard — SM90 only |
| `flashmla_sparse` / `flashmla_kv` / `fa3` | Hopper-oriented; untried |
| `aiter` | ROCm only |
| `tilelang` | **fp8 KV is ROCm-only; on CUDA it requires bf16 KV** |

The tilelang restriction is enforced up front by
`_check_tilelang_dsa_fp8_kv`, which is why fp8 and tilelang cannot be
combined at all:

> The tilelang DSA prefill/decode kernels only support an fp8_e4m3 KV cache
> on ROCm/HIP; on CUDA they require a bfloat16 KV cache.

So for this checkpoint, on this hardware, in this image, `bf16 KV +
tilelang` is the one combination not excluded by a guard or a measurement.

## Why tilelang is the right thing to try rather than a guess

The sources behind the published row report four GB10 fixes, and one of
them is a **DSA tilelang shared-memory overflow at 8-token verify**, with a
tuning of `block_I=32, num_stages=1, threads=128`. An overflow is not an
architecture rejection: it means the tilelang kernel compiled and ran on
GB10 and asked for more shared memory than the part allows (this project
measured 169,984 B requested against a 101,376 B
`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`). That demand scales with the verify
width, so the first run disables speculation.

That also gives something honest to compare against: the sources publish a
**no-speculation** ladder — c1 14.5 · c2 27.1 · c4 36.8 · c8 55.1 · c12
55.0 — which the campaign's code cells at 1/2/4/8 users line up with
directly. Those are decode rates and this project reports served rate, so
they still may not share a table (rule 2).

## Consequence for the tuple

The GLM result recorded here has **KV cache: bfloat16**, not the published
fp8. Under rule 1 that is a different tuple and must never be set beside
the fp8 row as though tuning explained the gap. The reason it is bf16 is
not a choice; it is the only combination this checkpoint's architecture
leaves available on GB10.
