# What the engines on these nodes can actually do

Rule 1 puts engine, version and image in the tuple "because the fixes live in
specific builds". This file records what each image on the nodes was measured
to support, so a configuration is chosen against capability rather than against
a recipe written for somebody else's build.

Everything here was read out of the images themselves on 2026-09-04, with no
GPU attached — `docker run --rm --entrypoint python3 <image> -c ...`. That
matters: a capability believed from a README is a plan, a capability read from
the build is a fact.

## The images

| image | engine | version | on |
|---|---|---|---|
| `sglang-qwen38fn:sm121-ours` | SGLang | `0.0.0.dev1+gd91c3682b` | both |
| `lmsysorg/sglang:dev-cu13-v4f-2dgx` | SGLang | `0.0.0.dev1+g6b57ed348` | both |
| `lmsysorg/sglang:glm-5.3-flash` | SGLang | — | both |
| `vllm-glm53-flash:sm121-v11-dflash2` | vLLM | `0.1.dev20051+g487ecf187` | both |
| `vllm/vllm-openai:latest` | vLLM | — | worker only |

All carry torch `2.13.0+cu130`. The two SGLang dev images carry FlashInfer
0.6.17 and tilelang 0.1.11; the vLLM image carries FlashInfer
`0.6.18.dev20260819` and tilelang 0.1.12.

## `sglang-qwen38fn:sm121-ours` — the Qwen image

Its own labels give the provenance, which is worth reading before treating it
as an unknown local build:

```
ai.radixark.base.image        lmsysorg/sglang:nightly-dev-cu13-20260817-d91c3682
ai.radixark.overlay.commits   3ea3a37a1,12070370f
ai.radixark.overlay.prs       Qiaolin-Yu/sglang-qwen-next#38
```

and its final layer patches `rotary_triton.py`, bounding a `t_mask` that was
unbounded:

```python
t_mask = ~(h_mask | w_mask)                       # before
t_mask = ~(h_mask | w_mask) & (cos_offsets < half_rd)   # after
```

That is the mrope fix, so this image is the local equivalent of the
`sm121-qsa-mrope1` tag `BASELINE.md` names. The patch script refuses to
double-patch and refuses to patch if the source has drifted, which is why the
provenance can be trusted.

Every flag the published Qwen recipe needs is present: `--max-total-tokens`,
`--mem-fraction-static`, `--cuda-graph-max-bs`, `--disable-cuda-graph-padding`,
`--ple-offload-embedding`, `--disable-radix-cache`, `--max-mamba-cache-size`,
`--sampling-backend`, `--context-length`.

**QSA is present and its successor is not.** The image has the `qsa` attention
backend and `qsa_kv_pool.py`. The pr36845 Triton packed-varlen kernel is not in
the attention path — the only `packed_varlen` hits in the image are under
`multimodal_gen`, an unrelated subsystem. So the silent long-context corruption
`BASELINE.md` warns about applies to this image.

Note what that implies about the published recipe: it passes
`--context-length 262144` while the same source says QSA corrupts past roughly
120K. Both cannot be safe. `scripts/launch-qwen-sglang.sh` therefore defaults to
120,000 and *refuses* anything larger unless `QWEN_ALLOW_LONG_QSA=1` is set —
and the agentic workload's needle check is what turns that warning into a
measurement rather than a superstition.

KV cache dtypes this build accepts: `auto, fp8_e5m2, fp8_e4m3, mxfp8, bf16,
nvfp4, fp4_mx_block16, fp4_e2m1`.

## `vllm-glm53-flash:sm121-v11-dflash2` — closer to the DeepSeek baseline than its name suggests

Built for GLM, but it registers `DeepseekV4ForCausalLM` and `DeepSeekV4MTPModel`,
and its speculative-decoding methods include both `dflash` (GLM's drafter) and
**`dspark`** (DeepSeek's). So the DeepSeek baseline's speculative method is
available here without building anything.

Two gaps stop it being the published configuration:

| the baseline wants | this image has |
|---|---|
| `--kv-cache-dtype nvfp4_ds_mla` | `fp8_ds_mla`, `nvfp4`, `nvfp4_4over6` — **no `nvfp4_ds_mla`** |
| `VLLM_USE_B12X_MOE=1` / `--moe-backend b12x` | neither: no `B12X` symbol in `vllm.envs` |

`nvfp4_ds_mla` appears nowhere in the package. `fp8_ds_mla` does, with a
documented packed layout — "512 NoPE + 16 scales + 128 RoPE" — and there is a
`flashinfer_mla_sparse_sm120.py` backend that *requires* that packed dtype,
which is the Blackwell sparse-MLA path this hardware needs. So a DeepSeek run is
reachable on this image today at `fp8_ds_mla`, with a smaller KV pool than the
published one, and it is a different tuple rather than a failed reproduction.

Since `VLLM_USE_B12X_MOE` is not a variable this build knows, exporting it would
do nothing and log nothing — the exact silent-failure shape the baseline warns
about, one level further out.

## b12x is present, in the SGLang DeepSeek image

`lmsysorg/sglang:dev-cu13-v4f-2dgx` installs
`b12x @ github.com/local-inference-lab/b12x@85d3681`, with `nvidia-cutlass-dsl
4.7.0` under `/opt/dsl470`. The package has `moe`, `gemm`, `attention`, `comm`,
`norm` and `quantization` submodules.

A README summary described B12X as an MXFP4 MoE kernel. Reading the package
contradicts that as a general claim: across its Python sources, `nvfp4`/`NVFP4`
appears 438 times against 35 for `mxfp4`/`MXFP4`, alongside 1,089 `e4m3` and 603
`e8m0`. It supports both scale formats, with NVFP4 the dominant path. This is
worth settling by measurement rather than by either description, because it is
the hypothesis for why two DeepSeek checkpoints carrying *identical* 4-bit
expert bytes serve at different speeds (`docs/checkpoints.md`).

## What this means for each model

**Qwen3.8-Flash-Next** — reproducible now on `sglang-qwen38fn:sm121-ours`,
capped at 120K context until the QSA question is settled.

**GLM-5.3-Flash** — the SGLang path is blocked in the way `docs/baselines.md`
records: the tilelang DSA kernel asks for 169,984 bytes of dynamic shared
memory and GB10 allows 101,376. Survivors among the nine `--dsa-*-backend`
values are `flashinfer_sparse_mla` and `fa3`. There is also an untried vLLM
path: `vllm-glm53-flash:sm121-v11-dflash2` supports `Glm5NextForCausalLM`,
`glm5_next_mtp` and `dflash`, which is the drafter the baseline calls a 2.15x.

**DeepSeek-V4-Flash** — no image here matches the published tuple. The
reachable configuration is the vLLM image at `fp8_ds_mla` with `dspark`
speculation; closing the gap to the published one means the DSpark fork's vLLM
build, which is a build project rather than a flag.
