#!/usr/bin/env python3
"""Let vLLM load DeepSeek-V4-Flash-Vision-Exp as a text model.

Two files, because there are two weight loaders. The target model
(`nvidia/model.py`) and the DSpark draft model (`nvidia/dspark.py`) each read
the same checkpoint through their own code path, and a skip written into one
has no effect on the other.

Target model: `DeepseekV4ForCausalLM.load_weights`
---------------------------------------------------
It runs `AutoWeightsLoader(self, skip_substrs=["mtp."])`, and AutoWeightsLoader
raises `ValueError: There is no module or parameter named ...` for every name
it cannot place. Vision-Exp carries 316 tensors that `0731` does not, and 313
of them have nowhere to go in a text-only `DeepseekV4ForCausalLM`:

    vision.*                        259   the 44-block ViT
    layers.N.ffn.gate.bias_vl        43   a SECOND expert-routing bias
    aligner.w{1,2}.{weight,bias}      4   ViT -> LM projection
    image_{start,end,pad,newline}     4   learned special-token embeddings
    layers.{0,1,2}.ffn.gate.bias      3   the hash-MoE layers; see below
    mtp.N.ffn.gate.bias_vl            3   skipped HERE by "mtp." -- but see
                                          the draft model, which does not skip

So the checkpoint cannot be loaded by the stock image at all. This patch adds
those prefixes to the existing skip list.

Two spellings, and only one of them works
-----------------------------------------
`AutoWeightsLoader.load_weights` applies the `WeightsMapper` FIRST and filters
on `skip_substrs` SECOND (`vllm/model_executor/models/utils.py`: `weights =
mapper.apply(weights)`, then the `_can_skip(name)` generator). So every entry
here is matched against the name the mapper produced, not the name in the
safetensors file.

That matters for exactly one group. `_make_deepseek_v4_weights_mapper` carries
`orig_to_new_suffix={".ffn.gate.bias": ".ffn.gate.e_score_correction_bias"}`,
so the checkpoint's `layers.0.ffn.gate.bias` is already
`layers.0.ffn.gate.e_score_correction_bias` by the time the filter sees it. A
skip list written in the checkpoint's spelling matches nothing and the load
dies in `DeepseekV4Model.load_weights` at `param = params_dict[name]` with
`KeyError: 'layers.0.ffn.gate.e_score_correction_bias'` -- which is what
happened on the first attempt (2026-09-04, `vexp-textonly-k6`).

The other four entries are unaffected: `vision.`/`aligner.`/`image_` have no
mapping at all, and `.bias_vl` survives because the gate-bias rule is a
*suffix* rule, so it does not fire on a name ending `.bias_vl`.

Why these three layers and not the other forty
----------------------------------------------
`DeepseekV4MoE.__init__` sets `is_hash_moe = extract_layer_index(prefix) <
config.num_hash_layers`; a hash-MoE layer registers `gate.tid2eid` and leaves
`gate.e_score_correction_bias` as `None`. There is no parameter for a routing
bias to land in, on any DeepSeek-V4 checkpoint. Vision-Exp ships one anyway;
`0731` does not, which is why only this checkpoint trips it. The count is read
from `config.num_hash_layers` rather than hard-coded, so a checkpoint with a
different number of hash layers is handled or fails loudly, not silently.

Draft model: `DeepseekV4DSpark.load_weights`
--------------------------------------------
It does not use AutoWeightsLoader at all. It walks the same weight iterator,
maps each name through `_remap_dspark_name` -- which returns `None` for
everything that is not `mtp.{i}.*`, so the ViT, the aligner, the image tokens
and the 43 `layers.N.ffn.gate.bias_vl` are all dropped for free -- and then
does a bare `param = params_dict[name]`.

What is left is the three `mtp.{i}.ffn.gate.bias_vl`, which `_remap_dspark_name`
rewrites to `model.layers.{i}.ffn.gate.bias_vl`, and which the draft model has
no parameter for. That was the SECOND failure of this bring-up
(`KeyError: 'model.layers.0.ffn.gate.bias_vl'`), reached only after the target
model loaded cleanly, because the draft is loaded afterwards by
`Speculator.load_model`. Skipping the target model's copy of a tensor says
nothing about the draft model's copy of it.

What this means for a result
----------------------------
It must be recorded in the tuple's flags: the vision tower is NOT loaded, and
neither is the vision routing bias, in the target or in the draft. Text tokens
route on `layers.N.ffn.gate.bias`, which is present in this checkpoint and in
`0731`, so the text path is complete; the image path is absent. That is the
right shape for this comparison anyway -- BASELINE.md establishes that the
published 84.3 / 197.3 tok/s figures are text throughput and not a vision
result -- but a run produced this way is a text-only load of a vision
checkpoint and must never be written down as anything else.

`.bias_vl` is deliberately matched as a substring rather than by a `layers.`
prefix, so one spelling covers both loaders.
"""
import sys

MARKER = "PATCHED-VLLM-DSV4-TEXTONLY"
ROOT = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia"

MODEL_ANCHOR = 'AutoWeightsLoader(self, skip_substrs=["mtp."])'
MODEL_REPLACEMENT = (
    'AutoWeightsLoader(\n'
    '            self,\n'
    '            # Text-only load of a vision checkpoint: see\n'
    '            # docker/patch-vllm-dsv4-textonly.py. Without these, loading\n'
    '            # DeepSeek-V4-Flash-Vision-Exp raises on its first vision\n'
    '            # tensor. A run made this way has no image path.\n'
    '            #\n'
    '            # These match the name AFTER hf_to_vllm_mapper runs, because\n'
    '            # AutoWeightsLoader maps first and filters second.\n'
    '            skip_substrs=[\n'
    '                "mtp.", "vision.", "aligner.", "image_", ".bias_vl",\n'
    '            ] + [\n'
    '                # Hash-MoE layers route by gate.tid2eid and leave\n'
    '                # gate.e_score_correction_bias as None, so the checkpoint\n'
    '                # bias the mapper renames into it has nowhere to land.\n'
    '                f"layers.{i}.ffn.gate.e_score_correction_bias"\n'
    '                for i in range(getattr(self.config, "num_hash_layers", 0))\n'
    '            ],\n'
    '        )'
)

DSPARK_ANCHOR = (
    '            mapped = self._remap_dspark_name(name)\n'
    '            if mapped is None:\n'
    '                continue\n'
    '            name = mapped\n'
)
DSPARK_REPLACEMENT = DSPARK_ANCHOR + (
    '            # Text-only load of a vision checkpoint: see\n'
    '            # docker/patch-vllm-dsv4-textonly.py. The draft model does not\n'
    '            # go through AutoWeightsLoader, so the target model\'s\n'
    '            # skip_substrs never reach here. _remap_dspark_name has\n'
    '            # already dropped everything outside mtp.{i}.*; what survives\n'
    '            # is the three mtp.{i}.ffn.gate.bias_vl, and the draft has no\n'
    '            # parameter for a vision routing bias.\n'
    '            if ".bias_vl" in name:\n'
    '                continue\n'
)

PATCHES = [
    (f"{ROOT}/model.py", MODEL_ANCHOR, MODEL_REPLACEMENT),
    (f"{ROOT}/dspark.py", DSPARK_ANCHOR, DSPARK_REPLACEMENT),
]


def main() -> None:
    for target, anchor, replacement in PATCHES:
        src = open(target).read()
        if MARKER in src:
            print(f"already patched {target}")
            continue
        n = src.count(anchor)
        if n != 1:
            sys.exit(f"refusing: expected the anchor exactly once in {target}, "
                     f"found {n}. The upstream loader changed; re-read it "
                     f"before patching.")
        src = f"# {MARKER}\n" + src.replace(anchor, replacement)
        open(target, "w").write(src)
        print(f"patched {target}")


if __name__ == "__main__":
    main()
