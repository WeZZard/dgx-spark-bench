#!/usr/bin/env python3
"""Let vLLM load DeepSeek-V4-Flash-Vision-Exp as a text model.

Why: `DeepseekV4ForCausalLM.load_weights` runs
`AutoWeightsLoader(self, skip_substrs=["mtp."])`, and AutoWeightsLoader raises
`ValueError: There is no module or parameter named ...` for every name it
cannot place. Vision-Exp carries 316 tensors that `0731` does not, and 313 of
them have nowhere to go in a text-only `DeepseekV4ForCausalLM`:

    vision.*                        259   the 44-block ViT
    layers.N.ffn.gate.bias_vl        43   a SECOND expert-routing bias
    aligner.w{1,2}.{weight,bias}      4   ViT -> LM projection
    image_{start,end,pad,newline}     4   learned special-token embeddings
    layers.{0,1,2}.ffn.gate.bias      3   the hash-MoE layers; see below
    mtp.N.ffn.gate.bias_vl            3   already covered by "mtp."

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
happened on the first attempt (2026-09-04, vexp-textonly-k6).

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

What that means for a result, and it must be recorded in the tuple's flags:
the vision tower is NOT loaded, and neither is the vision routing bias. Text
tokens route on `layers.N.ffn.gate.bias`, which is present in this checkpoint
and in `0731`, so the text path is complete; the image path is absent. That
is the right shape for this comparison anyway -- BASELINE.md establishes that
the published 84.3 / 197.3 tok/s figures are text throughput and not a vision
result -- but a run produced this way is a text-only load of a vision
checkpoint and must never be written down as anything else.

`.bias_vl` is deliberately matched as a substring rather than by a `layers.`
prefix, so it also covers the three `mtp.` ones if the mtp skip is ever
narrowed.
"""
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py"
ANCHOR = 'AutoWeightsLoader(self, skip_substrs=["mtp."])'
REPLACEMENT = (
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

MARKER = "PATCHED-VLLM-DSV4-TEXTONLY"


def main() -> None:
    src = open(TARGET).read()
    if MARKER in src:
        print("already patched")
        return
    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(f"refusing: expected the anchor exactly once, found {n}. "
                 f"The upstream loader changed; re-read it before patching.")
    src = src.replace(ANCHOR, REPLACEMENT)
    src = f"# {MARKER}\n" + src
    open(TARGET, "w").write(src)
    print(f"patched {TARGET}")


if __name__ == "__main__":
    main()
