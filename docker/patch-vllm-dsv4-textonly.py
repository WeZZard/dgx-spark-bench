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
    layers.{0,1}.ffn.gate.bias        3
    mtp.N.ffn.gate.bias_vl            3   already covered by "mtp."

So the checkpoint cannot be loaded by the stock image at all. This patch adds
those prefixes to the existing skip list.

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
import re
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
    '            skip_substrs=["mtp.", "vision.", "aligner.", "image_", ".bias_vl"],\n'
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
