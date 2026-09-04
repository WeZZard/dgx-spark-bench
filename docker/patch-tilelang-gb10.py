"""Apply the GB10 tilelang tuning to SGLang's DSA sparse-attention kernel.

Run inside the image at build time. Idempotent: refuses to double-apply.

Why: on GB10 the default tuning asks for more dynamic shared memory than the
part has. Measured on sm_121, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101,376 B,
by compiling the kernel standalone at GLM-5.3-Flash's real shapes
(num_heads 32 after attn_tp 2, dim 512, tail_dim 0, topk 2048):

    block_I=64 num_stages=2 threads=256   ->  169,984 B   (the default)
    block_I=64 num_stages=1 threads=256   ->  104,448 B
    block_I=32 num_stages=1 threads=128   ->  runs, output finite
    block_I=32 num_stages=2 threads=256   ->  invalid fragment layout
    block_I=16 num_stages=1 threads=128   ->  invalid fragment layout

So the working setting is not a preference among several; it is the only one
of these that both fits and has a valid layout. It is also exactly the
setting the published GB10 report gives, arrived at here independently.

The patch is gated on the device's own shared-memory limit rather than on a
compute capability, so a part that can afford the default still gets it.
Only the v1 path is touched: v2 (tail_dim != 0) hardcodes threads and takes
no num_stages.
"""

import sys

PATH = "/sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"

OLD = """        kernel = kernel_factory(
            num_heads, d_v, tail_dim, topk, sm_scale=sm_scale, return_lse=return_lse
        )"""

NEW = '''        # --- dgx-spark-bench GB10 patch ---
        # The default (block_I=64, num_stages=2, threads=256) requests 169,984 B
        # of dynamic shared memory. GB10 allows 101,376 B, so the launch fails
        # with "Failed to set the allowed dynamic shared memory size to 169984".
        # Gate on the device limit, not on the compute capability.
        _tune = {}
        if kernel_factory is sparse_attention_fwd_kernel_v1:
            _optin = torch.cuda.get_device_properties(
                q.device.index or 0
            ).shared_memory_per_block_optin
            if _optin < 169984:
                _tune = dict(block_I=32, num_stages=1, threads=128)
        kernel = kernel_factory(
            num_heads, d_v, tail_dim, topk, sm_scale=sm_scale,
            return_lse=return_lse, **_tune
        )
        # --- end GB10 patch ---'''

src = open(PATH).read()

if "dgx-spark-bench GB10 patch" in src:
    print("already patched; nothing to do")
    sys.exit(0)

if src.count(OLD) != 1:
    print(f"REFUSING: anchor found {src.count(OLD)} times, expected exactly 1", file=sys.stderr)
    print("Upstream moved; re-derive the patch instead of forcing it.", file=sys.stderr)
    sys.exit(1)

open(PATH, "w").write(src.replace(OLD, NEW))
print("patched", PATH)
