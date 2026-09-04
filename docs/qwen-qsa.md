# The QSA long-context problem, and how this project treats it

## What is claimed

`BASELINE.md` carries a warning from the Qwen source: the SM121 QSA kernel
guard causes **silent output corruption past about 120K context**, and its
successor is the Triton packed-varlen kernel in `pr36845.diff`.

Silent is the whole difficulty. A kernel that crashes costs an hour. A kernel
that returns confident wrong tokens costs every long-context measurement taken
after it, and there is nothing in the log to say which ones.

## What is true of the image we have

`sglang-qwen38fn:sm121-ours` has the QSA backend — `qsa` is one of its
`--attention-backend` values and `srt/mem_cache/qsa_kv_pool.py` is present.

It does **not** have the successor. The only `packed_varlen` matches anywhere in
the image are under `python/sglang/multimodal_gen/`, an unrelated subsystem
(vision encoders and diffusion transformers). Nothing in the attention path.

So the warning applies to this image as written.

## The contradiction in the published recipe

The recipe in `BASELINE.md` passes `--context-length 262144`. The same source
says QSA corrupts past roughly 120K. Both statements cannot describe a safe
configuration, and reproducing the flags exactly would mean measuring inside
the corruption zone while reporting the numbers as throughput.

`scripts/launch-qwen-sglang.sh` therefore defaults to `--context-length 120000`
and refuses anything larger:

```
refusing to launch: --context-length 131072 exceeds the QSA safe limit of 120000.
```

This is a deliberate departure from the published flags, and it is recorded in
the tuple's `flags` field on every run so no result silently claims to be the
published configuration.

## Turning the warning into a measurement

"About 120K" is a number from somebody else's report. This project can do
better than trust it, because the harness already carries the instrument.

Every `agentic-*` workload plants a build key early in the context —

```
<note>The build key for this session is BUILD-KEY-41904B04FE79. Quote it exactly when asked.</note>
```

— fills the middle with code, tool output and prose, and ends by asking for the
key back verbatim. `sparkbench measure` records `needle_hit_rate` for the run.
A configuration that has begun corrupting its long-context attention cannot
retrieve a key it read 100K tokens ago, so the hit rate collapses where the
kernel does.

The experiment, when there is machine time for it:

```bash
QWEN_ALLOW_LONG_QSA=1 QWEN_CTX=262144 bash scripts/bring-up.sh scripts/launch-qwen-sglang.sh
# then a ladder, watching needle_hit_rate rather than tok/s
scripts/sparkbench measure --workload agentic-33k  ...   # expect 1.0
scripts/sparkbench measure --workload agentic-131k ...   # the question
scripts/sparkbench measure --workload agentic-262k ...   # expect collapse
```

Two things make this sound rather than a fishing trip. The needle is retrieval
from *early* context, which is precisely what a sparse-attention kernel with a
broken guard would lose first. And the same ladder run at `agentic-33k` — well
inside the safe zone — is the control: if the hit rate is already below 1.0
there, the instrument is measuring the model's competence rather than the
kernel's correctness, and the long-context result means nothing.

Until that ladder has been run, no Qwen measurement in this project is taken
above 120,000 tokens of context, and none is quoted as reproducing the
published `--context-length 262144`.

## If the successor is wanted

`pr36845.diff` would have to be applied to a matching SGLang tree and the image
rebuilt. That is a build project, not a flag, and it is worth doing only if the
ladder above shows the corruption is real on this hardware and is costing
useful context. Measure first.

## The same kernel refuses an fp8 KV cache, and takes the server with it

`--kv-cache-dtype fp8_e4m3` is the obvious thing to want here: measured on
this checkpoint it doubles the KV pool, 166,016 tokens to **332,480** at the
same `--max-total-tokens 600000`. The `[mtp+fp8kv]` campaign in `REPORT.md`
is what happened when it was tried.

Short and single-user cells served normally — 23.7 tok/s at `short @ 1`,
84.0 at `code @ 6`, 21.8 at `agentic-4k @ 1`, all with 100% needle
retrieval. Then `agentic-4k @ 6 users` killed the server:

```
[2026-09-03 23:03:16 TP0] Prefill batch, #new-seq: 1, #new-token: 4352, ... #queue-req: 2
[2026-09-03 23:03:18 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File ".../triton/language/semantic.py", line 1500, in dot
    assert rhs.dtype in (tl.int8, tl.uint8, tl.float16, tl.bfloat16, tl.float32,
AssertionError: Unsupported rhs dtype fp8e4nv
[2026-09-03 23:03:18] kill_process_tree called: parent_pid=1, include_parent=True, pid=1
```

The frame below the assertion is
`srt/layers/attention/qsa/sparse_attn.py:275`, in
`sparse_gqa_fwd_interface_triton_ck` — the QSA chunked-prefill kernel. Both
remaining cells of that campaign then failed for the ordinary reason that
there was no longer a server, which is why the config has 7 rows and not 9.

This is not a tuning problem, and the reason is legible in
`srt/layers/attention/qwen_sparse_attn_backend.py`. `forward_extend` picks
between two kernels on one condition:

```python
if not any(prefix_lens):
    output = sparse_gqa_fwd_interface_triton(q, k[:num_valid_rows], v[:num_valid_rows], ...)
    return self._pad_extend_output(output, num_output_rows)

# otherwise: "The validated chunk-prefill kernel consumes tightly packed
# full-context K/V."
k_buffer = pool.get_key_buffer(layer.layer_id)
v_buffer = pool.get_value_buffer(layer.layer_id)
...
output = sparse_gqa_fwd_interface_triton_ck(q, torch.cat(k_parts), torch.cat(v_parts), ...)
```

With no prefix, `k` and `v` are the freshly projected tensors in the layer's
own dtype — bfloat16 — and Triton's `tl.dot` accepts them. With any prefix,
the `_ck` kernel reads K/V **straight out of the paged pool**, and the pool
is whatever `--kv-cache-dtype` made it. Set it to fp8 and `tl.dot` is handed
an `fp8e4nv` right-hand side that this Triton build does not accept.

So the trigger is not concurrency as such: it is a non-zero prefix length,
which is exactly what chunked prefill produces from the second chunk onward.
`chunked_prefill_size` is 8192. Every cell that survived had a prompt that
fit in one chunk; the 6-user cell had three 3,891-token prompts contending
for that budget and one of them was split.

The prediction that follows is that **a single 33k-token prompt would crash
it too**, with no concurrency at all, because 30,301 tokens cannot be one
chunk. That cell was never measured — the server was already dead when the
campaign reached it.

The operational conclusion does not depend on settling that: an fp8 KV cache
on this build is unusable for the long-prompt work this project exists to
measure, and it fails by killing the process rather than by degrading. Every
Qwen number in `REPORT.md` from `[qwen-rdma-mtp-eos]` onward is bfloat16 KV
for this reason, at half the KV pool.
