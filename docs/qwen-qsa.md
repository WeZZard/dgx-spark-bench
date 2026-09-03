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
