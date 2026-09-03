# Measured baselines

Recorded 2026-08-27 during Phase 0 provisioning. Node names are `node-a`
(head) and `node-b` (worker).

## Nodes (identical)

- NVIDIA GB10, compute capability 12.1, driver 580.173.02, CUDA 13.0
- 20-core ARM, 121 GiB unified memory, 15 GiB swap, 3.7 TB NVMe (3.5 TB free)
- Ubuntu 24.04.4, kernel 6.17.0-1031-nvidia (aarch64)
- GB10 telemetry: `power.draw`, SM clock, temp, GPU util available;
  `memory.total` / power-limit / clock control **not** exposed. Memory is
  sampled from `/proc/meminfo` instead. CPU governor: `performance`.

## Fabric (ConnectX-7 back-to-back, first port pair, MTU 9000)

| Measurement | Result |
|---|---|
| Link speed (PHY) | 200 Gb/s both pairs, both nodes |
| Cross-connected pair | first port pair (the second also has carrier, unused) |
| Ping RTT (jumbo) | 0.9 ms avg |
| iperf3 TCP, 4 streams, 5 s | **110 Gb/s** |
| ib_write_bw RoCE, 64 KiB msgs | **109 Gb/s** |
| NFS write (head→worker mount, 1 MiB direct) | 161 MB/s (sync export) |

## WAN

~14 MB/s sustained (Wi-Fi, −47 dBm). Downloads are the schedule's
critical path; every artifact is fetched once on the head and shared
over NFS.

## Gotchas found during provisioning

- **HF Xet backend fails on these nodes** (`CAS Client Error: Request
  middleware error` from `hf download`, huggingface_hub 1.28.0, aarch64).
  Auth and plain HTTPS are fine. Fix: `HF_HUB_DISABLE_XET=1` — downloads
  then run at the WAN limit. `HF_HUB_ENABLE_HF_TRANSFER` is deprecated in
  hub 1.x and was removed from the env file.
- **llama.cpp cmake target for the RPC server is `ggml-rpc-server`**, not
  `rpc-server`, and the binary it produces is also named `ggml-rpc-server`.

## Engine support reality (2026-08-27 evening)

| Arch | llama.cpp master | Notes |
|---|---|---|
| `deepseek_v4` + `dflash` draft | ✅ supported | draft refuses standalone load ("dflash requires ctx_other") — by design |
| `qwen4exp` | ❌ PR #27742 (open) | open bug #27780: graph-builder abort on **SM121** under sustained load — affects us directly |
| `glm5next` | ❌ PR #27752 (draft) | builds clean on GB10; untested against weights yet |

First tuning target switched to DeepSeek accordingly. Three failed sweep
attempts (24 runs) stay recorded in the DB: YAML `on`→True booleanization,
split-GGUF first-shard path, then unsupported arch — all three now guarded.

## Incident 2026-08-29: double machine death by memory exhaustion

Both nodes went fully dark (no display) and needed physical power cycles.
Cause per previous-boot journals: SGLang misdetects usable memory on GB10
unified memory ("Failed to get GPU memory capacity from nvidia-smi",
falls back to a guess), then giant ladder prompts (235k+, KV retained by
the prefix cache) pushed allocation past physical memory. The kernel
OOM-killer killed unattributable small daemons (audio, Xorg) while the
real consumer survived; each node ground to an unresponsive halt over
~2 hours (bottom 05:36–07:52, top 09:29–12:07 — not simultaneous).
The PSI guard did not fire: it only killed llama.cpp binaries by name,
and SGLang runs inside Docker. Fixes: guard now also `docker kill`s
sglang/vllm containers; inference containers run with a hard `--memory`
cgroup cap so the container dies instead of the host; SGLang gets an
explicit conservative `--mem-fraction-static`.

## Download path (2026-08-29, owner network context)

Weights route through hf-mirror.com, with DIRECT (non-VPN) rules on the
gateway proxy for hf-mirror.com, modelscope.cn, aliyuncs.com and hf.co.
Measured ~12 MB/s, zero VPN traffic. Plain huggingface.co is not
reachable from the nodes without the VPN — the `hf.co` CDN hosts were
silently consuming VPN traffic before this change.

Corrected 2026-09-04. This section previously said `HF_ENDPOINT` was set
"in the node env". It is not: it appears in no rc file on the head, and a
non-interactive `ssh spark-left env` shows it unset. Whoever ran the
earlier downloads set it on the command line. That is a trap rather than a
detail — a script that assumes the env is already right downloads from
huggingface.co over the VPN and never says so. Two more facts found the
same way:

- `hf` and `huggingface-cli` live in `~/.local/bin`, which is **not** on a
  non-interactive ssh PATH. A download script that works when pasted into
  a login shell fails from automation with "command not found".
- `huggingface-cli` is deprecated in huggingface_hub 1.x (1.28.0 here) and
  **no longer downloads at all**: it prints a warning and exits 0-ish
  without fetching anything. `hf download` is the current form.

`scripts/fetch-checkpoint.sh` pins all three — endpoint, PATH and CLI —
rather than inheriting them.

## Incident 2026-08-31: the nine-hour zombie container

A vLLM worker rank failed compiling a GPU kernel ("ninja: build stopped")
30 minutes after launch, but the container stayed *running* — holding
90 GB, GPU idle at 0% — for a further 8.5 hours. Health checks (container
running? OOM-killed?) all reported "fine". Lessons applied:
1. Every watcher gets a hard deadline; "still running" is not progress.
2. Watch the GPU, not just the process: an idle GPU on a node that should
   be loading is the earliest true failure signal.
3. Container liveness is not workload liveness.

## Incident 2026-09-03: the head node stalled, and it was not swap

The campaign tore down the Qwen server on both nodes, slept ten seconds, and
launched DeepSeek-V4-Flash-0731. The worker was fine throughout: weights loaded
in 111.6 s (`Load weight end. elapsed=111.60 s, quant=fp8, avail mem=36.80 GB,
mem usage=71.55 GB`), 84 GB used, load 0.10, and it then waited for rank 0 for
over half an hour. The head never got there.

**What the head looked like from outside.** Port 22 accepted the TCP connection
and no SSH banner ever arrived, on the LAN address and on the 200 GbE address
alike, at 25 s, at 120 s and at 600 s. Its kernel was alive: ICMP over the fast
link answered in 0.2 ms. But an NFS `readdir` served by that kernel took 31.7 s,
then 105.7 s, then did not return at all. Load average peaked above 107. The
node was not dead; it was scheduling almost nothing.

**What the kernel recorded.** The ring buffer settles it. At 06:19:29, when the
global OOM killer finally fired about fifty minutes in:

```
Node 0 Normal   free 43,840 kB of 126,901,560 kB managed
active_anon 5,613,836 kB     inactive_anon 6,506,252 kB
active_file 0 kB             inactive_file 1,496 kB
shmem 2,671,356 kB           slab_unreclaimable 583,227 pages
```

Add those up and about 17 GB is accounted for. The other 104 GB is held by the
NVIDIA driver as unified memory and appears in no counter the OOM killer
scores. Four consequences, each measured:

- **The page cache was gone** — `active_file 0 kB`. That is why sshd could not
  fault its own text back in and no banner ever arrived, while ICMP, handled in
  the kernel and touching no page cache, kept answering in 0.2 ms.
- **The cgroup cap never fired.** The kill line reads
  `constraint=CONSTRAINT_NONE ... global_oom`, not a memcg OOM, even though the
  container ran under `--memory 116g`. A cgroup cap does not bound what the
  driver allocates on GB10.
- **The OOM killer had nothing worth killing.** The largest RSS in the whole
  process table was about 5 GB. Its first victim was `(sd-pam)`, a one-megabyte
  process picked for its `oom_score_adj` of 100. It needed two more passes and
  37 more seconds to reach `sglang::schedul` itself.
- **The node never swapped.** `SwapFree` did not move and the `swapents` column
  is 0 for both server processes. An earlier version of this section blamed
  swap for the stall. That was a guess made while the node was unreachable, and
  it was wrong.

**Why the memory was outstanding at all.** `docker rm -f` returned at 05:29:59,
and the previous server was still alive at 06:19. Its scheduler thread was
blocked inside the driver — `INFO: task sglang::schedul:301733 blocked for more
than 1228 seconds`, `state:D`, tgid 301520 — and SIGKILL cannot reap a task
blocked in the kernel. Two `sglang::schedul` processes in two different docker
scopes were present in the OOM dump at once. So the campaign did not merely
start too soon after a teardown; it started a second 70 GB server while the
first one was still holding its memory and could not be made to let go.

**Fixes applied.**

1. `scripts/campaign-three-models.sh` waits for `MemAvailable` to cross a floor
   on both nodes after teardown, up to four minutes, instead of sleeping ten
   seconds. `MemAvailable` is the one counter that does track driver memory —
   it read 43 MB free at the OOM — so this is the check that works.
2. Each launcher sources `scripts/lib/memguard.sh`, which checks
   `MemAvailable` after the old container is gone and the page cache is
   dropped, waits up to five minutes, and exits 4 rather than starting.
   Refusing to launch is far cheaper than wedging a node that then cannot be
   reached to fix it.
3. When that wait fails, `report_stuck_servers` names any process in state D,
   because that memory does not come back until the node reboots and a silent
   wait looks like progress.
4. `--memory` and `--memory-swap` are computed as `MemTotal` minus a named
   12 GiB host reserve instead of typed per script. This does **not** prevent
   the failure above — the cap is not what bounds driver memory — but the old
   values of 116, 112 and 110 GiB on a 121.7 GiB machine left the host as
   little as 5.7 GiB, and nothing should be that tight by accident.
5. The campaign's liveness checks run under `timeout 20`, because `docker
   inspect` on a stalled node blocks and turns a bounded health wait into an
   unbounded one.

**How it ended.** No power cycle was needed. The global OOM killer reached the
stuck server at 06:20:06, and a patient SSH from the worker over the fast link
(a 300-second connect timeout, which covers the banner exchange) got a shell at
06:20:17 and ran the recovery script.

**What is still open.** Swap is not implicated by any measurement here, so the
question of whether to keep 15 GiB of it is no longer part of this incident.
The real open item is that GB10 driver memory is invisible to both the cgroup
accounting and the OOM killer's scoring, so a runaway server on these nodes
cannot be contained by any container setting — only by not starting it.

## Incident 2026-09-03: the results database came back malformed

`results/runs.sqlite` opened with `database disk image is malformed (11)`.
`PRAGMA integrity_check` reported `btreeInitPage()` errors across 30-odd pages
of two b-trees. Last write was 2026-09-02 21:39, which puts it inside the
window of the memory-exhaustion machine deaths recorded above.

**What was recovered.** `.recover` is unavailable in this sqlite build (3.45.1,
no `sqlite_dbpage`), so recovery went through `.dump`, which reads what it can
and ends in `ROLLBACK; -- due to errors`. Rewriting that last line as `COMMIT;`
and replaying gives a clean database: **164 runs, 826 rows, integrity_check
ok**. The corrupt original is kept beside it as
`results/runs.sqlite.corrupt-20260903`.

**What was lost.** Every recovered row is `engine=llamacpp`. No SGLang or vLLM
run is in the recovered set, so either they were never recorded or they sat in
the damaged pages. The SGLang and vLLM figures in `TUNING-MATRIX.md` come from
run logs, not from this database — that is why they survived, and it is also
the gap: a number that lives only in a log is one `rm` from being unsupported.

**The fix.** `RunStore` now opens with `journal_mode=WAL` and
`synchronous=FULL`. A benchmark run takes hours and these machines can die from
memory exhaustion mid-write; the journal has to be crash-safe by construction,
not by luck.

**The habit this should change.** Every measurement that reaches
`TUNING-MATRIX.md` should also reach the database through the harness, rather
than being read off a terminal. `scripts/endpoint-ladder.py` does this; ad-hoc
probes did not.

## The results database exists in two copies, and neither is a superset

Every entry point in this repo opens `results/runs.sqlite` by a **relative**
path — `src/sparkbench/cli.py` defaults all six subcommands to it,
`scripts/endpoint-ladder.py` and `scripts/sglang-ladder.py` hardcode it, and
so do the campaign scripts. So the machine that runs a phase writes its own
copy. Phases driven from the Mac against a remote server landed in the Mac's
copy; phases driven on the head landed in the head's.

Measured on 2026-09-03, before any merge:

| | runs | metrics |
|---|---|---|
| head `/home/station/dgx-spark-bench/results/runs.sqlite` | 170 | — |
| Mac `results/runs.sqlite` | 314 | 1840 |
| shared ids | 164 | |
| head only | 6 | 90 |
| Mac only | 150 | |

The six the head alone held were sglang qwen ladder rows from 21:22:03 to
21:23:53. The 150 the Mac alone held were the entire llama.cpp ladder for all
three models, the vllm GLM ladder, the whole optuna study, and every quality
run. **Neither copy was a superset**, so copying either one over the other
would have destroyed real measurements — and the HTML report reads one
database, so it would have shown a clean table with whole phases missing and
no sign that anything was gone.

Merge with `scripts/merge-runs-db.py`, which inserts by run id, inserts
metrics by `(run_id, name)`, and resolves a genuine conflict by the later
`finished_at` while printing every such decision. Take the source snapshot
with sqlite's own `.backup` rather than `scp` on a live file; a benchmark can
be writing to it.

**Run the merge before generating any report.** After the merge above:
320 runs, 1930 metrics.

## The two ranks were launched with different configurations (2026-09-03)

A two-node SGLang server is one server. If the two ranks disagree about how
the model is sharded, the launch is not a slower or worse measurement -- it
is not a measurement.

`bring_up()` in `scripts/lib/campaign.sh` started rank 1 with

    ssh "$WORKER" "cd $REPO && git pull -q; bash $launcher 1"

and rank 0 with a plain `bash "$launcher" 0` in the local shell. Every tuning
knob in this project travels as an environment variable -- `SG_SPEC`,
`SG_RADIX`, `SG_QSA_PATCH`, `GLM_EP`, `GLM_CTX` -- and ssh carries no
environment. So a knob written in front of a `bring_up` call reached rank 0
through the local shell and stopped at the ssh boundary.

The first run of round two set `GLM_EP=on` to test whether expert parallelism
avoids the NVFP4 packed-shard bug. Rank 0 came up tagged `TP0 EP0` and loaded
its weights past the point where the previous attempt had failed, which read
as the theory being right. Rank 1 was still on plain tensor parallelism and
died on the original error:

    RuntimeError: The size of tensor a (512) must match the size of tensor b
    (1024) at non-singleton dimension 1

Rank 0 then hung holding 109 GiB waiting for a rank that was gone. Nothing
about expert parallelism was learned; the run was discarded and repeated.

Two things make this worth writing down rather than just fixing. The failure
looked exactly like the hypothesis being wrong, so the natural next step
would have been to abandon a correct idea. And it was only visible by reading
the worker's log -- the head's log showed a healthy load.

The fix derives the forwarded set from the launchers' variable prefixes
instead of listing names, so a knob added later travels without anyone
remembering this function, and prints the chosen environment at launch:

    ######## rank 1 environment: GLM_EP=on

Verified after the fix by reading the flag back off the worker's own
container rather than trusting the launcher: `docker inspect sglang_glm53`
on the worker contains `--ep-size 2`.

Nothing measured before this date is affected. Round one set no knobs in
front of `bring_up`, and `scripts/finish-longctx.sh`, which launches its two
ranks by hand, exports its knobs (`export SG_MEM SG_TOTAL SG_SPEC`) so rank 0
inherits them.

## GLM-5.3-Flash on SGLang: the sparse-attention kernel does not fit GB10 (2026-09-03)

With expert parallelism on both ranks the NVFP4 weights load cleanly, so the
loader problem recorded above is settled. GLM then dies at the first decode
step instead, inside the sparse-attention path:

```
File ".../sglang/kernels/ops/attention/dsa/tilelang_kernel.py", line 1417,
  in tilelang_sparse_fwd
tvm.error.InternalError: Failed to set the allowed dynamic shared memory
  size to 169984
```

That number is the whole story. The kernel asks the driver for 169,984 bytes
of dynamic shared memory per block. Measured on this GB10 with the CUDA driver
API (`cuDeviceGetAttribute`, no context needed):

| Attribute | Value |
|---|---|
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 101,376 bytes (99.0 KiB) |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 102,400 bytes (100.0 KiB) |
| compute capability | 12.1 |

The request is 1.68 times what the hardware allows, so the driver refuses it.
169,984 bytes sits just under Hopper's 227 KiB per-block budget, which is the
size this kernel was written against. No flag makes a kernel fit a shared
memory budget it overruns by two thirds; the only way past it is a different
sparse-attention backend.

The launcher already carries the knob (`GLM_DSA`, `scripts/launch-glm-sglang.sh`),
and its comment had guessed this outcome before it happened: "tilelang has no
sm121 build of its own and may fall back or fail here."

Which backend to try next was settled without spending a model load. The build
accepts nine values for `--dsa-prefill-backend` / `--dsa-decode-backend`:
`flashmla_sparse`, `flashmla_sparse_q8`, `flashmla_kv`, `flashmla_auto`,
`flashinfer_sparse_mla`, `fa3`, `tilelang`, `aiter`, `trtllm`. Test-importing
the packages behind them inside the image, with no GPU attached, rules out six:

| Package | In `lmsysorg/sglang:glm-5.3-flash` | Rules out |
|---|---|---|
| `flash_mla` | missing | the four `flashmla_*` backends |
| `aiter` | missing (AMD ROCm anyway) | `aiter` |
| `tensorrt_llm` | missing | `trtllm` |
| `flashinfer` | present, 0.6.17 | — |
| `flash_attn` | present | — |
| `tilelang` | present, 0.1.12 | the one that overruns |

`flashinfer_sparse_mla` and `fa3` are the two survivors, and both are real code
paths in `dsa_backend.py` rather than stubs (13 and 29 references). FlashInfer
is the first candidate because it ships Blackwell kernels; FA3 is Hopper-first
and is the fallback.

## What `power.draw` measures on GB10, and what it does not (2026-09-03)

The agreed tie-break for equally fast configurations is tokens per joule. The
only energy source on these machines is `nvidia-smi --query-gpu=power.draw`,
which the 1 Hz sampler in `scripts/telemetry.sh` records. That number is **GPU
domain power, not whole-module and not wall power**, so every energy figure in
this project must carry that label.

Measured on the head node with `nvidia-smi -q -d POWER`:

```
GPU Power Readings
    Average Power Draw        : 12.01 W
    Instantaneous Power Draw  : 12.18 W
    Current Power Limit       : N/A
GPU Memory Power Readings
    Average Power Draw        : N/A
Module Power Readings
    Average Power Draw        : N/A
    Current Power Limit       : N/A
```

`Module Power Readings` is the field that would cover the whole GB10 package —
the 20 CPU cores and the unified memory alongside the GPU — and the driver
reports it as `N/A`. Memory power is `N/A` too. So the GPU figure omits the
CPU cores that run the scheduler and the tokenizer, and omits the memory
system that a long-context run leans on hardest.

There is no second source to fall back on. `/sys/class/power_supply` is empty,
and a search of `/sys/bus/i2c/devices` and `/sys/devices/platform` for
`in*_input`, `power*_input` and `curr*_input` returns nothing — GB10 exposes
none of the INA-style rail sensors that Jetson boards carry. A wall meter is
out of scope by the owner's ruling.

The sensor itself works and tracks load. Measured on the head node while
DeepSeek-V4-Flash served its 131k-token ladder step:

| state | power.draw | SM clock | GPU util |
|---|---|---|---|
| idle, model loaded | 12.0 W | 2398 MHz | ~50% |
| prefill at 131k tokens | 47.9 W | 2444 MHz | 96% |

So the reading is live, not stuck. It is also small: 48 W at full utilization
is a minority of what the whole module draws, which is the clearest evidence
that the CPU cores and the memory system sit outside this number.

Consequences, and they are worth stating plainly:

- Tokens per joule here means **tokens per GPU joule**. It is sound for
  comparing two configurations on the same machines, which is exactly what the
  tie-break needs, and it is not comparable to any published whole-system
  number.
- The absolute figure understates energy use, and understates it unevenly:
  llama.cpp keeps more work on the CPU than SGLang does, so a CPU-heavy engine
  looks better on this metric than it would on wall power.

The same command also confirms, again, that every power limit field is `N/A`.
That is the measured basis for the earlier finding that GB10 offers no
power-cap or clock-lock knob: power is measured on these machines, never tuned.
