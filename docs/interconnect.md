# The container could not see the RDMA devices, and it cost 1.74x

Measured 2026-09-04 on Qwen3.8-Flash-Next. One variable changed between the two
sweeps; everything else — checkpoint, revision, quantization, KV dtype, pool,
flags, image, load generator — is identical, and both sets are in
`results/runs.sqlite` tagged `control` and `rdma`.

## What was wrong

These nodes are joined back-to-back at 200 GbE and carry four RoCE devices.
`docs/baselines.md` measured 109 Gb/s with `ib_write_bw` on the host during
provisioning, so the fabric was known to work. It was not reaching the
containers.

```
host:      /dev/infiniband/{uverbs0,uverbs1,uverbs2,uverbs3,rdma_cm}   present
container: /dev/infiniband                                             ABSENT
container: /sys/class/infiniband -> rocep1s0f0 rocep1s0f1 ...          present
```

The sysfs entries being visible is what makes this hard to catch. NCCL
enumerates four RoCE HCAs, finds them all listed exactly as expected, and then
cannot open a single verbs character device — so it falls back to TCP sockets
over the same wire. At the default log level it reports neither the attempt nor
the fallback. Nothing in the server log, no warning, no degraded-mode notice.
The link is fast either way, so bandwidth-style checks look fine.

## What it cost

Served rate, this project's formula, at every cell of the campaign:

| workload | prompt tok | users | NCCL over TCP | NCCL over RoCE | ratio |
|---|---:|---:|---:|---:|---:|
| short | 74 | 1 | 10.4 | **16.8** | 1.62x |
| short | 74 | 6 | 42.3 | **74.8** | 1.77x |
| code | 87 | 1 | 10.0 | **17.4** | 1.74x |
| code | 87 | 6 | 41.0 | **72.8** | 1.78x |
| prose | 81 | 1 | 9.6 | **17.0** | 1.77x |
| agentic-4k | ~4,000 | 1 | 9.3 | **15.7** | 1.69x |
| agentic-4k | ~3,900 | 6 | 30.0 | **50.9** | 1.70x |
| agentic-33k | ~30,300 | 1 | 5.9 | **10.9** | 1.85x |
| agentic-33k | ~30,200 | 6 | 11.6 | **20.4** | 1.76x |

Zero failed requests in either sweep. Needle retrieval 100% in both, so nothing
was traded away for the speed.

The ratio holds from 74-token prompts to 30,000-token ones and from one user to
six, which is what a fixed per-layer cost looks like: it is paid on every token
regardless of what else is going on.

## Why the interconnect and not something else

Decode over TCP ran at about 100 ms a token. The model has 48 layers, so 2.1 ms
a layer. Three things rule out the alternatives:

- **Not kernel launch overhead.** SGLang's own log reports `cuda graph: True` on
  every decode batch, so the decode path is captured.
- **Not arithmetic.** 3B active parameters across 48 layers is roughly 0.12
  GFLOP a layer, tens of microseconds at any plausible rate.
- **Not weight bandwidth.** 3B active NVFP4 parameters is ~1.5 GB a token, half
  of that per node at TP2, which at this hardware's memory bandwidth is a few
  milliseconds for the whole forward pass, not 2.1 ms per layer.

That leaves per-layer synchronisation, which is what tensor parallelism across
two machines does on every layer of every token. Confirming it took one flag:
with `NCCL_DEBUG=INFO` all eight channels now report

```
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0
Channel 00/0 : 1[0] -> 0[0] [receive] via NET/IB/0
```

against sockets before.

## The fix

`scripts/lib/rdma.sh`, used by all three launchers. Three parts, all needed:

1. Pass the verbs devices through — `--device /dev/infiniband/uverbs0` and so
   on, plus `rdma_cm`.
2. `--cap-add IPC_LOCK` and `--ulimit memlock=-1:-1`. RDMA registers pinned
   memory regions and registration fails without the capability.
3. `NCCL_IB_HCA=rocep1s0f0`, binding NCCL to the port pair that is actually
   cross-connected rather than letting it probe all four.

A host with no RDMA devices gets no arguments and runs over TCP, rather than
failing to launch.

## The habit

`scripts/check-rdma.sh` asks a running container which transport it got, and
`scripts/sweep-qwen.sh` runs it after every bring-up. It exits non-zero when
`NCCL_DEBUG` is too quiet to answer, because absence of evidence of sockets is
not evidence of RoCE — which is precisely how this survived a whole provisioning
phase that had already measured the fabric at 109 Gb/s.

## What is still unexplained

With RoCE, single-stream decode is ~57 ms a token, still about 1.2 ms a layer.
An RDMA all-reduce of a 5 KB tensor is on the order of ten microseconds, so the
interconnect no longer accounts for it. The leading candidate is
`--ple-offload-embedding`: the checkpoint carries 47.68 GiB of per-layer n-gram
embedding tables and the flag moves them to host memory, which means 48
per-layer gathers from a 2.5-million-row table off-device on every token.
`QWEN_PLE_OFFLOAD=off` exists to measure that, and the memory arithmetic says it
fits without the flag.
