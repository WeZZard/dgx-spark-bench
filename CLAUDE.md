# dgx-spark-bench

**Read `BASELINE.md` before running anything.** It holds the published
two-node configurations this project has to reproduce, each as a complete
six-field tuple with its source.

Two machines: NVIDIA DGX Spark (GB10, sm_121), 121 GiB unified memory each,
back-to-back 200 GbE, NFS model store on the head. Hardware facts and the
traps found the hard way are in `docs/baselines.md`.

## Rule 1 — a result is a tuple, never a number

Every measurement records all six fields, and a report shows them:

**(model, weight, KV cache, engine, decoding users, KV pool)**

- **model** — the checkpoint and revision, not the family name. A vision
  variant is not the text model. `0731` is not `Vision-Exp`.
- **weight** — FP8, NVFP4, EXL3 4bpw, or the GGUF quant string.
- **KV cache** — the dtype, written out. If no flag is passed, record
  `default` and go and find out what the default actually was.
- **engine** — engine, version and image. The fixes that matter live in
  specific builds, not in the project name.
- **decoding users** — how many requests were in flight.
- **KV pool** — the measured pool in tokens, read from the running server.

A number missing any field is not comparable with anything and must not be
put in a table beside one that has them. This rule exists because the previous
attempt compared a text model on FP8 with default KV against a vision model on
NVFP4 with a 4-bit KV cache, and read the difference as a tuning gap.

## Rule 2 — one speed formula for the whole project

Two rates can be computed from any run here and they differ by more than
twofold on a long prompt:

- **served rate** — all output tokens divided by the whole wall clock, so the
  wait for the first token counts against it.
- **decode rate** — 1000 divided by the mean gap between output tokens, so the
  prompt read time is removed.

**Served rate is this project's formula.** Use it for every headline number,
every comparison and every ranking. Decode rate may appear only where the
question is how generation holds up as the prompt grows, and it is labelled
as decode rate every time it appears.

Never quote one where the other belongs. Never switch formula to make a
comparison line up. When an outside figure was measured under different
conditions — most published figures use a 22-token prompt — measure the
matching cell instead of changing the formula.

## Rule 3 — never run an unproven launcher unattended

A launcher with no successful run behind it is run while someone is watching,
with `scripts/lib/memguard.sh` in front of it. On GB10 a container memory cap
does not bound what the driver allocates; the guard's header explains why and
what the real signal is. Ignoring this hung both machines for ninety minutes
and needed a power cycle.

## Public repo

No secrets, ever. Credentials live in `~/.config/zsh/secrets.zsh` and in 0600
env files on the nodes. LAN addresses come from an untracked
`inventory.local.yml`. Reports are scrubbed of hostnames and IPs before
commit — `node-a` and `node-b` in anything published.
