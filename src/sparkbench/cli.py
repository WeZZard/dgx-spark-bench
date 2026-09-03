"""Command line for the harness.

    sparkbench probe    --base URL [--container NAME]
    sparkbench measure  --base URL --model-dir DIR --workload W --users N ...
    sparkbench report   [--db PATH]
    sparkbench selftest

`measure` is the only way a number enters the database, and it assembles the
six-field tuple from measurements rather than from arguments wherever it can:
the checkpoint's revision and weight come from its own safetensors headers, and
the engine, its version, the KV cache dtype and the KV pool come from the
running server. What is left for the caller to state is the container image and
the launcher flags, neither of which the server reports faithfully.

If any field cannot be established, the run is not written. That is the
intended behaviour, not a rough edge: `docs/baselines.md` records a whole phase
of this project whose figures lived only in run logs and were lost, and Rule 1
exists because a number missing a field is not comparable with anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sparkbench import checkpoint, rates, workload  # noqa: E402
from sparkbench.store import DEFAULT_DB, RunStore, RunTuple  # noqa: E402

# `load` and `probe` need `requests`, which is present on the nodes but not
# necessarily on a laptop. `report` and `selftest` are useful in both places,
# so the HTTP-touching modules are imported only by the commands that drive a
# server. Keeping the selftest runnable anywhere is the point: it is what
# checks that the project's one speed formula still says what it should.


def cmd_probe(a) -> int:
    from sparkbench import probe

    print(f"base: {a.base}")
    try:
        engine, version = probe.probe_engine(a.base)
        print(f"  engine         {engine} {version}")
    except probe.ProbeFailed as e:
        print(f"  engine         UNREADABLE: {e}")
        return 1
    dtype, dsrc = probe.probe_kv_cache_dtype(a.base, a.container, a.ssh_host)
    print(f"  kv cache dtype {dtype}   [{dsrc}]")
    try:
        pool, psrc = probe.probe_kv_pool(a.base, a.container, a.ssh_host)
        print(f"  kv pool        {pool:,} tokens   [{psrc}]")
    except probe.ProbeFailed as e:
        print(f"  kv pool        UNREADABLE\n{e}")
        return 1
    return 0


def cmd_measure(a) -> int:
    from sparkbench import load, probe

    desc = checkpoint.describe(a.model_dir)
    if not desc["revision"] and not a.model_revision:
        print(f"refusing to measure: no revision found under {a.model_dir} and "
              f"--model-revision not given. Rule 1 wants the exact checkpoint.",
              file=sys.stderr)
        return 2

    engine, version = probe.probe_engine(a.base)
    kv_dtype, kv_dtype_src = probe.probe_kv_cache_dtype(a.base, a.container, a.ssh_host)
    if a.kv_cache:  # a flag the launcher actually passed beats an unread default
        kv_dtype, kv_dtype_src = a.kv_cache, "launcher flag"
    pool, pool_src = probe.probe_kv_pool(a.base, a.container, a.ssh_host)

    prompts = workload.build(a.workload, a.requests or a.users, seed=a.seed)
    print(f"# {a.workload}: {len(prompts)} prompts, {a.users} users, "
          f"max_tokens={a.max_tokens}")
    print(f"# engine {engine} {version} | kv {kv_dtype} | pool {pool:,} tokens")

    t0 = time.time()
    res = load.run_load(a.base, a.served_model, prompts, a.users,
                        max_tokens=a.max_tokens, temperature=a.temperature,
                        timeout=a.timeout, ignore_eos=a.ignore_eos)
    failed = [t for t in res.traces if not t.ok]
    served = rates.served(res.traces)
    print(f"\n{served.label()}   ({served.output_tokens:,} output tokens / "
          f"{served.wall_s:.1f} s wall, {len(res.traces)} requests, "
          f"{len(failed)} failed)")

    measured = [served]
    try:
        dec = rates.decode(res.traces)
        print(f"{dec.label()}   (for prompt-growth comparison only)")
        measured.append(dec)
    except ValueError as e:
        print(f"decode rate: not available ({e})")

    prompt_tokens = None
    if res.prompt_tokens_seen:
        prompt_tokens = sum(res.prompt_tokens_seen) // len(res.prompt_tokens_seen)
        print(f"prompt tokens, as the engine counted them: {prompt_tokens:,} mean")
    if not res.usage_reported:
        print("WARNING: the engine did not return usage; output tokens are chunk "
              "counts and decode rate assumes one token a chunk")
    if res.needle_total:
        print(f"needle retrieval: {res.needle_hits}/{res.needle_total} verbatim")

    extra: dict[str, tuple[float, str]] = {"requests_failed": (len(failed), "requests")}
    ttft = rates.ttft_p50_ms(res.traces)
    if ttft is not None:
        extra["ttft_p50"] = (ttft, "ms")
    if res.needle_total:
        extra["needle_hit_rate"] = (res.needle_hits / res.needle_total, "fraction")

    rt = RunTuple(
        model=a.model_id or desc["name"],
        model_revision=a.model_revision or desc["revision"],
        weight=a.weight or desc["weight"],
        kv_cache=kv_dtype,
        engine=engine,
        engine_version=version,
        engine_image=a.engine_image,
        users=a.users,
        kv_pool_tokens=pool,
        kv_pool_source=pool_src,
        kv_cache_source=kv_dtype_src,
        workload=a.workload,
        prompt_tokens=prompt_tokens,
        max_output_tokens=a.max_tokens,
        nodes=a.nodes,
        flags=a.flags,
        notes=a.notes,
    )
    store = RunStore(a.db)
    run_id = store.record(rt, measured, res.traces, extra, started_at=t0)
    store.close()
    print(f"recorded run {run_id} in {a.db}")
    return 0 if not failed else 1


def cmd_report(a) -> int:
    import sqlite3
    db = sqlite3.connect(a.db)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT r.*, ms.value AS served, md.value AS decode
        FROM runs r
        LEFT JOIN metrics ms ON ms.run_id = r.id AND ms.name = 'served_rate'
        LEFT JOIN metrics md ON md.run_id = r.id AND md.name = 'decode_rate'
        ORDER BY r.model, r.users, r.started_at
    """).fetchall()
    if not rows:
        print(f"no runs in {a.db}")
        return 0
    hdr = ("model", "weight", "KV cache", "engine", "users", "KV pool",
           "workload", "served rate")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for r in rows:
        print("| {} | {} | {} | {} | {} | {:,} | {} | {:.1f} tok/s |".format(
            f"{r['model']} @{(r['model_revision'] or '')[:7]}",
            r["weight"], r["kv_cache"],
            f"{r['engine']} {r['engine_version']}",
            r["users"], r["kv_pool_tokens"], r["workload"],
            r["served"] if r["served"] is not None else float("nan")))
    print("\nAll figures are **served rate** (output tokens / whole wall clock).")
    return 0


def cmd_selftest(a) -> int:
    from sparkbench.rates import RequestTrace, decode, served
    ok = True

    def check(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: got {got:.6f} want {want:.6f}")

    # One request, 100 tokens, 10 s wall, of which 4 s was prefill.
    t = RequestTrace(started_at=0.0, finished_at=10.0, output_tokens=100,
                     input_tokens=2457,
                     token_times=[4.0 + i * (6.0 / 99) for i in range(100)])
    check("served rate counts prefill against the number", served([t]).value, 10.0)
    check("decode rate removes prefill", decode([t]).value, 99 / 6.0, 1e-6)

    # The two differ by more than twofold on a long prompt -- the failure that
    # produced 22.6 and 40.3 tok/s for one run in the previous attempt.
    ratio = decode([t]).value / served([t]).value
    print(f"  INFO  decode/served on this trace = {ratio:.2f}x "
          f"(they are different quantities)")

    # Concurrency: served rate is aggregate over one shared wall clock.
    a1 = RequestTrace(0.0, 10.0, 100, 10, [1.0, 9.0])
    a2 = RequestTrace(0.0, 8.0, 60, 10, [1.0, 7.0])
    check("aggregate served rate over the shared clock", served([a1, a2]).value, 16.0)

    # A failed request costs wall clock but contributes no tokens.
    bad = RequestTrace(0.0, 20.0, 0, 10, [], ok=False, error="boom")
    check("failures are not hidden", served([a1, bad]).value, 100 / 20.0)

    # A rate always carries its formula name.
    lbl = served([t]).label()
    good = "served rate" in lbl
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  a rate prints its formula: {lbl!r}")

    # The store refuses an incomplete tuple.
    from sparkbench.store import IncompleteTuple, RunTuple
    try:
        RunTuple(model="m", model_revision="r", weight="w", kv_cache="",
                 engine="e", engine_version="v", engine_image="i", users=1,
                 kv_pool_tokens=1, kv_pool_source="s", kv_cache_source="s",
                 workload="w").validate()
        print("  FAIL  an empty KV cache field was accepted")
        ok = False
    except IncompleteTuple:
        print("  PASS  an empty KV cache field is refused")
    try:
        RunTuple(model="m", model_revision="r", weight="w", kv_cache="fp8",
                 engine="e", engine_version="v", engine_image="i", users=1,
                 kv_pool_tokens=0, kv_pool_source="s", kv_cache_source="s",
                 workload="w").validate()
        print("  FAIL  a zero KV pool was accepted")
        ok = False
    except IncompleteTuple:
        print("  PASS  a zero KV pool is refused")

    # Workloads cannot degenerate.
    ps = workload.build("agentic-4k", 8)
    good = len({p.text for p in ps}) == 8
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  workload produced 8 distinct prompts")

    print("\nselftest:", "all passed" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sparkbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="ask a running server what it is")
    p.add_argument("--base", required=True)
    p.add_argument("--container")
    p.add_argument("--ssh-host")
    p.set_defaults(fn=cmd_probe)

    m = sub.add_parser("measure", help="run a workload and record the tuple")
    m.add_argument("--base", required=True)
    m.add_argument("--served-model", default="default",
                   help="the model name the server answers to on /v1")
    m.add_argument("--model-dir", required=True,
                   help="checkpoint on disk; its revision and weight are read from it")
    m.add_argument("--model-id", help="repo id, e.g. nvidia/DeepSeek-V4-Flash-nvfp4-DSpark")
    m.add_argument("--model-revision")
    m.add_argument("--weight", help="override only if the audit cannot see it")
    m.add_argument("--kv-cache", help="the dtype the launcher actually passed")
    m.add_argument("--engine-image", required=True)
    m.add_argument("--container", help="for reading the KV pool out of the engine log")
    m.add_argument("--ssh-host")
    m.add_argument("--workload", required=True)
    m.add_argument("--users", type=int, required=True)
    m.add_argument("--requests", type=int,
                   help="total requests; defaults to --users, i.e. one full round")
    m.add_argument("--max-tokens", type=int, default=256)
    m.add_argument("--temperature", type=float, default=0.0)
    m.add_argument("--timeout", type=float, default=1800.0)
    m.add_argument("--ignore-eos", action="store_true")
    m.add_argument("--seed", type=int, default=20260904)
    m.add_argument("--nodes", type=int, default=2)
    m.add_argument("--flags")
    m.add_argument("--notes")
    m.add_argument("--db", default=DEFAULT_DB)
    m.set_defaults(fn=cmd_measure)

    r = sub.add_parser("report", help="render every recorded tuple")
    r.add_argument("--db", default=DEFAULT_DB)
    r.set_defaults(fn=cmd_report)

    s = sub.add_parser("selftest", help="check the formulas and the tuple rule")
    s.set_defaults(fn=cmd_selftest)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
