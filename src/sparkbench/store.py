"""The results database, and the rule that a result is a six-field tuple.

Rule 1 in CLAUDE.md: a result records model (checkpoint *and* revision),
weight, KV cache dtype, engine (with version and image), decoding users, and
the measured KV pool in tokens. A number missing any field must not sit in a
table beside one that has them.

This module refuses to store the number at all. `RunTuple.validate()` raises on
any empty field, so an incomplete run cannot enter the database and therefore
cannot reach a report. That is the point: the previous attempt kept the rule in
prose and broke it.

Two operational lessons from `docs/baselines.md` are also built in.

* The database opens `journal_mode=WAL` with `synchronous=FULL`. A run takes
  hours and these machines can die from memory exhaustion mid-write; on
  2026-09-03 `results/runs.sqlite` came back `database disk image is malformed`
  and this sqlite build has no `.recover`. Crash safety has to be by
  construction.
* There is exactly one canonical database and it lives on the head node, at
  the path `SPARKBENCH_DB` names. The previous attempt let every entry point
  open `results/runs.sqlite` by a relative path, so the Mac and the head each
  accumulated runs the other did not have, and neither copy was a superset.
  Driving the load from the head is also the measurement-correct choice: served
  rate counts the whole wall clock, so a load generator on the far side of
  Wi-Fi would charge its own network latency to the engine.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .rates import Rate, RequestTrace

DEFAULT_DB = os.environ.get(
    "SPARKBENCH_DB", "/home/station/dgx-spark-bench/results/runs.sqlite"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    -- the six fields of the tuple, all mandatory
    model             TEXT NOT NULL,
    model_revision    TEXT NOT NULL,
    weight            TEXT NOT NULL,
    kv_cache          TEXT NOT NULL,
    engine            TEXT NOT NULL,
    engine_version    TEXT NOT NULL,
    engine_image      TEXT NOT NULL,
    users             INTEGER NOT NULL,
    kv_pool_tokens    INTEGER NOT NULL,
    -- how the KV pool and dtype were established, so 'default' is traceable
    kv_pool_source    TEXT NOT NULL,
    kv_cache_source   TEXT NOT NULL,
    -- context
    workload          TEXT NOT NULL,
    prompt_tokens     INTEGER,
    max_output_tokens INTEGER,
    nodes             INTEGER,
    host              TEXT,
    flags             TEXT,
    notes             TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id  TEXT NOT NULL,
    name    TEXT NOT NULL,
    value   REAL,
    unit    TEXT,
    PRIMARY KEY (run_id, name),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS requests (
    run_id        TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    ttft_ms       REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    ok            INTEGER,
    error         TEXT,
    PRIMARY KEY (run_id, idx),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model, users);
"""

# Every field the tuple demands, with the message shown when it is missing.
REQUIRED = {
    "model": "the checkpoint, not the family name",
    "model_revision": "the exact revision; audit it with scripts/audit-checkpoint.py",
    "weight": "the weight quantization as measured, e.g. NVFP4 (block 16)",
    "kv_cache": "the KV cache dtype written out; 'default' only with kv_cache_source saying what the default was",
    "engine": "the engine name",
    "engine_version": "the engine version string reported by the build itself",
    "engine_image": "the container image and tag",
    "users": "how many requests were in flight",
    "kv_pool_tokens": "the pool in tokens, read from the running server",
    "kv_pool_source": "how the pool was read, so the number is traceable",
    "kv_cache_source": "flag / observed / engine-default",
    "workload": "which workload produced these tokens",
}


class IncompleteTuple(ValueError):
    """Raised instead of storing a number that is not a result."""


@dataclass
class RunTuple:
    model: str
    model_revision: str
    weight: str
    kv_cache: str
    engine: str
    engine_version: str
    engine_image: str
    users: int
    kv_pool_tokens: int
    kv_pool_source: str
    kv_cache_source: str
    workload: str
    prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    nodes: int | None = None
    flags: str | None = None
    notes: str | None = None

    def validate(self) -> None:
        missing = []
        for f, why in REQUIRED.items():
            v = getattr(self, f)
            if v is None or (isinstance(v, str) and not v.strip()):
                missing.append(f"  {f}: {why}")
            if f in ("users", "kv_pool_tokens") and isinstance(v, int) and v <= 0:
                missing.append(f"  {f}: must be a positive measured integer, got {v}")
        if missing:
            raise IncompleteTuple(
                "this is not a result -- Rule 1 fields are missing or empty:\n"
                + "\n".join(missing)
            )


class RunStore:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        # See the module docstring: the 2026-09-03 corruption is why both.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def record(
        self,
        rt: RunTuple,
        rates: list[Rate],
        traces: list[RequestTrace] | None = None,
        extra: dict[str, tuple[float, str]] | None = None,
        started_at: float | None = None,
    ) -> str:
        rt.validate()
        run_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start_iso = (
            datetime.fromtimestamp(started_at, timezone.utc).isoformat(timespec="seconds")
            if started_at
            else now
        )
        d = asdict(rt)
        d.update(id=run_id, started_at=start_iso, finished_at=now, host=socket.gethostname())
        cols = ", ".join(d)
        marks = ", ".join("?" for _ in d)
        self.db.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(d.values()))

        for r in rates:
            # The Rate carries its own formula name, so the column cannot be
            # mislabelled by a call site: "served_rate" is written by served().
            self.db.execute(
                "INSERT OR REPLACE INTO metrics (run_id, name, value, unit) VALUES (?,?,?,?)",
                (run_id, f"{r.kind}_rate", r.value, "tok/s"),
            )
        for r in rates[:1]:
            self.db.execute(
                "INSERT OR REPLACE INTO metrics (run_id, name, value, unit) VALUES (?,?,?,?)",
                (run_id, "wall_clock", r.wall_s, "s"),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO metrics (run_id, name, value, unit) VALUES (?,?,?,?)",
                (run_id, "output_tokens", r.output_tokens, "tokens"),
            )
        for name, (value, unit) in (extra or {}).items():
            self.db.execute(
                "INSERT OR REPLACE INTO metrics (run_id, name, value, unit) VALUES (?,?,?,?)",
                (run_id, name, value, unit),
            )
        for i, t in enumerate(traces or []):
            self.db.execute(
                "INSERT OR REPLACE INTO requests "
                "(run_id, idx, started_at, finished_at, ttft_ms, input_tokens, "
                " output_tokens, ok, error) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, i, t.started_at, t.finished_at, t.ttft_ms, t.input_tokens,
                 t.output_tokens, int(t.ok), t.error),
            )
        self.db.commit()
        return run_id

    def close(self) -> None:
        self.db.close()
