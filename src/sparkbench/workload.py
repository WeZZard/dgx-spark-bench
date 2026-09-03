"""The prompts, built deterministically and never downloaded.

Two constraints shape this module.

**It must not be able to fail silently into a degenerate corpus.** A sibling
project's prompt loader returned ten texts repeated thirty times whenever its
download failed, cached that, and produced a healthy-looking thousand-prompt
fit averaging ten distinct prompts. So nothing here fetches anything. Prompts
are synthesised from a seeded RNG, and `build()` asserts distinctness before
returning.

**The prompt length that gets recorded must be the measured one.** These
workloads target an approximate token count by construction, but the number
stored in the database is `usage.prompt_tokens` as the engine reports it. That
matters most for the short cell: `BASELINE.md` notes every published figure was
taken at a 22-token prompt, and the previous attempt compared against those
figures while sending 2,457 tokens. `short` exists to make that comparison
honest, and it is only honest if its length is read back rather than asserted.

The agentic workloads imitate what this project is actually for: a long
multi-turn context of file listings, source, diffs and tool output, ending in a
question that needs something from early in the context. That last property is
deliberate -- it keeps a prefix-cache configuration from looking good by
answering without reading, and it doubles as the needle-retrieval check.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

# Roughly four characters to a token for English prose and code alike. Only
# used to *aim*; the recorded length is always the engine's own count.
CHARS_PER_TOKEN = 4

_WORDS = (
    "scheduler allocator kernel tensor gradient checkpoint parser resolver "
    "buffer pipeline throughput latency quantization dispatch registry cache "
    "eviction residual attention embedding projection normalisation routing "
    "expert batch prefill decode token sequence context window budget"
).split()

_MODULES = ("loader", "router", "cache", "runtime", "codec", "planner",
            "sampler", "telemetry", "fabric", "store")


@dataclass(frozen=True)
class Prompt:
    text: str
    needle: str | None = None      # the exact string a correct answer must contain
    label: str = ""


def _para(rng: random.Random, n_words: int) -> str:
    w = [rng.choice(_WORDS) for _ in range(n_words)]
    w[0] = w[0].capitalize()
    return " ".join(w) + "."


def _code_block(rng: random.Random, mod: str, n_lines: int) -> str:
    out = [f"# ---- src/{mod}.py " + "-" * 40, f"class {mod.capitalize()}:"]
    for i in range(n_lines):
        f = rng.choice(_WORDS)
        out.append(f"    def {f}_{i}(self, {rng.choice(_WORDS)}):")
        out.append(f"        # {_para(rng, 6)}")
        out.append(f"        return self._{f}({rng.randint(1, 4096)})")
    return "\n".join(out)


def _tool_output(rng: random.Random, n: int) -> str:
    out = ["<tool_result name=\"grep\">"]
    for _ in range(n):
        out.append(f"  src/{rng.choice(_MODULES)}.py:{rng.randint(1, 900)}: "
                   f"{_para(rng, 8)}")
    out.append("</tool_result>")
    return "\n".join(out)


def _agentic(rng: random.Random, target_tokens: int, seed_label: str) -> Prompt:
    """A long agent context with a needle planted early and asked for at the end."""
    key = hashlib.sha256(seed_label.encode()).hexdigest()[:12].upper()
    needle = f"BUILD-KEY-{key}"
    parts = [
        "You are a coding agent working in a large repository. "
        "The transcript below is your context so far.\n",
        f"<note>The build key for this session is {needle}. "
        "Quote it exactly when asked.</note>\n",
    ]
    target_chars = target_tokens * CHARS_PER_TOKEN
    while sum(len(p) for p in parts) < target_chars:
        pick = rng.random()
        if pick < 0.45:
            parts.append(_code_block(rng, rng.choice(_MODULES), rng.randint(6, 18)))
        elif pick < 0.75:
            parts.append(_tool_output(rng, rng.randint(4, 14)))
        else:
            parts.append(_para(rng, rng.randint(40, 120)))
        parts.append("\n\n")
    parts.append(
        "\n\nQuestion: state the build key for this session exactly as it was "
        "given, then in two sentences say what the router module appears to do."
    )
    return Prompt("".join(parts), needle=needle, label=seed_label)


def build(name: str, n: int, seed: int = 20260904) -> list[Prompt]:
    """Return `n` distinct prompts for the named workload.

    Raises if the workload is unknown, or if it somehow produced duplicates --
    the check exists because a degenerate corpus is the failure that looks
    healthy.
    """
    rng = random.Random(seed)
    prompts: list[Prompt] = []

    if name == "short":
        # Matches the conditions every published figure in BASELINE.md was
        # taken under: a ~22-token instruction, no context to read. The point
        # is a like-for-like cell, so the text is deliberately plain.
        for i in range(n):
            prompts.append(Prompt(
                f"Count from 1 to 300, separated by spaces. Start now. (run {i})",
                label="short"))
    elif name == "prose":
        # The published DSpark acceptance floor: ~34% on prose. Judge agentic
        # results against this, not against "count to 300".
        for i in range(n):
            topic = rng.choice(_WORDS)
            prompts.append(Prompt(
                f"Write eight paragraphs of continuous prose about the design of a "
                f"{topic} in a distributed inference server. No lists, no headings. "
                f"(variation {i})", label="prose"))
    elif name == "code":
        for i in range(n):
            mod = rng.choice(_MODULES)
            prompts.append(Prompt(
                f"Write a complete, runnable Python module implementing a {mod} "
                f"with an LRU eviction policy, type hints, docstrings and unit "
                f"tests. Output only code. (variation {i})", label="code"))
    elif name.startswith("agentic"):
        # agentic-4k, agentic-33k, agentic-131k, agentic-262k
        suffix = name.split("-", 1)[1] if "-" in name else "33k"
        target = int(float(suffix.rstrip("k")) * 1024)
        for i in range(n):
            prompts.append(_agentic(rng, target, f"{name}-{i}"))
    else:
        raise ValueError(
            f"unknown workload {name!r}; known: short, prose, code, agentic-<N>k")

    texts = [p.text for p in prompts]
    if len(set(texts)) != len(texts):
        raise RuntimeError(
            f"workload {name!r} produced {len(texts)} prompts but only "
            f"{len(set(texts))} distinct ones -- a degenerate corpus would make "
            "every prefix-cache measurement meaningless")
    return prompts


def checks_needle(name: str) -> bool:
    return name.startswith("agentic")
