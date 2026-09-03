"""sparkbench -- the measurement harness for dgx-spark-bench.

Three rules from CLAUDE.md are enforced here in code rather than in prose,
because the previous attempt kept them in prose and broke all three:

* Rule 1, the six-field tuple: `store.RunTuple.validate()` refuses to write a
  run with an empty field, and `checkpoint.py` / `probe.py` read four of those
  fields off the checkpoint and the running server rather than accepting them
  from a launcher.
* Rule 2, one speed formula: `rates.py` is the only place either rate is
  computed, and a `Rate` carries the name of the formula that made it.
* Rule 3, never run an unproven launcher unattended: `scripts/lib/memguard.sh`
  fronts every launcher, and `preflight.py` refuses a launch that has no
  recorded successful run behind it.
"""
__all__ = ["rates", "store", "probe", "load", "workload", "checkpoint"]
