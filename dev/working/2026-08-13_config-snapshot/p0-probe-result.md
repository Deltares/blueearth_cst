# P0 probe result — Snakemake lifecycle handlers

**Date:** 2026-08-13 · **Brief:** `p0-probe.task.md` · **Design under test:** `design-v3.md` §5.2 (implementation step 0), §5.7

## Verdict

**The journal contract in §5.2 does NOT hold as written. STOP at Gate 1.**

The falsifier the brief told us to hunt for is present: **a "Nothing to be done"
no-op appends no line at all** — not the terminal `onsuccess` line the design
calls the contract, and not the best-effort `onstart` line either. Snakemake
fires lifecycle handlers only when the scheduler executes **at least one job**.

This is not a tuning problem. It is an unguarded early `return` in the pinned
Snakemake, quoted below, and no CLI flag reaches past it.

## Environment

| Item | Value |
|---|---|
| Snakemake | **9.6.2** (`pixi run snakemake --version`) |
| Pin | `pixi.lock` → PyPI wheel `snakemake-9.6.2-py3-none-any.whl` |
| Platform | win-64, `WSU-5CG4013KNZ` |
| Probe | `.tmp/p0-probe/Snakefile` (throwaway, deleted after the run) |

## Results

Handlers append one JSON line each to `journal.jsonl`, a path **no rule
declares as an output** — mirroring the real contract (§5.7's silent-truncation
finding). The invocation id is minted once at parse time in a module-level
binding all three handlers close over, as §5.7 specifies.

| # | Invocation | `onstart` | `onsuccess` | `onerror` |
|---|---|---|---|---|
| A | Fresh — outputs absent, 3 jobs execute | **fired** | **fired** | — |
| B | **No-op — "Nothing to be done"** (ran twice) | **NOT fired** | **NOT fired** | — |
| C | Failing rule (`exit 1`) | **fired** | — | **fired** |
| D | `--dry-run` with jobs pending | **NOT fired** | **NOT fired** | — |
| E | DAG-build failure (`MissingInputException`) | **NOT fired** | — | **NOT fired** |
| F | `--forcerun` of an up-to-date target | **fired** | **fired** | — |

Rows A–D are the brief's required 4×3. **E and F are additions**, and both
carry a finding — see below. Journal line count moved 0 → 2 → 2 → 2 → 4 → 4 → 6
across the legs, so every "NOT fired" is a measured absence, not an unread file.

### Against the design's two predictions

- (a) *"handlers fire on a normal invocation and on a 'Nothing to be done'
  no-op"* — **half true.** Normal: yes (A, C, F). No-op: **no** (B).
- (b) *"none fires under `--dry-run`"* — **true** (D).

### Mechanism

`snakemake/workflow.py:1375-1377`, on the non-dry path:

```python
                else:
                    logger.info(NOTHING_TO_BE_DONE_MSG)
                    return
```

The `else` belongs to `if len(self.dag):` (line 1313) — the DAG length is the
count of jobs needing execution. When it is zero, Snakemake logs the message and
**returns**, twenty lines before `self._onstart(...)` at 1395-1396 and seventy
before `self._onsuccess(...)` at 1446. The return is unconditional: no setting,
no CLI flag and no exec mode guards it. The only related flag is `--no-hooks`
(`cli.py:1512`), which disables handlers further.

Leg E fails even earlier — `MissingInputException` is raised while the DAG is
being built, before this block is reached at all.

## What this costs the design

§5.2 assigns hooks three cases that params threading cannot cover:

| Case §5.2 assigns to hooks | Covered? |
|---|---|
| the failed invocation | **yes** (C) — but only once a job has started; a DAG-build failure records nothing (E) |
| the no-op invocation | **no** (B) |
| the same-config re-run | **no** — with everything up to date this *is* the no-op of B |

So hooks cover exactly "an invocation in which at least one job executed", which
is the set params threading already covers by writing `run_record.yml`. **The
mechanism as designed adds outcome and failure information, but does not add the
invocation coverage it was introduced for.** Requirement R5's "record every
invocation" is unmet for the case where the answer is "nothing needed doing" —
which, on a mature project tree, is the common case.

Leg E is a second, smaller gap the brief did not enumerate: a Snakefile whose
rule declares an output no script writes fails at DAG build, and the journal
would not record that class of failure. The master brief names exactly this
failure mode as a live risk for P4.

## Scope of the blockage

- **Blocks P4** (Snakefile wiring — the hooks are wired there) and the §5.2 /
  §5.7 text, plus design open item 10.
- **Does not block P1, P2, P3.** `append_journal_line()` is a caller-agnostic
  helper; only *who calls it, and when* is in question. Rows A, C and F show the
  handler plumbing itself works and the undeclared-path append accumulates
  correctly across invocations (§5.7's mandatory accumulation test passes here:
  six lines from three successful and one failed invocation).

No workaround is proposed, per the brief and Gate 1.

## Exact commands

All run with `.tmp/p0-probe/` as the working directory, so the repo-root
`profiles/default/config.yaml` (`quiet: reason`) is not auto-discovered and
`.snakemake/` stays inside the throwaway tree.

```bash
cd .tmp/p0-probe
rm -f journal.jsonl out_*.txt && rm -rf .snakemake

# A — fresh
pixi run snakemake all -c 1 --config mode=ok
# B — no-op (identical command, run twice)
pixi run snakemake all -c 1 --config mode=ok
pixi run snakemake all -c 1 --config mode=ok
# D — dry-run with jobs pending
rm -f out_b.txt
pixi run snakemake all -c 1 --config mode=ok --dry-run
# C — failing rule
pixi run snakemake all -c 1 --config mode=fail
# E — DAG-build failure
pixi run snakemake all -c 1 --config mode=missinginput
# F — forced re-execution of an up-to-date target
pixi run snakemake all -c 1 --config mode=ok --forcerun make_a
```

A `--dry-run` against an already up-to-date tree was also run before leg D; it
prints "Nothing to be done" and appends nothing, but it cannot separate the
dry-run rule from the no-op rule, which is why leg D deletes an output first.
