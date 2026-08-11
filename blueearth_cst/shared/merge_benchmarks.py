"""Merge a workflow's per-rule Snakemake benchmark TSVs into one Markdown table.

Snakemake writes one benchmark TSV per rule (per job) under
``<parts_dir>/<W.NN_rule>[/…].tsv`` (TSV is Snakemake's fixed benchmark
format). This gather step (one job per workflow) collects that workflow's parts
into a single readable ``benchmarks/wf<W>_benchmarks.md`` — a Markdown table with
a ``rule`` column and a bold ``TOTAL`` row — regenerated fresh each run.

WF1 and WF2 both pass ``benchmarks/_parts``, so parts are filtered by the ``W.``
numbering prefix on the first path component under ``parts_dir``. WF3 passes
``benchmarks/_parts/<experiment>`` instead: its part names are rule numbers,
identical across experiments, so a shared directory would let one experiment's
stranded part be merged into another's table and deleted with it.

That nests WF3's parts INSIDE the directory WF1 and WF2 share, which is safe
only because the prefix filter is applied to the glob before anything else uses
it: WF1's gather sees ``<experiment>/3.16_….tsv`` as first component
``<experiment>``, which does not start with ``1.``, so those parts are neither
rowed nor deleted. The filtered list is the SAME list ``_remove_parts`` deletes —
keep it that way, or a filter that gates only the table would have WF1's gather
silently eating WF3's parts.
"""

import glob
import os

import pandas as pd

from blueearth_cst.shared.snake_utils import _log_header_lines

# How the TOTAL row aggregates each Snakemake benchmark column.
_SUM = ["s", "io_in", "io_out", "cpu_time"]  # additive across jobs
_MAX = ["max_rss", "max_vms", "max_uss", "max_pss"]  # peak across jobs
_MEAN = ["mean_load"]  # average across jobs

# Legend appended under the table. Snakemake's benchmark columns are a fixed set
# (psutil-sampled); this explains what each means and how to read it.
_LEGEND = """\

**How to read this** — one row per rule (fan-out rules aggregate all their jobs).
The **TOTAL** row *sums* the time / IO / CPU columns but takes the **peak** of the
memory columns, since peak usage in different rules does not happen at once.

| column | meaning | interpretation |
| --- | --- | --- |
| `s` / `h:m:s` | wall-clock runtime (seconds / h:m:s) | the headline cost of each step |
| `max_rss` | peak physical RAM held (MB) | the "does it fit in memory?" number |
| `max_vms` | peak virtual address space (MB) | usually far larger than RSS and rarely actionable |
| `max_uss` | peak RAM unique to the process (MB) | memory reclaimed if it died — a tight lower bound |
| `max_pss` | peak RAM incl. a share of shared pages (MB) | Linux-only; on Windows a proxy equal to `max_uss` |
| `io_in` / `io_out` | disk read / written (MB) | spots IO-bound steps; may under-report on Windows |
| `mean_load` | average CPU load (%), 100 ≈ one core busy | >100 ⇒ used multiple cores; well under 100 ⇒ waited on IO |
| `cpu_time` | total CPU seconds (user + system) | vs `s`: cpu_time ≫ s ⇒ parallel, ≈ s ⇒ single-threaded |

Metrics come from Snakemake's `benchmark:` sampler. On Windows `max_pss` mirrors
`max_uss` and the `io_*` columns can under-report (psutil limitation).
"""


def _hms(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def merge_benchmarks(parts_dir, workflow_num, out_path):
    """Concatenate workflow ``workflow_num``'s benchmark parts + a TOTAL row."""
    prefix = f"{workflow_num}."
    tsvs = sorted(
        p
        for p in glob.glob(os.path.join(parts_dir, "**", "*.tsv"), recursive=True)
        if os.path.relpath(p, parts_dir)
        .replace("\\", "/")
        .split("/")[0]
        .startswith(prefix)
    )
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    frames = []
    for path in tsvs:
        rule = os.path.splitext(os.path.relpath(path, parts_dir))[0].replace("\\", "/")
        df = pd.read_csv(path, sep="\t")
        df.insert(0, "rule", rule)
        frames.append(df)

    if not frames:  # no parts (e.g. nothing ran)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(  # same provenance header as rule logs, rendered for Markdown
                _log_header_lines(
                    out_path, kind="benchmark", time_label="generated", markdown=True
                )
            )
            handle.write(
                f"# wf{workflow_num} benchmarks\n\n(no benchmark parts found)\n"
            )
        return

    merged = pd.concat(frames, ignore_index=True)
    total = {"rule": "TOTAL"}
    for col in _SUM:
        if col in merged:
            total[col] = round(merged[col].sum(skipna=True), 4)
    for col in _MAX:
        if col in merged:
            total[col] = merged[col].max(skipna=True)
    for col in _MEAN:
        if col in merged:
            total[col] = round(merged[col].mean(skipna=True), 4)
    if "s" in total and "h:m:s" in merged.columns:
        total["h:m:s"] = _hms(total["s"])

    merged = pd.concat([merged, pd.DataFrame([total])], ignore_index=True)
    merged.loc[merged["rule"] == "TOTAL", "rule"] = "**TOTAL**"  # emphasise the total
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(  # same provenance header as rule logs, rendered for Markdown
            _log_header_lines(
                out_path, kind="benchmark", time_label="generated", markdown=True
            )
        )
        handle.write(f"# wf{workflow_num} benchmarks\n\n")
        handle.write(merged.to_markdown(index=False, floatfmt=".2f"))
        handle.write("\n")
        handle.write(_LEGEND)

    _remove_parts(tsvs, parts_dir)


def _remove_parts(tsvs, parts_dir):
    """Delete merged benchmark parts and prune now-empty part subdirs.

    The merged ``.md`` is the durable artifact; the per-rule TSV parts are
    scratch. Only ``tsvs`` (already prefix-filtered to this workflow) are
    removed — WF1 and WF2 share ``_parts`` and WF3 sits in a subdir of it — and
    directory pruning only ever removes *empty* dirs. A shared root is therefore
    removed only after its last workflow part is merged, and WF1 pruning an
    emptied ``_parts/<experiment>/`` is harmless: the next WF3 run recreates it.
    Note: a later *partial* re-run regenerates parts only for the rules that
    re-ran, so its merged table reflects just those (a full run stays complete).
    """
    for path in tsvs:
        try:
            os.remove(path)
        except OSError:
            pass
    for root, _dirs, _files in os.walk(parts_dir, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        merge_benchmarks(sm.params.parts_dir, str(sm.params.workflow_num), sm.output[0])
