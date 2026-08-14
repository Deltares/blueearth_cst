"""Merge a workflow's per-rule log parts into ONE log for the whole workflow.

Every logging rule writes its log under ``<logs>/_parts/<W.NN>_<rule>[/<member>].log``.
This gather step -- one job per workflow, scheduled after every logging rule --
concatenates those parts, in rule order, into a single ``<logs>/<workflow>.log``,
then deletes the parts it merged. All three workflows use it:

=========================== ==================================================
``build_model.smk``         ``logs/wf1_build_model.log`` (1.17)
``analyze_projections.smk`` ``logs/wf2_analyze_projections.log`` (2.09)
``run_stress_test.smk``     ``logs/wf3_run_stress_test_<experiment>.log`` (3.18)
=========================== ==================================================

Shape of the merged file (the same pattern ``merge_benchmarks.py`` applies to the
benchmark tables):

- ONE provenance header at the top, from ``snake_utils._log_header_lines``. The
  per-part headers are **stripped**: a near-identical three-line block repeated
  once per rule was the bulk of the old merged file and none of its information.
- One ``==`` banner per rule, carrying the same ``W.NN  name`` tag the live
  console announces (``snake_utils.rule_banner``), so a section is greppable by
  the number it ran under.
- One ``--`` sub-header per member of a fan-out rule (WF2's ``{series_key}``,
  WF3's ``rlz_N_st_M`` / ``batch_N``) -- those are the parts a reader has to tell
  apart, and the member id is what the stripped header used to carry.

The caller passes an ordered list of RULE LABELS, not part paths, and members are
discovered by listing the rule's part dir. Deriving the member set a second time
in the Snakefile would mean re-deriving WF3's fan-out arithmetic (``ST_START``,
``RLZ_NUM``, the ``_batches`` split) somewhere it could drift from the rules that
actually own it. Scoping discovery to the label list is what keeps this from
being a blind glob: an orphan dir left by a renamed rule (``test_local`` still
holds ``2.04_monthly_change/``) is not a label, so it is never read and never
deleted.

A rule with no part is reported rather than silently skipped, so the merged log
says which sections its run actually covers. Absent is NORMAL, not a fault: a
rule already up to date does not re-run and writes no part, and a rule shared
between workflows (WF2's 2.11 is WF1's 1.10) usually logged over there.
"""

import os
import re

from blueearth_cst.shared.snake_utils import _log_header_lines

WIDTH = 80
_HEADER_MARK = "# BlueEarth-CST"
_RULE_NUMBER = re.compile(r"^\d+\.\d+[a-z]?$")  # 2.01, and WF3's 3.00b
_DIGITS = re.compile(r"(\d+)")


def _rule_tag(label):
    """Render a part label ``2.04_fetch_gcm_slice`` as the banner tag ``2.04  fetch_gcm_slice``.

    Mirrors ``snake_utils.rule_banner`` so the merged log and the console use one
    spelling. A label that is not ``<W.NN>_<name>`` is passed through as-is.
    """
    number, sep, name = label.partition("_")
    if sep and name and _RULE_NUMBER.match(number):
        return f"{number}  {name}"
    return label


def _natural_key(text):
    """Sort key that orders digit runs numerically: ``rlz_2`` before ``rlz_10``.

    WF3 fans out to ``RLZ_NUM x ST_NUM`` members and batches them, so plain
    lexicographic order would interleave ``rlz_10`` into the middle of the single
    digits -- in exactly the workflow with the most members to read.
    """
    return [int(p) if p.isdigit() else p for p in _DIGITS.split(text)]


def _strip_part_header(text):
    """Drop the provenance header block a part inherited from ``tee_to_log``.

    Only strips when the part actually starts with one (``# BlueEarth-CST``):
    parts written by R scripts, or by a job that died before its first write,
    have no header and must survive untouched. Body rows are
    ``HH:MM:SS - module - msg`` (``_compact_log_line``, which shows the level
    only when it is not INFO), so consuming the leading run of ``#`` lines
    cannot eat log content.
    """
    if not text.startswith(_HEADER_MARK):
        return text
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    if i < len(lines) and not lines[i].strip():
        i += 1  # the blank line the header block ends with
    return "".join(lines[i:])


def _members(parts_dir, label):
    """Return ``[(member_id, path), ...]`` for one rule label, in natural order.

    Three cases, probed on the filesystem rather than inferred from the rule name:
    ``<parts_dir>/<label>.log`` is a single-job rule (member ``None``);
    ``<parts_dir>/<label>/`` is a fan-out; neither means the rule left no part.

    The fan-out walk is RECURSIVE and derives the member id from the path
    relative to the label dir, so a wildcard carrying a ``/`` -- a CMIP6
    ``{model}`` is ``NOAA-GFDL/GFDL-ESM4`` (WF2 overview obs. 8) -- reads as one
    member of the right rule instead of vanishing or landing in its own section.
    """
    flat = os.path.join(parts_dir, f"{label}.log")
    if os.path.isfile(flat):
        return [(None, flat)]
    rule_dir = os.path.join(parts_dir, label)
    if not os.path.isdir(rule_dir):
        return []
    found = []
    for root, _dirs, files in os.walk(rule_dir):
        for name in files:
            if not name.endswith(".log"):
                continue
            path = os.path.join(root, name)
            member = os.path.relpath(path, rule_dir).replace(os.sep, "/")
            found.append((os.path.splitext(member)[0], path))
    return sorted(found, key=lambda item: _natural_key(item[0]))


def _remove_parts(paths, parts_dir):
    """Delete the merged parts, then prune the now-empty part dirs.

    The merged log is the durable artifact; the parts are scratch. Only the paths
    actually merged are removed, so an orphan dir from a renamed rule is left
    exactly as found -- deleting what this run does not own is not this rule's
    call. Directory pruning only ever removes *empty* dirs, ``parts_dir`` itself
    included, which is what makes ``logs/_parts/`` disappear on a clean full run.

    Pruning ``parts_dir`` is safe here only because a caller's ``parts_dir``
    holds nothing another run owns. WF1 and WF2 do share
    ``<project_dir>/logs/_parts``, so each prunes a dir the other may be about to
    write -- harmless, because pruning removes only EMPTY dirs and each run
    recreates what it needs. WF3 is given its own
    ``<project_dir>/logs/_parts/<experiment>``, which is stronger: its part names
    are rule numbers, identical across experiments, so a shared dir would let one
    experiment's stranded part be merged into another's log and deleted with it.

    ``merge_benchmarks._remove_parts`` pointedly does NOT prune its own
    ``parts_dir`` -- the same reasoning reaching a different answer, since all
    three workflows write into one ``benchmarks/_parts/``.
    """
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
    if not parts_dir:
        return
    for root, _dirs, _files in os.walk(parts_dir, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass


def merge_logs(rules, out_path, parts_dir, remove_parts=False):
    """Write ``out_path`` as the banner-delimited merge of every rule's log parts.

    Parameters
    ----------
    rules : list of str
        Rule labels (``"2.04_fetch_gcm_slice"``) in the order their sections should
        appear. Rule-number order, matching the rule map and the benchmark table.
    out_path : str
        The merged log. Regenerated whole on every run.
    parts_dir : str
        Root of the part tree, ``<logs>/_parts``.
    remove_parts : bool, optional
        Delete the merged parts and prune the emptied dirs afterwards.

    Notes
    -----
    A leftover member part -- ``rlz_5_st_1`` after ``realizations_num`` drops to
    3 -- IS merged, since it sits in a live rule's dir. It is then deleted with
    the rest, so the condition shows in one log and heals itself; the same trade
    ``merge_benchmarks`` makes for a renamed rule's row (``dev/followups-archive.md``
    R7-9). Not worth teaching this script the current wildcard ranges.
    """
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    bar = "=" * WIDTH
    merged = []
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(_log_header_lines(out_path, time_label="merged"))
        for label in rules:
            out.write(f"{bar}\n== {_rule_tag(label)}\n{bar}\n\n")
            members = _members(parts_dir, label)
            if not members:
                out.write("# (no part from this run — rule was already up to date)\n\n")
                continue
            for member, path in members:
                if member is not None:
                    pad = max(3, WIDTH - len(member) - 4)
                    out.write(f"-- {member} {'-' * pad}\n")
                with open(path, encoding="utf-8", errors="replace") as f:
                    out.write(_strip_part_header(f.read()))
                out.write("\n")
                merged.append(path)
    if remove_parts:
        _remove_parts(merged, parts_dir)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        merge_logs(
            list(sm.params.rules),
            sm.output[0],
            sm.params.parts_dir,
            remove_parts=True,
        )
