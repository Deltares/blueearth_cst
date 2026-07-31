"""Merge a workflow's per-rule log parts into ONE log for the whole workflow.

Every logging rule writes its log under ``logs/_parts/<W.NN>_<rule>[/<member>].log``.
This gather step -- one job per workflow, scheduled after every logging rule --
concatenates those parts, in rule order, into a single ``logs/<workflow>.log``,
then deletes the parts it merged.

Shape of the merged file (the same pattern ``merge_benchmarks.py`` applies to the
benchmark tables):

- ONE provenance header at the top, from ``snake_utils._log_header_lines``. The
  per-part headers are **stripped**: a near-identical three-line block repeated
  once per rule was the bulk of the old merged file and none of its information.
- One ``==`` banner per rule, carrying the same ``W.NN  name`` tag the live
  console announces (``snake_utils.rule_banner``), so a section is greppable by
  the number it ran under.
- One ``--`` sub-header per member of a fan-out rule (the ``{series_key}`` of a
  per-series part) -- those are the parts a reader has to tell apart, and the
  member id is what the stripped header used to carry.

A part listed but absent is reported rather than silently skipped, so the merged
log says which sections its run actually covers. Absent is NORMAL, not a fault:
a rule already up to date does not re-run and writes no part, and a rule shared
with another workflow (WF2's 2.11 is WF1's 1.10) usually logged over there.
"""
import os

from blueearth_cst.shared.snake_utils import _log_header_lines

WIDTH = 80
_HEADER_MARK = "# BlueEarth-CST"


def _rule_tag(label):
    """Render a part label ``2.01_fetch_gcm_raw`` as the banner tag ``2.01  fetch_gcm_raw``.

    Mirrors ``snake_utils.rule_banner`` so the merged log and the console use one
    spelling. A label that is not ``<number>_<name>`` is passed through as-is.
    """
    number, sep, name = label.partition("_")
    if sep and name and number.replace(".", "").isdigit():
        return f"{number}  {name}"
    return label


def _strip_part_header(text):
    """Drop the provenance header block a part inherited from ``tee_to_log``.

    Only strips when the part actually starts with one (``# BlueEarth-CST``):
    parts written by R scripts, or by a job that died before its first write,
    have no header and must survive untouched. Body rows are
    ``HH:MM:SS - module - LEVEL - msg`` (``_compact_log_line``), so consuming the
    leading run of ``#`` lines cannot eat log content.
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


def _sections(part_paths, parts_dir):
    """Group ordered part paths into ``[(rule_label, [(member, path), ...]), ...]``.

    A part directly in ``parts_dir`` is a whole-rule log (no member); one a level
    below it is a member of that rule's fan-out. Grouping is over *consecutive*
    runs, so the caller's order -- the Snakefile's rule order -- is what the
    merged file shows, with no re-sorting.
    """
    grouped = []
    for path in part_paths:
        parent = os.path.dirname(path)
        flat = parts_dir is not None and os.path.normpath(parent) == os.path.normpath(parts_dir)
        stem = os.path.splitext(os.path.basename(path))[0]
        if flat or parts_dir is None:
            label, member = stem, None
        else:
            label, member = os.path.basename(parent), stem
        if grouped and grouped[-1][0] == label:
            grouped[-1][1].append((member, path))
        else:
            grouped.append((label, [(member, path)]))
    return grouped


def _remove_parts(part_paths, parts_dir):
    """Delete the merged parts, then prune the now-empty part dirs.

    The merged log is the durable artifact; the parts are scratch. Only the paths
    actually listed are removed -- an orphan left by a renamed rule is neither
    merged nor deleted, so it never shows up as a phantom section. Directory
    pruning only ever removes *empty* dirs, ``parts_dir`` itself included, which
    is what makes ``logs/_parts/`` disappear on a clean full run.

    That last part DIVERGES from ``merge_benchmarks._remove_parts``, which
    pointedly keeps its ``parts_dir``: ``benchmarks/_parts/`` is shared by all
    three workflows, while ``logs/_parts/`` has exactly one owner (WF1 logs flat;
    WF3's fan-out logs live under ``<exp_dir>/logs/3.NN_<rule>/``). Give
    ``logs/_parts/`` a second writer and that guard has to come back.
    """
    for path in part_paths:
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


def merge_logs(part_paths, out_path, parts_dir=None, remove_parts=False):
    """Write ``out_path`` as the banner-delimited merge of ``part_paths``.

    Parameters
    ----------
    part_paths : list of str
        Part logs in the order they should appear -- rule order, and within a
        fan-out rule, member order.
    out_path : str
        The merged log. Regenerated whole on every run.
    parts_dir : str, optional
        Root of the part tree. Tells a whole-rule part (directly inside it) from
        a fan-out member (one level below). ``None`` treats every part as a
        whole-rule log.
    remove_parts : bool, optional
        Delete the merged parts and prune the emptied dirs afterwards.
    """
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    rule = "=" * WIDTH
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(_log_header_lines(out_path, time_label="merged"))
        for label, members in _sections(part_paths, parts_dir):
            out.write(f"{rule}\n== {_rule_tag(label)}\n{rule}\n\n")
            for member, path in members:
                if member is not None:
                    pad = max(3, WIDTH - len(member) - 4)
                    out.write(f"-- {member} {'-' * pad}\n")
                if os.path.exists(path):
                    with open(path, encoding="utf-8", errors="replace") as f:
                        out.write(_strip_part_header(f.read()))
                else:
                    out.write("# (no part from this run — rule was already up to date)\n")
                out.write("\n")
    if remove_parts:
        _remove_parts(part_paths, parts_dir)


if __name__ == "__main__":
    if "snakemake" in globals():
        sm = globals()["snakemake"]
        merge_logs(
            list(sm.params.parts),
            sm.output[0],
            parts_dir=sm.params.parts_dir,
            remove_parts=True,
        )
