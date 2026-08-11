"""Readable per-variable tables derived from wflow's ``output.csv``.

Wflow writes ONE csv, at full float precision, with columns in its own internal
order and ISO-8601 timestamps: ``time,Q_101,Q_1040,Q_1020,...``. That file is not
ours to reformat -- ``plot_results.py`` reads it back through
``WflowSbmModel.output_csv``, so its layout is an interface with hydromt's reader
(AGENTS.md: do not re-engineer how hydromt handles data).

So this DERIVES rather than rewrites. The raw file stays byte-for-byte as wflow
wrote it; the tables here are a product, and being a product is what makes the
per-variable split free: the variable moves into the filename, which is what lets
the columns carry bare station ids without ``Q_1`` becoming ambiguous against a
positional outlet (the ambiguity ``shared/gauges.py`` exists to prevent).

Three differences from the raw file, each fixing something measured on a real
basin (``C:/TESTS/CST/gabon_1008``, 2026-08-10):

* **Five decimal places.** Values carried up to 21 characters
  (``0.00343887581965451``). Sub-millimetre precision on discharge from an
  uncalibrated global-data model is noise, and it cost 58% of the file. The cap
  is on DECIMAL PLACES rather than significant digits, so wflow's spin-up
  values around 1e-39 read as ``0`` instead of as forty leading zeros -- see
  ``_format_value``.
* **No scientific notation.** 502 of 7670 rows held values like ``9.86e-5``.
  Excel renders those as ``9.87E-05``, which is a display most people then have
  to fight. The decimal cap removes the need for it entirely.
* **Timestamps Excel parses.** ``2000-01-02T00:00:00`` imports as TEXT, not a
  date, because of the ``T``. A space makes it a datetime everywhere.

Columns are sorted numerically, not lexically: as text, ``1010`` sorts before
``101``, and station ids are integers.
"""

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

#: Wflow names a mapped column ``<header>_<id>``; see the vendored guide
#: (docs/wflow-user-guide/03-toml-file.md, the `[[output.csv.column]]` section).
_COLUMN = re.compile(r"^(?P<var>.+?)_(?P<station>\d+)$")

#: Decimal places kept in a derived table. Five is the owner ruling: it puts a
#: hard floor under the noise wflow emits during spin-up, so a 1e-39 recharge
#: reads as `0` rather than as forty leading zeros.
DEFAULT_DECIMALS = 5


def _format_value(value, decimals: int) -> str:
    """Render one value as plain decimal text, capped at ``decimals`` places.

    DECIMAL PLACES, not significant digits. Significant digits keep every value
    at constant relative precision, which sounds better and reads worse here:
    wflow's recharge spins up through ~1e-39, and five significant digits of
    that is ``0.000000000000000000000000000000000000001054`` -- 44 characters of
    floating-point noise, in a table whose whole purpose is being readable. A
    value that small IS zero at any hydrological scale, and a decimal cap says
    so (owner ruling, 2026-08-10).

    The cost, stated because it is real: anything under 5e-6 renders as ``0``.
    On this basin discharge runs 1e-5..1e-1 and recharge -5.1..38.6, so nothing
    meaningful is lost -- but a catchment with genuinely microscopic flows would
    want `decimals` raised, which is why it stays a parameter.

    Trailing zeros are trimmed, so the column is as narrow as its values allow
    rather than padded to a fixed width.
    """
    if pd.isna(value):
        return ""
    text = f"{float(value):.{decimals}f}"
    # Catches the underflow case AND normalises "-0.00000" to a bare "0".
    if float(text) == 0:
        return "0"
    return text.rstrip("0").rstrip(".") if "." in text else text


def slugify(variable: str) -> str:
    """Filename-safe form of a wflow column's variable part.

    Wflow headers carry the SEMANTIC label verbatim, spaces included -- a config
    asking for ``groundwater recharge`` yields columns named
    ``groundwater recharge_basavg_101``. Left alone that produces a filename with
    a space in it, which contradicts naming.md and is awkward on every shell.
    """
    return re.sub(r"[^a-z0-9]+", "_", str(variable).lower()).strip("_")


def split_columns(columns) -> Dict[str, List[str]]:
    """Group ``<var>_<station>`` column names by variable, stations sorted.

    Columns that do not match the grammar (``time``, and any ``<var>_basavg``
    aggregate, which is per-subcatchment rather than per-station) are left out:
    they have no station id to key on, so they cannot join a per-station table.
    """
    grouped = {}
    for name in columns:
        match = _COLUMN.match(str(name))
        if not match:
            continue
        grouped.setdefault(match["var"], []).append(
            (int(match["station"]), str(name))
        )
    return {
        var: [name for _, name in sorted(pairs)]
        for var, pairs in sorted(grouped.items())
    }


def tidy_tables(
    frame: pd.DataFrame, decimals: int = DEFAULT_DECIMALS
) -> Dict[str, pd.DataFrame]:
    """One Excel-ready frame per variable, keyed by the variable name.

    The index is the formatted timestamp; columns are bare station ids in
    numeric order. Values are strings, deliberately: the rounding has to survive
    being written, and a float column would be re-expanded by pandas on write.
    """
    time_col = frame.columns[0]
    stamps = pd.to_datetime(frame[time_col]).dt.strftime("%Y-%m-%d %H:%M:%S")

    tables = {}
    for var, names in split_columns(frame.columns).items():
        data = {
            _COLUMN.match(name)["station"]: [
                _format_value(v, decimals) for v in frame[name]
            ]
            for name in names
        }
        table = pd.DataFrame(data)
        table.insert(0, "time", stamps.to_numpy())
        tables[var] = table
    return tables


def write_tidy_tables(
    csv_path, out_dir, prefix: str = "output", decimals: int = DEFAULT_DECIMALS
) -> List[Path]:
    """Read wflow's csv and write ``<prefix>_<var>.csv`` per variable.

    Returns the paths written, sorted. A run whose csv holds no
    ``<var>_<station>`` column writes nothing and returns an empty list rather
    than raising -- a model configured with only basin-average outputs is
    unusual but not wrong.
    """
    frame = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for var, table in tidy_tables(frame, decimals).items():
        path = out_dir / f"{prefix}_{slugify(var)}.csv"
        table.to_csv(path, index=False)
        written.append(path)

    # Drop tables this run did NOT produce, so the set on disk is a function of
    # the current config rather than the union of every config ever run. Rule
    # 1.14b declares only the discharge table, so Snakemake cannot clean the
    # rest: renaming the recharge header left `output_groundwater_recharge_
    # basavg.csv` (0.32 MB) beside its replacement, holding the same numbers
    # under the retired name (measured 2026-08-10). The glob needs the
    # underscore -- `output.csv` is wflow's own and must never be touched here.
    keep = {p.name for p in written}
    for stale in out_dir.glob(f"{prefix}_*.csv"):
        if stale.name not in keep:
            stale.unlink()

    return sorted(written)


if __name__ == "__main__":
    # Snakemake `script:` entry point: reads snakemake.input/output, never argv.
    sm = snakemake  # noqa: F821 - injected by Snakemake
    write_tidy_tables(sm.input.csv_path, Path(sm.output.q_table).parent)
