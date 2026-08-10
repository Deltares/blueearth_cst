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

* **Significant digits.** Values carried up to 21 characters
  (``0.00343887581965451``). Sub-metre-per-second precision on discharge from an
  uncalibrated global-data model is noise, and it cost 58% of the file.
* **No scientific notation.** 502 of 7670 rows held values like ``9.86e-5``.
  Excel renders those as ``9.87E-05``, which is a display most people then have
  to fight.
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

#: Significant digits kept in a derived table. Not decimal places: `round(5)`
#: turns 9.86e-5 into 0.0001 and destroys small flows, while a significant-digit
#: cap is scale-invariant. Same reasoning as `export_wflow_results._format_value`.
DEFAULT_SIGNIFICANT_DIGITS = 5


def _format_value(value, digits: int) -> str:
    """Render one value as PLAIN DECIMAL text with ``digits`` significant digits.

    ``f"{v:.5g}"`` is the obvious spelling and is wrong here: it still emits
    ``9.8596e-05``. Formatting through a Decimal quantized to the significant
    figure keeps small values readable as decimals, which is the whole point.
    """
    if pd.isna(value):
        return ""
    value = float(value)
    if value == 0:
        return "0"
    from decimal import Decimal

    quantized = float(f"%.{digits}g" % value)
    text = format(Decimal(repr(quantized)), "f")
    # Decimal keeps a trailing ".0" on integral values; drop it for readability.
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
    frame: pd.DataFrame, digits: int = DEFAULT_SIGNIFICANT_DIGITS
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
                _format_value(v, digits) for v in frame[name]
            ]
            for name in names
        }
        table = pd.DataFrame(data)
        table.insert(0, "time", stamps.to_numpy())
        tables[var] = table
    return tables


def write_tidy_tables(
    csv_path, out_dir, prefix: str = "output", digits: int = DEFAULT_SIGNIFICANT_DIGITS
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
    for var, table in tidy_tables(frame, digits).items():
        path = out_dir / f"{prefix}_{slugify(var)}.csv"
        table.to_csv(path, index=False)
        written.append(path)
    return sorted(written)


if __name__ == "__main__":
    # Snakemake `script:` entry point: reads snakemake.input/output, never argv.
    sm = snakemake  # noqa: F821 - injected by Snakemake
    write_tidy_tables(sm.input.csv_path, Path(sm.output.q_table).parent)
