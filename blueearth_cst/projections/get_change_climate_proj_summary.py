"""Coordinate-safe in-memory combination of aligned change-factor datasets."""

import xarray as xr


def preprocess_coords(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess function to remove unwanted coords, and stop string TRUNCATION.

    The string coords arrive as numpy fixed-width dtypes (`<U13`, `<U19`, …) whose
    width is set by the longest value in *that one file*. Concatenating files with
    different widths silently truncates every value to the FIRST file's width:
    `NOAA-GFDL/GFDL-ESM4` (19) became `NOAA-GFDL/GFD` (13) whenever an
    `INM/INM-CM4-8` file was merged first. Silent, because a truncated model name
    is still a plausible-looking string.

    It corrupted the wide summary's `model` coordinate, and through it the tidy
    table's `model` column — and, because the per-combination window lookup is
    keyed on that name, the truncated rows also missed their effective-window
    override and fell back to the run-level one. Found at S8-04 when the column
    read `GFD`; the truncation predates it and was already in the `dataset` column
    of the wide summary.

    Casting to object dtype makes each value independent of the others' lengths.
    """
    coords_to_remove = ["height"]
    for coord in coords_to_remove:
        if coord in ds.coords:
            ds = ds.drop_vars(coord)
    for name, coord in ds.coords.items():
        if coord.dtype.kind in ("U", "S"):
            ds = ds.assign_coords({name: coord.astype(object)})
    return ds


def combine_changes(datasets):
    """Combine basin-scalar results, preserving xarray's concatenate-all policy."""
    return xr.combine_by_coords(
        [preprocess_coords(ds) for ds in datasets],
        data_vars="all",
        coords="minimal",
        compat="no_conflicts",
        join="outer",
        combine_attrs="override",
    )


# NOTE: this module no longer runs as a Snakemake `script:`. Step 4d merged rules
# 2.04/2.05 into `derive_change_factors`, which imports the functions above and
# owns the orchestration. The former `__main__` block is deleted rather than left
# dead: it was a second copy of the per-point procedure, and two copies of the
# same arithmetic is how they drift apart. The functions stay here — they are the
# tested surface (tests/test_get_change_climate_proj*.py).
