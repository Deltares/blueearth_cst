"""User-selected precision for retained scalar response tables."""


def normalize_csv_output(output: object) -> dict[str, int]:
    """Resolve a closed output.csv block; five decimal places are the default."""
    if output is None:
        return {"decimal_places": 5}
    if not isinstance(output, dict) or set(output) - {"csv", "netcdf_grid"}:
        raise ValueError("output accepts only csv and netcdf_grid")
    csv = output.get("csv", {})
    if not isinstance(csv, dict) or set(csv) - {"decimal_places"}:
        raise ValueError("output.csv accepts only decimal_places")
    places = csv.get("decimal_places", 5)
    if type(places) is not int or not 0 <= places <= 15:
        raise ValueError("output.csv.decimal_places must be an integer from 0 to 15")
    return {"decimal_places": places}
