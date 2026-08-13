"""Streamflow metrics — pure pandas, one value per water year.

Every annual reduction here takes an ``anchor``: the pandas resample rule for
the water year in force, produced by
``snake_utils.water_year_end_anchor(shared.water_year_start)``.

It is a PARAMETER rather than a module constant because these are the response
surface. A Jan-Dec year splits a flood season that crosses New Year across two
years and understates the annual maximum, which is exactly what a water year
exists to prevent — so the basin's own year has to reach the arithmetic, not a
default someone forgot.

``YE-DEC`` (a January water year) is identical to the bare ``YE`` these
functions used before, so the default changes no recorded number.
"""

from blueearth_cst.shared.snake_utils import DEFAULT_WATER_YEAR_ANCHOR


## High flows
def Q7d_maxyear(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    return df.rolling(7).mean().resample(anchor).max().mean()


def Q7d_total(df):
    return df.rolling(7).mean()


def highpulse(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    return df[df > df.quantile(0.75)].resample(anchor).count().mean()


def wetmonth_mean(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    """Mean flow in the wettest month, averaged over years.

    Takes ``anchor`` for uniformity, but is water-year INVARIANT by
    construction: a calendar month never straddles a water-year boundary, so
    the chosen month's per-year mean is the same however the year is cut. Only
    which EDGE years are complete can differ. Kept as a parameter rather than
    dropped so every metric here has one signature.
    """
    monthlysum = df.groupby(df.index.month).sum()
    wetmonth = monthlysum.idxmax().iloc[0]
    df_wetmonth = df[df.index.month == wetmonth]
    return df_wetmonth.resample(anchor).mean().mean()


## Low flows
def Q7d_min(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    return df.rolling(7).mean().resample(anchor).min().mean()


def lowpulse(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    return df[df < df.quantile(0.25)].resample(anchor).count().mean()


def drymonth_mean(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    """Mean flow in the drytest month, averaged over years.

    Takes ``anchor`` for uniformity, but is water-year INVARIANT by
    construction: a calendar month never straddles a water-year boundary, so
    the chosen month's per-year mean is the same however the year is cut. Only
    which EDGE years are complete can differ. Kept as a parameter rather than
    dropped so every metric here has one signature.
    """
    monthlysum = df.groupby(df.index.month).sum()
    drymonth = monthlysum.idxmin().iloc[0]
    df_drymonth = df[df.index.month == drymonth]
    return df_drymonth.resample(anchor).mean().mean()


def BFI(df, anchor=DEFAULT_WATER_YEAR_ANCHOR):
    Q7d = df.rolling(7).mean().resample(anchor).min()
    annmean = df.resample(anchor).mean()
    return (Q7d / annmean).mean()
