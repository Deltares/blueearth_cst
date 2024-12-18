import xarray as xr
from xclim.indices.stats import frequency_analysis


## High flows
def returninterval(df, T):
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs["units"] = "m3/d"
    Q_interval = frequency_analysis(ds, t=T, dist="genextreme", mode="max", freq="YS")
    df_interval = xr.Dataset.to_dataframe(Q_interval)
    return df_interval.transpose().iloc[:, 0]


def returnintervalmulti(df):
    ds = xr.Dataset.from_dataframe(df)
    ds.attrs["units"] = "m3/d"
    all_T = [2, 5, 10, 20, 50, 100, 200]
    Q_rps = frequency_analysis(ds, t=all_T, dist="genextreme", mode="max", freq="YS")
    return Q_rps


def Q7d_maxyear(df):
    return df.rolling(7).mean().resample("YE").max().mean()


def Q7d_total(df):
    return df.rolling(7).mean()


def highpulse(df):
    return df[df > df.quantile(0.75)].resample("YE").count().mean()


def wetmonth_mean(df):
    monthlysum = df.groupby(df.index.month).sum()
    wetmonth = monthlysum.idxmax().iloc[0]
    df_wetmonth = df[df.index.month == wetmonth]
    return df_wetmonth.resample("YE").mean().mean()


## Low flows
def returninterval_Q7d(df, T):
    df7D = df.rolling(7).mean()
    ds = xr.Dataset.from_dataframe(df7D)
    ds.attrs["units"] = "m3/d"
    Q_interval = frequency_analysis(ds, t=T, dist="genextreme", mode="min", freq="YS")
    df_interval = xr.Dataset.to_dataframe(Q_interval)
    return df_interval.transpose().iloc[:, 0]


def Q7d_min(df):
    return df.rolling(7).mean().resample("YE").min().mean()


def lowpulse(df):
    return df[df < df.quantile(0.25)].resample("YE").count().mean()


def drymonth_mean(df):
    monthlysum = df.groupby(df.index.month).sum()
    drymonth = monthlysum.idxmin().iloc[0]
    df_drymonth = df[df.index.month == drymonth]
    return df_drymonth.resample("YE").mean().mean()


def BFI(df):
    Q7d = df.rolling(7).mean().resample("YE").min()
    annmean = df.resample("YE").mean()
    return (Q7d / annmean).mean()
