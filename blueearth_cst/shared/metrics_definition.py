## High flows
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
