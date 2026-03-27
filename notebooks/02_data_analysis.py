# %%
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns

from ohlc_dss_model.data.integrity import sort_data
from ohlc_dss_model.utils.dt_utils import (
    convert_to_timezone,
    intraday_session_tagging,
    session_tagging,
)

# %%
data_path = "../data/raw/NAS100_30m.parquet"

# %%
df = pl.read_parquet(data_path)
df = convert_to_timezone(df)
print(df.head(10))

# %%
# The data is inverted so we must resort them so it goes from earliest -> latest
# This will make our job easier later on
df = sort_data(df)
print(df.head(5))

# %%
# We will then check for any duplicated rows
print(df.filter(df.is_duplicated()))

# %%
# Next we will handle session tagging which we will use to seperate different trading days
# Or simply we just define which row belongs to a daily candle
df = session_tagging(df)

# since these dates doesnt have a complete day (cutoff from the dataset) we will ommit these
df = df.filter(
    (pl.col("Session") != date(2016, 11, 15)), (pl.col("Session") != date(2025, 10, 1))
)
print(df.select(["DateTime", "Session"]).head(5))
print(df.select(["DateTime", "Session"]).tail(5))

# %%
# Next we will also need to tag intraday session for each candle such as asia london and new york
df = intraday_session_tagging(df)
print(df.select(["DateTime", "Intraday_Session"]).head(10))

# %%

descriptive_stats = df.with_columns(
    (pl.col("Close") / pl.col("Close").shift(1)).log().alias("Log_Return_Close"),
    (pl.col("High") / pl.col("Low")).log().alias("Candle_Range"),
)
print(
    descriptive_stats.select(
        ["DateTime", "Session", "Intraday_Session", "Log_Return_Close", "Candle_Range"]
    ).tail(10)
)

# %%
descriptive_analysis = descriptive_stats.group_by("Intraday_Session").agg(
    pl.col("Log_Return_Close").std().alias("Std_Deviation"),
    pl.col("Log_Return_Close").skew().alias("Skewness"),
    pl.col("Log_Return_Close").kurtosis().alias("Kurtosis"),
    pl.col("Candle_Range").mean().alias("Mean_Candle_Range"),
    pl.col("Candle_Range").max().alias("Max_Candle_Range"),
)

print(descriptive_analysis.filter(pl.col("Intraday_Session") != "Closed"))
# What we can get from this data? both std deviation and mean candle ranges for new york
# are ~2 times the amount of london and asia these proofs Andersen-Bollerslev intraday periodicity hypothesis
# The return distribution of the dataset across all session astronomical values (86, 170, 193) shows clear
# leptokurtic behavior this means there are events such as the black swan events frequently present
# meaning that simple linear models are insufficient.

# %%
# we will do violin plot to better visualize cnadle ranges, but we have a problem
# we see that our maximum candle range is such a big number in comparison to the mean it would dwarf
# the rest of the candles in the plot so we have to essentially treat these as an outlier and ommit them
# in the plotting process, we can ommit them by just filtering the 99 precentile
limit = descriptive_stats["Candle_Range"].quantile(0.99)
candle_range_plot = descriptive_stats.filter(pl.col("Candle_Range") < limit)
candle_range_plot = candle_range_plot.to_pandas()

plt.figure(figsize=(12, 8))

sns.violinplot(
    data=candle_range_plot,
    x="Intraday_Session",
    y="Candle_Range",
    order=["Asia", "London", "New York"],
    inner="quart",
    density_norm="width",
    palette="dark:#5A9_r",
    hue="Intraday_Session",
)

plt.title(
    "Distribution of 30 Minute Candle Ranges Aggregated by Intraday Sessions (NAS100)"
)
plt.ylabel("Candle Range (High/Low)")
plt.xlabel("Intraday Session")
plt.show()
# Based on this we can clearly see the volatility differences between each session, where new york session
# shows clear lead where the belly are much higher and wider, also it has much higher median compared to other session
# whereas asia and london are much more squashed, this essentially just tells us that new york have more candles that are larger.
