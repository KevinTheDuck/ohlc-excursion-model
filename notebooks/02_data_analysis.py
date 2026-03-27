# %%
import polars as pl

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
print(df.select(["DateTime", "Session"]).head(5))

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
